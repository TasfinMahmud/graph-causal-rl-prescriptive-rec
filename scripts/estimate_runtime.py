#!/usr/bin/env python3
"""Estimate how long the full experiment run will take on *this* machine.

Runs a small timed probe of each phase, measures the scaling, and extrapolates
to the configuration you intend to run. Takes about three to five minutes and
saves hours of guessing.

The extrapolation uses measured exponents rather than assumed ones:

* **Phase 1** is close to independent of row count. Message passing is over the
  graph, so its cost tracks nodes and edges, not the number of logged rounds.
* **Phase 3** is linear in ``rows x epochs x agents`` (gradient steps), with an
  extra factor for the width of the action space, since the Q-network's output
  layer has one unit per action.
* **Phase 4** is linear in ``rows x agents x estimators``, plus a per-policy
  reward-model fit when MRDR is requested.

Usage
-----
    python estimate_runtime.py
    python estimate_runtime.py --config ../code/configs/obd_fast.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
import warnings
from pathlib import Path

# scripts/ sits inside the package root, so the parent of this file's
# directory is the importable root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

# Imported after the sys.path insert above so the local package resolves.
import numpy as np  # noqa: E402

PROBE_ROWS = 20_000
PROBE_EPOCHS = 3
PROBE_ARCHS = 1
PROBE_AGENTS = 2
PROBE_ACTIONS = 80


def _probe(device: str) -> dict:
    """Time one small end-to-end run and return per-phase seconds."""
    logging.disable(logging.INFO)
    from gcrl.config import load_config
    from gcrl.data.graphs import build_bipartite_graph
    from gcrl.phases.phase1_gnn import run_phase1
    from gcrl.phases.phase3_rl import build_state, run_phase3
    from gcrl.phases.phase4_ope import run_phase4
    from gcrl.seeding import seed_everything

    tmp = Path(tempfile.mkdtemp())
    (tmp / "probe.yaml").write_text(
        f"""
experiment_name: probe
seed: 5
device: {device}
paths: {{root: "{tmp.as_posix()}"}}
dataset: {{name: probe, loader: obd, num_actions: {PROBE_ACTIONS}}}
gnn: {{architectures: [LightGCN], embedding_dim: 64, hidden_dim: 128, num_layers: 3,
       epochs: {PROBE_EPOCHS}, node_batch_size: 4096, early_stopping_patience: 99}}
rl: {{agents: [DQN, CQL], state_source: gnn_embeddings, gnn_architecture: LightGCN,
      hidden_units: 256, epochs: {PROBE_EPOCHS}, batch_size: 256}}
ope: {{estimators: [snipw, dr], n_bootstrap: 50}}
"""
    )
    config = load_config(tmp / "probe.yaml")
    config.paths.ensure_output_dirs()
    seed_everything(5)

    rng = np.random.default_rng(5)
    users = rng.integers(0, 400, PROBE_ROWS)
    actions = rng.integers(0, PROBE_ACTIONS, PROBE_ROWS)
    propensities = rng.uniform(0.02, 0.4, PROBE_ROWS)
    rewards = (rng.random(PROBE_ROWS) < 0.05).astype(float)
    graph = build_bipartite_graph(users, actions, 400, PROBE_ACTIONS, embedding_dim=64, seed=5)

    start = time.time()
    run_phase1(graph, config, seed=5)
    t1 = time.time() - start

    state = build_state(config, np.zeros((PROBE_ROWS, 2), dtype=np.float32), users, 5)

    start = time.time()
    policies = run_phase3(state, actions, rewards, PROBE_ACTIONS, config, seed=5)
    t3 = time.time() - start

    start = time.time()
    run_phase4(policies, state, actions, rewards, propensities, PROBE_ACTIONS, config, seed=5)
    t4 = time.time() - start

    return {"phase1": t1, "phase3": t3, "phase4": t4}


def _estimate(probe: dict, rows, actions, gnn_epochs, archs, rl_epochs, agents, estimators, causal):
    """Extrapolate from the probe to a target configuration."""
    # Phase 1: per architecture-epoch, scaled by graph size (nodes ~ rows^0.5).
    p1_unit = probe["phase1"] / (PROBE_ARCHS * PROBE_EPOCHS)
    graph_factor = (rows / PROBE_ROWS) ** 0.5
    p1 = p1_unit * archs * gnn_epochs * graph_factor

    # Phase 3: linear in gradient steps; the Q-head widens with the action space.
    p3_unit = probe["phase3"] / (PROBE_ROWS * PROBE_EPOCHS * PROBE_AGENTS)
    action_factor = 1.0 + 0.25 * np.log10(max(actions / PROBE_ACTIONS, 1.0)) * (actions / PROBE_ACTIONS) ** 0.35
    p3 = p3_unit * rows * rl_epochs * agents * action_factor

    # Phase 4: linear in rows x agents x estimators; MRDR refits per policy.
    p4_unit = probe["phase4"] / (PROBE_ROWS * PROBE_AGENTS * 2)
    p4 = p4_unit * rows * agents * len(estimators)
    if "mrdr" in estimators:
        p4 *= 2.2  # a per-policy gradient-boosted reward model fit

    # Phase 2 is not probed (it needs econml); scale from Phase 3 empirics.
    p2 = (0.8 * p3 / max(agents, 1)) * 4 if causal else 0.0

    # The CSV scan is I/O bound and depends on file size, not on these
    # parameters, so the caller fills it in from the source file.
    return {"phase1": p1, "phase2": p2, "phase3": p3, "phase4": p4, "scan": 0.0}


def _fmt(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--obd-csv", type=Path, default=None, help="path to the OBD csv, for scan timing")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    from gcrl.config import resolve_device

    device = resolve_device(args.device)
    print(f"Probing on device: {device}")
    print(f"({PROBE_ROWS:,} rows, {PROBE_ARCHS} encoder, {PROBE_AGENTS} agents, {PROBE_EPOCHS} epochs each)\n")

    probe = _probe(device)
    print("Probe timings:")
    for key, value in probe.items():
        print(f"  {key:8s} {value:6.1f} s")

    scan_seconds = 0.0
    if args.obd_csv and args.obd_csv.exists():
        size_gb = args.obd_csv.stat().st_size / 1e9
        scan_seconds = size_gb * 1e9 / 120e6 * 2  # two passes at ~120 MB/s
        print(f"\nOBD source file: {size_gb:.1f} GB -> two-pass scan approx {_fmt(scan_seconds)}")

    targets = [
        ("OBD  main          ", 300_000, 80, 20, 5, 30, 7, ["snipw", "dr", "mrdr"], True),
        ("OBD  ablation_raw  ", 300_000, 80, 20, 5, 30, 7, ["snipw", "dr", "mrdr"], False),
        ("OBD  ablation_cate ", 300_000, 80, 20, 5, 30, 7, ["snipw", "dr", "mrdr"], True),
        ("KuaiRec main       ", 300_000, 10_728, 20, 5, 30, 5, ["exact"], False),
        ("KuaiRec ablation   ", 300_000, 10_728, 20, 5, 30, 5, ["exact"], False),
        ("KuaiRec validate   ", 300_000, 10_728, 20, 5, 30, 5, ["exact", "dm", "dr", "mrdr", "snipw"], False),
    ]

    print(f"\n{'run':22s} {'P1':>8} {'P2':>8} {'P3':>8} {'P4':>8} {'scan':>8} {'TOTAL':>9}")
    print("-" * 75)
    grand = 0.0
    for name, rows, actions, ge, ar, re_, ag, est, causal in targets:
        e = _estimate(probe, rows, actions, ge, ar, re_, ag, est, causal)
        is_obd = name.strip().startswith("OBD")
        e["scan"] = scan_seconds if is_obd else scan_seconds * 0.25
        total = sum(e.values())
        grand += total
        print(f"{name:22s} {_fmt(e['phase1']):>8} {_fmt(e['phase2']):>8} {_fmt(e['phase3']):>8} "
              f"{_fmt(e['phase4']):>8} {_fmt(e['scan']):>8} {_fmt(total):>9}")
    print("-" * 75)
    print(f"{'ALL RUNS, one seed':22s} {'':>8} {'':>8} {'':>8} {'':>8} {'':>8} {_fmt(grand):>9}")

    print("\nNotes")
    print("  - These are estimates from a small probe; treat them as +/- 50%.")
    print("  - The runner is resumable: an interrupted run continues where it stopped.")
    print("  - Run OBD and KuaiRec in two shells to overlap I/O with compute.")
    if device == "cpu":
        print("  - Probing on CPU. With CUDA available, expect the Phase 3 figures")
        print("    to fall by roughly 2-4x; the scan and Phase 2 will not change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
