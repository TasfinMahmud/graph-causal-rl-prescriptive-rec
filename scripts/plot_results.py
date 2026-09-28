#!/usr/bin/env python3
"""Generate the paper's figures from the JSON results each phase writes.

Figures are produced *only* from result files. There is no path by which a
number reaches a plot without a run behind it, and a missing result file raises
rather than being drawn as zero.

Design decisions, and why
-------------------------
**Palette.** Slots 1 and 2 of the reference categorical palette (blue ``#2a78d6``,
orange ``#eb6834``). Validated: worst adjacent CVD separation is Delta-E 24.7
(protan), normal-vision 33.6, and both clear 3:1 contrast against the surface.
The seaborn defaults previously used here put red against green at Delta-E 7.3
under deuteranopia -- inside the floor band, and the wrong choice for a figure
that will be read by people with colour vision deficiency.

**Hatching.** IEEE proceedings are frequently printed in greyscale, where blue and
orange differ in relative luminance by a factor of only 1.48. Every multi-series
figure therefore carries a hatch pattern as secondary encoding, so the series
remain distinguishable with no colour at all.

**Recessive chrome.** Thin spines, dashed low-alpha gridlines on the value axis
only, and direct value labels on bars, so the reader does not trace back to an
axis.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Validated categorical slots. Assigned in fixed order, never cycled.
COLOR_PRIMARY = "#2a78d6"
COLOR_SECONDARY = "#eb6834"
HATCH_PRIMARY = ""
HATCH_SECONDARY = "///"

INK = "#1b1f24"
MUTED = "#6b7680"
GRID = "#c8d2cb"

#: Value-label type size, in points. The collision threshold is derived from it.
LABEL_FONT_PT = 7.5

ESTIMATOR_LABELS = {
    "snipw": "SNIPW", "ipw": "IPW", "dm": "DM",
    "dr": "DR", "mrdr": "MRDR", "exact": "Exact policy value",
}


def _style(ax) -> None:
    """Recessive axes: no top/right spines, gridlines on the value axis only."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45, color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK)


def _label_bars(ax, bars, values, fmt: str = "{:.4f}", offsets=None) -> None:
    """Direct value labels, so the reader never traces back to the axis.

    ``offsets`` gives a per-bar vertical offset in points. Grouped charts use it
    to lift one label of a pair clear of the other; see :func:`_pair_offsets`.
    """
    for index, (bar, value) in enumerate(zip(bars, values, strict=True)):
        if not np.isfinite(value):
            continue
        ax.annotate(
            fmt.format(value),
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            textcoords="offset points",
            xytext=(0, 3.0 if offsets is None else offsets[index]),
            ha="center", va="bottom", fontsize=LABEL_FONT_PT, color=INK,
        )


def _pair_offsets(ax, a_values, b_values, min_gap_pt: float | None = None):
    """Per-pair label offsets that keep a grouped chart's labels from colliding.

    Two labels collide when the bars they sit on end at similar heights, since
    each label is wider than its bar. Heights are compared in display points --
    not data units -- because whether two labels overlap depends on the rendered
    geometry, not on the magnitude of the values.

    Only the pairs that actually collide are staggered; lifting every second
    label would push a short bar's label up toward its tall neighbour and create
    the collision it was meant to prevent.

    The threshold is derived from the type size rather than hardcoded. A value
    label is far wider than the bar it sits on -- roughly 27pt against a 22pt
    spacing between paired bar centres -- so paired labels always overlap
    horizontally, and vertical separation is the only remedy. Two labels are
    therefore treated as colliding whenever their bars end within about two line
    heights of each other.
    """
    if min_gap_pt is None:
        min_gap_pt = LABEL_FONT_PT * 1.9

    transform = ax.transData
    y0 = transform.transform((0, 0))[1]
    scale = 72.0 / ax.figure.dpi  # pixels -> points

    a_off, b_off = [], []
    for a, b in zip(a_values, b_values, strict=True):
        a_pt = (transform.transform((0, a))[1] - y0) * scale
        b_pt = (transform.transform((0, b))[1] - y0) * scale
        if abs(a_pt - b_pt) < min_gap_pt:
            # Lift the label on the TALLER bar. Lifting the lower one would
            # raise it toward its neighbour and worsen the collision.
            a_off.append(3.0 + min_gap_pt if a_pt >= b_pt else 3.0)
            b_off.append(3.0 if a_pt >= b_pt else 3.0 + min_gap_pt)
        else:
            a_off.append(3.0)
            b_off.append(3.0)
    return a_off, b_off


#: Mirrors ``scripts/generate_paper_tables.py``. Phase 4 writes
#: ``phase4_ope_{experiment_name}_seed{seed}.json`` and puts the experiment name
#: nowhere inside the rows, so the filename is the only record of which run
#: produced them. Two configs can share a ``dataset.name`` while differing by an
#: order of magnitude in scale, so the dataset is not an experiment identity.
_RESULT_NAME = re.compile(r"^phase4_ope_(?P<experiment>.+)_seed(?P<seed>-?\d+)\.json$")


#: Suffixes ``run_paper_experiments.py`` appends for the arms of one ablation.
#: Arms sharing a stem belong in one ablation figure; anything else is a
#: different experiment and gets its own figure.
VARIANT_SUFFIXES = ("_main", "_ablation_raw", "_ablation_cate")


def experiment_family(experiment: str) -> str:
    for suffix in VARIANT_SUFFIXES:
        if experiment.endswith(suffix) and len(experiment) > len(suffix):
            return experiment[: -len(suffix)]
    return experiment


def experiment_from_filename(name: str) -> str:
    match = _RESULT_NAME.match(name)
    if not match:
        raise ValueError(
            f"result file {name!r} does not follow phase4_ope_<experiment>_seed<seed>.json; "
            f"its experiment cannot be identified, and pooling it with another run would "
            f"plot a bar no run measured"
        )
    return match.group("experiment")


def load_phase4(results_dir: Path, dataset: str) -> list[dict]:
    """Load every Phase 4 row for one dataset, across variants and seeds.

    Result files are named by experiment (which encodes the variant), so the
    dataset is read from the rows rather than from the filename -- but the
    experiment is read from the filename and kept, because rows from different
    experiments must never be averaged into one bar.
    """
    rows = []
    for path in sorted(results_dir.glob("phase4_ope_*_seed*.json")):
        experiment = experiment_from_filename(path.name)
        for row in json.loads(path.read_text(encoding="utf-8")):
            if row.get("dataset") == dataset:
                row = dict(row)
                row["_experiment"] = experiment
                rows.append(row)
    if not rows:
        raise FileNotFoundError(
            f"no Phase 4 results for dataset {dataset!r} in {results_dir}. "
            f"Run the experiments first; a missing result is not plotted as zero."
        )
    return rows


def load_run_records(results_dir: Path, experiments: set[str]) -> dict:
    """Seeds ``run_paper_experiments.py`` attempted, from experiment_runs.json.

    A seed that died before Phase 4 leaves no result file, so it is invisible in
    the rows. Without this the figure draws a solid bar over the seeds that
    happened to survive while the table reports ``n/m`` -- the two disagreeing
    about the same cell.
    """
    path = results_dir / "experiment_runs.json"
    out: dict[str, dict] = defaultdict(dict)
    if not path.exists():
        return out
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return out
    for record in records if isinstance(records, list) else []:
        variant, seed = record.get("variant"), record.get("seed")
        if variant is None or seed is None:
            continue
        for experiment in experiments:
            if experiment == variant or experiment.endswith(f"_{variant}"):
                out[experiment][int(seed)] = record.get("status", "unknown")
    return out


def _aggregate(
    rows: list[dict], estimator: str, state_source: str, lam: float,
    attempted: dict | None = None,
) -> dict:
    """Per-agent statistics for one (estimator, configuration) cell.

    Every agent that was *evaluated* appears in the result, including the ones
    that produced no value. Dropping them made an agent the table prints as
    ``n.i.`` disappear from the figure with no gap and no label, so figure and
    table disagreed and only the table admitted the gap. What is returned is a
    dict per agent with:

    ``mean``/``std``   NaN when nothing was measured -- never 0.0.
    ``status``         ``"ok"``, ``"partial"``, or the status the run recorded.
    ``n_usable``/``n_attempted``  seeds behind the bar, and seeds tried.
    """
    grouped = defaultdict(list)
    for row in rows:
        if (
            row["estimator"] == estimator
            and row.get("state_source") == state_source
            and float(row.get("cate_reward_weight", 0.0)) == lam
        ):
            grouped[row["agent"]].append(row)

    out = {}
    for agent, entries in grouped.items():
        sources = sorted({e.get("_experiment", "?") for e in entries})
        if len(sources) > 1:
            # Two runs are two measurements. Averaging them produces a bar no
            # run ever measured -- the figure's version of the pooled cell.
            raise ValueError(
                f"rows for agent {agent!r} ({estimator}, {state_source}, lambda={lam}) come "
                f"from {len(sources)} different experiments ({', '.join(sources)}). They are "
                f"separate measurements and must be plotted separately, not averaged."
            )
        usable = [e for e in entries if e.get("status", "ok") == "ok"]
        statuses = sorted({
            e.get("status", "ok") for e in entries if e.get("status", "ok") != "ok"
        })
        seen = {e["seed"] for e in entries}
        run_log = {
            s for s, st in (attempted or {}).get(sources[0], {}).items()
            if st != "skipped"
        }
        n_attempted = len(seen | run_log)
        n_usable = len({e["seed"] for e in usable})
        if not usable:
            out[agent] = {
                "mean": float("nan"), "std": float("nan"),
                "status": statuses[0] if len(statuses) == 1 else "mixed_failure",
                "n_usable": 0, "n_attempted": n_attempted,
            }
            continue
        values = np.array([e["value"] for e in usable], dtype=float)
        out[agent] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "status": "ok" if n_usable == n_attempted else "partial",
            "n_usable": n_usable, "n_attempted": n_attempted,
        }
    return out


#: Short text drawn in the gap where a bar would be.
_ABSENT_MARKERS = {
    "not_identifiable": "n.i.",
    "mixed_failure": "no value",
}


def _absent_label(stats: dict) -> str:
    status = stats["status"]
    return _ABSENT_MARKERS.get(status, status.replace("_", " "))


def _draw_absent(ax, x, stats: dict, width: float, top: float) -> None:
    """Draw the gap where an agent was evaluated but produced no value.

    An empty hatched slot of fixed height, plus the reason written in it. ``top``
    is passed in rather than read from the axes so that every slot in one figure
    is identical: a slot read from a shifting ``ylim`` would differ in height
    between agents and could be mistaken for a value.
    """
    ax.bar(
        x, top, width=width, color="none", edgecolor=MUTED,
        hatch="xx", linewidth=0.7, alpha=0.5, zorder=0,
    )
    ax.annotate(
        _absent_label(stats),
        (x, top * 0.5), ha="center", va="center", rotation=90,
        fontsize=LABEL_FONT_PT, color=INK,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
              "edgecolor": "none", "alpha": 0.9},
    )


def plot_policy_values(
    results_dir: Path, dataset: str, out_dir: Path, estimator: str | None = None
) -> list[Path]:
    """Bar chart of policy value per agent, one figure per estimator.

    One figure per *experiment*: two runs of the same dataset are two different
    measurements and are never averaged into one bar.
    """
    rows = load_phase4(results_dir, dataset)
    estimators = [estimator] if estimator else sorted({r["estimator"] for r in rows})
    experiments = sorted({r["_experiment"] for r in rows})
    attempted = load_run_records(results_dir, set(experiments))
    written = []

    for est in estimators:
        # Only the experiments that actually have a main-arm cell for this
        # estimator. The filename keeps its stable form when there is one, and
        # is disambiguated only when two runs would otherwise overwrite it.
        per_experiment = {}
        for experiment in experiments:
            exp_rows = [r for r in rows if r["_experiment"] == experiment]
            cell = _aggregate(exp_rows, est, "gnn_embeddings", 0.0, attempted)
            if cell:
                per_experiment[experiment] = cell

        for experiment, stats in per_experiment.items():
            suffix = "" if len(per_experiment) == 1 else f"_{experiment}"
            # Measured bars first, by value; then the agents with no value, so
            # the gap is visible rather than the agent simply being absent.
            measured = sorted(
                (kv for kv in stats.items() if np.isfinite(kv[1]["mean"])),
                key=lambda kv: -kv[1]["mean"],
            )
            absent = sorted(kv for kv in stats.items() if not np.isfinite(kv[1]["mean"]))
            ordered = measured + absent
            agents = [a for a, _ in ordered]
            means = [s["mean"] for _, s in ordered]
            errors = [0.0 if not np.isfinite(s["std"]) else s["std"] for _, s in ordered]

            fig, ax = plt.subplots(figsize=(7.0, 3.6))
            width = 0.62
            bars = ax.bar(
                agents, [0.0 if not np.isfinite(m) else m for m in means],
                yerr=errors if any(errors) else None,
                capsize=3, color=COLOR_PRIMARY, width=width,
                error_kw={"linewidth": 1.0, "ecolor": MUTED},
            )
            # Hatch a bar that rests on fewer seeds than were attempted, so it
            # cannot be read as a settled measurement.
            for bar, (_, s) in zip(bars, ordered, strict=True):
                if s["status"] == "partial":
                    bar.set_hatch(HATCH_SECONDARY)
                    bar.set_edgecolor("white")
                    bar.set_linewidth(0.8)
            _label_bars(ax, bars, means)
            partial_any = any(s["status"] == "partial" for _, s in ordered)
            if partial_any:
                # Headroom for the seed count that sits above the value label.
                ax.set_ylim(top=ax.get_ylim()[1] * 1.10)
            for index, (_, s) in enumerate(ordered):
                if s["status"] == "partial":
                    ax.annotate(
                        f"{s['n_usable']}/{s['n_attempted']} seeds",
                        (bars[index].get_x() + width / 2, bars[index].get_height()),
                        textcoords="offset points", xytext=(0, 3 + LABEL_FONT_PT * 1.6),
                        ha="center", va="bottom", fontsize=LABEL_FONT_PT - 0.5,
                        color=INK, fontstyle="italic",
                    )
            # Freeze the scale on the measured bars before drawing any slot, so
            # every slot is the same height and none of them changes the scale.
            finite = [m for m in means if np.isfinite(m)]
            if finite and absent:
                ax.set_ylim(top=max(ax.get_ylim()[1], max(finite) * 1.12))
            slot_top = ax.get_ylim()[1]
            for index, (_, s) in enumerate(ordered):
                if not np.isfinite(s["mean"]):
                    _draw_absent(ax, index, s, width, slot_top)
            ax.set_ylim(top=slot_top)

            ax.set_ylabel(ESTIMATOR_LABELS.get(est, est.upper()), fontsize=10, color=INK)
            ax.set_xlabel("")
            if absent or partial_any:
                ax.set_title(
                    "hatched: fewer seeds than attempted   |   "
                    "crosshatched slot: evaluated, no value produced",
                    fontsize=LABEL_FONT_PT, color=MUTED, loc="left", pad=6,
                )
            _style(ax)
            plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
            fig.tight_layout()

            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"policy_values_{dataset}_{est}{suffix}.png"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            written.append(path)
            print(f"wrote {path}")
    return written


def plot_encoder_comparison(results_dir: Path, dataset: str, out_dir: Path) -> Path | None:
    """Validation BPR loss per architecture -- the held-out metric."""
    entries = []
    for path in sorted(results_dir.glob("phase1_gnn_*_seed*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries.extend(e for e in payload if e.get("dataset") == dataset)
    if not entries:
        print(f"no Phase 1 results for {dataset}; skipping encoder figure")
        return None

    grouped = defaultdict(list)
    for entry in entries:
        grouped[entry["architecture"]].append(entry["best_val_bpr_loss"])
    ordered = sorted(grouped.items(), key=lambda kv: np.mean(kv[1]))

    names = [n for n, _ in ordered]
    means = [float(np.mean(v)) for _, v in ordered]
    errors = [float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for _, v in ordered]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bars = ax.bar(
        names, means, yerr=errors if any(errors) else None, capsize=3,
        color=COLOR_PRIMARY, width=0.6, error_kw={"linewidth": 1.0, "ecolor": MUTED},
    )
    _label_bars(ax, bars, means)
    ax.set_ylabel("Validation BPR loss (lower is better)", fontsize=10, color=INK)
    _style(ax)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"encoder_comparison_{dataset}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    return path


def plot_ablation(
    results_dir: Path, dataset: str, out_dir: Path, kind: str, estimator: str | None = None
) -> Path | None:
    """Grouped bars comparing two configurations that differ in one value.

    ``kind`` is ``"gnn"`` (graph state versus raw features) or ``"cate"``
    (causal reward shaping on versus off). Output filenames are the stable names
    the paper references: ``ablation_gnn.png`` and ``ablation_cate.png``.
    """
    all_rows = load_phase4(results_dir, dataset)
    est = estimator or ("exact" if any(r["estimator"] == "exact" for r in all_rows) else
                        "snipw" if any(r["estimator"] == "snipw" for r in all_rows) else "dr")

    # An ablation compares the arms of ONE experiment family. Arms of different
    # families are different experiments and never share a figure.
    families = sorted({experiment_family(r["_experiment"]) for r in all_rows})
    written_paths = []
    for family in families:
        rows = [r for r in all_rows if experiment_family(r["_experiment"]) == family]
        suffix = "" if len(families) == 1 else f"_{family}"
        path = _plot_one_ablation(
            rows, dataset, out_dir, kind, est, suffix,
            load_run_records(results_dir, {r["_experiment"] for r in rows}),
        )
        if path is not None:
            written_paths.append(path)
    return written_paths[0] if len(written_paths) == 1 else (written_paths or None)


def _plot_one_ablation(
    rows: list[dict], dataset: str, out_dir: Path, kind: str, est: str, suffix: str,
    attempted: dict | None = None,
) -> Path | None:
    """Draw the ablation figure for the arms of one experiment family."""
    baseline = _aggregate(rows, est, "gnn_embeddings", 0.0, attempted)
    if kind == "gnn":
        contrast = _aggregate(rows, est, "raw_features", 0.0, attempted)
        labels = ("GNN state", "Raw features")
    else:
        lams = sorted({float(r.get("cate_reward_weight", 0.0)) for r in rows} - {0.0})
        if not lams:
            print(f"no CATE-shaped runs for {dataset}; skipping the {kind} ablation figure")
            return None
        contrast = _aggregate(rows, est, "gnn_embeddings", lams[0], attempted)
        labels = (r"$\lambda=0$", rf"$\lambda={lams[0]:g}$")

    agents = sorted(set(baseline) & set(contrast))
    if not agents:
        print(f"no overlapping agents for the {kind} ablation on {dataset}; skipping")
        return None

    x = np.arange(len(agents))
    width = 0.36
    # NaN where nothing was measured. It is never replaced by 0.0: a bar of
    # height zero is a measurement, and no measurement was made.
    a_means = [baseline[a]["mean"] for a in agents]
    b_means = [contrast[a]["mean"] for a in agents]
    a_err = [0.0 if not np.isfinite(baseline[a]["std"]) else baseline[a]["std"] for a in agents]
    b_err = [0.0 if not np.isfinite(contrast[a]["std"]) else contrast[a]["std"] for a in agents]
    a_plot = [0.0 if not np.isfinite(m) else m for m in a_means]
    b_plot = [0.0 if not np.isfinite(m) else m for m in b_means]

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    # Hatch is secondary encoding: the two series stay distinguishable when the
    # figure is printed in greyscale, where these hues differ in luminance by
    # only a factor of 1.48.
    bars_a = ax.bar(x - width / 2, a_plot, width, yerr=a_err if any(a_err) else None,
                    capsize=3, label=labels[0], color=COLOR_PRIMARY,
                    hatch=HATCH_PRIMARY, edgecolor="white", linewidth=0.8,
                    error_kw={"linewidth": 1.0, "ecolor": MUTED})
    bars_b = ax.bar(x + width / 2, b_plot, width, yerr=b_err if any(b_err) else None,
                    capsize=3, label=labels[1], color=COLOR_SECONDARY,
                    hatch=HATCH_SECONDARY, edgecolor="white", linewidth=0.8,
                    error_kw={"linewidth": 1.0, "ecolor": MUTED})
    a_off, b_off = _pair_offsets(ax, a_plot, b_plot)
    _label_bars(ax, bars_a, a_means, offsets=a_off)
    _label_bars(ax, bars_b, b_means, offsets=b_off)
    if max(max(a_off), max(b_off)) > 3.0:
        # Staggered labels need headroom above the tallest bar.
        ax.set_ylim(top=ax.get_ylim()[1] * 1.10)

    # An arm that produced no value gets a marked gap in its own half of the
    # pair, so the reader sees which side is missing rather than a bar of zero.
    absent = False
    slot_top = ax.get_ylim()[1]
    for index, agent in enumerate(agents):
        for offset, side in ((-width / 2, baseline[agent]), (width / 2, contrast[agent])):
            if not np.isfinite(side["mean"]):
                _draw_absent(ax, x[index] + offset, side, width, slot_top)
                absent = True
            elif side["status"] == "partial":
                bar_top = side["mean"]
                ax.annotate(
                    f"{side['n_usable']}/{side['n_attempted']}",
                    (x[index] + offset, bar_top), textcoords="offset points",
                    xytext=(0, 3 + LABEL_FONT_PT * 1.6),
                    ha="center", va="bottom", fontsize=LABEL_FONT_PT - 0.5,
                    color=INK, fontstyle="italic",
                )
                absent = True
    ax.set_ylim(top=slot_top)
    if absent:
        ax.set_title(
            "n/m: fewer seeds than attempted   |   crosshatched slot: evaluated, "
            "no value produced",
            fontsize=LABEL_FONT_PT, color=MUTED, loc="left", pad=6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=20, ha="right")
    ax.set_ylabel(ESTIMATOR_LABELS.get(est, est.upper()), fontsize=10, color=INK)
    legend = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for text in legend.get_texts():
        text.set_color(INK)
    _style(ax)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ablation_{kind}{suffix}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--estimator", default=None, help="restrict to one estimator")
    parser.add_argument(
        "--figures", nargs="+", default=["policy", "encoder", "ablation"],
        choices=["policy", "encoder", "ablation"],
    )
    args = parser.parse_args()

    out_dir = args.figures_dir or (args.results_dir / "figures")

    if "policy" in args.figures:
        plot_policy_values(args.results_dir, args.dataset, out_dir, args.estimator)
    if "encoder" in args.figures:
        plot_encoder_comparison(args.results_dir, args.dataset, out_dir)
    if "ablation" in args.figures:
        plot_ablation(args.results_dir, args.dataset, out_dir, "gnn", args.estimator)
        plot_ablation(args.results_dir, args.dataset, out_dir, "cate", args.estimator)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
