#!/usr/bin/env python3
"""Build the result figures from the files in ``results/``.

Both figures are drawn from the result JSON rather than from transcribed
numbers, so rerunning the experiments changes the figures with the results.
Output is written as PDF and PNG.

Usage
-----
    python scripts/make_figures.py --results-dir results --out-dir docs/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

#: Colourblind-safe categorical hues (Okabe-Ito), assigned in a fixed order.
BLUE, ORANGE, GREEN, PURPLE = "#0072B2", "#E69F00", "#009E73", "#8B5FA8"
INK, MUTED, GRID = "#1a1a1a", "#555555", "#d9d9d9"

#: IEEE single-column width, in inches.
COL_W = 3.4

RC = {
    # Type 42 embeds TrueType outlines rather than matplotlib's default Type 3
    # bitmap fonts, which IEEE Xplore's PDF compliance check rejects and which
    # leave figure text unsearchable.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.linewidth": 0.6,
    "figure.dpi": 400,
}


def _save(fig, out_dir: Path, stem: str) -> None:
    for suffix in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.pdf and {stem}.png")


def value_ladder(results: Path, out_dir: Path) -> None:
    """Every policy's exact value against the three reference policies.

    A horizontal bar is the right form here: the task is magnitude comparison
    across named entities, read against fixed thresholds. The thresholds are
    drawn as rules rather than bars so they cannot be mistaken for policies,
    and they stop below the label band so that no rule crosses its own label.
    """
    main = json.loads((results / "reference_policies_kuairec_main_seed42.json")
                      .read_text(encoding="utf-8"))
    sage = json.loads((results / "reference_policies_kuairec_sage_bandits_seed42.json")
                      .read_text(encoding="utf-8"))
    ope = json.loads((results / "phase4_ope_kuairec_main_seed42.json")
                     .read_text(encoding="utf-8"))
    ref = main["reference"]
    exact = {r["agent"]: r["value"] for r in ope if r["estimator"] == "exact"}

    def entry(name, source=main):
        agent = source["agents"][name]
        # The headline run reports its exact value through Phase 4; the
        # GraphSAGE arm carries its own recomputed value.
        value = exact[name] if source is main else agent["exact_value_recomputed"]
        return value, agent["n_distinct_actions"]

    agents = [(name, *entry(name)) for name in
              ("NeuralUCB", "BCQ", "IQL", "LinUCB", "DQN", "CQL")]
    # Random is stochastic, so it has no deterministic action support to report.
    agents.append(("Random", exact["Random"], None))
    agents.append(("NeuralUCB\n(GraphSAGE)", *entry("NeuralUCB", sage)))
    agents.sort(key=lambda a: a[1])
    n = len(agents)

    fig, ax = plt.subplots(figsize=(COL_W, 2.30))
    ys = list(range(n))
    ax.barh(ys, [a[1] for a in agents], height=0.62, color=BLUE,
            edgecolor="white", linewidth=0.8, zorder=3)

    for y, (_, value, actions) in zip(ys, agents):
        label = f"{value:.3f}"
        if actions:
            label += f" ({actions} item{'s' if actions != 1 else ''})"
        if value > 0.35:
            ax.text(value - 0.028, y, label, va="center", ha="right",
                    fontsize=6.2, color="white", zorder=5)
        else:
            ax.text(value + 0.012, y, label, va="center", ha="left",
                    fontsize=6.2, color=INK)

    bar_top, low, high = n - 0.50, n - 0.32, n + 0.50
    rules = [
        (ref["train_popularity"]["highest_logged_reward_rate"]["exact_value"],
         "popularity", PURPLE, (1.5, 1.5), "right", -0.016, low),
        (ref["best_constant"]["value"], "best constant", ORANGE, (4, 2), "left", 0.016, high),
        (ref["oracle"]["value"], "oracle", GREEN, (4, 2), "center", 0.0, low),
    ]
    for x, name, colour, dash, align, dx, base in rules:
        ax.vlines(x, -0.62, bar_top, color=colour, linewidth=1.0,
                  linestyle=(0, dash), zorder=4)
        ax.annotate(f"{name}\n{x:.3f}", xy=(x + dx, base), fontsize=5.8,
                    color=colour, ha=align, va="bottom",
                    annotation_clip=False, linespacing=1.0)

    ax.set_yticks(ys)
    ax.set_yticklabels([a[0] for a in agents], fontsize=6.8)
    ax.set_xlim(0, 1.12)
    ax.set_ylim(-0.62, n + 1.30)
    ax.set_xlabel("Exact policy value (fully observed block)")
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.25)
    _save(fig, out_dir, "fig_value_ladder")


def estimator_error(results: Path, out_dir: Path) -> None:
    """Signed estimator error against the known policy value.

    Polarity is what matters, so the encoding is a zero rule with dots either
    side rather than bars rising from an arbitrary base.
    """
    rows = json.loads((results / "phase4_ope_kuairec_validate_ope_full_seed42.json")
                      .read_text(encoding="utf-8"))
    truth = {r["agent"]: r["value"] for r in rows if r["estimator"] == "exact"}
    estimators = [("snipw", "SNIPW", BLUE), ("dm", "DM", ORANGE),
                  ("dr", "DR", GREEN), ("mrdr", "MRDR", PURPLE)]
    order = ["Random", "CQL", "DQN", "LinUCB", "IQL", "BCQ", "NeuralUCB"]

    fig, ax = plt.subplots(figsize=(COL_W, 1.90))
    ax.axvline(0, color=INK, linewidth=0.9, zorder=3)
    for y, agent in enumerate(order):
        ax.axhline(y, color=GRID, linewidth=0.5, zorder=0)
        for k, (key, _, colour) in enumerate(estimators):
            value = next(r["value"] for r in rows
                         if r["agent"] == agent and r["estimator"] == key)
            ax.plot(value - truth[agent], y + (k - 1.5) * 0.17, "o", ms=4.0,
                    color=colour, markeredgecolor="white", markeredgewidth=0.6,
                    zorder=4)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=6.8)
    ax.set_xlabel("Estimate minus true value")
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v:+.2f}".replace("+0.00", "0")))

    handles = [plt.Line2D([], [], marker="o", ls="", ms=4.0, color=colour,
                          markeredgecolor="white", label=name)
               for _, name, colour in estimators]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=6.2,
              handletextpad=0.2, borderpad=0.1, labelspacing=0.2)
    fig.tight_layout(pad=0.25)
    _save(fig, out_dir, "fig_estimator_error")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RC):
        value_ladder(args.results_dir, args.out_dir)
        estimator_error(args.results_dir, args.out_dir)


if __name__ == "__main__":
    main()
