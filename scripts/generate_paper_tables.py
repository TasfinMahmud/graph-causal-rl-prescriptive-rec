#!/usr/bin/env python3
"""Turn result files into paper-ready LaTeX.

Everything emitted here is read from a result artefact or an effective-config
file that a run wrote. Nothing is typed in. That is the point: it removes the
transcription step where the paper and the code drift apart, and it means the
implementation-details paragraph describes the run that actually happened rather
than the run that was intended.

Outputs (written to --out-dir, default results/paper):
    table_ope.tex             main results, mean +/- std across seeds
    table_datasets.tex        dataset statistics
    implementation.tex        the Implementation Details paragraph
    claims_check.md           which of the paper's claims the numbers support

Usage
-----
    python scripts/generate_paper_tables.py --results-dir results
    python scripts/generate_paper_tables.py --results-dir results --datasets obd kuairec
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ESTIMATOR_LABELS = {
    "snipw": "SNIPW",
    "ipw": "IPW",
    "dm": "DM",
    "dr": "DR",
    "mrdr": "MRDR",
    "exact": "Exact",
}

AGENT_ORDER = ["Random", "LinUCB", "NeuralUCB", "IQL", "DQN", "CQL", "BCQ"]


def _escape(text: str) -> str:
    """Escape the LaTeX specials that appear in agent and dataset names."""
    for char, replacement in [("_", r"\_"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#")]:
        text = text.replace(char, replacement)
    return text


#: Suffixes that ``run_paper_experiments.py`` appends to a base config stem to
#: name an ablation arm. Arms sharing a stem are arms of ONE experiment and
#: belong in one ablation table; everything else is a separate experiment.
VARIANT_SUFFIXES = ("_main", "_ablation_raw", "_ablation_cate")

_RESULT_NAME = re.compile(r"^phase4_ope_(?P<experiment>.+)_seed(?P<seed>-?\d+)\.json$")


def experiment_from_filename(name: str) -> str:
    """Recover ``experiment_name`` from a Phase 4 result filename.

    Phase 4 writes ``phase4_ope_{experiment_name}_seed{seed}.json`` but does not
    put ``experiment_name`` inside the rows, so the filename is the only record
    of which experiment produced them. Two configs can share a
    ``dataset.name`` (``kuairec_validate_ope`` and ``kuairec_validate_ope_full``
    both say ``kuairec``) while differing in scale by an order of magnitude, so
    the dataset is not an experiment identity and must never be used as one.
    """
    match = _RESULT_NAME.match(name)
    if not match:
        raise ValueError(
            f"result file {name!r} does not follow phase4_ope_<experiment>_seed<seed>.json; "
            f"its experiment cannot be identified, and pooling it with another run would "
            f"average numbers that were never measured together"
        )
    return match.group("experiment")


def experiment_family(experiment: str) -> str:
    """The group of experiments that belong in one ablation table.

    The three arms ``X_main``, ``X_ablation_raw`` and ``X_ablation_cate`` differ
    in exactly one config value and are meant to sit side by side as columns, so
    they share a family. Any other experiment name is its own family: it is a
    different run and its numbers are not comparable cell-for-cell.
    """
    for suffix in VARIANT_SUFFIXES:
        if experiment.endswith(suffix) and len(experiment) > len(suffix):
            return experiment[: -len(suffix)]
    return experiment


#: Experiments that must never produce a table in the paper. Smoke and fast runs
#: exist to exercise the pipeline on a few thousand rows; `sage` is the
#: encoder-sensitivity check over three agents, reported as prose in the
#: diagnostics section rather than as a fourth policy-value table. Leaving them
#: in emitted a "20,000 held-out rounds" table beside the 4.7-million-round one,
#: which invites exactly the comparison the two do not support.
NON_REPORTABLE = ("smoke", "fast", "sage")


def is_reportable(experiment: str) -> bool:
    return not any(marker in experiment for marker in NON_REPORTABLE)


def load_ope_results(results_dir: Path, dataset: str) -> list[dict]:
    """Load every reportable Phase 4 result file for one dataset."""
    rows: list[dict] = []
    for path in sorted(results_dir.glob("phase4_ope_*_seed*.json")):
        experiment = experiment_from_filename(path.name)
        if not is_reportable(experiment):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload:
            # Rows carry their own dataset, so variant files can share a glob.
            if row.get("dataset") != dataset:
                continue
            row["_source"] = path.name
            row["_experiment"] = experiment
            rows.append(row)
    return rows


def load_run_records(results_dir: Path, experiments: set[str]) -> dict:
    """Seeds that ``run_paper_experiments.py`` attempted, from experiment_runs.json.

    A run that died before Phase 4 leaves no result file at all, so its seed is
    invisible to a generator that reads only result files: the cell silently
    reports fewer seeds than were run and says nothing about the rest. That
    record is the only place the attempt survives.

    Returns ``{experiment_name: {seed: status}}`` plus a special key
    ``"_unattributed"`` for failed records whose variant matches no experiment
    that produced results at all.
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
        status = record.get("status", "unknown")
        if variant is None or seed is None:
            continue
        # experiment_name is "<base stem>_<variant name>"; match on that suffix.
        matched = [e for e in experiments if e == variant or e.endswith(f"_{variant}")]
        if matched:
            for experiment in matched:
                out[experiment][int(seed)] = status
        elif status not in ("ok", "skipped"):
            out["_unattributed"][(variant, int(seed))] = record.get("error") or status
    return out


def aggregate(rows: list[dict], attempted: dict | None = None) -> dict:
    """Group by (experiment, agent, state_source, lambda, estimator).

    ``experiment`` leads the key so that two runs of the same dataset can never
    land in the same cell. Seeds are pooled only *within* one experiment, which
    is the only pooling that corresponds to a repetition of the same measurement.

    ``attempted`` maps an experiment to ``{seed: status}`` from
    ``experiment_runs.json``. Seeds listed there but absent from the results are
    counted as attempted-and-lost, so a cell can report ``n/m seeds`` honestly
    instead of silently shrinking to the seeds that happened to survive.
    """
    attempted = attempted or {}
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("_experiment", "?"),
            row["agent"],
            row.get("state_source", "?"),
            float(row.get("cate_reward_weight", 0.0)),
            row["estimator"],
        )
        grouped[key].append(row)

    out = {}
    for key, entries in grouped.items():
        experiment = key[0]
        usable = [e for e in entries if e.get("status", "ok") == "ok"]
        usable_seeds = sorted({e["seed"] for e in usable})
        seen_seeds = {e["seed"] for e in entries}

        # Seeds that were launched for this experiment but never reached Phase 4.
        run_log = {s: st for s, st in attempted.get(experiment, {}).items() if st != "skipped"}
        lost_seeds = sorted(set(run_log) - seen_seeds)
        attempted_seeds = sorted(seen_seeds | set(run_log))

        # Statuses of every seed that produced no usable value, with its reason.
        failures: dict[str, list] = defaultdict(list)
        for entry in entries:
            status = entry.get("status", "ok")
            if status != "ok":
                failures[status].append(entry["seed"])
        for seed in lost_seeds:
            failures[f"run_{run_log[seed]}"].append(seed)
        failures = {k: sorted(v) for k, v in sorted(failures.items())}

        details = sorted({
            str(e["detail"]) for e in entries
            if e.get("status", "ok") != "ok" and e.get("detail")
        })

        n_attempted = len(attempted_seeds)
        n_usable = len(usable_seeds)
        # Every ESS that was actually recorded. A missing one is not a zero:
        # "or 0.0" turned "this run reported no ESS" into "this run measured an
        # ESS of zero", which is a fabricated measurement.
        ess_values = [
            float(e["effective_sample_size"]) for e in usable
            if e.get("effective_sample_size") is not None
        ]
        n_tests = sorted({int(e.get("n_test_rounds", 0)) for e in (usable or entries)})

        common = {
            "experiment": experiment,
            "n_attempted": n_attempted,
            "n_usable": n_usable,
            "attempted_seeds": attempted_seeds,
            "seeds": usable_seeds,
            "failed_seeds": sorted(set(attempted_seeds) - set(usable_seeds)),
            "failures": failures,
            "details": details,
            "n_test": max(n_tests) if n_tests else 0,
            "n_test_values": n_tests,
        }

        if not usable:
            # No seed produced a value. Report the status each seed actually
            # reported -- relabelling them all "not_identifiable" would print
            # them under a footnote that asserts a cause nobody measured.
            statuses = sorted(failures)
            out[key] = {
                **common,
                "mean": float("nan"), "std": 0.0,
                "ci_lower": float("nan"), "ci_upper": float("nan"),
                "ess": None, "ess_missing": True,
                "n_seeds": 0,
                "partial": False,
                "status": statuses[0] if len(statuses) == 1 else "mixed_failure",
            }
            continue

        values = np.array([e["value"] for e in usable], dtype=float)
        out[key] = {
            **common,
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "n_seeds": n_usable,
            "ci_lower": float(np.mean([e["ci_lower"] for e in usable])),
            "ci_upper": float(np.mean([e["ci_upper"] for e in usable])),
            "ess": float(np.mean(ess_values)) if ess_values else None,
            "ess_missing": len(ess_values) < len(usable),
            "partial": n_usable < n_attempted,
            "status": "partial" if n_usable < n_attempted else "ok",
        }
    return out


def group_for_tables(aggregated: dict) -> list[tuple[str, dict]]:
    """Split the 5-tuple cells into groups that may share one table.

    One group is one experiment family: the ablation arms of a single base
    config, which differ in exactly one switch and belong side by side. Within a
    group each ``(state_source, lambda)`` column maps to exactly one experiment,
    so the cells can be re-keyed to the 4-tuple the table builders use without
    any two experiments colliding. If that mapping is not one-to-one -- two
    experiments claiming the same column -- the family is split into one group
    per experiment rather than letting them overwrite each other.
    """
    families: dict[str, set[str]] = defaultdict(set)
    for key in aggregated:
        families[experiment_family(key[0])].add(key[0])

    groups: list[tuple[str, set[str]]] = []
    for family, experiments in sorted(families.items()):
        columns: dict[tuple, set[str]] = defaultdict(set)
        for key in aggregated:
            if key[0] in experiments:
                columns[(key[2], key[3])].add(key[0])
        if all(len(owners) == 1 for owners in columns.values()):
            groups.append((family, experiments))
        else:
            groups += [(e, {e}) for e in sorted(experiments)]

    out = []
    for label, experiments in groups:
        out.append((
            label,
            {k[1:]: v for k, v in aggregated.items() if k[0] in experiments},
        ))
    return out


def one_group(aggregated: dict) -> dict:
    """Re-key an :func:`aggregate` result for the table builders.

    The builders index cells as ``(agent, state_source, lambda, estimator)``.
    :func:`aggregate` keys on the experiment as well, so passing its output
    straight to a builder would compare a string against a float and, worse,
    could let one experiment's cells stand in for another's. Cells spanning more
    than one experiment are refused here rather than quietly rendered: choosing
    which one to keep is :func:`group_for_tables`'s job, and there is no correct
    silent answer.
    """
    if not aggregated:
        return {}
    arity = {len(k) for k in aggregated}
    if arity == {4}:
        return aggregated
    if arity != {5}:
        raise ValueError(f"mixed cell key shapes: {sorted(arity)}")
    experiments = sorted({k[0] for k in aggregated})
    if len(experiments) > 1:
        raise ValueError(
            f"these cells span {len(experiments)} experiments "
            f"({', '.join(experiments)}). One table may only describe one "
            f"experiment; split them with group_for_tables() first."
        )
    return {k[1:]: v for k, v in aggregated.items()}


#: Marker -> the footnote text that explains it. ``not_identifiable`` is the
#: only status whose cause is known well enough to state; every other failure
#: prints the status the run actually recorded.
NOT_IDENTIFIABLE_NOTE = (
    "$^{\\dagger}$n.i.\\ = not identifiable: the evaluation policy selects no "
    "logged action, so all importance weights are zero."
)


#: Below this many effective observations, an importance-weighted estimate is
#: reported with its ESS and marked, and its interval is suppressed. Stated here
#: and in Section IV rather than only in a caption, so it cannot look like a
#: threshold chosen after seeing which cells it flattered. A hundred effective
#: observations is the point below which a percentile bootstrap over the same
#: rounds stops being trustworthy.
LOW_ESS_THRESHOLD = 100.0

LOW_ESS_NOTE = (
    "$^{\\S}$effective sample size below "
    + f"{LOW_ESS_THRESHOLD:.0f}"
    + ": the importance weights concentrate on so few rounds that the estimate "
    "is not supported by the data, whatever its nominal value."
)


def _is_weighted(estimator: str) -> bool:
    """Whether an estimator consumes importance weights, and so has an ESS."""
    return estimator in {"ipw", "snipw", "dr", "mrdr"}


def _status_label(status: str) -> str:
    """Human-readable form of a recorded failure status."""
    return _escape(status.replace("run_", "run ").replace("_", " "))


def _format(stats: dict, precision: int = 5) -> str:
    """``mean ± std`` over the seeds that produced a value, or a failure marker.

    A cell that lost seeds is never printed as a bare point estimate: that is
    indistinguishable from a run that was only ever meant to have one seed. It
    carries its own ``n/m seeds`` count, so no caption can overstate it.
    """
    status = stats.get("status", "ok")
    if status == "not_identifiable":
        return "n.i.$^{\\dagger}$"
    if status not in ("ok", "partial"):
        # A real, different failure. Marked distinctly so it is not swept under
        # the "all importance weights are zero" footnote.
        return "n/a$^{\\ddagger}$"
    if stats["n_seeds"] > 1:
        text = f"{stats['mean']:.{precision}f} $\\pm$ {stats['std']:.{precision}f}"
    else:
        text = f"{stats['mean']:.{precision}f}"
    if stats.get("partial"):
        text += (
            f"$^{{*}}$\\,{{\\scriptsize({stats['n_seeds']}/{stats['n_attempted']} seeds)}}"
        )
    return text


def _cell_footnotes(cells: list[dict]) -> list[str]:
    """Footnote lines for whatever markers the given cells actually used."""
    notes: list[str] = []
    if any(c.get("status") == "not_identifiable" for c in cells):
        notes.append(NOT_IDENTIFIABLE_NOTE)

    other: dict[str, set[str]] = defaultdict(set)
    for cell in cells:
        if cell.get("status") in ("ok", "partial", "not_identifiable"):
            continue
        for status in cell.get("failures", {}) or {cell.get("status", "unknown"): []}:
            if status == "not_identifiable":
                continue
            other[status].update(cell.get("details", []))
    if other:
        parts = []
        for status, details in sorted(other.items()):
            reason = f" ({_escape(sorted(details)[0])})" if details else ""
            parts.append(f"{_status_label(status)}{reason}")
        notes.append(
            "$^{\\ddagger}$no value was produced; the run recorded: "
            + "; ".join(parts)
            + ". This is not a non-identifiability result."
        )

    partial = [c for c in cells if c.get("partial")]
    if partial:
        lost = sorted({
            f"{_status_label(s)} (seeds {', '.join(map(str, seeds))})"
            for c in partial for s, seeds in (c.get("failures") or {}).items()
        })
        notes.append(
            "$^{*}$fewer seeds produced a value than were attempted; the cell gives "
            "$n/m$. A bracketed cell is not comparable with a full-seed cell and is "
            "never marked best. Lost seeds: " + ("; ".join(lost) if lost else "cause not recorded")
            + "."
        )
    return notes


def _seed_phrase(cells: list[dict]) -> str:
    """Caption clause describing the seeds, driven by per-cell counts.

    The old text took the maximum ``n_seeds`` over the whole table, so one
    full-seed cell made the caption claim that every cell was averaged over that
    many seeds. The caption may only state a number that holds for every cell it
    covers; otherwise it states the range and defers to the per-cell counts.
    """
    counts = sorted({c["n_seeds"] for c in cells if c.get("status") in ("ok", "partial")})
    if not counts:
        return ", no seed produced a usable value"
    if len(counts) == 1:
        if counts[0] == 1:
            return ", single seed"
        return f", mean $\\pm$ std over {counts[0]} seeds"
    return (
        f", seed count varies by cell ({counts[0]}--{counts[-1]}); "
        f"each cell gives its own count"
    )


def _rounds_phrase(cells: list[dict]) -> str:
    """Caption clause for held-out rounds, honest when cells disagree."""
    values = sorted({c["n_test"] for c in cells if c.get("n_test")})
    if not values:
        return "held-out rounds not recorded"
    if len(values) == 1:
        return f"{values[0]:,} held-out rounds"
    return f"{values[0]:,}--{values[-1]:,} held-out rounds"


def build_ope_table(
    aggregated: dict, dataset: str, variant_state: str = "gnn_embeddings",
    variant_lambda: float = 0.0, label_suffix: str = "", experiment: str | None = None,
) -> str:
    """LaTeX table of policy value per agent, one column per estimator."""
    aggregated = one_group(aggregated)
    estimators = sorted(
        {k[3] for k in aggregated if k[1] == variant_state and k[2] == variant_lambda},
        key=lambda e: list(ESTIMATOR_LABELS).index(e) if e in ESTIMATOR_LABELS else 99,
    )
    if not estimators:
        return f"% no results for {dataset} ({variant_state}, lambda={variant_lambda})\n"

    agents = [
        a for a in AGENT_ORDER
        if any(k[0] == a and k[1] == variant_state and k[2] == variant_lambda for k in aggregated)
    ]
    extra = sorted(
        {k[0] for k in aggregated if k[1] == variant_state and k[2] == variant_lambda}
        - set(agents)
    )
    agents += extra

    in_table = [
        v for k, v in aggregated.items()
        if k[1] == variant_state and k[2] == variant_lambda
    ]
    experiment = experiment or next(
        iter(sorted({v.get("experiment", "?") for v in in_table})), "?"
    )

    # Best value per estimator, for bolding. A cell that lost seeds is excluded:
    # bolding a one-surviving-seed number as the winner over a full-seed
    # measurement is exactly the comparison the missing seeds would settle.
    # A cell flagged for low effective sample size is also excluded. Bolding a
    # number as the winner while a footnote says the data does not support it
    # is the misreading this table exists to prevent: the eye takes the bold and
    # never reaches the dagger.
    def _eligible_for_best(est: str, stats: dict) -> bool:
        if stats.get("status") != "ok" or stats.get("partial"):
            return False
        ess = stats.get("ess")
        if _is_weighted(est) and ess is not None and ess < LOW_ESS_THRESHOLD:
            return False
        return True

    best = {
        est: max(
            (v["mean"] for k, v in aggregated.items()
             if k[3] == est and k[1] == variant_state and k[2] == variant_lambda
             and _eligible_for_best(est, v)),
            default=float("-inf"),
        )
        for est in estimators
    }

    lines = [
        "% Generated by scripts/generate_paper_tables.py -- do not edit by hand.",
        f"% Source: experiment {experiment}, dataset {dataset}, state={variant_state}, "
        f"lambda={variant_lambda}, {_rounds_phrase(in_table)}.",
        "% Only seeds of this one experiment are pooled; no other run contributes.",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\scriptsize",
        "\\begin{tabular}{l" + "c" * len(estimators) + "r}",
        "\\toprule",
        "\\textbf{Agent} & "
        + " & ".join(f"\\textbf{{{ESTIMATOR_LABELS.get(e, e.upper())}}}" for e in estimators)
        + " & \\textbf{ESS} \\\\",
        "\\midrule",
    ]

    shown: list[dict] = []
    for agent in agents:
        cells = []
        for est in estimators:
            stats = aggregated.get((agent, variant_state, variant_lambda, est))
            if stats is None:
                cells.append("")
                continue
            shown.append(stats)
            text = _format(stats)
            ess = stats.get("ess")
            if (
                _is_weighted(est) and ess is not None
                and ess < LOW_ESS_THRESHOLD and stats.get("status") == "ok"
            ):
                stats = {**stats, "_low_ess": True}
                shown[-1] = stats
            if (
                stats.get("status") == "ok" and not stats.get("partial")
                and abs(stats["mean"] - best[est]) < 1e-12
            ):
                text = f"\\textbf{{{text}}}"
            cells.append(text)
        # One ESS per agent: the weights depend on the policy and the log, not
        # on which estimator consumes them, so every weighted cell in this row
        # rests on the same number of effective observations.
        row_ess = [
            aggregated[(agent, variant_state, variant_lambda, est)].get("ess")
            for est in estimators
            if (agent, variant_state, variant_lambda, est) in aggregated
        ]
        row_ess = [e for e in row_ess if e is not None]
        ess_cell = f"{max(row_ess):,.0f}" if row_ess else "--"
        lines.append(f"{_escape(agent)} & " + " & ".join(cells) + f" & {ess_cell} \\\\")

    lines += [
        "\\bottomrule", "\\end{tabular}",
        # The caption states its own provenance: how many held-out rounds the
        # table covers and how many seeds each cell rests on. A caption that
        # quietly took the maximum seed count across cells would claim more
        # measurement than was made.
        f"\\caption{{Off-policy evaluation on {_escape(dataset.upper())}, "
        f"{_rounds_phrase(in_table)}{_seed_phrase(in_table)}.}}",
        f"\\label{{tab:ope_{dataset}{label_suffix}}}",
    ]
    for index, note in enumerate(_cell_footnotes(shown)):
        prefix = "\\\\[2pt]\\footnotesize " if index == 0 else "\\\\ \\footnotesize "
        lines.append(prefix + note)
    lines += ["\\end{table}", ""]
    return "\n".join(lines)


def build_ablation_table(
    aggregated: dict, dataset: str, estimator: str, label_suffix: str = ""
) -> str:
    """LaTeX table comparing configurations that differ in one switch.

    ``aggregated`` is one experiment *family*: arms that differ in exactly one
    config value, each arm its own experiment. Because each column maps to
    exactly one experiment (see :func:`group_for_tables`), no column averages
    across runs.
    """
    aggregated = one_group(aggregated)
    configurations = sorted({(k[1], k[2]) for k in aggregated if k[3] == estimator})
    if len(configurations) < 2:
        return (
            f"% Only one configuration was run for {dataset}; an ablation table needs at\n"
            f"% least two. Run: python scripts/run_paper_experiments.py --variants all\n"
        )

    agents = [a for a in AGENT_ORDER if any(k[0] == a for k in aggregated)]
    labels = {
        ("gnn_embeddings", 0.0): "GNN state",
        ("raw_features", 0.0): "Raw features",
    }
    lam_symbol = "$\\lambda$"

    def _config_label(config: tuple) -> str:
        if config in labels:
            return labels[config]
        state, lam = config
        # _escape, not the raw name: an unlabelled configuration falls through
        # to the config value itself, and "gnn_embeddings" carries an underscore
        # that puts LaTeX into math mode and kills the compile with
        # "Missing $ inserted" -- a fatal error from a column heading.
        return f"{_escape(str(state))}, {lam_symbol}={lam:g}"

    lines = [
        "% Generated by scripts/generate_paper_tables.py -- do not edit by hand.",
        "% Each column is a separate run differing in exactly one config value.",
        "\\begin{table}[htbp]",
        f"\\caption{{Ablation on {_escape(dataset.upper())} ({ESTIMATOR_LABELS.get(estimator, estimator.upper())}).}}",
        f"\\label{{tab:ablation_{dataset}{label_suffix}}}",
        "\\centering",
        "\\footnotesize",
        "\\begin{tabular}{l" + "c" * len(configurations) + "}",
        "\\toprule",
        "\\textbf{Agent} & "
        + " & ".join(
            "\\textbf{" + _config_label(c) + "}" for c in configurations
        )
        + " \\\\",
        "\\midrule",
    ]
    shown: list[dict] = []
    for agent in agents:
        cells = []
        for state, lam in configurations:
            stats = aggregated.get((agent, state, lam, estimator))
            if stats:
                shown.append(stats)
            cells.append(_format(stats) if stats else "--")
        lines.append(f"{_escape(agent)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    for index, note in enumerate(_cell_footnotes(shown)):
        prefix = "\\\\[2pt]\\footnotesize " if index == 0 else "\\\\ \\footnotesize "
        lines.append(prefix + note)
    lines += ["\\end{table}", ""]
    return "\n".join(lines)


def build_ope_validation_table(
    aggregated: dict, dataset: str, label_suffix: str = ""
) -> str:
    """Compare each estimator against exact ground truth.

    Only meaningful where the reward matrix is fully observed, so the `exact`
    column is a true value rather than an estimate. This is the table behind the
    estimator-validation contribution: it reports how far SNIPW, DR and MRDR
    depart from a value that is known rather than inferred.
    """
    aggregated = one_group(aggregated)
    exact = {
        k[0]: v for k, v in aggregated.items()
        if k[3] == "exact" and k[1] == "gnn_embeddings" and k[2] == 0.0
        and v.get("status") == "ok"
    }
    if not exact:
        return (
            "% No exact-evaluation results found. This table requires a dataset with a\n"
            "% fully observed reward matrix. Run: configs/kuairec_validate_ope.yaml\n"
        )

    estimators = [
        e for e in ("snipw", "dm", "dr", "mrdr")
        if any(k[3] == e for k in aggregated)
    ]
    if not estimators:
        return "% Only exact values available; no estimators to validate against them.\n"

    agents = [a for a in AGENT_ORDER if a in exact] + sorted(set(exact) - set(AGENT_ORDER))

    lines = [
        "% Generated by scripts/generate_paper_tables.py -- do not edit by hand.",
        "% Exact is ground truth from the fully observed matrix; the rest are estimates.",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\scriptsize",
        "\\begin{tabular}{l" + "c" * (1 + len(estimators)) + "}",
        "\\toprule",
        "\\textbf{Agent} & \\textbf{Exact (true)} & "
        + " & ".join(
            f"\\textbf{{{ESTIMATOR_LABELS.get(e, e.upper())}}}" for e in estimators
        )
        + " \\\\",
        "\\midrule",
    ]

    shown: list[dict] = []
    for agent in agents:
        truth = exact[agent]["mean"]
        cells = [f"{truth:.5f}"]
        for est in estimators:
            stats = aggregated.get((agent, "gnn_embeddings", 0.0, est))
            if stats is None:
                cells.append("")
                continue
            shown.append(stats)
            status = stats.get("status", "ok")
            if status == "not_identifiable":
                cells.append("n.i.$^{\\dagger}$")
            elif status not in ("ok", "partial"):
                # A different failure. It gets its own marker and its own
                # footnote rather than borrowing the non-identifiability one.
                cells.append("n/a$^{\\ddagger}$")
            else:
                error = stats["mean"] - truth
                text = f"{error:+.5f}"
                if stats.get("partial"):
                    text += (
                        f"$^{{*}}$\\,{{\\scriptsize"
                        f"({stats['n_seeds']}/{stats['n_attempted']} seeds)}}"
                    )
                cells.append(text)
        lines.append(f"{_escape(agent)} & " + " & ".join(cells) + " \\\\")

    # Mean absolute error per estimator, across agents.
    lines.append("\\midrule")
    mae_cells = [""]
    for est in estimators:
        errors = [
            abs(aggregated[(a, "gnn_embeddings", 0.0, est)]["mean"] - exact[a]["mean"])
            for a in agents
            if (a, "gnn_embeddings", 0.0, est) in aggregated
            and aggregated[(a, "gnn_embeddings", 0.0, est)].get("status") == "ok"
            and not aggregated[(a, "gnn_embeddings", 0.0, est)].get("partial")
        ]
        mae_cells.append(f"{np.mean(errors):.5f}" if errors else "")
    lines.append("\\textbf{MAE} & " + " & ".join(mae_cells) + " \\\\")

    lines += [
        "\\bottomrule", "\\end{tabular}",
        f"\\caption{{Estimator error against exact ground truth on "
        f"{_escape(dataset.upper())}.}}",
        f"\\label{{tab:ope_validation_{dataset}{label_suffix}}}",
    ]
    for index, note in enumerate(_cell_footnotes(shown)):
        prefix = "\\\\[2pt]\\footnotesize " if index == 0 else "\\\\ \\footnotesize "
        lines.append(prefix + note)
    lines += ["\\end{table}", ""]
    return "\n".join(lines)


def build_dataset_table(results_dir: Path, datasets: list[str]) -> str:
    # kuairec_sage is the same data under a different encoder; as a third column
    # it only widened the table past the text block.
    datasets = [d for d in datasets if is_reportable(d)]
    """Dataset statistics, read from what each run actually loaded."""
    summaries = {}
    for dataset in datasets:
        path = results_dir / f"dataset_summary_{dataset}.json"
        if path.exists():
            summaries[dataset] = json.loads(path.read_text(encoding="utf-8"))

    if not summaries:
        return "% No dataset_summary_*.json found. Run Phase 1 to generate them.\n"

    fields = [
        ("Interactions", "n_interactions", "{:,}"),
        ("Users", "n_users", "{:,}"),
        ("Items (actions)", "n_actions", "{:,}"),
        ("Mean reward", "mean_reward", "{:.5f}"),
        ("Reward definition", "reward_definition", "{}"),
        ("Train / Val / Test", None, "{}"),
        ("Split strategy", "split_strategy", "{}"),
        ("Logged propensities", None, "{}"),
    ]

    names = list(summaries)
    lines = [
        "% Generated by scripts/generate_paper_tables.py from the loaded data.",
        "\\begin{table}[htbp]",
        "\\caption{Dataset statistics, as loaded by the experiment pipeline.}",
        "\\label{tab:datasets}",
        "\\centering",
        "\\footnotesize",
        "\\begin{tabular}{l" + "c" * len(names) + "}",
        "\\toprule",
        "\\textbf{Property} & " + " & ".join(f"\\textbf{{{_escape(n.upper())}}}" for n in names) + " \\\\",
        "\\midrule",
    ]
    for label, key, fmt in fields:
        cells = []
        for name in names:
            summary = summaries[name]
            if label == "Train / Val / Test":
                cells.append(
                    f"{summary['n_train']:,} / {summary['n_validation']:,} / {summary['n_test']:,}"
                )
            elif label == "Logged propensities":
                cells.append("No (uniform)" if summary.get("propensity_is_constant") else "Yes")
            else:
                cells.append(_escape(fmt.format(summary.get(key, "--"))))
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def build_implementation_paragraph(
    results_dir: Path, datasets: list[str], actual_seeds: dict[str, list[int]] | None = None
) -> str:
    """The Implementation Details paragraph, generated from the effective configs.

    The seed count is taken from the runs that produced results, not from the
    config's ``seeds:`` list. Those differ whenever fewer seeds were run than
    planned, and reporting the planned number would be the exact
    paper-versus-reality drift this generator exists to prevent.
    """
    import yaml

    # Experiment names carry the variant, not the dataset, so match on the
    # dataset recorded inside each config rather than on the filename.
    configs = {}
    for path in sorted(results_dir.glob("effective_config_*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        name = payload.get("dataset", {}).get("name")
        if name not in datasets:
            continue
        # Describe the main arm, not an ablation arm.
        is_main = payload.get("rl", {}).get("state_source") == "gnn_embeddings" and not payload.get(
            "rl", {}
        ).get("cate_reward_weight")
        if name not in configs or is_main:
            configs[name] = payload

    if not configs:
        return "% No effective_config_*.yaml found. Run any phase to generate one.\n"

    dataset = next(iter(configs))
    cfg = configs[dataset]
    gnn, rl, split, data = cfg["gnn"], cfg["rl"], cfg["split"], cfg["dataset"]

    subsample = data.get("subsample_rows")
    subsample_text = (
        f"We sample {subsample:,} logged rounds uniformly at random from the full "
        f"log using the run's seed, spanning the whole campaign rather than a "
        f"temporal prefix. "
        if subsample else ""
    )
    state_description = (
        "graph embedding of each round's user"
        if rl["state_source"] == "gnn_embeddings"
        else "raw tabular feature vector"
    )
    estimator_names = ", ".join(
        ESTIMATOR_LABELS.get(e, e.upper()) for e in cfg["ope"]["estimators"]
    )
    # Percentages are formatted as plain integers and the literal LaTeX "\%" is
    # appended, since "{:.0%}" already emits an unescaped "%".
    train_pct = f"{split['train_ratio'] * 100:.0f}"
    val_pct = f"{split['val_ratio'] * 100:.0f}"
    test_pct = f"{split['test_ratio'] * 100:.0f}"
    conf_pct = f"{cfg['ope']['confidence_level'] * 100:.0f}"

    seeds = (actual_seeds or {}).get(dataset) or []
    planned = cfg.get("seeds", [])
    if len(seeds) > 1:
        seed_text = (
            f"All results are reported as the mean $\\pm$ standard deviation over "
            f"{len(seeds)} seeds ({', '.join(map(str, seeds))})."
        )
    elif len(seeds) == 1:
        seed_text = (
            f"All results use a single seed ({seeds[0]}); point estimates are "
            f"indicative rather than precise."
        )
    else:
        # No run produced a usable result. The config's ``seed:`` default is the
        # seed that WOULD have been used; printing it as the seed results are
        # reported for describes a measurement that does not exist.
        seed_text = (
            "%% NOTE: no seed produced a usable result for this dataset, so there is no "
            "seed to report. Do not publish this paragraph until at least one run "
            "succeeds."
        )
    if planned and len(planned) > len(seeds):
        seed_text += (
            f" %% NOTE: the config lists {len(planned)} seeds but only {len(seeds)} "
            f"produced results. Run the rest, or keep this sentence as written."
        )

    text = f"""% Generated by scripts/generate_paper_tables.py from the effective run config.
% Every figure below is what the code executed, not what was intended.
% This is a fragment: the paper supplies the \\subsection heading.

{subsample_text}Graph encoders are trained with {gnn['embedding_dim']}-dimensional
embeddings across {gnn['num_layers']} message-passing layers, hidden width
{gnn['hidden_dim']}, using Adam (learning rate ${gnn['learning_rate']:g}$, batch size
{gnn['node_batch_size']}) for up to {gnn['epochs']} epochs with early stopping on
validation BPR loss (patience {gnn['early_stopping_patience']}). Validation uses a
10\\% held-out edge sample; message passing during validation uses training edges
only, so held-out edges never enter the representation that scores them.

Offline RL agents use a {rl['num_layers']}-layer MLP with {rl['hidden_units']} hidden
units, trained for {rl['epochs']} epochs with learning rate ${rl['learning_rate']:g}$ and
batch size {rl['batch_size']}. BCQ uses an action flexibility of
{rl['bcq_action_flexibility']}; CQL uses $\\alpha = {rl['cql_alpha']}$; IQL uses expectile
{rl['iql_expectile']} and $\\beta = {rl['iql_beta']}$. The agent state is the
{state_description}.

Data is partitioned {train_pct}\\%/{val_pct}\\%/{test_pct}\\%
into train/validation/test using a {split['strategy']} split; the test partition is
read only at evaluation time. Off-policy evaluation uses
{estimator_names} with
{cfg['ope']['n_bootstrap']} bootstrap replicates at the
{conf_pct}\\% level. {seed_text}
"""
    return text


def build_claims_check(aggregated: dict, dataset: str, experiment: str | None = None) -> str:
    """Report which comparative claims the numbers support.

    This exists so a claim in the paper can be checked against the table
    mechanically, rather than by rereading prose written before the numbers.
    """
    aggregated = one_group(aggregated)
    experiment = experiment or next(
        iter(sorted({v.get("experiment", "?") for v in aggregated.values()})), "?"
    )
    lines = [
        f"# Claims check — {dataset.upper()} — experiment `{experiment}`",
        "",
        "Seeds are pooled only within this one experiment. Numbers from another "
        "experiment on the same dataset are reported separately below, never averaged in.",
        "",
    ]
    estimators = sorted({k[3] for k in aggregated})

    for est in estimators:
        entries = {
            k[0]: v for k, v in aggregated.items()
            if k[3] == est and k[1] == "gnn_embeddings" and k[2] == 0.0
            and v.get("status") in ("ok", "partial")
        }
        if not entries:
            continue
        ranked = sorted(entries.items(), key=lambda kv: -kv[1]["mean"])
        label = ESTIMATOR_LABELS.get(est, est.upper())
        lines += [
            f"## {label}", "",
            "| Rank | Agent | Value | Seeds used / attempted | Complete? |",
            "|---|---|---|---|---|",
        ]
        for i, (agent, stats) in enumerate(ranked, 1):
            spread = f" ± {stats['std']:.5f}" if stats["n_seeds"] > 1 else ""
            complete = "no — **seeds lost**" if stats.get("partial") else "yes"
            lines.append(
                f"| {i} | {agent} | {stats['mean']:.5f}{spread} | "
                f"{stats['n_seeds']} / {stats['n_attempted']} | {complete} |"
            )
        lines.append("")

        incomplete = {a: s for a, s in entries.items() if s.get("partial")}
        if incomplete:
            for agent, stats in sorted(incomplete.items()):
                reasons = "; ".join(
                    f"{status} (seeds {', '.join(map(str, seeds))})"
                    for status, seeds in (stats.get("failures") or {}).items()
                ) or "cause not recorded"
                lines.append(
                    f"- **{agent} rests on {stats['n_seeds']} of {stats['n_attempted']} "
                    f"attempted seeds.** Lost: {reasons}. Its value is not comparable with a "
                    f"full-seed value and no ranking involving it is established."
                )
            lines.append("")

        complete_ranked = [kv for kv in ranked if not kv[1].get("partial")]
        if complete_ranked:
            best_agent, best_stats = complete_ranked[0]
            lines.append(f"**Best under {label}: {best_agent}** ({best_stats['mean']:.5f}).")
            if incomplete:
                lines.append(
                    f"  (Ranked over full-seed cells only; "
                    f"{', '.join(sorted(incomplete))} excluded as incomplete.)"
                )
        else:
            lines.append(f"**Best under {label}: not established** — every cell lost seeds.")
        lines.append("")

        for constrained in ("BCQ", "CQL"):
            if constrained in entries and "DQN" in entries:
                a, b = entries[constrained]["mean"], entries["DQN"]["mean"]
                incomplete_side = [
                    n for n in (constrained, "DQN") if entries[n].get("partial")
                ]
                if incomplete_side:
                    verdict = (
                        "**NOT ESTABLISHED** — "
                        + ", ".join(
                            f"{n} used {entries[n]['n_seeds']} of "
                            f"{entries[n]['n_attempted']} seeds"
                            for n in incomplete_side
                        )
                    )
                else:
                    verdict = "SUPPORTED" if a > b else "**NOT SUPPORTED**"
                lines.append(
                    f"- Claim *\"{constrained} outperforms unconstrained DQN\"*: {verdict} "
                    f"({constrained}={a:.5f} vs DQN={b:.5f})."
                )
        if "Random" in entries:
            baseline = entries["Random"]["mean"]
            if entries["Random"].get("partial"):
                lines.append(
                    f"- The random baseline itself rests on "
                    f"{entries['Random']['n_seeds']} of {entries['Random']['n_attempted']} "
                    f"seeds, so no comparison against it is established."
                )
            else:
                comparable = {
                    a: s for a, s in entries.items()
                    if a != "Random" and not s.get("partial")
                }
                beat = [a for a, s in comparable.items() if s["mean"] > baseline]
                lost = [a for a, s in comparable.items() if s["mean"] <= baseline]
                lines.append(f"- Agents beating the random baseline ({baseline:.5f}): {', '.join(beat) or 'none'}.")
                if lost:
                    lines.append(f"- Agents **not** beating random: {', '.join(lost)}. Any claim of general improvement must exclude these.")
                undecided = sorted(set(entries) - set(comparable) - {"Random"})
                if undecided:
                    lines.append(
                        f"- Undecided against random (lost seeds): {', '.join(undecided)}."
                    )

        # ESS is a property of ONE agent's importance weights. Averaging it
        # across agents lets a healthy agent hide an agent whose estimate rests
        # on a handful of rounds -- and that caveat is the reader's only signal
        # that the number is unreliable. Test each agent against its own rounds.
        if est in {"ipw", "snipw", "dr", "mrdr"}:
            starved = []
            unknown = []
            for agent, stats in sorted(entries.items()):
                n_test = stats["n_test"]
                if stats["ess"] is None:
                    unknown.append(agent)
                elif n_test and stats["ess"] < 0.01 * n_test:
                    starved.append(
                        f"{agent}: ESS {stats['ess']:.1f} of {n_test:,} rounds "
                        f"({100 * stats['ess'] / n_test:.3f}%)"
                    )
            for agent, stats in sorted(entries.items()):
                if stats.get("ess_missing") and agent not in unknown:
                    unknown.append(agent)
            if starved:
                lines.append(
                    "- **Caveat:** these agents' importance-weighted estimates rest on very "
                    "few rounds and must be reported with this caveat — "
                    + "; ".join(starved) + "."
                )
            if unknown:
                lines.append(
                    f"- **Effective sample size was not recorded** for: "
                    f"{', '.join(unknown)}. Reliability cannot be assessed for these; a "
                    f"missing ESS is not an ESS of zero."
                )
        lines.append("")

    # Ablation contrasts
    for est in estimators:
        gnn_side = {k[0]: v for k, v in aggregated.items() if k[3] == est and k[1] == "gnn_embeddings" and k[2] == 0.0}
        raw_side = {k[0]: v for k, v in aggregated.items() if k[3] == est and k[1] == "raw_features" and k[2] == 0.0}
        shaped = {k[0]: v for k, v in aggregated.items() if k[3] == est and k[2] > 0.0}

        def _verdict(left: dict, right: dict, yes: str, no: str) -> str:
            if left.get("status") not in ("ok", "partial") or right.get("status") not in ("ok", "partial"):
                return "**no verdict** (one side produced no value)"
            if left.get("partial") or right.get("partial"):
                return "**no verdict** (one side lost seeds)"
            return yes if right["mean"] > left["mean"] else no

        if gnn_side and raw_side:
            lines += [f"## Ablation: graph state vs raw features ({ESTIMATOR_LABELS.get(est, est)})", ""]
            for agent in sorted(set(gnn_side) & set(raw_side)):
                g, r = gnn_side[agent]["mean"], raw_side[agent]["mean"]
                lines.append(
                    f"- {agent}: GNN={g:.5f} vs raw={r:.5f} -> "
                    + _verdict(raw_side[agent], gnn_side[agent],
                               "graph helps", "**graph does not help**")
                )
            lines.append("")
        if gnn_side and shaped:
            lines += [f"## Ablation: CATE reward shaping ({ESTIMATOR_LABELS.get(est, est)})", ""]
            for agent in sorted(set(gnn_side) & set(shaped)):
                base, sh = gnn_side[agent]["mean"], shaped[agent]["mean"]
                lines.append(
                    f"- {agent}: lambda=0 {base:.5f} vs shaped {sh:.5f} -> "
                    + _verdict(gnn_side[agent], shaped[agent],
                               "shaping helps", "**shaping does not help**")
                )
            lines.append("")

    if len(lines) <= 2:
        lines.append("No results found. Run the experiments first.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    args = parser.parse_args()

    results_dir = args.results_dir
    out_dir = args.out_dir or results_dir / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.datasets:
        datasets = args.datasets
    else:
        datasets = sorted({
            row["dataset"]
            for p in results_dir.glob("phase4_ope_*_seed*.json")
            for row in json.loads(p.read_text(encoding="utf-8"))
        })

    if not datasets:
        print(f"No phase4_ope_*.json found in {results_dir}. Run the experiments first.")
        return 1

    print(f"Datasets found: {', '.join(datasets)}")
    ope_tex, ablation_tex, claims, validation_tex = [], [], [], []
    actual_seeds: dict[str, list[int]] = {}

    for dataset in datasets:
        rows = load_ope_results(results_dir, dataset)
        if not rows:
            continue
        experiments = {r["_experiment"] for r in rows}
        attempted = load_run_records(results_dir, experiments)
        aggregated = aggregate(rows, attempted)
        # Only seeds that produced a usable value may be named as the seeds the
        # results are reported for.
        actual_seeds[dataset] = sorted({
            r["seed"] for r in rows if r.get("status", "ok") == "ok"
        })

        groups = group_for_tables(aggregated)
        if len(groups) > 1:
            print(
                f"  {dataset}: {len(groups)} separate experiments "
                f"({', '.join(label for label, _ in groups)}); "
                f"each gets its own table -- their numbers are not pooled"
            )
        for label, cells in groups:
            # A suffix only when the dataset has more than one experiment, so a
            # single-experiment run keeps the label the paper already cites.
            suffix = "" if len(groups) == 1 else "_" + label.replace("-", "_")
            ope_tex.append(build_ope_table(aggregated=cells, dataset=dataset,
                                           label_suffix=suffix, experiment=label))
            estimators = sorted({k[3] for k in cells})
            # The ablation arms only exist for the main family; the validation
            # run has no raw-feature or shaped-reward counterpart, so building a
            # table for it produced a one-column "ablation" of nothing.
            if estimators and "validate_ope" not in label and "kuairec" not in dataset:
                ablation_tex.append(
                    build_ablation_table(cells, dataset, estimators[0], label_suffix=suffix)
                )
            validation_tex.append(
                build_ope_validation_table(cells, dataset, label_suffix=suffix)
            )
            claims.append(build_claims_check(cells, dataset, experiment=label))

        unattributed = attempted.get("_unattributed") or {}
        if unattributed:
            note = "\n".join(
                f"% run '{v}' seed {s} failed and produced no results at all: {reason}"
                for (v, s), reason in sorted(unattributed.items())
            )
            ope_tex.append("% NOTE: runs missing entirely from this table --\n" + note + "\n")
        print(f"  {dataset}: {len(rows)} result rows, {len(aggregated)} cells")

    (out_dir / "table_ope.tex").write_text("\n".join(ope_tex), encoding="utf-8")
    (out_dir / "table_ablation.tex").write_text("\n".join(ablation_tex), encoding="utf-8")
    (out_dir / "table_ope_validation.tex").write_text("\n".join(validation_tex), encoding="utf-8")
    (out_dir / "table_datasets.tex").write_text(build_dataset_table(results_dir, datasets), encoding="utf-8")
    (out_dir / "implementation.tex").write_text(
        build_implementation_paragraph(results_dir, datasets, actual_seeds), encoding="utf-8"
    )
    (out_dir / "claims_check.md").write_text("\n\n".join(claims), encoding="utf-8")

    print(f"\nWrote to {out_dir}/:")
    for name in ("table_ope.tex", "table_ablation.tex", "table_ope_validation.tex",
                 "table_datasets.tex", "implementation.tex", "claims_check.md"):
        print(f"  {name}")
    print("\nRead claims_check.md before rewriting the results section.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
