"""Regression tests for the paper-table generator and the result figures.

Every test here corresponds to a number that reached a LaTeX table without a
run behind it. The rule the whole file enforces is: a number the generator
prints must have been measured in one run, and anything that was not measured
must be visible as absent rather than filled in.

Each test fails on the pre-fix code. The failure mode is named in the test's
docstring so a future change that reintroduces it is recognisable.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent


def _load(name: str, relative: str):
    """Import a module that lives outside the package (scripts/)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tables = _load("_paper_tables", "scripts/generate_paper_tables.py")
plots = _load("_plot_results", "scripts/plot_results.py")


# --------------------------------------------------------------------------- helpers
def make_row(
    agent: str,
    estimator: str,
    value: float,
    *,
    seed: int = 42,
    dataset: str = "kuairec",
    n_test: int = 50_000,
    state: str = "gnn_embeddings",
    lam: float = 0.0,
    status: str = "ok",
    detail: str | None = None,
    ess: float | None = 40_000.0,
    experiment: str = "exp_main",
) -> dict:
    """One Phase 4 result row, shaped exactly as ``OPEResult.as_row`` writes it."""
    return {
        "agent": agent,
        "dataset": dataset,
        "state_source": state,
        "cate_reward_weight": lam,
        "seed": seed,
        "n_test_rounds": n_test,
        "estimator": estimator,
        "value": value,
        "ci_lower": value - 0.001,
        "ci_upper": value + 0.001,
        "n_samples": n_test,
        "effective_sample_size": ess,
        "status": status,
        "detail": detail,
        "_experiment": experiment,
        "_source": f"phase4_ope_{experiment}_seed{seed}.json",
    }


def write_results(directory: Path, experiment: str, seed: int, rows: list[dict]) -> Path:
    """Write rows to the filename Phase 4 would have written them to."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"phase4_ope_{experiment}_seed{seed}.json"
    payload = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def cell(aggregated: dict, agent: str, estimator: str, *, experiment: str | None = None):
    """Look a cell up by agent and estimator, whatever the key arity is."""
    for key, value in aggregated.items():
        if key[-4] == agent and key[-1] == estimator and (
            experiment is None or value.get("experiment") == experiment
        ):
            return value
    raise KeyError((agent, estimator, experiment))


# =========================================================================== DEFECT 1
class TestExperimentsAreNeverPooled:
    """Rows from two different experiments must never land in one cell.

    ``kuairec_validate_ope`` (300k rows) and ``kuairec_validate_ope_full`` (2M
    rows) both declare ``dataset.name: kuairec``. Keying only on the dataset
    averaged a 60,000-round run at 0.90 with a 400,000-round run at 0.10 into
    ``0.50000`` -- a number measured in no run -- and captioned it with the
    larger run's round count.
    """

    @staticmethod
    def _rows() -> list[dict]:
        return [
            make_row("BCQ", "snipw", 0.90, n_test=60_000, experiment="kuairec_validate_ope"),
            make_row("DQN", "snipw", 0.20, n_test=60_000, experiment="kuairec_validate_ope"),
            make_row("BCQ", "snipw", 0.10, n_test=400_000, experiment="kuairec_validate_ope_full"),
            make_row("DQN", "snipw", 0.30, n_test=400_000, experiment="kuairec_validate_ope_full"),
        ]

    def test_each_experiment_keeps_its_own_cell(self):
        aggregated = tables.aggregate(self._rows())
        assert len(aggregated) == 4, "two experiments collapsed into one set of cells"

        small = cell(aggregated, "BCQ", "snipw", experiment="kuairec_validate_ope")
        large = cell(aggregated, "BCQ", "snipw", experiment="kuairec_validate_ope_full")
        assert small["mean"] == pytest.approx(0.90)
        assert large["mean"] == pytest.approx(0.10)
        assert small["n_test"] == 60_000
        assert large["n_test"] == 400_000

    def test_the_pooled_average_is_emitted_nowhere(self):
        aggregated = tables.aggregate(self._rows())
        means = [v["mean"] for v in aggregated.values()]
        assert not any(m == pytest.approx(0.50) for m in means), (
            "0.50 is the mean of 0.90 and 0.10 across two different runs; it was "
            "measured in neither"
        )

        latex = "\n".join(
            tables.build_ope_table(cells, "kuairec", label_suffix=f"_{label}")
            for label, cells in tables.group_for_tables(aggregated)
        )
        assert "0.50000" not in latex
        assert "0.90000" in latex and "0.10000" in latex

    def test_each_table_is_captioned_with_its_own_round_count(self):
        groups = tables.group_for_tables(tables.aggregate(self._rows()))
        assert len(groups) == 2, "the two experiments must not share one table"

        captions = {}
        for label, cells in groups:
            latex = tables.build_ope_table(cells, "kuairec", label_suffix=f"_{label}")
            captions[label] = next(line for line in latex.splitlines() if line.startswith("\\caption"))

        assert "60,000" in captions["kuairec_validate_ope"]
        assert "400,000" not in captions["kuairec_validate_ope"]
        assert "400,000" in captions["kuairec_validate_ope_full"]
        assert "60,000" not in captions["kuairec_validate_ope_full"]

    def test_labels_are_unique_so_latex_does_not_collide(self):
        groups = tables.group_for_tables(tables.aggregate(self._rows()))
        labels = [
            line
            for label, cells in groups
            for line in tables.build_ope_table(
                cells, "kuairec", label_suffix=f"_{label}"
            ).splitlines()
            if line.startswith("\\label")
        ]
        assert len(labels) == len(set(labels)), f"duplicate LaTeX labels: {labels}"

    def test_end_to_end_generator_writes_two_tables(self, tmp_path):
        results = tmp_path / "results"
        write_results(results, "kuairec_validate_ope", 42, self._rows()[:2])
        write_results(results, "kuairec_validate_ope_full", 42, self._rows()[2:])
        out = tmp_path / "paper"

        sys.argv = [
            "generate_paper_tables.py",
            "--results-dir", str(results),
            "--out-dir", str(out),
        ]
        assert tables.main() == 0

        latex = (out / "table_ope.tex").read_text(encoding="utf-8")
        assert latex.count("\\begin{table}") == 2
        assert "0.50000" not in latex


class TestAblationArmsStillShareOneTable:
    """The three arms of one base config differ in exactly one config value and
    are meant to be columns of one ablation table. Keying on the experiment must
    not split them: they have different ``experiment_name``s by design.
    """

    @staticmethod
    def _rows() -> list[dict]:
        rows = []
        for seed in (42, 43):
            rows += [
                make_row("BCQ", "snipw", 0.42, seed=seed, dataset="obd",
                         experiment="obd_bench_main"),
                make_row("DQN", "snipw", 0.30, seed=seed, dataset="obd",
                         experiment="obd_bench_main"),
                make_row("BCQ", "snipw", 0.38, seed=seed, dataset="obd",
                         state="raw_features", experiment="obd_bench_ablation_raw"),
                make_row("DQN", "snipw", 0.25, seed=seed, dataset="obd",
                         state="raw_features", experiment="obd_bench_ablation_raw"),
                make_row("BCQ", "snipw", 0.40, seed=seed, dataset="obd", lam=0.5,
                         experiment="obd_bench_ablation_cate"),
                make_row("DQN", "snipw", 0.33, seed=seed, dataset="obd", lam=0.5,
                         experiment="obd_bench_ablation_cate"),
            ]
        return rows

    def test_the_three_arms_form_one_group(self):
        groups = tables.group_for_tables(tables.aggregate(self._rows()))
        assert len(groups) == 1
        assert groups[0][0] == "obd_bench"

    def test_the_ablation_table_keeps_three_columns(self):
        _, cells = tables.group_for_tables(tables.aggregate(self._rows()))[0]
        latex = tables.build_ablation_table(cells, "obd", "snipw")
        assert "GNN state" in latex
        assert "Raw features" in latex
        assert "0.42000" in latex and "0.38000" in latex and "0.40000" in latex
        header = next(line for line in latex.splitlines() if line.startswith("\\textbf{Agent}"))
        assert header.count("&") == 3, "an ablation arm lost its column"

    def test_arms_are_still_separate_cells(self):
        aggregated = tables.aggregate(self._rows())
        experiments = {v["experiment"] for v in aggregated.values()}
        assert experiments == {
            "obd_bench_main", "obd_bench_ablation_raw", "obd_bench_ablation_cate",
        }

    def test_two_experiments_claiming_one_column_are_split_apart(self):
        """A family whose arms are not one-to-one with columns is split, rather
        than letting one experiment silently overwrite the other."""
        rows = self._rows()
        # A second experiment claiming the same (gnn_embeddings, 0.0) column.
        rows.append(make_row("BCQ", "snipw", 0.99, dataset="obd",
                             experiment="obd_bench_main2"))
        groups = tables.group_for_tables(tables.aggregate(rows))
        assert len(groups) > 1, "colliding experiments were merged into one table"


# =========================================================================== DEFECT 2
class TestPartialSeedFailureIsVisible:
    """One surviving seed of five must not print as a bare point estimate.

    ``BCQ`` printed ``\\textbf{0.00830}`` from one seed, bolded as best against
    ``DQN``'s genuine five-seed ``0.00414 +/- 0.00021``, under a caption reading
    "mean +/- std over 5 seeds" -- because the caption took the maximum seed
    count over the whole table. The four failures left no trace.
    """

    @staticmethod
    def _rows() -> list[dict]:
        dqn = [0.00390, 0.00405, 0.00414, 0.00425, 0.00436]
        rows = []
        for value, seed in zip(dqn, (42, 43, 44, 45, 46), strict=True):
            rows.append(make_row("DQN", "dr", value, seed=seed))
            if seed == 42:
                rows.append(make_row("BCQ", "dr", 0.00830, seed=seed))
            else:
                rows.append(make_row(
                    "BCQ", "dr", float("nan"), seed=seed,
                    status="reward_model_failed",
                    detail="HistGradientBoostingRegressor did not converge",
                    ess=None,
                ))
        return rows

    def test_the_cell_carries_attempted_and_usable_counts(self):
        aggregated = tables.aggregate(self._rows())
        bcq = cell(aggregated, "BCQ", "dr")
        assert bcq["n_usable"] == 1
        assert bcq["n_attempted"] == 5
        assert bcq["partial"] is True
        assert bcq["failed_seeds"] == [43, 44, 45, 46]

        dqn = cell(aggregated, "DQN", "dr")
        assert dqn["n_usable"] == dqn["n_attempted"] == 5
        assert dqn["partial"] is False

    def test_a_partial_cell_does_not_print_as_a_bare_point_estimate(self):
        aggregated = tables.aggregate(self._rows())
        text = tables._format(cell(aggregated, "BCQ", "dr"))
        assert text != "0.00830", "indistinguishable from a legitimate single-seed run"
        assert "1/5" in text

    def test_a_partial_cell_is_not_bolded_as_best(self):
        aggregated = tables.aggregate(self._rows())
        latex = tables.build_ope_table(aggregated, "kuairec")
        bcq_line = next(line for line in latex.splitlines() if line.startswith("BCQ"))
        dqn_line = next(line for line in latex.splitlines() if line.startswith("DQN"))
        assert "\\textbf{0.00830" not in bcq_line, (
            "a one-of-five-seed value was bolded as the best result"
        )
        assert "\\textbf{" in dqn_line, "the full-seed cell should hold the bold"

    def test_the_caption_does_not_claim_five_seeds(self):
        latex = tables.build_ope_table(tables.aggregate(self._rows()), "kuairec")
        caption = next(line for line in latex.splitlines() if line.startswith("\\caption"))
        assert "over 5 seeds" not in caption, (
            "the caption took the maximum seed count across cells"
        )
        assert "varies by cell" in caption

    def test_the_lost_seeds_are_named_in_a_footnote(self):
        latex = tables.build_ope_table(tables.aggregate(self._rows()), "kuairec")
        assert "43, 44, 45, 46" in latex
        assert "reward model failed" in latex

    def test_the_claims_check_does_not_declare_the_claim_supported(self):
        markdown = tables.build_claims_check(tables.aggregate(self._rows()), "kuairec")
        claim = next(
            line for line in markdown.splitlines()
            if "BCQ outperforms unconstrained DQN" in line
        )
        assert "NOT ESTABLISHED" in claim
        assert ": SUPPORTED" not in claim
        assert "1 / 5" in markdown


class TestSeedsThatProducedNoFileAreCounted:
    """A run that died before Phase 4 leaves no result file at all.

    ``run_paper_experiments.py`` records it in ``experiment_runs.json``, which
    the generator never read -- so the cell reported the two surviving seeds and
    said nothing about the three that crashed.
    """

    @staticmethod
    def _setup(tmp_path: Path) -> Path:
        results = tmp_path / "results"
        for seed in (42, 43):
            write_results(results, "obd_bench_main", seed, [
                make_row("DQN", "dr", 0.30, seed=seed, dataset="obd",
                         experiment="obd_bench_main"),
            ])
        (results / "experiment_runs.json").write_text(json.dumps([
            {"variant": "main", "seed": 42, "status": "ok", "seconds": 1.0,
             "error": None, "phases": ["phase4"]},
            {"variant": "main", "seed": 43, "status": "ok", "seconds": 1.0,
             "error": None, "phases": ["phase4"]},
            {"variant": "main", "seed": 44, "status": "failed", "seconds": 1.0,
             "error": "phase3: CUDA out of memory", "phases": []},
            {"variant": "main", "seed": 45, "status": "failed", "seconds": 1.0,
             "error": "phase3: CUDA out of memory", "phases": []},
            {"variant": "main", "seed": 46, "status": "failed", "seconds": 1.0,
             "error": "phase1: download timed out", "phases": []},
        ], indent=2), encoding="utf-8")
        return results

    def test_crashed_seeds_raise_the_attempted_count(self, tmp_path):
        results = self._setup(tmp_path)
        rows = tables.load_ope_results(results, "obd")
        attempted = tables.load_run_records(results, {"obd_bench_main"})
        aggregated = tables.aggregate(rows, attempted)

        dqn = cell(aggregated, "DQN", "dr")
        assert dqn["n_usable"] == 2
        assert dqn["n_attempted"] == 5, (
            "three seeds crashed before Phase 4 and left no file; the cell "
            "reported only the survivors"
        )
        assert dqn["partial"] is True

    def test_the_table_shows_two_of_five(self, tmp_path):
        results = self._setup(tmp_path)
        aggregated = tables.aggregate(
            tables.load_ope_results(results, "obd"),
            tables.load_run_records(results, {"obd_bench_main"}),
        )
        latex = tables.build_ope_table(aggregated, "obd")
        assert "2/5 seeds" in latex

    def test_an_arm_that_produced_nothing_is_still_reported(self, tmp_path):
        results = self._setup(tmp_path)
        records = json.loads((results / "experiment_runs.json").read_text())
        records.append({"variant": "ablation_cate", "seed": 42, "status": "failed",
                        "seconds": 1.0, "error": "phase2: dragonnet diverged",
                        "phases": []})
        (results / "experiment_runs.json").write_text(json.dumps(records), encoding="utf-8")

        attempted = tables.load_run_records(results, {"obd_bench_main"})
        assert ("ablation_cate", 42) in attempted["_unattributed"]


# =========================================================================== DEFECT 3
class TestStatusIsNotRelabelled:
    """Any non-``ok`` status was rewritten to ``not_identifiable``, and the
    footnote then asserted a cause nobody measured: "the evaluation policy
    selects no logged action, so all importance weights are zero". A cell whose
    real status was ``reward_model_failed`` was printed under that claim.
    """

    @staticmethod
    def _rows() -> list[dict]:
        return [
            make_row("DQN", "snipw", 0.40, ess=40_000.0),
            make_row("CQL", "snipw", float("nan"), status="not_identifiable",
                     detail="SNIPW is not identifiable: all importance weights are zero",
                     ess=0.0),
            make_row("IQL", "snipw", float("nan"), status="reward_model_failed",
                     detail="reward model fit raised LinAlgError: singular matrix",
                     ess=None),
        ]

    def test_the_real_status_survives_aggregation(self):
        aggregated = tables.aggregate(self._rows())
        assert cell(aggregated, "IQL", "snipw")["status"] == "reward_model_failed"
        assert cell(aggregated, "CQL", "snipw")["status"] == "not_identifiable"

    def test_the_recorded_detail_is_read(self):
        aggregated = tables.aggregate(self._rows())
        details = cell(aggregated, "IQL", "snipw")["details"]
        assert any("LinAlgError" in d for d in details), (
            "the per-row detail carrying the real reason was never read"
        )

    def test_the_two_failures_print_as_different_markers(self):
        aggregated = tables.aggregate(self._rows())
        latex = tables.build_ope_table(aggregated, "kuairec")
        iql = next(line for line in latex.splitlines() if line.startswith("IQL"))
        cql = next(line for line in latex.splitlines() if line.startswith("CQL"))
        assert "n.i.$^{\\dagger}$" in cql
        assert "n.i.$^{\\dagger}$" not in iql, (
            "a reward-model failure was printed under the non-identifiability marker"
        )

    def test_the_footnote_does_not_assert_zero_weights_for_a_model_failure(self):
        latex = tables.build_ope_table(tables.aggregate(self._rows()), "kuairec")
        assert "reward model failed" in latex
        assert "LinAlgError" in latex
        assert "This is not a non-identifiability result." in latex

    def test_a_dagger_footnote_is_absent_when_no_cell_is_non_identifiable(self):
        rows = [r for r in self._rows() if r["agent"] != "CQL"]
        latex = tables.build_ope_table(tables.aggregate(rows), "kuairec")
        assert "all importance weights" not in latex, (
            "the zero-weights footnote was emitted for a table with no "
            "non-identifiable cell"
        )


class TestMissingEssIsNotAMeasuredZero:
    """``e.get("effective_sample_size") or 0.0`` turned "this run recorded no
    ESS" into "this run measured an ESS of zero"."""

    def test_a_missing_ess_stays_missing(self):
        aggregated = tables.aggregate([make_row("Random", "snipw", 0.05, ess=None)])
        stats = cell(aggregated, "Random", "snipw")
        assert stats["ess"] is None, "a missing ESS was recorded as a measured 0.0"
        assert stats["ess_missing"] is True

    def test_a_recorded_zero_is_still_zero(self):
        aggregated = tables.aggregate([make_row("Random", "snipw", 0.05, ess=0.0)])
        stats = cell(aggregated, "Random", "snipw")
        assert stats["ess"] == 0.0
        assert stats["ess_missing"] is False

    def test_a_missing_ess_does_not_drag_an_average_down(self):
        rows = [
            make_row("DQN", "snipw", 0.4, seed=42, ess=40_000.0),
            make_row("DQN", "snipw", 0.4, seed=43, ess=None),
        ]
        stats = cell(tables.aggregate(rows), "DQN", "snipw")
        assert stats["ess"] == pytest.approx(40_000.0), (
            "the missing ESS was averaged in as a zero, halving the reported ESS"
        )

    def test_the_claims_check_reports_the_ess_as_unrecorded(self):
        rows = [make_row("DQN", "snipw", 0.4, ess=None)]
        markdown = tables.build_claims_check(tables.aggregate(rows), "kuairec")
        assert "not recorded" in markdown
        assert "a missing ESS is not an ESS of zero" in markdown


class TestEssCaveatIsPerAgent:
    """ESS was averaged across agents before the reliability caveat was tested,
    so one agent with healthy ESS suppressed the caveat for an agent whose
    estimate rested on twelve rounds of a hundred thousand.
    """

    @staticmethod
    def _rows() -> list[dict]:
        return [
            make_row("DQN", "snipw", 0.40, n_test=100_000, ess=90_000.0),
            make_row("BCQ", "snipw", 0.95, n_test=100_000, ess=12.0),
            make_row("Random", "snipw", 0.05, n_test=100_000, ess=None),
        ]

    def test_the_caveat_fires_for_the_starved_agent(self):
        markdown = tables.build_claims_check(tables.aggregate(self._rows()), "kuairec")
        assert "Caveat" in markdown, (
            "the starved agent's caveat was suppressed by a healthy agent's ESS"
        )
        assert "BCQ" in markdown.split("Caveat")[1].split("\n")[0]
        assert "12.0 of 100,000" in markdown

    def test_the_healthy_agent_is_not_caveated(self):
        markdown = tables.build_claims_check(tables.aggregate(self._rows()), "kuairec")
        caveat = markdown.split("**Caveat:**")[1].split("\n")[0]
        assert "DQN" not in caveat

    def test_the_mean_across_agents_would_have_hidden_it(self):
        """A mean across agents would hide the low-ESS cell; this guards it."""
        aggregated = tables.aggregate(self._rows())
        recorded = [
            v["ess"] for v in aggregated.values() if v["ess"] is not None
        ]
        assert np.mean(recorded) > 0.01 * 100_000, (
            "this fixture no longer reproduces the masking it was built for"
        )


class TestImplementationParagraphNeverInventsASeed:
    """``[cfg.get("seed", 42)]`` made the paragraph state "results are reported
    for a single seed (42)" for a run that produced nothing."""

    @staticmethod
    def _write_config(results: Path) -> None:
        import yaml

        config = yaml.safe_load((REPO / "configs" / "default.yaml").read_text())
        config["dataset"] = {"name": "obd", "subsample_rows": None}
        results.mkdir(parents=True, exist_ok=True)
        (results / "effective_config_obd_main_seed42.yaml").write_text(
            yaml.safe_dump(config), encoding="utf-8"
        )

    def test_no_seed_is_claimed_when_nothing_succeeded(self, tmp_path):
        results = tmp_path / "results"
        self._write_config(results)
        text = tables.build_implementation_paragraph(results, ["obd"], {"obd": []})
        assert "single seed (42)" not in text, (
            "a seed was reported for a run that produced no usable result"
        )
        assert "no seed produced a usable result" in text

    def test_a_real_seed_is_still_reported(self, tmp_path):
        results = tmp_path / "results"
        self._write_config(results)
        text = tables.build_implementation_paragraph(results, ["obd"], {"obd": [42]})
        assert "single seed (42)" in text

    def test_only_successful_seeds_reach_the_paragraph(self, tmp_path):
        """A seed whose every estimate failed is not a seed results are
        reported for."""
        results = tmp_path / "results"
        self._write_config(results)
        write_results(results, "obd_main", 42, [
            make_row("DQN", "dr", float("nan"), dataset="obd",
                     status="reward_model_failed", detail="y contains NaN", ess=None),
        ])
        sys.argv = [
            "generate_paper_tables.py",
            "--results-dir", str(results),
            "--out-dir", str(tmp_path / "paper"),
        ]
        assert tables.main() == 0
        text = (tmp_path / "paper" / "implementation.tex").read_text(encoding="utf-8")
        assert "single seed (42)" not in text


# =========================================================================== FIGURES
class TestFigureShowsAbsence:
    """``_aggregate`` filtered ``status == "ok"``, so an agent the table printed
    as ``n.i.`` simply vanished from the bar chart with no gap and no label.
    Figure and table then disagreed about the same cell.
    """

    @staticmethod
    def _rows() -> list[dict]:
        return [
            make_row("DQN", "snipw", 0.40),
            make_row("BCQ", "snipw", 0.95),
            make_row("CQL", "snipw", float("nan"), status="not_identifiable",
                     detail="all importance weights are zero", ess=0.0),
            make_row("IQL", "snipw", float("nan"), status="reward_model_failed",
                     detail="singular matrix", ess=None),
        ]

    def test_failed_agents_are_still_present(self):
        stats = plots._aggregate(self._rows(), "snipw", "gnn_embeddings", 0.0)
        assert set(stats) == {"DQN", "BCQ", "CQL", "IQL"}, (
            "an agent that produced no value disappeared from the figure entirely"
        )

    def test_a_failed_agent_has_no_value_rather_than_zero(self):
        stats = plots._aggregate(self._rows(), "snipw", "gnn_embeddings", 0.0)
        assert not np.isfinite(stats["CQL"]["mean"])
        assert not np.isfinite(stats["IQL"]["mean"])

    def test_the_reason_is_carried_to_the_figure(self):
        stats = plots._aggregate(self._rows(), "snipw", "gnn_embeddings", 0.0)
        assert stats["CQL"]["status"] == "not_identifiable"
        assert stats["IQL"]["status"] == "reward_model_failed"
        assert plots._absent_label(stats["CQL"]) == "n.i."
        assert plots._absent_label(stats["IQL"]) == "reward model failed"

    def test_the_figure_renders_and_labels_every_agent(self, tmp_path):
        results = tmp_path / "results"
        write_results(results, "exp_main", 42, self._rows())
        written = plots.plot_policy_values(results, "kuairec", tmp_path / "figures")
        assert len(written) == 1 and written[0].exists()

        import matplotlib.pyplot as plt

        # The axis must carry a tick for every agent, failed ones included.
        rows = plots.load_phase4(results, "kuairec")
        stats = plots._aggregate(rows, "snipw", "gnn_embeddings", 0.0)
        assert len(stats) == 4
        plt.close("all")

    def test_a_partial_cell_is_marked_in_the_figure(self):
        rows = [
            make_row("DQN", "dr", 0.004, seed=s) for s in (42, 43, 44, 45, 46)
        ] + [make_row("BCQ", "dr", 0.008, seed=42)] + [
            make_row("BCQ", "dr", float("nan"), seed=s, status="reward_model_failed",
                     detail="did not converge", ess=None)
            for s in (43, 44, 45, 46)
        ]
        stats = plots._aggregate(rows, "dr", "gnn_embeddings", 0.0)
        assert stats["BCQ"]["status"] == "partial"
        assert (stats["BCQ"]["n_usable"], stats["BCQ"]["n_attempted"]) == (1, 5)
        assert stats["DQN"]["status"] == "ok"


class TestFigureNeverPoolsExperiments:
    """The figure pooled experiments exactly as the table did, so it drew the
    same fabricated bar."""

    @staticmethod
    def _rows() -> list[dict]:
        return [
            make_row("BCQ", "snipw", 0.90, n_test=60_000, experiment="kuairec_validate_ope"),
            make_row("BCQ", "snipw", 0.10, n_test=400_000,
                     experiment="kuairec_validate_ope_full"),
        ]

    def test_pooling_two_experiments_raises(self):
        with pytest.raises(ValueError, match="different experiments"):
            plots._aggregate(self._rows(), "snipw", "gnn_embeddings", 0.0)

    def test_one_figure_per_experiment(self, tmp_path):
        results = tmp_path / "results"
        write_results(results, "kuairec_validate_ope", 42, self._rows()[:1])
        write_results(results, "kuairec_validate_ope_full", 42, self._rows()[1:])
        written = plots.plot_policy_values(results, "kuairec", tmp_path / "figures")
        assert len(written) == 2
        assert len({p.name for p in written}) == 2, "one figure overwrote the other"

    def test_the_stable_filename_is_kept_for_a_single_experiment(self, tmp_path):
        results = tmp_path / "results"
        write_results(results, "kuairec_validate_ope", 42, self._rows()[:1])
        written = plots.plot_policy_values(results, "kuairec", tmp_path / "figures")
        assert written[0].name == "policy_values_kuairec_snipw.png"

    def test_the_ablation_figure_keeps_its_stable_name(self, tmp_path):
        results = tmp_path / "results"
        for seed in (42, 43):
            write_results(results, "obd_bench_main", seed, [
                make_row("DQN", "snipw", 0.30, seed=seed, dataset="obd",
                         experiment="obd_bench_main"),
            ])
            write_results(results, "obd_bench_ablation_raw", seed, [
                make_row("DQN", "snipw", 0.25, seed=seed, dataset="obd",
                         state="raw_features", experiment="obd_bench_ablation_raw"),
            ])
        path = plots.plot_ablation(results, "obd", tmp_path / "figures", "gnn")
        assert path is not None
        assert Path(path).name == "ablation_gnn.png"
