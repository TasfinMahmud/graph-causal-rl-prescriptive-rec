"""Tests for the causal estimation phase."""

import numpy as np
import pytest

from gcrl.config import load_config
from gcrl.models.causal import CausalValidationError
from gcrl.phases.phase2_causal import run_phase2


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        f"""
experiment_name: t
seed: 3
device: cpu
paths:
  root: "{tmp_path.as_posix()}"
dataset:
  name: t
  loader: obd
  num_actions: 4
causal:
  enabled: true
  estimators: [SLearner]
  primary_estimator: SLearner
  treatment_column: treated
  outcome_column: outcome
  n_estimators: 5
  max_depth: 3
gnn:
  architectures: [LightGCN]
"""
    )
    cfg = load_config(path)
    cfg.paths.ensure_output_dirs()
    return cfg


@pytest.fixture
def valid_data():
    rng = np.random.default_rng(0)
    n = 800
    X = rng.normal(size=(n, 4))
    T = (rng.random(n) < 0.5).astype(int)
    Y = X[:, 0] + T * (1.5 + X[:, 1]) + 0.2 * rng.normal(size=n)
    mask = np.zeros(n, dtype=bool)
    mask[: int(0.8 * n)] = True
    return Y, T, X, mask


class TestRunPhase2:
    def test_produces_cate_for_each_estimator(self, config, valid_data):
        Y, T, X, mask = valid_data
        cate = run_phase2(Y, T, X, mask, config, seed=3)
        assert set(cate) == {"SLearner"}
        assert len(cate["SLearner"]) == len(Y)
        assert np.isfinite(cate["SLearner"]).all()

    def test_writes_results_and_cate_artefacts(self, config, valid_data):
        Y, T, X, mask = valid_data
        run_phase2(Y, T, X, mask, config, seed=3)
        assert (config.paths.results / "phase2_causal_t_seed3.json").exists()
        # Ask the code where it puts the artefact rather than hardcoding the
        # name. The path carries a fingerprint of everything that produced the
        # CATE, so a hardcoded name goes stale the moment a setting changes --
        # and a stale assertion is a test that stops testing.
        from gcrl.cli import cate_artefact_path

        assert cate_artefact_path(config, "SLearner", 3).exists()

    def test_refuses_a_circular_setup(self, config):
        """The phase must not estimate anything from a treatment derived from the outcome."""
        rng = np.random.default_rng(0)
        n = 600
        video_duration = rng.uniform(5, 60, n)
        play_duration = rng.uniform(0, 120, n)
        T = ((play_duration / video_duration) > 1.0).astype(int)
        X = np.column_stack([video_duration, rng.normal(size=n)])
        mask = np.ones(n, dtype=bool)

        config.causal.treatment_column = "treated"
        config.causal.outcome_column = "play_duration"
        provenance = {"treated": ["watch_ratio"], "watch_ratio": ["play_duration", "video_duration"]}

        with pytest.raises(CausalValidationError, match="derived from the outcome"):
            run_phase2(
                play_duration, T, X, mask, config, seed=3, provenance=provenance
            )

    def test_records_the_covariate_source(self, config, valid_data):
        """Which representation fed the estimator is the ablation's independent variable."""
        import json

        Y, T, X, mask = valid_data
        run_phase2(Y, T, X, mask, config, seed=3, covariate_source="gnn_embeddings:LightGCN")
        payload = json.loads((config.paths.results / "phase2_causal_t_seed3.json").read_text())
        assert payload["covariate_source"] == "gnn_embeddings:LightGCN"
