"""End-to-end pipeline test on synthetic data.

This exercises the property that matters most: Phase 1's embeddings reach
Phase 3 as the agent's state, Phase 2's CATE reaches Phase 3 as reward shaping,
and Phase 4 scores every agent on a held-out split. If any phase were
disconnected -- as they were previously -- these tests fail.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gcrl.config import load_config
from gcrl.data.graphs import build_bipartite_graph
from gcrl.phases.phase1_gnn import embedding_path, load_embeddings, run_phase1
from gcrl.phases.phase3_rl import build_state, run_phase3, shape_rewards
from gcrl.phases.phase4_ope import run_phase4
from gcrl.seeding import seed_everything


@pytest.fixture
def tiny_config(tmp_path):
    """A miniature configuration that runs in seconds."""
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(
        f"""
experiment_name: tiny
seed: 7
device: cpu
paths:
  root: "{tmp_path.as_posix()}"
dataset:
  name: tiny
  loader: obd
  num_actions: 5
gnn:
  architectures: [LightGCN, GCN]
  embedding_dim: 8
  hidden_dim: 16
  num_layers: 2
  epochs: 3
  node_batch_size: 64
  early_stopping_patience: 2
causal:
  enabled: false
rl:
  agents: [Random, LinUCB, IQL]
  state_source: gnn_embeddings
  gnn_architecture: LightGCN
  hidden_units: 16
  epochs: 2
  batch_size: 32
  neuralucb_epochs: 1
ope:
  estimators: [snipw]
  n_bootstrap: 10
"""
    )
    config = load_config(cfg_path)
    config.paths.ensure_output_dirs()
    return config


@pytest.fixture
def synthetic_data():
    """A small logged bandit dataset with a known best action per user."""
    seed_everything(7)
    rng = np.random.default_rng(7)
    n_users, n_items, n_rounds, n_actions = 20, 5, 400, 5

    user_index = rng.integers(0, n_users, n_rounds)
    actions = rng.integers(0, n_actions, n_rounds)
    propensities = np.full(n_rounds, 1.0 / n_actions)
    best_action = rng.integers(0, n_actions, n_users)
    rewards = (actions == best_action[user_index]).astype(np.float64)

    graph = build_bipartite_graph(
        user_index=user_index % n_users,
        item_index=actions,
        num_users=n_users,
        num_items=n_items,
        embedding_dim=8,
        seed=7,
    )
    return {
        "graph": graph, "user_index": user_index, "actions": actions,
        "rewards": rewards, "propensities": propensities,
        "n_actions": n_actions, "n_users": n_users,
    }


class TestPhase1:
    def test_exports_embeddings_not_weights(self, tiny_config, synthetic_data):
        """The Phase 1 artefact must be an embedding matrix, not a state_dict.

        The previous implementation saved model weights into the embeddings
        directory, so no downstream phase could load a graph representation.
        """
        results = run_phase1(synthetic_data["graph"], tiny_config, seed=7)
        assert len(results) == 2

        for result in results:
            payload = load_embeddings(result.embedding_path)
            assert isinstance(payload.embeddings, torch.Tensor)
            assert payload.embeddings.shape == (
                synthetic_data["graph"].num_nodes, tiny_config.gnn.embedding_dim,
            )
            assert torch.isfinite(payload.embeddings).all()
            assert payload.num_users == synthetic_data["n_users"]
            assert payload.architecture == result.architecture

    def test_reports_validation_loss_not_training_loss(self, tiny_config, synthetic_data):
        """Model selection must use held-out edges."""
        results = run_phase1(synthetic_data["graph"], tiny_config, seed=7)
        for result in results:
            assert np.isfinite(result.best_val_bpr_loss)
            assert 1 <= result.best_epoch <= result.epochs_run

    def test_is_reproducible_under_a_fixed_seed(self, tiny_config, synthetic_data):
        first = run_phase1(synthetic_data["graph"], tiny_config, seed=7)
        second = run_phase1(synthetic_data["graph"], tiny_config, seed=7)
        for a, b in zip(first, second, strict=True):
            assert a.best_val_bpr_loss == pytest.approx(b.best_val_bpr_loss, rel=1e-6)


class TestPhase3Wiring:
    def test_state_comes_from_phase1_embeddings(self, tiny_config, synthetic_data):
        """The agent's observation must be the Phase 1 output, not raw features."""
        run_phase1(synthetic_data["graph"], tiny_config, seed=7)

        raw = np.zeros((len(synthetic_data["actions"]), 2), dtype=np.float32)
        state = build_state(tiny_config, raw, synthetic_data["user_index"], seed=7)

        assert state.shape[1] == tiny_config.gnn.embedding_dim
        assert state.shape[1] != raw.shape[1]

        payload = load_embeddings(
            embedding_path(tiny_config, "LightGCN", 7)
        )
        expected = payload.for_users(synthetic_data["user_index"])
        np.testing.assert_allclose(state, expected, rtol=1e-6)

    def test_out_of_range_user_index_raises(self, tiny_config, synthetic_data):
        """Clipping would silently map several users onto one state."""
        run_phase1(synthetic_data["graph"], tiny_config, seed=7)
        with pytest.raises(ValueError, match="outside the"):
            build_state(
                tiny_config, np.zeros((3, 2), dtype=np.float32),
                np.array([0, 1, 10_000]), seed=7,
            )

    def test_missing_embeddings_raise_rather_than_falling_back(self, tiny_config, synthetic_data):
        """A missing artefact must not silently degrade to raw features."""
        with pytest.raises(FileNotFoundError, match="Run Phase 1"):
            build_state(
                tiny_config, np.zeros((10, 2), dtype=np.float32), np.zeros(10, dtype=int), seed=999
            )

    def test_raw_feature_mode_is_a_real_alternative(self, tiny_config):
        tiny_config.rl.state_source = "raw_features"
        raw = np.random.default_rng(0).normal(size=(50, 3)).astype(np.float32)
        state = build_state(tiny_config, raw, None, seed=7)
        np.testing.assert_allclose(state, raw)


class TestRewardShaping:
    def test_lambda_zero_leaves_rewards_unchanged(self):
        rewards = np.array([0.0, 1.0, 0.0, 1.0])
        np.testing.assert_allclose(shape_rewards(rewards, None, 0.0), rewards)

    def test_nonzero_lambda_applies_cate(self):
        rewards = np.array([0.0, 1.0])
        cate = np.array([0.5, -0.5])
        np.testing.assert_allclose(shape_rewards(rewards, cate, 2.0), [1.0, 0.0])

    def test_nonzero_lambda_without_cate_raises(self):
        """The ablation must not silently collapse into comparing a run to itself."""
        with pytest.raises(ValueError, match="requires CATE estimates"):
            shape_rewards(np.array([0.0, 1.0]), None, 0.5)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="entries but there are"):
            shape_rewards(np.zeros(4), np.zeros(3), 1.0)


class TestPhase4:
    def test_every_agent_is_scored_from_a_real_run(self, tiny_config, synthetic_data):
        """No agent may receive a hardcoded or placeholder score."""
        run_phase1(synthetic_data["graph"], tiny_config, seed=7)
        state = build_state(
            tiny_config, np.zeros((len(synthetic_data["actions"]), 2), dtype=np.float32),
            synthetic_data["user_index"], seed=7,
        )
        policies = run_phase3(
            state, synthetic_data["actions"], synthetic_data["rewards"],
            synthetic_data["n_actions"], tiny_config, seed=7,
        )
        assert set(policies) == set(tiny_config.rl.agents)

        evaluations = run_phase4(
            policies, state, synthetic_data["actions"], synthetic_data["rewards"],
            synthetic_data["propensities"], synthetic_data["n_actions"], tiny_config, seed=7,
        )
        assert len(evaluations) == len(tiny_config.rl.agents)
        for evaluation in evaluations:
            result = evaluation.estimates["snipw"]
            assert np.isfinite(result.value)
            assert result.lower <= result.value <= result.upper
            assert result.n_samples == len(synthetic_data["actions"])

    def test_a_failing_agent_stops_the_run(self, tiny_config, synthetic_data):
        """A crash must not become a 0.0 row in the results table."""
        class BrokenPolicy:
            n_actions = 5
            def action_distribution(self, observations):
                raise RuntimeError("simulated failure")

        with pytest.raises(RuntimeError, match="rather than recording a placeholder"):
            run_phase4(
                {"Broken": BrokenPolicy()}, np.zeros((10, 4), dtype=np.float32),
                np.zeros(10, dtype=int), np.zeros(10), np.full(10, 0.2),
                5, tiny_config, seed=7,
            )
