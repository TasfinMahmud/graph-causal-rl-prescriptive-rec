"""Tests for the dataset loaders and pipeline preparation."""

import numpy as np
import pandas as pd
import pytest

from gcrl.config import load_config
from gcrl.data.kuairec import KUAIREC_PROVENANCE, load_kuairec, load_social_edges
from gcrl.data.obd import build_user_personas, load_obd, to_bandit_feedback
from gcrl.pipeline import prepare_dataset, summarise_dataset


@pytest.fixture
def obd_root(tmp_path):
    """A miniature OBD layout with genuine logged propensities."""
    rng = np.random.default_rng(0)
    n = 400
    frame = pd.DataFrame({
        "timestamp": np.arange(n),
        "item_id": rng.integers(0, 8, n),
        "position": rng.integers(1, 4, n),
        "click": rng.integers(0, 2, n),
        "propensity_score": rng.uniform(0.02, 0.4, n),
        "user_feature_0": rng.choice(["a", "b"], n),
        "user_feature_1": rng.choice(["x", "y"], n),
        "user-item_affinity_0": rng.normal(size=n),
        "user-item_affinity_1": rng.normal(size=n),
    })
    for policy in ("random", "bts"):
        directory = tmp_path / policy / "all"
        directory.mkdir(parents=True)
        frame.to_csv(directory / "all.csv")
    return tmp_path


@pytest.fixture
def kuairec_root(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    play = rng.uniform(0, 100, n)
    duration = rng.uniform(10, 60, n)
    frame = pd.DataFrame({
        "user_id": rng.integers(0, 20, n),
        "video_id": rng.integers(0, 15, n),
        "play_duration": play,
        "video_duration": duration,
        "watch_ratio": play / duration,
        "timestamp": np.arange(n, dtype=float),
    })
    directory = tmp_path / "data"
    directory.mkdir(parents=True)
    for name in ("big_matrix", "small_matrix"):
        frame.to_csv(directory / f"{name}.csv", index=False)
    pd.DataFrame({
        "user_id": [0, 1, 2, 3],
        "friend_list": ["[1, 2]", "[0]", "[]", "[0, 1]"],
    }).to_csv(directory / "social_network.csv", index=False)
    return tmp_path


class TestOBDLoader:
    def test_loads_the_requested_policy(self, obd_root):
        frame = load_obd(obd_root, policy="bts", campaign="all")
        assert len(frame) == 400
        assert "propensity_score" in frame.columns

    def test_rejects_an_unknown_policy(self, obd_root):
        with pytest.raises(ValueError, match="policy must be one of"):
            load_obd(obd_root, policy="greedy")

    def test_missing_propensity_column_raises(self, tmp_path):
        """A constant propensity cannot be silently substituted."""
        directory = tmp_path / "bts" / "all"
        directory.mkdir(parents=True)
        pd.DataFrame({
            "item_id": [0, 1], "click": [0, 1], "position": [1, 2], "timestamp": [0, 1]
        }).to_csv(directory / "all.csv")
        with pytest.raises(ValueError, match="propensity"):
            load_obd(tmp_path, policy="bts", campaign="all")

    def test_subsampling_is_reproducible(self, obd_root):
        a = load_obd(obd_root, policy="bts", subsample_rows=100, seed=1)
        b = load_obd(obd_root, policy="bts", subsample_rows=100, seed=1)
        pd.testing.assert_frame_equal(a, b)

    def test_personas_are_stable_across_processes(self, obd_root):
        """Personas are built from the feature tuple, not a randomised hash."""
        frame = load_obd(obd_root, policy="bts")
        a = build_user_personas(frame, ["user_feature_0", "user_feature_1"])
        b = build_user_personas(frame.copy(), ["user_feature_0", "user_feature_1"])
        pd.testing.assert_series_equal(a, b)
        assert a.nunique() <= 4

    def test_bandit_feedback_rejects_out_of_range_actions(self, obd_root):
        frame = load_obd(obd_root, policy="bts")
        with pytest.raises(ValueError, match="exceeds the declared action space"):
            to_bandit_feedback(frame, n_actions=3, feature_columns=["user-item_affinity_0"])

    def test_bandit_feedback_can_use_embeddings_as_context(self, obd_root):
        frame = load_obd(obd_root, policy="bts")
        user_index = np.zeros(len(frame), dtype=np.int64)
        embeddings = np.random.default_rng(0).normal(size=(1, 16))
        feedback = to_bandit_feedback(
            frame, 8, ["user-item_affinity_0"],
            embedding_lookup=embeddings, user_index=user_index,
        )
        assert feedback.context.shape[1] == 16
        assert feedback.metadata["context_source"] == "gnn_embeddings"


class TestKuaiRecLoader:
    def test_applies_the_declared_reward_rule(self, kuairec_root):
        data = load_kuairec(kuairec_root, matrix="small", reward_threshold=2.0)
        expected = (data.interactions["watch_ratio"] >= 2.0).astype(float)
        np.testing.assert_array_equal(data.interactions["reward"], expected)
        assert data.reward_threshold == 2.0

    def test_threshold_changes_the_reward(self, kuairec_root):
        low = load_kuairec(kuairec_root, matrix="small", reward_threshold=0.5)
        high = load_kuairec(kuairec_root, matrix="small", reward_threshold=5.0)
        assert low.interactions["reward"].mean() > high.interactions["reward"].mean()

    def test_rejects_an_unknown_matrix(self, kuairec_root):
        with pytest.raises(ValueError, match="matrix must be"):
            load_kuairec(kuairec_root, matrix="medium")

    def test_parses_the_social_graph(self, kuairec_root):
        edges = load_social_edges(kuairec_root / "data")
        assert edges.shape[0] == 2
        assert edges.shape[1] == 5

    def test_provenance_records_the_circular_dependency(self):
        """This lineage is what lets the causal validator refuse a watch_ratio treatment."""
        assert "play_duration" in KUAIREC_PROVENANCE["watch_ratio"]

    def test_exact_matrix_requires_the_fully_observed_block(self, kuairec_root):
        from gcrl.data.kuairec import build_full_reward_matrix
        from gcrl.encoders import IdentifierIndexer

        data = load_kuairec(kuairec_root, matrix="big")
        users = IdentifierIndexer("u").fit(data.interactions["user_id"])
        items = IdentifierIndexer("i").fit(data.interactions["video_id"])
        with pytest.raises(ValueError, match="fully observed small_matrix"):
            build_full_reward_matrix(data, users, items)


class TestPrepareDataset:
    def test_graph_is_built_from_training_edges_only(self, obd_root):
        """Including held-out edges would leak them into the representation."""
        config = load_config("configs/obd.yaml")
        config.dataset.num_actions = 8
        config.dataset.subsample_rows = None
        frame = load_obd(obd_root, policy="bts")
        frame["user_persona"] = build_user_personas(
            frame, ["user_feature_0", "user_feature_1"]
        ).astype("category").cat.codes

        prepared = prepare_dataset(
            frame, config, "user_persona", "item_id", "click", "propensity_score"
        )
        n_train_edges = len(prepared.split.train)
        assert prepared.graph.edge_index.size(1) <= 2 * n_train_edges

    def test_uses_logged_propensities(self, obd_root):
        config = load_config("configs/obd.yaml")
        config.dataset.num_actions = 8
        config.dataset.subsample_rows = None
        frame = load_obd(obd_root, policy="bts")
        frame["user_persona"] = 0
        prepared = prepare_dataset(
            frame, config, "user_persona", "item_id", "click", "propensity_score"
        )
        assert prepared.frame["__propensity"].nunique() > 1

    def test_missing_propensity_raises_when_ope_needs_it(self, obd_root):
        config = load_config("configs/obd.yaml")
        config.dataset.num_actions = 8
        config.dataset.subsample_rows = None
        frame = load_obd(obd_root, policy="bts").drop(columns=["propensity_score"])
        frame["user_persona"] = 0
        with pytest.raises(ValueError, match="require logged propensities"):
            prepare_dataset(frame, config, "user_persona", "item_id", "click", "propensity_score")

    def test_summary_is_derived_from_loaded_data(self, obd_root):
        """The dataset table should be generated, not transcribed."""
        config = load_config("configs/obd.yaml")
        config.dataset.num_actions = 8
        config.dataset.subsample_rows = None
        frame = load_obd(obd_root, policy="bts")
        frame["user_persona"] = 0
        prepared = prepare_dataset(
            frame, config, "user_persona", "item_id", "click", "propensity_score"
        )
        summary = summarise_dataset(prepared, config)
        assert summary["n_interactions"] == len(frame)
        assert summary["propensity_is_constant"] is False
        assert summary["n_train"] + summary["n_validation"] + summary["n_test"] == len(frame)
