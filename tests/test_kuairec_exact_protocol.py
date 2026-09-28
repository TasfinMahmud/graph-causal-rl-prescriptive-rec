"""The KuaiRec two-matrix protocol and the ``exact`` estimator.

The defect these tests pin: ``exact`` was unreachable. ``command_phase4`` never
passed ``true_rewards``, ``build_full_reward_matrix`` was called from nowhere but
a test, and ``dataset.eval_file`` was declared in the config schema and read by
no code -- so every KuaiRec config asked for ``exact`` and aborted in Phase 4,
and the protocol the docstrings described (train on the sparse ``big_matrix``,
evaluate exactly on the fully observed ``small_matrix``) was not implemented at
all.

The fixtures here are synthetic but structurally identical to the published
files: ``small``'s users and items are strict subsets of ``big``'s, ``small`` is
almost fully observed with a deliberate hole, and the raw identifiers are
non-contiguous so that a per-file indexer assigns the same raw id a *different*
code in each file. That last property is what gives the alignment tests teeth.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gcrl.cli import main
from gcrl.config import ConfigError, load_config
from gcrl.data.kuairec import (
    ExactRewardLookup,
    build_exact_reward_lookup,
    build_full_reward_matrix,
    load_kuairec_train_eval,
)
from gcrl.encoders import IdentifierIndexer

N_BIG_USERS, N_BIG_ITEMS = 60, 40
N_SMALL_USERS, N_SMALL_ITEMS = 12, 15
SMALL_HOLES = 4


def write_kuairec_fixture(root, seed: int = 0, overlap: bool = False):
    """A miniature KuaiRec layout with the real files' structure.

    The structural fact that matters, and that the first version of this
    protocol got wrong: the evaluation users DO appear in the sparse log, and
    the evaluation items DO appear in the sparse log, but **not together**.
    KuaiRec's authors removed the fully observed block's (user, item) pairs from
    the log to prevent leakage, so the intersection is empty by construction. On
    the published files:

        small users                          1,411
        ...present in big_matrix             1,411   (all of them)
        their rows in big_matrix           571,061   over 6,334 OTHER items
        small items                          3,327
        ...rows in big_matrix           10,272,525   from 5,765 OTHER users
        (small user, small item) rows in big     0   <-- the killer

    ``overlap=True`` builds the same layout WITHOUT that property, so the tests
    can show the code reports a different regime when the intersection is not
    empty rather than assuming KuaiRec's shape.
    """
    rng = np.random.default_rng(seed)
    # Non-contiguous, non-zero-based ids: a per-file index then yields a
    # different code for the same raw id in each file.
    big_users = np.arange(100, 100 + 3 * N_BIG_USERS, 3)
    big_items = np.arange(500, 500 + 2 * N_BIG_ITEMS, 2)
    small_users = np.sort(rng.choice(big_users, N_SMALL_USERS, replace=False))
    small_items = np.sort(rng.choice(big_items, N_SMALL_ITEMS, replace=False))
    other_users = np.setdiff1d(big_users, small_users)
    other_items = np.setdiff1d(big_items, small_items)

    def rows_for(users, items, n_rows):
        return rng.choice(users, n_rows), rng.choice(items, n_rows)

    def outcomes(u, i):
        duration = rng.uniform(10, 60, len(u))
        # watch_ratio is a function of the (user, item) CELL, so a misaligned
        # index produces wrong rewards rather than merely different noise.
        base = ((u * 7 + i * 13) % 11) / 4.0
        play = duration * np.clip(base + rng.normal(0, 0.15, len(u)), 0.05, None)
        return pd.DataFrame({
            "user_id": u, "video_id": i, "play_duration": play,
            "video_duration": duration, "watch_ratio": play / duration,
            "timestamp": np.sort(rng.uniform(0, 1e6, len(u))),
        })

    # The sparse log: the OTHER users see the whole catalogue (so the evaluation
    # actions are well represented), while the evaluation users see only items
    # OUTSIDE the evaluation action space -- unless `overlap` says otherwise.
    log_parts = [outcomes(*rows_for(other_users, big_items, 1800))]
    log_parts.append(
        outcomes(*rows_for(small_users, big_items if overlap else other_items, 600))
    )
    log = pd.concat(log_parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    # The evaluation block: every (user, item) pair, minus a few deliberate holes.
    u, i = np.meshgrid(small_users, small_items, indexing="ij")
    u, i = u.ravel(), i.ravel()
    keep = np.ones(len(u), dtype=bool)
    keep[rng.choice(len(u), SMALL_HOLES, replace=False)] = False
    block = outcomes(u[keep], i[keep])

    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    log.to_csv(data / "big_matrix.csv", index=False)
    block.to_csv(data / "small_matrix.csv", index=False)
    pd.DataFrame({
        "user_id": big_users[:5],
        "friend_list": [f"[{big_users[1]}, {big_users[2]}]"] * 5,
    }).to_csv(data / "social_network.csv", index=False)
    return root


@pytest.fixture
def kuairec_root(tmp_path):
    return write_kuairec_fixture(tmp_path / "raw" / "kuairec")


@pytest.fixture
def kuairec_config(tmp_path, kuairec_root):
    """A miniature config that runs the two-matrix protocol in seconds."""
    path = tmp_path / "kuairec_tiny.yaml"
    path.write_text(f"""
experiment_name: kuairec_tiny
seed: 5
device: cpu
paths:
  root: "{tmp_path.as_posix()}"
  raw_data: "{(tmp_path / 'raw').as_posix()}"
dataset:
  name: kuairec
  loader: kuairec
  train_file: big_matrix.csv
  eval_file: small_matrix.csv
  feature_columns: [video_duration]
  num_actions: {N_SMALL_ITEMS}
gnn:
  architectures: [LightGCN]
  embedding_dim: 8
  hidden_dim: 16
  num_layers: 2
  epochs: 2
  node_batch_size: 64
  early_stopping_patience: 2
causal:
  enabled: false
rl:
  agents: [Random, IQL]
  state_source: gnn_embeddings
  gnn_architecture: LightGCN
  hidden_units: 16
  epochs: 2
  batch_size: 32
ope:
  estimators: [exact, dm, dr, mrdr, snipw]
  behaviour_policy: estimated
  cross_fitting_folds: 3
  n_bootstrap: 10
""")
    config = load_config(path)
    config.paths.ensure_output_dirs()
    return path, config


@pytest.fixture
def prepared(kuairec_config, monkeypatch):
    from gcrl.cli import _build_dataset

    monkeypatch.setenv("GCRL_NO_DATASET_CACHE", "1")
    _, config = kuairec_config
    return _build_dataset(config, seed=5)


class TestTwoMatrixProtocol:
    """``dataset.eval_file`` must actually select the evaluation block."""

    def test_test_partition_is_the_evaluation_block(self, prepared, kuairec_root):
        small = pd.read_csv(kuairec_root / "data" / "small_matrix.csv")
        assert len(prepared.split.test) == len(small)
        assert prepared.split.strategy.endswith("+eval_block")

    def test_the_test_rows_are_the_evaluation_file_and_nothing_else(self, prepared, kuairec_root):
        """The test partition is small_matrix verbatim; training is big_matrix."""
        small = pd.read_csv(kuairec_root / "data" / "small_matrix.csv")
        test = prepared.split.test
        assert (
            sorted(zip(test["user_id"], test["video_id"], strict=True))
            == sorted(zip(small["user_id"], small["video_id"], strict=True))
        )
        # Every logged row of the training file lands in train or validation.
        assert len(prepared.split.train) + len(prepared.split.validation) + len(test) == len(
            prepared.frame
        )
        assert len(prepared.split.train) > 0 and len(prepared.split.validation) > 0

    def test_action_space_is_the_evaluation_block_item_universe(self, prepared, kuairec_root):
        """Only actions whose true reward is known may be selectable."""
        small = pd.read_csv(kuairec_root / "data" / "small_matrix.csv")
        big = pd.read_csv(kuairec_root / "data" / "big_matrix.csv")
        assert big["video_id"].nunique() > small["video_id"].nunique()
        assert len(prepared.item_indexer) == small["video_id"].nunique()
        assert prepared.n_actions == N_SMALL_ITEMS

    def test_the_training_log_is_never_filtered_to_the_action_space(self, kuairec_root):
        """Filtering it is what deleted every evaluation user. It must not happen."""
        big = pd.read_csv(kuairec_root / "data" / "big_matrix.csv")
        bundle = load_kuairec_train_eval(kuairec_root)
        assert len(bundle.train.interactions) == len(big)
        outside = set(bundle.train.interactions["video_id"]) - set(bundle.action_universe)
        assert outside, "the log must keep the items the evaluation block cannot score"

    def test_policy_rows_are_the_action_space_subset_of_the_log(self, prepared, kuairec_root):
        """An agent cannot be trained on an action outside its own action set."""
        small = pd.read_csv(kuairec_root / "data" / "small_matrix.csv")
        universe = set(small["video_id"])
        assert set(prepared.split.train["video_id"]).issubset(universe)
        assert set(prepared.split.validation["video_id"]).issubset(universe)
        # ...but the representation saw strictly more than that.
        assert prepared.representation_rows > prepared.policy_rows
        assert prepared.policy_rows == len(prepared.split.train)

    def test_subsample_rows_counts_whole_log_rows(self, kuairec_root):
        bundle = load_kuairec_train_eval(kuairec_root, subsample_rows=200)
        assert len(bundle.train.interactions) == 200

    def test_subsampling_rounds_does_not_shrink_the_ground_truth(self, kuairec_root):
        bundle = load_kuairec_train_eval(kuairec_root, eval_subsample_rows=50)
        assert len(bundle.eval_rounds.interactions) == 50
        # The ground-truth block keeps every recorded cell.
        assert len(bundle.eval_block.interactions) == N_SMALL_USERS * N_SMALL_ITEMS - SMALL_HOLES
        assert len(bundle.action_universe) == N_SMALL_ITEMS

    def test_evaluating_on_the_training_matrix_is_refused(self, kuairec_root):
        with pytest.raises(ValueError, match="must be different matrices"):
            load_kuairec_train_eval(kuairec_root, train_matrix="big", eval_matrix="big")

    def test_a_mistyped_matrix_name_raises(self, kuairec_config, monkeypatch):
        from gcrl.cli import _build_dataset

        monkeypatch.setenv("GCRL_NO_DATASET_CACHE", "1")
        _, config = kuairec_config
        config.dataset.eval_file = "smal_matrix.csv"
        with pytest.raises(ValueError, match="does not name a KuaiRec matrix"):
            _build_dataset(config, seed=5)


class TestSharedIndexers:
    """A per-file index is the failure this protocol is most exposed to."""

    def test_one_index_covers_both_matrices(self, prepared, kuairec_root):
        big = pd.read_csv(kuairec_root / "data" / "big_matrix.csv")
        small = pd.read_csv(kuairec_root / "data" / "small_matrix.csv")
        assert set(small["user_id"]).issubset(set(big["user_id"]))
        assert set(small["video_id"]).issubset(set(big["video_id"]))
        # Users from BOTH blocks are addressable by the one user indexer.
        assert set(small["user_id"]).issubset(prepared.user_indexer.mapping)
        assert set(prepared.split.train["user_id"]).issubset(prepared.user_indexer.mapping)

    def test_the_same_raw_id_gets_the_same_index_in_both_files(self, prepared, kuairec_root):
        """One index across both files, even where they share no rows.

        Under the cold-item regime the policy-training partition holds none of
        the evaluation users, so the check has to go back to the raw log rather
        than compare two partitions.
        """
        big = pd.read_csv(kuairec_root / "data" / "big_matrix.csv")
        test = prepared.split.test
        shared = sorted(set(big["user_id"]) & set(test["user_id"]))
        assert len(shared) == N_SMALL_USERS, "every evaluation user is in the log"
        for raw in shared[:5]:
            from_block = test.loc[test["user_id"] == raw, "__user_index"].unique().tolist()
            from_log = prepared.user_indexer.transform(
                big.loc[big["user_id"] == raw, "user_id"]
            )
            assert from_block == [prepared.user_indexer.mapping[raw]]
            assert set(from_log.tolist()) == set(from_block)

    def test_a_per_file_index_would_assign_a_different_code(self, kuairec_root):
        """The premise of the alignment check: the two indexes really do differ."""
        big = pd.read_csv(kuairec_root / "data" / "big_matrix.csv")
        small = pd.read_csv(kuairec_root / "data" / "small_matrix.csv")
        shared = IdentifierIndexer("item").fit(np.sort(small["video_id"].unique()))
        per_file = IdentifierIndexer("item").fit(big["video_id"])
        raw = np.sort(small["video_id"].unique())
        assert not np.array_equal(shared.transform(raw), per_file.transform(raw))

    def test_ground_truth_reproduces_every_logged_evaluation_reward(self, prepared, kuairec_root):
        test = prepared.arrays("test")
        lookup = build_exact_reward_lookup(
            _eval_block(kuairec_root), prepared.user_indexer, prepared.item_indexer,
            round_user_index=test["user_index"], n_actions=prepared.n_actions,
            n_users=len(prepared.user_indexer),
        )
        # Every evaluation round is itself a recorded cell, so this must hold
        # exactly. It is the only thing that would notice an unshared index.
        lookup.assert_reproduces_logged_rewards(test["actions"], test["rewards"])

    def test_an_unshared_item_index_is_caught(self, prepared, kuairec_root):
        big = pd.read_csv(kuairec_root / "data" / "big_matrix.csv")
        per_file = IdentifierIndexer("item").fit(big["video_id"])
        test = prepared.arrays("test")
        lookup = build_exact_reward_lookup(
            _eval_block(kuairec_root), prepared.user_indexer, per_file,
            round_user_index=test["user_index"], n_actions=len(per_file),
            n_users=len(prepared.user_indexer),
        )
        with pytest.raises(ValueError, match="indexer"):
            lookup.assert_reproduces_logged_rewards(test["actions"], test["rewards"])

    def test_a_shifted_round_to_user_mapping_is_caught(self, prepared, kuairec_root):
        """Same cells observed, wrong rows: only the value check can see this."""
        test = prepared.arrays("test")
        lookup = build_exact_reward_lookup(
            _eval_block(kuairec_root), prepared.user_indexer, prepared.item_indexer,
            round_user_index=np.roll(test["user_index"], 1), n_actions=prepared.n_actions,
            n_users=len(prepared.user_indexer),
        )
        with pytest.raises(ValueError):
            lookup.assert_reproduces_logged_rewards(test["actions"], test["rewards"])


def _eval_block(root):
    from gcrl.data.kuairec import load_kuairec

    return load_kuairec(
        root, matrix="small", subsample_rows=None,
        load_social_graph=False, load_user_features=False,
    )


class TestUnobservedCells:
    """An unobserved cell is masked out of the expectation, never imputed as 0."""

    def _lookup(self):
        rewards = np.ones((3, 4), dtype=np.float32)
        observed = np.ones((3, 4), dtype=bool)
        observed[0, 0] = False          # user 0 has no record for action 0
        return ExactRewardLookup(rewards, observed, np.array([0, 1, 2]), {"duplicate_cells": 0})

    def test_masked_not_zero_imputed(self):
        from gcrl.evaluation.ope import DeterministicPolicy, exact_value

        lookup = self._lookup()
        value = exact_value(DeterministicPolicy(np.array([0, 0, 0]), 4), lookup)
        # Zero-imputation would give (0 + 1 + 1) / 3 = 0.6667. Masking the one
        # unrecorded cell gives the mean over the two cells that were recorded.
        assert value == pytest.approx(1.0)

    def test_uniform_policy_averages_only_observed_actions(self):
        from gcrl.evaluation.ope import UniformPolicy, exact_value

        lookup = self._lookup()
        assert exact_value(UniformPolicy(3, 4), lookup) == pytest.approx(1.0)

    def test_matches_a_dense_masked_reference(self):
        from gcrl.evaluation.ope import DeterministicPolicy, exact_value

        rng = np.random.default_rng(3)
        rewards = rng.random((6, 5)).astype(np.float32)
        observed = rng.random((6, 5)) > 0.25
        rows = rng.integers(0, 6, 40)
        lookup = ExactRewardLookup(rewards, observed, rows, {"duplicate_cells": 0})
        dense = np.ma.MaskedArray(rewards[rows].astype(np.float64), mask=~observed[rows])
        actions = rng.integers(0, 5, 40)
        got = exact_value(DeterministicPolicy(actions, 5), lookup)
        want = float(dense[np.arange(40), actions].mean())
        assert got == pytest.approx(want)

    def test_coverage_is_recorded(self, prepared, kuairec_root):
        test = prepared.arrays("test")
        lookup = build_exact_reward_lookup(
            _eval_block(kuairec_root), prepared.user_indexer, prepared.item_indexer,
            round_user_index=test["user_index"], n_actions=prepared.n_actions,
            n_users=len(prepared.user_indexer),
        )
        report = lookup.report
        assert 0.9 < report["coverage"] < 1.0
        assert report["cells_observed"] < report["cells_consulted"]
        assert report["unobserved_policy"] == "masked_out_of_expectation"
        assert report["duplicate_cells"] == 0

    def test_a_policy_with_no_observed_cell_raises(self, kuairec_config):
        """Averaging over an empty set must stop the run, not return NaN."""
        from gcrl.evaluation.ope import DeterministicPolicy
        from gcrl.phases.phase4_ope import run_phase4

        _, config = kuairec_config
        config.ope.estimators = ["exact"]
        rewards = np.ones((2, 3), dtype=np.float32)
        observed = np.zeros((2, 3), dtype=bool)
        observed[:, 1:] = True
        lookup = ExactRewardLookup(rewards, observed, np.array([0, 1, 0, 1]), {})

        class PicksAction0:
            n_actions = 3

            def action_distribution(self, observations):
                return DeterministicPolicy(np.zeros(len(observations), dtype=np.int64), 3)

        with pytest.raises(RuntimeError, match="not defined"):
            run_phase4(
                {"Zero": PicksAction0()}, np.zeros((4, 2), dtype=np.float32),
                np.array([1, 1, 2, 2]), np.ones(4), np.full(4, 0.5), 3, config,
                seed=1, true_rewards=lookup,
            )

    def test_masked_rounds_are_counted_in_the_result_row(self, kuairec_config):
        """A policy that selects an unrecorded cell must say how often it did."""
        from gcrl.evaluation.ope import DeterministicPolicy
        from gcrl.phases.phase4_ope import run_phase4

        _, config = kuairec_config
        config.ope.estimators = ["exact"]
        rewards = np.ones((2, 3), dtype=np.float32)
        observed = np.ones((2, 3), dtype=bool)
        observed[0, 0] = False                      # user 0 has no record for action 0
        rows = np.array([0, 1, 0, 1])
        lookup = ExactRewardLookup(rewards, observed, rows, {"duplicate_cells": 0})

        class PicksAction0:
            n_actions = 3

            def action_distribution(self, observations):
                return DeterministicPolicy(np.zeros(len(observations), dtype=np.int64), 3)

        [evaluation] = run_phase4(
            {"Zero": PicksAction0()}, np.zeros((4, 2), dtype=np.float32),
            np.array([1, 1, 2, 2]), np.ones(4), np.full(4, 0.5), 3, config,
            seed=1, true_rewards=lookup,
        )
        [row] = evaluation.as_rows()
        assert row["exact_rounds_masked"] == 2      # the two rounds whose user is 0
        assert row["exact_rounds_scored"] == 2
        assert row["value"] == pytest.approx(1.0)   # not 0.5, which zero-filling would give

    def test_build_full_reward_matrix_still_refuses_the_sparse_log(self, kuairec_root):
        from gcrl.data.kuairec import load_kuairec

        data = load_kuairec(kuairec_root, matrix="big")
        users = IdentifierIndexer("u").fit(data.interactions["user_id"])
        items = IdentifierIndexer("i").fit(data.interactions["video_id"])
        with pytest.raises(ValueError, match="fully observed small_matrix"):
            build_full_reward_matrix(data, users, items)

    def test_matrix_cannot_be_narrower_than_the_index(self, kuairec_root):
        data = _eval_block(kuairec_root)
        users = IdentifierIndexer("u").fit(data.interactions["user_id"])
        items = IdentifierIndexer("i").fit(data.interactions["video_id"])
        with pytest.raises(ValueError, match="Widening the matrix is safe"):
            build_full_reward_matrix(data, users, items, n_actions=len(items) - 1)


class TestExactReachesPhase4:
    """The headline contribution has to be producible from the CLI."""

    def test_exact_is_computed_end_to_end(self, kuairec_config, monkeypatch, tmp_path):
        monkeypatch.setenv("GCRL_NO_DATASET_CACHE", "1")
        path, config = kuairec_config
        assert main(["run-all", "--config", str(path), "--log-level", "ERROR"]) == 0

        rows = json.loads(
            (tmp_path / "results" / "phase4_ope_kuairec_tiny_seed5.json").read_text()
        )
        exact = [r for r in rows if r["estimator"] == "exact"]
        assert exact, "the exact estimator produced no rows"
        assert {r["agent"] for r in exact} == set(config.rl.agents)
        for row in exact:
            assert row["status"] == "ok"
            assert np.isfinite(row["value"])
            assert 0.0 <= row["value"] <= 1.0
            assert row["exact_rounds_scored"] > 0
            assert 0.9 < row["exact_coverage"] < 1.0
            assert row["exact_unobserved_policy"] == "masked_out_of_expectation"
            assert row["propensity_source"].startswith("eval_block:enumerated_empirical")
            # The block enumerates its contexts, so the folds partition those
            # contexts and q holds one row per context; the string says so.
            assert row["q_model"] == "cross_fitted(k=3, folds=contexts, q=per_context)"

    def test_exact_without_an_evaluation_block_raises(self, kuairec_config, monkeypatch):
        """No silent disappearance of the estimator when there is no ground truth."""
        from gcrl.phases.phase4_ope import run_phase4

        _, config = kuairec_config
        config.ope.estimators = ["exact"]
        with pytest.raises(ValueError, match="no ground-truth reward matrix"):
            run_phase4(
                {}, np.zeros((4, 2), dtype=np.float32), np.zeros(4, dtype=np.int64),
                np.zeros(4), np.full(4, 0.5), 3, config, seed=1, true_rewards=None,
            )


class TestEstimatedBehaviourPolicy:
    """The sparse log has no logged propensity, so one is fitted and labelled."""

    def test_propensities_are_not_a_constant(self, prepared):
        propensities = prepared.arrays("test")["propensities"]
        assert propensities.min() > 0
        assert propensities.max() <= 1

    def test_source_names_both_data_generating_processes(self, prepared):
        """The block and the log were generated differently and must say so."""
        assert prepared.propensity_source.startswith("eval_block:enumerated_empirical")
        assert "log:estimated:" in prepared.propensity_source
        assert "shrinkage" in prepared.propensity_source

    def test_it_is_a_proper_distribution_over_actions(self, prepared):
        from gcrl.pipeline import fit_behaviour_policy

        train = prepared.arrays("train")
        policy = fit_behaviour_policy(
            train["user_index"], train["actions"],
            n_users=len(prepared.user_indexer), n_actions=prepared.n_actions, shrinkage=10.0,
        )
        for user in np.unique(train["user_index"])[:6]:
            actions = np.arange(prepared.n_actions)
            total = policy.score(np.full(prepared.n_actions, user), actions).sum()
            assert total == pytest.approx(1.0, abs=1e-9)

    def test_it_is_fitted_on_training_rounds_only(self, prepared):
        """Fitting on the rounds it then weights is the same in-sample error."""
        from gcrl.pipeline import fit_behaviour_policy

        train = prepared.arrays("train")
        fitted = fit_behaviour_policy(
            train["user_index"], train["actions"],
            n_users=len(prepared.user_indexer), n_actions=prepared.n_actions, shrinkage=10.0,
        )
        np.testing.assert_allclose(
            fitted.score(train["user_index"], train["actions"]),
            train["propensities"], rtol=1e-12,
        )
        contaminated = fit_behaviour_policy(
            np.concatenate([train["user_index"], prepared.arrays("test")["user_index"]]),
            np.concatenate([train["actions"], prepared.arrays("test")["actions"]]),
            n_users=len(prepared.user_indexer), n_actions=prepared.n_actions, shrinkage=10.0,
        )
        assert not np.allclose(
            contaminated.score(train["user_index"], train["actions"]), train["propensities"]
        )

    def test_diagnostics_are_reported(self, prepared):
        from gcrl.pipeline import summarise_dataset

        config = load_config("configs/kuairec.yaml")
        config.dataset.name = "kuairec"
        summary = summarise_dataset(prepared, config)
        assert "log:estimated:" in summary["propensity_source"]
        assert summary["propensity_is_constant"] is False
        assert "train_mean_log_likelihood" in summary
        assert summary["block_rows_per_cell"] == pytest.approx(1.0)
        assert summary["representation_rows"] > summary["policy_rows"]
        assert summary["evaluation_regime"].startswith("cold_item")

    def test_a_constant_propensity_is_never_passed_off_as_a_policy(self, kuairec_config, monkeypatch):
        from gcrl.cli import _build_dataset

        monkeypatch.setenv("GCRL_NO_DATASET_CACHE", "1")
        _, config = kuairec_config
        config.ope.estimators = ["exact"]
        config.ope.behaviour_policy = "logged"
        prepared = _build_dataset(config, seed=5)
        assert "log:constant:uniform" in prepared.propensity_source
        # The sparse log's rows get the labelled constant; the evaluation rounds
        # still get the block's own known propensities, which is not a constant.
        assert len(np.unique(prepared.arrays("train")["propensities"])) == 1

    def test_importance_weighted_estimators_refuse_a_constant_without_a_block(self):
        """Without a fully observed block there is nothing to read pi_b off."""
        from gcrl.pipeline import prepare_dataset

        config = load_config("configs/kuairec.yaml")
        config.dataset.num_actions = 4
        config.ope.estimators = ["snipw"]
        config.ope.behaviour_policy = "logged"
        config.ope.use_logged_propensity = True
        frame = pd.DataFrame({
            "user_id": [1, 1, 2, 2, 3, 3], "video_id": [0, 1, 2, 3, 0, 1],
            "reward": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0], "timestamp": range(6),
            "video_duration": np.linspace(1, 6, 6),
        })
        with pytest.raises(ValueError, match="ope.behaviour_policy: estimated"):
            prepare_dataset(frame, config, "user_id", "video_id", "reward", seed=1)


class TestConfigs:
    """The shipped KuaiRec configs must describe what actually runs."""

    @pytest.mark.parametrize(
        "name",
        ["kuairec", "kuairec_fast", "kuairec_validate_ope", "kuairec_validate_ope_full"],
    )
    def test_every_kuairec_config_can_produce_exact(self, name):
        config = load_config(f"configs/{name}.yaml")
        assert "exact" in config.ope.estimators
        assert config.dataset.eval_file, "exact needs a fully observed evaluation block"
        assert config.dataset.num_actions == 3327, "the action space is small_matrix's universe"
        assert config.ope.behaviour_policy == "estimated"
        assert config.ope.cross_fitting_folds >= 2

    def test_cross_fitting_folds_is_validated(self):
        config = load_config("configs/kuairec.yaml")
        config.ope.cross_fitting_folds = 1
        with pytest.raises(ConfigError, match="cross_fitting_folds"):
            config.validate()

    def test_behaviour_policy_is_validated(self):
        config = load_config("configs/kuairec.yaml")
        config.ope.behaviour_policy = "uniform"
        with pytest.raises(ConfigError, match="behaviour_policy"):
            config.validate()

    def test_behaviour_policy_changes_the_dataset_cache_key(self):
        import copy

        from gcrl.cli import _dataset_cache_key

        base = load_config("configs/kuairec.yaml")
        other = copy.deepcopy(base)
        other.ope.behaviour_policy = "logged"
        assert _dataset_cache_key(base, 42)[0] != _dataset_cache_key(other, 42)[0]
        shrunk = copy.deepcopy(base)
        shrunk.ope.behaviour_policy_shrinkage = 1.0
        assert _dataset_cache_key(base, 42)[0] != _dataset_cache_key(shrunk, 42)[0]


@pytest.fixture
def overlapping_root(tmp_path):
    """The same layout WITHOUT KuaiRec's empty intersection."""
    return write_kuairec_fixture(tmp_path / "warm" / "kuairec", overlap=True)


class TestEmptyIntersection:
    """The structural fact that broke the first version of this protocol.

    KuaiRec's authors removed the fully observed block's (user, item) pairs from
    the sparse log to prevent leakage. Both populations are present in the log;
    only their intersection is empty. Filtering the log to the evaluation action
    space therefore deletes precisely the users the block exists to evaluate.
    These tests pin that so it cannot be silently reintroduced.
    """

    def test_the_fixture_reproduces_kuairec_structure(self, kuairec_root):
        big = pd.read_csv(kuairec_root / "data" / "big_matrix.csv")
        small = pd.read_csv(kuairec_root / "data" / "small_matrix.csv")
        eval_users, eval_items = set(small["user_id"]), set(small["video_id"])

        assert eval_users <= set(big["user_id"]), "every evaluation user is in the log"
        assert eval_items <= set(big["video_id"]), "every evaluation item is in the log"
        in_both = big["user_id"].isin(eval_users) & big["video_id"].isin(eval_items)
        assert int(in_both.sum()) == 0, "the intersection must be empty, as in the real files"

    def test_filtering_the_log_by_item_would_delete_every_evaluation_user(self, kuairec_root):
        """This is the bug, reproduced as an assertion.

        The first implementation kept only the log rows whose item is in the
        evaluation action space. On this structure that leaves a training log
        and an evaluation set with DISJOINT user populations, which is what the
        reachability guard reported on the real data.
        """
        big = pd.read_csv(kuairec_root / "data" / "big_matrix.csv")
        small = pd.read_csv(kuairec_root / "data" / "small_matrix.csv")
        filtered = big[big["video_id"].isin(set(small["video_id"]))]

        assert not filtered.empty, "the actions themselves are well represented"
        assert set(filtered["user_id"]).isdisjoint(set(small["user_id"])), (
            "filtering the log by item removes every evaluation user -- which is why "
            "the log must never be filtered by item"
        )

    def test_overlap_is_measured_not_assumed(self, kuairec_root):
        bundle = load_kuairec_train_eval(kuairec_root)
        overlap = bundle.overlap
        assert overlap.eval_users == N_SMALL_USERS
        assert overlap.eval_users_in_log == N_SMALL_USERS
        assert overlap.eval_user_log_rows > 0
        assert overlap.eval_items == N_SMALL_ITEMS
        assert overlap.eval_item_log_rows > 0
        assert overlap.eval_item_log_users > 0
        assert overlap.overlap_rows == 0
        assert overlap.is_cold_item is True
        assert "cold_item" in overlap.regime

    def test_a_non_empty_intersection_is_reported_as_warm(self, overlapping_root):
        """The regime is read off the data, not hardcoded for KuaiRec."""
        bundle = load_kuairec_train_eval(overlapping_root)
        assert bundle.overlap.overlap_rows > 0
        assert bundle.overlap.is_cold_item is False
        assert bundle.overlap.regime.startswith("warm")

    def test_the_regime_reaches_the_result_rows(self, prepared):
        assert prepared.evaluation_regime.startswith("cold_item")

    def test_the_loader_refuses_a_log_without_the_evaluation_users(
        self, kuairec_config, monkeypatch
    ):
        """The guard that caught this must keep working on the case it is for."""
        from gcrl.cli import _build_dataset

        monkeypatch.setenv("GCRL_NO_DATASET_CACHE", "1")
        _, config = kuairec_config
        data = Path(config.paths.raw_data) / "kuairec" / "data"
        big = pd.read_csv(data / "big_matrix.csv")
        small = pd.read_csv(data / "small_matrix.csv")
        big[~big["user_id"].isin(set(small["user_id"]))].to_csv(
            data / "big_matrix.csv", index=False
        )
        with pytest.raises(ValueError, match="no policy could be given a trained"):
            _build_dataset(config, seed=5)

    def test_prepare_dataset_refuses_unreachable_evaluation_users(self, kuairec_root):
        """The same guard one layer down, where the graph rows are known.

        This is the exact failure the first version produced on the real files:
        a training log and an evaluation block with disjoint user populations.
        """
        from gcrl.pipeline import prepare_dataset

        config = load_config("configs/kuairec.yaml")
        config.dataset.num_actions = N_SMALL_ITEMS
        config.dataset.feature_columns = ["video_duration"]
        config.ope.estimators = ["exact"]
        big = pd.read_csv(kuairec_root / "data" / "big_matrix.csv")
        small = pd.read_csv(kuairec_root / "data" / "small_matrix.csv")
        for part in (big, small):
            part["reward"] = (part["watch_ratio"] >= 2.0).astype(float)
        # Reproduce the old behaviour: filter the log to the action space.
        filtered = big[big["video_id"].isin(set(small["video_id"]))]
        with pytest.raises(ValueError, match="representation was never trained"):
            prepare_dataset(
                filtered, config, "user_id", "video_id", "reward", seed=1,
                eval_frame=small, action_universe=np.sort(small["video_id"].unique()),
                evaluation_block="small_matrix.csv",
            )


class TestRepresentationVersusPolicy:
    """The graph and the policy are trained on different rows, deliberately."""

    def test_the_graph_is_trained_on_every_logged_row(self, prepared, kuairec_root):
        big = pd.read_csv(kuairec_root / "data" / "big_matrix.csv")
        # train + validation of the log, before the action-space filter.
        assert prepared.representation_rows > prepared.policy_rows
        assert prepared.representation_rows <= len(big)
        assert prepared.graph_item_indexer is not None
        assert len(prepared.graph_item_indexer) > prepared.n_actions

    def test_every_evaluation_user_has_graph_edges(self, prepared):
        """Without this their embedding is the untrained node feature."""
        sources = prepared.graph.edge_index[0].numpy()
        users_with_edges = set(sources[sources < prepared.graph.num_users].tolist())
        evaluated = set(prepared.split.test["__user_index"].tolist())
        assert evaluated <= users_with_edges

    def test_no_edge_joins_an_evaluation_user_to_an_evaluation_action(self, prepared):
        """The cold-item structure, asserted at the level of the graph itself."""
        graph_items = prepared.graph_item_indexer
        offset = prepared.graph.num_users
        eval_action_nodes = {
            offset + graph_items.mapping[int(raw)]
            for raw in prepared.item_indexer.mapping
            if int(raw) in graph_items.mapping
        }
        evaluated = set(prepared.split.test["__user_index"].tolist())
        edges = prepared.graph.edge_index.numpy()
        joined = {
            (int(a), int(b))
            for a, b in zip(edges[0], edges[1], strict=True)
            if int(a) in evaluated and int(b) in eval_action_nodes
        }
        assert not joined, (
            "the log is not supposed to contain any (evaluation user, evaluation action) "
            "pair; a direct edge would mean the ground truth leaked into the representation"
        )

    def test_the_actions_are_represented_through_other_users(self, prepared):
        graph_items = prepared.graph_item_indexer
        offset = prepared.graph.num_users
        edges = prepared.graph.edge_index.numpy()
        targets = set(edges[1].tolist())
        reached = sum(
            1 for raw in prepared.item_indexer.mapping
            if int(raw) in graph_items.mapping
            and offset + graph_items.mapping[int(raw)] in targets
        )
        assert reached > 0, "no evaluation action appears in the graph at all"


class TestEvaluationBlockPropensities:
    """A fully observed block is an enumeration, so its pi_b is known exactly."""

    def test_they_are_read_off_the_enumeration(self, prepared):
        test = prepared.arrays("test")
        propensities = test["propensities"]
        assert propensities.min() > 0
        # Each user is enumerated over the action space minus its own holes, so
        # pi_b is 1/k(u), a long way from the log policy's near-zero weights.
        assert propensities.max() <= 1.0
        assert propensities.mean() == pytest.approx(1.0 / N_SMALL_ITEMS, rel=0.05)

    def test_they_sum_to_one_over_each_users_rounds(self, prepared):
        test = prepared.arrays("test")
        frame = pd.DataFrame({"u": test["user_index"], "p": test["propensities"]})
        totals = frame.groupby("u")["p"].sum()
        np.testing.assert_allclose(totals.to_numpy(), 1.0, rtol=1e-12)

    def test_no_outcome_enters_them(self, prepared):
        """They are a design parameter of the block, not a model of its rewards."""
        from gcrl.pipeline import enumerated_block_propensities

        test = prepared.arrays("test")
        a, _ = enumerated_block_propensities(test["user_index"], test["actions"])
        rng = np.random.default_rng(0)
        shuffled = rng.permutation(len(test["rewards"]))
        b, _ = enumerated_block_propensities(
            test["user_index"], test["actions"]
        )
        np.testing.assert_array_equal(a, b)
        assert len(shuffled) == len(a)

    def test_the_log_policy_does_not_describe_the_block(self, prepared):
        """Why the substitution matters, stated so it does not depend on scale.

        The block enumerates each action once per user, so any distribution that
        genuinely describes it must sum to one over that user's rounds. The
        policy fitted on the sparse log does not: it is a model of a different
        data-generating process, and weighting the block by it produces
        importance weights with nothing behind them. On the real files, where
        the log has seen none of these pairs, it can offer only its shrinkage
        term and the weights come out orders of magnitude too large.
        """
        from gcrl.pipeline import fit_behaviour_policy

        train, test = prepared.arrays("train"), prepared.arrays("test")
        from_log = fit_behaviour_policy(
            train["user_index"], train["actions"],
            n_users=len(prepared.user_indexer), n_actions=prepared.n_actions, shrinkage=10.0,
        ).score(test["user_index"], test["actions"])

        frame = pd.DataFrame({"u": test["user_index"], "log": from_log,
                              "block": test["propensities"]})
        totals = frame.groupby("u").sum()
        np.testing.assert_allclose(totals["block"].to_numpy(), 1.0, rtol=1e-12)
        assert not np.allclose(totals["log"].to_numpy(), 1.0, rtol=0.05), (
            "the log-fitted policy does not sum to one over the block's rounds, so it is "
            "not the distribution that generated them"
        )

    def test_snipw_recovers_the_exact_value_on_an_enumerated_block(self):
        """A consequence of a complete enumeration, not a coincidence.

        A deterministic, user-level policy agrees with exactly one round per
        user, so the self-normalised estimate is the mean true reward of the
        selected action over users -- the exact value. The estimator error this
        dataset reports for DM, DR and MRDR is therefore the reward model's
        alone.
        """
        from gcrl.evaluation.ope import DeterministicPolicy, exact_value, snipw
        from gcrl.pipeline import enumerated_block_propensities

        rng = np.random.default_rng(4)
        n_users, n_items = 6, 9
        truth = (rng.random((n_users, n_items)) < 0.4).astype(np.float64)
        users = np.repeat(np.arange(n_users), n_items)
        actions = np.tile(np.arange(n_items), n_users)
        rewards = truth[users, actions]
        propensities, _ = enumerated_block_propensities(users, actions)
        np.testing.assert_allclose(propensities, 1.0 / n_items)

        chosen = rng.integers(0, n_items, n_users)
        policy = DeterministicPolicy(chosen[users], n_items)
        lookup = ExactRewardLookup(
            truth.astype(np.float32), np.ones_like(truth, dtype=bool), users,
            {"duplicate_cells": 0},
        )
        assert snipw(actions, rewards, propensities, policy) == pytest.approx(
            exact_value(policy, lookup)
        )


class TestContextReplication:
    """A fully enumerated block replicates each context thousands of times."""

    def test_distinct_contexts_are_reported(self, kuairec_config, monkeypatch, tmp_path):
        monkeypatch.setenv("GCRL_NO_DATASET_CACHE", "1")
        path, _ = kuairec_config
        assert main(["run-all", "--config", str(path), "--log-level", "ERROR"]) == 0
        rows = json.loads(
            (tmp_path / "results" / "phase4_ope_kuairec_tiny_seed5.json").read_text()
        )
        for row in rows:
            assert row["n_distinct_contexts"] == N_SMALL_USERS
            assert row["n_test_rounds"] > row["n_distinct_contexts"]
            assert row["evaluation_regime"].startswith("cold_item")
            # The block is an enumeration, so the interval is resampled over
            # its users -- not over the rounds that replicate them.
            assert row["resampling_unit"] == "context"
            assert row["n_independent_units"] == N_SMALL_USERS
            assert row["bootstrap_method"] == "cluster_percentile"

    def test_the_bootstrap_caveat_is_raised(self, kuairec_config, caplog):
        """The interval is over rounds; the decisions are over contexts."""
        import logging

        from gcrl.evaluation.ope import DeterministicPolicy
        from gcrl.phases.phase4_ope import run_phase4

        _, config = kuairec_config
        config.ope.estimators = ["exact"]
        truth = np.ones((3, 4), dtype=np.float32)
        rounds = np.repeat(np.arange(3), 20)
        lookup = ExactRewardLookup(
            truth, np.ones_like(truth, dtype=bool), rounds, {"duplicate_cells": 0}
        )

        class Fixed:
            n_actions = 4

            def action_distribution(self, observations):
                return DeterministicPolicy(np.zeros(len(observations), dtype=np.int64), 4)

        with caplog.at_level(logging.WARNING):
            run_phase4(
                {"Fixed": Fixed()}, np.zeros((60, 2), dtype=np.float32),
                np.zeros(60, dtype=np.int64), np.ones(60), np.full(60, 0.25), 4, config,
                seed=1, true_rewards=lookup, round_context_id=rounds,
            )
        assert any("distinct contexts" in record.message for record in caplog.records)
