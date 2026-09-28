"""``q(x, a)`` is computed once per DISTINCT CONTEXT, and cross-fitted by context.

The defect these tests pin. KuaiRec's evaluation block is an enumeration: each
of 1,411 users was paired with each of 3,327 items, so its 4,676,570 rounds
carry 1,411 distinct contexts, each replicated ~3,314 times. Two consequences,
one about memory and one about correctness:

1. ``q(x_i, .)`` depends on round ``i`` only through its context, so the dense
   ``(n_rounds, n_actions)`` grid is 4,676,570 x 3,327 = 115.9 GiB of float64
   holding 1,411 x 3,327 = 37.6 MB of distinct values. The duplicates are
   bit-identical; computing one row per context and gathering it changes the
   memory and nothing else.

2. Cross-fitting partitioned ROUNDS. A fold boundary drawn through a replicated
   context leaves its identical twins -- same context vector, real rewards
   attached -- in the training fold, so the model that formally never saw round
   ``i`` has seen ``i``'s context and can read its reward level off. Every
   formal check still passes while the guarantee is gone.
   ``TestFoldsPartitionContexts`` measures the leak.

The bit-identity proofs are the other half: deduplication may not move a single
digit of any estimate, and the comparisons below are on ``float.hex()``.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from gcrl.config import load_config
from gcrl.evaluation.ope import (
    DENSE_Q_MODEL_CELL_BUDGET,
    REWARD_MODEL_FIT_CELL_BUDGET,
    DenseDistributionError,
    DeterministicPolicy,
    UniformPolicy,
    assert_dense_q_model_affordable,
    assert_reward_model_fit_affordable,
    evaluate_policy,
)
from gcrl.phases.phase4_ope import (
    ContextIndex,
    ReplicatedQEstimates,
    _fit_q_model,
    _predict_all_actions,
    context_fold_indices,
    cross_fitted_reward_model,
    run_phase4,
)

N_ACTIONS = 6
N_FEATURES = 10


# -- fixtures -------------------------------------------------------------


def replicated_rounds(
    n_contexts: int = 40, per_context: int = 25, seed: int = 0, n_actions: int = N_ACTIONS
):
    """An enumerated block in miniature: every context replicated identically.

    Each context carries its own reward level ``p_c``, drawn independently of
    its feature vector, and each of its rounds draws ``Bernoulli(p_c)``. The
    features say nothing about ``p_c``, so a model can only know a context's
    level by having seen that context's rounds -- which is exactly what a
    round-partitioned cross-fit lets it do.
    """
    rng = np.random.default_rng(seed)
    contexts = rng.normal(size=(n_contexts, N_FEATURES)).astype(np.float32)
    level = rng.uniform(0.05, 0.95, n_contexts)

    context_id = np.repeat(np.arange(n_contexts), per_context)
    rng.shuffle(context_id)                       # rounds are not grouped in the log
    n = len(context_id)
    return {
        "context": contexts[context_id],
        "context_id": context_id,
        "actions": rng.integers(0, n_actions, n),
        "rewards": (rng.random(n) < level[context_id]).astype(np.float64),
        "propensities": np.full(n, 1.0 / n_actions),
        "level": level[context_id],
        "n_contexts": n_contexts,
    }


def distinct_rounds(n: int = 400, seed: int = 1, n_actions: int = N_ACTIONS):
    """Rounds with no two contexts alike -- the case deduplication must not touch."""
    rng = np.random.default_rng(seed)
    context = rng.normal(size=(n, N_FEATURES)).astype(np.float32)
    return {
        "context": context,
        "context_id": np.arange(n),
        "actions": rng.integers(0, n_actions, n),
        "rewards": (rng.random(n) < 0.3).astype(np.float64),
        "propensities": np.full(n, 1.0 / n_actions),
        "n_contexts": n,
    }


def legacy_cross_fitted_reward_model(
    context, actions, rewards, n_actions, n_folds, seed=42, sample_weight=None
):
    """``cross_fitted_reward_model`` EXACTLY as it was before deduplication.

    Folds over rounds, one dense ``(n_rounds, n_actions)`` grid, one prediction
    per round. The reference for every bit-identity assertion below.
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    n = len(actions)
    indices = np.arange(n)
    if sample_weight is None:
        folds = list(KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(indices))
    else:
        positive = np.asarray(sample_weight) > 0
        folds = list(
            StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(
                indices, positive
            )
        )
    q = np.zeros((n, n_actions), dtype=np.float64)
    for fold, (fit_index, score_index) in enumerate(folds, start=1):
        weights = None if sample_weight is None else np.asarray(sample_weight)[fit_index]
        model = _fit_q_model(
            context[fit_index], actions[fit_index], rewards[fit_index],
            n_actions, seed + fold, weights,
        )
        q[score_index] = _predict_all_actions(model, context[score_index], n_actions)
    return q


def dense_q_under(folds, context, actions, rewards, n_actions, seed):
    """The REDUNDANT grid: one prediction per round, under the same fold assignment.

    This is what the deduplicated ``q`` claims to equal -- the same models, asked
    the same question once per round instead of once per context.
    """
    q = np.zeros((len(actions), n_actions), dtype=np.float64)
    for fold, (fit_index, score_index, _) in enumerate(folds, start=1):
        model = _fit_q_model(
            context[fit_index], actions[fit_index], rewards[fit_index],
            n_actions, seed + fold, None,
        )
        q[score_index] = _predict_all_actions(model, context[score_index], n_actions)
    return q


def assert_hex_identical(got, want, label: str) -> None:
    """Every cell equal as ``float.hex()`` -- not ``allclose``, not ``approx``."""
    got = np.asarray(got, dtype=np.float64).ravel()
    want = np.asarray(want, dtype=np.float64).ravel()
    assert got.shape == want.shape, (label, got.shape, want.shape)
    mismatched = [
        (i, float(a).hex(), float(b).hex())
        for i, (a, b) in enumerate(zip(got, want, strict=True))
        if float(a).hex() != float(b).hex()
    ]
    assert not mismatched, f"{label}: {len(mismatched)} cells moved, first {mismatched[:3]}"


@pytest.fixture(scope="module")
def replicated():
    return replicated_rounds()


@pytest.fixture(scope="module")
def distinct():
    return distinct_rounds()


# -- 1. the numbers may not move ------------------------------------------


class TestBitIdenticalWithoutReplication:
    """With one round per context, deduplication must be a no-op."""

    def test_q_is_bit_identical_to_the_pre_deduplication_model(self, distinct):
        got = cross_fitted_reward_model(
            distinct["context"], distinct["actions"], distinct["rewards"], N_ACTIONS,
            n_folds=5, seed=7,
        )
        want = legacy_cross_fitted_reward_model(
            distinct["context"], distinct["actions"], distinct["rewards"], N_ACTIONS,
            n_folds=5, seed=7,
        )
        assert isinstance(got, np.ndarray)       # nothing to deduplicate, nothing wrapped
        assert_hex_identical(got, want, "q without replication")

    def test_the_mrdr_weighted_fit_is_bit_identical_too(self, distinct):
        rng = np.random.default_rng(3)
        weights = np.where(rng.random(len(distinct["actions"])) < 0.4, rng.random(400), 0.0)
        got = cross_fitted_reward_model(
            distinct["context"], distinct["actions"], distinct["rewards"], N_ACTIONS,
            n_folds=4, seed=11, sample_weight=weights,
        )
        want = legacy_cross_fitted_reward_model(
            distinct["context"], distinct["actions"], distinct["rewards"], N_ACTIONS,
            n_folds=4, seed=11, sample_weight=weights,
        )
        assert_hex_identical(got, want, "MRDR-weighted q without replication")

    def test_the_folds_are_the_same_folds(self, distinct):
        """Not merely equivalent: the identical index arrays, fold for fold."""
        from sklearn.model_selection import KFold

        index = ContextIndex.of(distinct["context"])
        assert index.is_trivial
        legacy = list(KFold(n_splits=5, shuffle=True, random_state=7).split(np.arange(400)))
        for (fit_rounds, score_rounds, _), (want_fit, want_score) in zip(
            context_fold_indices(index, 5, seed=7), legacy, strict=True
        ):
            np.testing.assert_array_equal(fit_rounds, want_fit)
            np.testing.assert_array_equal(score_rounds, want_score)


class TestBitIdenticalWithReplication:
    """With replication, the gather must equal the recomputation, cell for cell."""

    @pytest.fixture(scope="class")
    def models(self, replicated):
        index = ContextIndex.of(replicated["context"], replicated["context_id"])
        folds = context_fold_indices(index, 5, seed=7)
        return {
            "index": index,
            "folds": folds,
            "deduplicated": cross_fitted_reward_model(
                replicated["context"], replicated["actions"], replicated["rewards"],
                N_ACTIONS, n_folds=5, seed=7, context_index=index,
            ),
            "redundant": dense_q_under(
                folds, replicated["context"], replicated["actions"], replicated["rewards"],
                N_ACTIONS, seed=7,
            ),
        }

    def test_q_is_stored_once_per_context(self, models, replicated):
        q = models["deduplicated"]
        assert isinstance(q, ReplicatedQEstimates)
        assert q.n_distinct_contexts == replicated["n_contexts"]
        assert q.shape == (len(replicated["actions"]), N_ACTIONS)

    def test_every_gathered_cell_is_bit_identical_to_the_redundant_grid(self, models):
        assert_hex_identical(
            models["deduplicated"].to_dense(), models["redundant"], "gathered q"
        )

    def test_every_estimator_is_bit_identical(self, models, replicated):
        """dm, dr, mrdr, exact, ipw and snipw, from the same rounds and policies."""
        rng = np.random.default_rng(5)
        n = len(replicated["actions"])
        truth = rng.random((n, N_ACTIONS))
        softmax = rng.random((n, N_ACTIONS))
        softmax /= softmax.sum(axis=1, keepdims=True)
        policies = {
            "deterministic": DeterministicPolicy(rng.integers(0, N_ACTIONS, n), N_ACTIONS),
            "uniform": UniformPolicy(n, N_ACTIONS),
            "dense": softmax,
        }
        names = ["ipw", "snipw", "dm", "dr", "mrdr", "exact"]
        for label, policy in policies.items():
            common = {
                "actions": replicated["actions"], "rewards": replicated["rewards"],
                "propensities": replicated["propensities"], "action_dist": policy,
                "true_rewards": truth, "estimators": names, "n_bootstrap": 5,
                "bootstrap_units": replicated["context_id"],
            }
            got = evaluate_policy(q_estimates=models["deduplicated"], **common)
            want = evaluate_policy(q_estimates=models["redundant"], **common)
            for name in names:
                assert got[name].value.hex() == want[name].value.hex(), (label, name)
                assert got[name].lower.hex() == want[name].lower.hex(), (label, name)
                assert got[name].upper.hex() == want[name].upper.hex(), (label, name)

    def test_the_uniform_policy_row_mean_is_bit_identical(self, models):
        assert_hex_identical(
            models["deduplicated"].mean(axis=1),
            models["redundant"].mean(axis=1),
            "row mean",
        )

    @pytest.mark.parametrize("cells_per_block", [1, 37, 10**9])
    def test_the_block_size_cannot_change_the_answer(self, models, replicated, cells_per_block):
        rng = np.random.default_rng(6)
        n = len(replicated["actions"])
        softmax = rng.random((n, N_ACTIONS))
        softmax /= softmax.sum(axis=1, keepdims=True)
        assert_hex_identical(
            models["deduplicated"].expected_under(softmax, cells_per_block=cells_per_block),
            np.sum(softmax * models["redundant"], axis=1),
            f"expected_under block={cells_per_block}",
        )

    def test_the_bootstrap_reslice_gathers_the_same_cells(self, models, replicated):
        rng = np.random.default_rng(8)
        index = rng.integers(0, len(replicated["actions"]), 500)
        assert_hex_identical(
            models["deduplicated"][index].to_dense(),
            models["redundant"][index],
            "resliced q",
        )


# -- 2. the folds partition contexts --------------------------------------


class TestFoldsPartitionContexts:
    """A context must belong to exactly one fold, or cross-fitting means nothing."""

    def test_no_training_fold_contains_a_scored_context(self, replicated):
        index = ContextIndex.of(replicated["context"], replicated["context_id"])
        seen = set()
        for fit_rounds, score_rounds, score_contexts in context_fold_indices(index, 5, seed=7):
            fitted_on = set(index.ids[fit_rounds].tolist())
            scored = set(index.ids[score_rounds].tolist())
            assert scored == set(score_contexts.tolist())
            assert not (fitted_on & scored), sorted(fitted_on & scored)
            seen |= scored
        assert seen == set(range(index.n_contexts))    # every context scored exactly once

    @pytest.mark.parametrize("seed", [0, 1])
    def test_round_partitioned_folds_leak_the_context_reward_level(self, seed):
        """The measurement that makes the change necessary, not merely tidy.

        The context's features say nothing about its reward level, so an honest
        ``q`` cannot predict it at all. A round-partitioned cross-fit predicts it
        well, because ~80% of each scored round's own context sits in the
        training fold with its rewards attached. 200 contexts are used so the
        correlation of an honest q is measured to about +-0.07.
        """
        data = replicated_rounds(n_contexts=200, per_context=10, seed=seed)
        context, actions = data["context"], data["actions"]
        rewards, level = data["rewards"], data["level"]

        def correlation(q):
            at_logged = np.asarray(q)[np.arange(len(actions)), actions]
            return float(np.corrcoef(at_logged, level)[0, 1])

        by_round = correlation(
            legacy_cross_fitted_reward_model(context, actions, rewards, N_ACTIONS, 5, seed=7)
        )
        by_context = correlation(
            cross_fitted_reward_model(
                context, actions, rewards, N_ACTIONS, 5, seed=7,
                context_index=ContextIndex.of(context, data["context_id"]),
            ).to_dense()
        )
        # Measured: 0.77 against -0.05 (seed 0) and 0.75 against -0.11 (seed 1).
        assert by_round > 0.5, by_round          # the leak, measured
        assert abs(by_context) < 0.25, by_context
        assert abs(by_context) < 0.4 * by_round, (by_context, by_round)

    def test_mrdr_stratifies_on_supported_contexts(self, replicated):
        """A context counts as supported when ANY of its rounds carries weight."""
        index = ContextIndex.of(replicated["context"], replicated["context_id"])
        weights = np.zeros(len(replicated["actions"]))
        # One supported round in each of six contexts: enough for 3 folds by
        # context, and the round-level count (6) says the same here.
        for context_id in range(6):
            weights[np.flatnonzero(index.ids == context_id)[0]] = 1.0
        folds = context_fold_indices(index, 3, seed=2, sample_weight=weights)
        for fit_rounds, _, _ in folds:
            assert weights[fit_rounds].sum() > 0

    def test_too_few_supported_contexts_is_refused(self, replicated):
        index = ContextIndex.of(replicated["context"], replicated["context_id"])
        weights = np.zeros(len(replicated["actions"]))
        # 50 supported ROUNDS, but all of them inside two contexts.
        weights[np.isin(index.ids, [0, 1])] = 1.0
        assert int(np.count_nonzero(weights)) > 5
        with pytest.raises(ValueError, match="too rarely for MRDR to be estimable"):
            context_fold_indices(index, 5, seed=2, sample_weight=weights)

    def test_more_folds_than_contexts_is_refused(self, replicated):
        """1,000 rounds over 40 contexts cannot be cross-fitted 50 ways."""
        index = ContextIndex.of(replicated["context"], replicated["context_id"])
        with pytest.raises(ValueError, match="cannot cross-fit"):
            context_fold_indices(index, 50, seed=2)


# -- 3. the context index -------------------------------------------------


class TestContextIndex:
    def test_an_identifier_is_used_when_it_holds(self, replicated):
        index = ContextIndex.of(replicated["context"], replicated["context_id"])
        assert index.source == "round_context_id"
        assert index.n_contexts == replicated["n_contexts"]

    def test_the_identifier_is_verified_not_trusted(self, replicated):
        """A context id that groups rounds with different contexts is rejected.

        ``state_source: raw_features`` is exactly this case: one user's rounds
        carry different feature vectors, and deduplicating them would change
        every DM/DR/MRDR number rather than merely save memory.
        """
        context = replicated["context"].copy()
        context[0] = context[0] + 1.0            # one round leaves its group
        index = ContextIndex.of(context, replicated["context_id"])
        assert index.source == "context_rows"
        assert index.n_contexts == replicated["n_contexts"] + 1
        # The bootstrap keeps the coarser grouping, which can only widen an interval.
        assert index.unit_source == "round_context_id"
        assert index.n_units == replicated["n_contexts"]

    def test_it_is_derived_from_the_context_rows_when_no_identifier_is_given(self, replicated):
        index = ContextIndex.of(replicated["context"])
        assert index.source == "context_rows"
        assert index.n_contexts == replicated["n_contexts"]
        np.testing.assert_array_equal(
            index.ids,
            ContextIndex.of(replicated["context"], replicated["context_id"]).ids,
        )

    def test_distinct_contexts_are_numbered_by_first_appearance(self, distinct):
        index = ContextIndex.of(distinct["context"])
        np.testing.assert_array_equal(index.ids, np.arange(len(distinct["actions"])))
        np.testing.assert_array_equal(index.representatives, np.arange(len(distinct["actions"])))
        assert index.is_trivial

    def test_a_mismatched_identifier_length_raises(self, replicated):
        with pytest.raises(ValueError, match="aligned round for round"):
            ContextIndex.of(replicated["context"], replicated["context_id"][:10])


# -- 4. memory ------------------------------------------------------------


class TestPeakMemory:
    """The scoring path never allocates a grid the size of the rounds.

    The 115.9 GiB was ``q`` itself and everything indexed out of it. Fitting the
    model is a separate cost that deduplication cannot remove -- every round is
    its own training example -- and it has its own guard; see
    :func:`gcrl.evaluation.ope.assert_reward_model_fit_affordable`.
    """

    @staticmethod
    def _replicated_q(n_rounds: int, n_actions: int, n_contexts: int = 20):
        rng = np.random.default_rng(21)
        q = rng.random((n_contexts, n_actions))
        return ReplicatedQEstimates(q, rng.integers(0, n_contexts, n_rounds)), q

    def test_q_holds_one_row_per_context(self):
        q, stored = self._replicated_q(200_000, 500)
        dense_bytes = 200_000 * 500 * 8
        assert stored.nbytes == 20 * 500 * 8
        assert stored.nbytes < dense_bytes / 1000

    def test_scoring_never_allocates_the_dense_grid(self):
        """DM, DR and the bootstrap, over a grid that would be 800 MB dense."""
        n_rounds, n_actions = 200_000, 500
        q, _ = self._replicated_q(n_rounds, n_actions)
        rng = np.random.default_rng(22)
        actions = rng.integers(0, n_actions, n_rounds)
        rewards = rng.random(n_rounds)
        propensities = np.full(n_rounds, 1.0 / n_actions)
        policy = DeterministicPolicy(rng.integers(0, n_actions, n_rounds), n_actions)

        tracemalloc.start()
        tracemalloc.reset_peak()
        try:
            results = evaluate_policy(
                actions, rewards, propensities, policy, q_estimates=q,
                estimators=["dm", "dr"], n_bootstrap=3,
            )
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert np.isfinite(results["dm"].value)
        grid_bytes = n_rounds * n_actions * 8          # 800 MB
        assert peak < grid_bytes / 10, f"peak {peak:,} B against a {grid_bytes:,} B grid"

    def test_a_dense_policy_is_bounded_by_the_block_not_the_rounds(self):
        """The one path that multiplies q by a full distribution, in row blocks."""
        n_actions = 400

        def peak_for(n_rounds: int) -> int:
            q, _ = self._replicated_q(n_rounds, n_actions)
            distribution = np.full((n_rounds, n_actions), 1.0 / n_actions)
            tracemalloc.start()
            tracemalloc.reset_peak()
            try:
                q.expected_under(distribution)
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            # ``distribution`` was allocated before tracing started, so the
            # peak below is what ``expected_under`` itself asks for.
            assert distribution.shape == (n_rounds, n_actions)
            return peak

        small, large = peak_for(50_000), peak_for(200_000)
        grid_bytes = 200_000 * n_actions * 8
        assert large < grid_bytes / 10, f"peak {large:,} B against a {grid_bytes:,} B grid"
        assert large < 5 * small, f"peak grew with the rounds: {small:,} -> {large:,}"


# -- 5. the guards --------------------------------------------------------


class TestTheGuards:
    """The budget rises exactly as far as deduplication justifies."""

    def test_kuairecs_full_block_fits_once_deduplicated(self):
        # 4,676,570 rounds x 3,327 actions = 115.9 GiB; 1,411 contexts = 37.6 MB.
        assert_dense_q_model_affordable(4_676_570, 3_327, n_contexts=1_411)
        assert DENSE_Q_MODEL_CELL_BUDGET / 400 > 1_411 * 3_327

    def test_the_same_grid_without_deduplication_is_still_refused(self):
        with pytest.raises(DenseDistributionError, match="reward model"):
            assert_dense_q_model_affordable(4_676_570, 3_327)

    def test_a_deduplicated_grid_that_is_still_impossible_is_refused(self):
        """Deduplication is not a licence: 2e6 distinct contexts still cannot be held."""
        with pytest.raises(DenseDistributionError) as raised:
            assert_dense_q_model_affordable(4_676_570, 3_327, n_contexts=2_000_000)
        assert "2,000,000 distinct contexts x 3,327 actions" in str(raised.value)
        assert "49.6 GiB as float64" in str(raised.value)

    def test_the_training_matrix_is_guarded_separately(self):
        """Deduplication does not shrink the fit: every round is its own example."""
        # KuaiRec's full block, 5 folds, 64-dimensional state.
        with pytest.raises(DenseDistributionError, match="must be FITTED") as raised:
            assert_reward_model_fit_affordable(3_741_256, 64, 3_327)
        assert "3,741,256 rounds x 3,391 features" in str(raised.value)
        # And the largest fit any shipped config asks for is allowed.
        assert_reward_model_fit_affordable(100_000, 64, 3_327)
        assert REWARD_MODEL_FIT_CELL_BUDGET / 2 > 100_000 * (64 + 3_327)


# -- 6. wired into Phase 4 ------------------------------------------------


def _phase4_config(tmp_path, estimators="[dm, dr]", folds=4):
    path = tmp_path / "dedup.yaml"
    path.write_text(f"""
experiment_name: dedup
seed: 3
device: cpu
paths:
  root: "{tmp_path.as_posix()}"
dataset:
  name: dedup
  loader: obd
  num_actions: {N_ACTIONS}
causal:
  enabled: false
rl:
  agents: [Random]
  state_source: raw_features
ope:
  estimators: {estimators}
  cross_fitting_folds: {folds}
  n_bootstrap: 10
""")
    config = load_config(path)
    config.paths.ensure_output_dirs()
    return config


class TestPhase4UsesTheDeduplicatedModel:
    def test_run_phase4_scores_with_the_context_cross_fitted_q(self, tmp_path, replicated):
        config = _phase4_config(tmp_path)
        chosen = np.asarray(replicated["actions"])

        class Fixed:
            n_actions = N_ACTIONS

            def action_distribution(self, observations):
                return DeterministicPolicy(chosen, N_ACTIONS)

        [evaluation] = run_phase4(
            {"Fixed": Fixed()}, replicated["context"], replicated["actions"],
            replicated["rewards"], replicated["propensities"], N_ACTIONS, config, seed=3,
            round_context_id=replicated["context_id"], independent_unit="context",
        )
        index = ContextIndex.of(replicated["context"], replicated["context_id"])
        q = cross_fitted_reward_model(
            replicated["context"], replicated["actions"], replicated["rewards"],
            N_ACTIONS, 4, seed=3, context_index=index,
        )
        expected = float(np.mean(q[np.arange(len(chosen)), chosen]))
        assert evaluation.estimates["dm"].value.hex() == expected.hex()
        assert evaluation.n_distinct_contexts == replicated["n_contexts"]

    def test_the_rounds_a_context_shares_all_get_the_same_q_row(self, tmp_path, replicated):
        index = ContextIndex.of(replicated["context"], replicated["context_id"])
        q = cross_fitted_reward_model(
            replicated["context"], replicated["actions"], replicated["rewards"],
            N_ACTIONS, 4, seed=3, context_index=index,
        ).to_dense()
        for context_id in range(replicated["n_contexts"]):
            rows = q[index.ids == context_id]
            assert np.array_equal(rows, np.repeat(rows[:1], len(rows), axis=0))
