"""Tests for off-policy estimators, checked against analytically known values."""

import numpy as np
import pytest

from gcrl.evaluation.ope import (
    WARNING_CI_UNAVAILABLE,
    WARNING_DR_REDUCES_TO_DM,
    WARNING_NO_OVERLAP,
    DeterministicPolicy,
    OPEError,
    UniformPolicy,
    direct_method,
    doubly_robust,
    effective_sample_size,
    evaluate_policy,
    exact_value,
    ipw,
    mrdr_weights,
    snipw,
)


def _deterministic_dist(actions, n_actions):
    dist = np.zeros((len(actions), n_actions))
    dist[np.arange(len(actions)), actions] = 1.0
    return dist


class TestIPW:
    def test_on_policy_recovers_mean_reward(self):
        """If the evaluation policy equals the behaviour policy, IPW is the mean reward."""
        rng = np.random.default_rng(0)
        n, n_actions = 500, 4
        actions = rng.integers(0, n_actions, n)
        rewards = rng.random(n)
        propensities = np.full(n, 1.0 / n_actions)
        action_dist = np.full((n, n_actions), 1.0 / n_actions)
        assert ipw(actions, rewards, propensities, action_dist) == pytest.approx(rewards.mean(), abs=1e-10)

    def test_unbiased_under_uniform_logging(self):
        """IPW recovers the true value of a deterministic policy under uniform logging."""
        rng = np.random.default_rng(1)
        n, n_actions = 40_000, 4
        true_values = np.array([0.1, 0.9, 0.3, 0.5])
        actions = rng.integers(0, n_actions, n)
        rewards = (rng.random(n) < true_values[actions]).astype(float)
        propensities = np.full(n, 1.0 / n_actions)
        target = np.ones(n, dtype=int)  # always take action 1
        estimate = ipw(actions, rewards, propensities, _deterministic_dist(target, n_actions))
        assert estimate == pytest.approx(true_values[1], abs=0.03)

    def test_raises_when_policies_never_agree(self):
        """Zero weights make IPW's arithmetic return 0.0; that zero is not a measurement.

        The counterpart of ``TestSNIPW::test_raises_when_policies_never_agree``.
        Without it, IPW published an exact 0.0 with ``status="ok"``, which the
        results table bolds as the best value with no caveat.
        """
        n, n_actions = 50, 3
        actions = np.zeros(n, dtype=int)
        action_dist = _deterministic_dist(np.ones(n, dtype=int), n_actions)
        with pytest.raises(OPEError, match="not identifiable"):
            ipw(actions, np.ones(n), np.full(n, 0.5), action_dist)


class TestSNIPW:
    def test_bounded_by_reward_range(self):
        """Self-normalisation keeps the estimate inside the observed reward range."""
        rng = np.random.default_rng(2)
        n, n_actions = 2000, 5
        actions = rng.integers(0, n_actions, n)
        rewards = rng.random(n)
        propensities = rng.uniform(0.01, 1.0, n)
        action_dist = rng.dirichlet(np.ones(n_actions), n)
        value = snipw(actions, rewards, propensities, action_dist)
        assert rewards.min() <= value <= rewards.max()

    def test_matches_mean_when_weights_constant(self):
        n, n_actions = 100, 4
        actions = np.zeros(n, dtype=int)
        rewards = np.linspace(0, 1, n)
        propensities = np.full(n, 0.25)
        action_dist = np.full((n, n_actions), 0.25)
        assert snipw(actions, rewards, propensities, action_dist) == pytest.approx(rewards.mean())

    def test_raises_when_policies_never_agree(self):
        """A policy that never selects a logged action has no identifiable value."""
        n, n_actions = 50, 3
        actions = np.zeros(n, dtype=int)
        action_dist = _deterministic_dist(np.ones(n, dtype=int), n_actions)
        with pytest.raises(OPEError, match="not identifiable"):
            snipw(actions, np.ones(n), np.full(n, 0.5), action_dist)


class TestDoublyRobust:
    def test_reduces_to_dm_with_perfect_reward_model(self):
        """With a perfect reward model the correction term vanishes."""
        rng = np.random.default_rng(3)
        n, n_actions = 300, 4
        actions = rng.integers(0, n_actions, n)
        q = rng.random((n, n_actions))
        rewards = q[np.arange(n), actions]  # model is exactly right
        propensities = rng.uniform(0.1, 1.0, n)
        action_dist = rng.dirichlet(np.ones(n_actions), n)
        assert doubly_robust(actions, rewards, propensities, action_dist, q) == pytest.approx(
            direct_method(action_dist, q)
        )

    def test_reduces_to_ipw_with_zero_reward_model(self):
        rng = np.random.default_rng(4)
        n, n_actions = 300, 4
        actions = rng.integers(0, n_actions, n)
        rewards = rng.random(n)
        propensities = rng.uniform(0.1, 1.0, n)
        action_dist = rng.dirichlet(np.ones(n_actions), n)
        q = np.zeros((n, n_actions))
        assert doubly_robust(actions, rewards, propensities, action_dist, q) == pytest.approx(
            ipw(actions, rewards, propensities, action_dist)
        )

    def test_consistent_with_wrong_reward_model(self):
        """Double robustness: correct propensities rescue a badly wrong reward model."""
        rng = np.random.default_rng(5)
        n, n_actions = 60_000, 3
        true_values = np.array([0.2, 0.8, 0.4])
        actions = rng.integers(0, n_actions, n)
        rewards = (rng.random(n) < true_values[actions]).astype(float)
        propensities = np.full(n, 1.0 / n_actions)
        target = np.ones(n, dtype=int)
        q_wrong = np.full((n, n_actions), 0.99)  # deliberately wrong
        estimate = doubly_robust(
            actions, rewards, propensities, _deterministic_dist(target, n_actions), q_wrong
        )
        assert estimate == pytest.approx(true_values[1], abs=0.03)

    def test_collapses_to_dm_when_policies_never_agree(self):
        """With all weights zero the correction vanishes and DR *is* DM.

        The value is real, so DR does not raise -- but it is a reward-model
        extrapolation, not a doubly-robust estimate, and
        :func:`evaluate_policy` has to say so.
        """
        n, n_actions = 40, 3
        actions = np.zeros(n, dtype=int)
        action_dist = _deterministic_dist(np.ones(n, dtype=int), n_actions)
        q = np.tile(np.array([0.1, 0.9, 0.4]), (n, 1))
        value = doubly_robust(actions, np.ones(n), np.full(n, 0.5), action_dist, q)
        assert value == pytest.approx(direct_method(action_dist, q))


class TestMRDRWeights:
    def test_weights_are_non_negative(self):
        rng = np.random.default_rng(6)
        n, n_actions = 200, 4
        actions = rng.integers(0, n_actions, n)
        propensities = rng.uniform(0.05, 1.0, n)
        action_dist = rng.dirichlet(np.ones(n_actions), n)
        assert np.all(mrdr_weights(actions, propensities, action_dist) >= 0)

    def test_weight_grows_as_propensity_shrinks(self):
        """Rare logged actions carry more variance and receive more weight."""
        actions = np.array([0, 0])
        action_dist = np.array([[1.0, 0.0], [1.0, 0.0]])
        weights = mrdr_weights(actions, np.array([0.5, 0.05]), action_dist)
        assert weights[1] > weights[0]


class TestExactValue:
    def test_matches_lookup_for_deterministic_policy(self):
        true_rewards = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
        dist = _deterministic_dist(np.array([1, 0, 1]), 2)
        assert exact_value(dist, true_rewards) == pytest.approx((1.0 + 1.0 + 0.5) / 3)


class TestSparsePolicies:
    """The sparse representations must convert to their dense distribution.

    Both define ``__len__`` and a ``__getitem__`` that returns another policy,
    so without an explicit array protocol NumPy treats them as nested
    sequences: ``UniformPolicy`` recurses until the process is killed, and
    ``DeterministicPolicy`` quietly produces a one-dimensional object array
    instead of a distribution.
    """

    def test_uniform_policy_converts_to_the_dense_distribution(self):
        dense = np.asarray(UniformPolicy(10, 5))
        assert dense.shape == (10, 5)
        np.testing.assert_allclose(dense, 0.2)

    def test_deterministic_policy_converts_to_the_one_hot_matrix(self):
        actions = np.array([1, 0, 2])
        dense = np.asarray(DeterministicPolicy(actions, 3))
        assert dense.shape == (3, 3)
        np.testing.assert_allclose(dense, _deterministic_dist(actions, 3))

    def test_conversion_honours_a_requested_dtype(self):
        assert np.asarray(UniformPolicy(4, 2), dtype=np.float32).dtype == np.float32

    def test_estimators_still_take_the_sparse_path(self, monkeypatch):
        """Densifying is now possible, so pin that the estimators never do it.

        The dense matrix these classes exist to avoid is 8.6 GB on KuaiRec. Any
        conversion inside an estimator fails this test, and the estimates must
        still match the dense computation exactly.
        """
        def _forbidden(*args, **kwargs):
            raise AssertionError("an estimator materialised a sparse policy")

        for cls in (UniformPolicy, DeterministicPolicy):
            monkeypatch.setattr(cls, "to_dense", _forbidden)
            monkeypatch.setattr(cls, "__array__", _forbidden)

        rng = np.random.default_rng(11)
        n, n_actions = 200, 4
        actions = rng.integers(0, n_actions, n)
        rewards = rng.random(n)
        propensities = rng.uniform(0.2, 0.9, n)
        q = rng.random((n, n_actions))
        target = rng.integers(0, n_actions, n)
        names = ["ipw", "snipw", "dm", "dr"]

        for sparse, dense in (
            (DeterministicPolicy(target, n_actions), _deterministic_dist(target, n_actions)),
            (UniformPolicy(n, n_actions), np.full((n, n_actions), 1.0 / n_actions)),
        ):
            got = evaluate_policy(actions, rewards, propensities, sparse,
                                  q_estimates=q, estimators=names, n_bootstrap=10)
            want = evaluate_policy(actions, rewards, propensities, dense,
                                   q_estimates=q, estimators=names, n_bootstrap=10)
            for name in names:
                assert got[name].value == pytest.approx(want[name].value)
                assert got[name].lower == pytest.approx(want[name].lower)


class TestEffectiveSampleSize:
    def test_equals_n_for_uniform_weights(self):
        assert effective_sample_size(np.ones(100)) == pytest.approx(100.0)

    def test_collapses_when_one_weight_dominates(self):
        weights = np.concatenate([[1000.0], np.full(99, 1e-6)])
        assert effective_sample_size(weights) < 2.0


class TestValidation:
    def test_rejects_zero_propensity(self):
        """A zero propensity violates overlap and must not be silently clipped."""
        with pytest.raises(OPEError, match="strictly positive"):
            ipw(np.array([0]), np.array([1.0]), np.array([0.0]), np.array([[1.0, 0.0]]))

    def test_rejects_unnormalised_action_dist(self):
        with pytest.raises(OPEError, match="sum to 1"):
            ipw(np.array([0]), np.array([1.0]), np.array([0.5]), np.array([[0.3, 0.3]]))

    def test_rejects_length_mismatch(self):
        with pytest.raises(OPEError, match="length mismatch"):
            ipw(np.array([0, 1]), np.array([1.0]), np.array([0.5, 0.5]), np.ones((2, 2)) / 2)

    def test_rejects_action_outside_space(self):
        with pytest.raises(OPEError, match="outside the action space"):
            ipw(np.array([5]), np.array([1.0]), np.array([0.5]), np.array([[0.5, 0.5]]))

    def test_missing_q_estimates_raises_not_returns_zero(self):
        """A missing reward model must raise, never yield a placeholder score."""
        rng = np.random.default_rng(7)
        n = 50
        with pytest.raises(OPEError, match="requires q_estimates"):
            evaluate_policy(
                rng.integers(0, 2, n), rng.random(n), np.full(n, 0.5),
                np.full((n, 2), 0.5), estimators=["dr"],
            )


class TestEvaluatePolicy:
    def test_returns_interval_containing_point_estimate(self):
        rng = np.random.default_rng(8)
        n, n_actions = 1000, 4
        actions = rng.integers(0, n_actions, n)
        rewards = rng.random(n)
        propensities = np.full(n, 1.0 / n_actions)
        action_dist = rng.dirichlet(np.ones(n_actions), n)
        results = evaluate_policy(
            actions, rewards, propensities, action_dist, estimators=["snipw"], n_bootstrap=50
        )
        r = results["snipw"]
        assert r.lower <= r.value <= r.upper
        assert r.n_samples == n

    def test_healthy_estimate_carries_no_warnings(self):
        """The negative control for the flags below."""
        rng = np.random.default_rng(12)
        n, n_actions = 400, 4
        results = evaluate_policy(
            rng.integers(0, n_actions, n), rng.random(n), rng.uniform(0.2, 0.9, n),
            rng.dirichlet(np.ones(n_actions), n), estimators=["snipw"], n_bootstrap=20,
        )
        r = results["snipw"]
        assert r.warnings == ()
        assert r.has_confidence_interval
        assert r.as_row()["warnings"] == ""

    def test_ipw_is_not_identifiable_rather_than_a_published_zero(self):
        """End to end: a table cell of 0.00000 here would be bolded as the best value."""
        n, n_actions = 80, 3
        actions = np.zeros(n, dtype=int)
        action_dist = _deterministic_dist(np.ones(n, dtype=int), n_actions)
        results = evaluate_policy(
            actions, np.ones(n), np.full(n, 0.5), action_dist,
            estimators=["ipw"], n_bootstrap=10,
        )
        r = results["ipw"]
        assert r.status == "not_identifiable"
        assert not r.is_identifiable
        assert np.isnan(r.value)

    def test_dr_and_mrdr_are_flagged_when_they_reduce_to_dm(self):
        """A DM value in the DR column must not pass as doubly robust."""
        n, n_actions = 60, 3
        actions = np.zeros(n, dtype=int)
        action_dist = _deterministic_dist(np.ones(n, dtype=int), n_actions)
        q = np.tile(np.array([0.1, 0.9, 0.4]), (n, 1))
        results = evaluate_policy(
            actions, np.ones(n), np.full(n, 0.5), action_dist, q_estimates=q,
            estimators=["dm", "dr", "mrdr"], n_bootstrap=10,
        )
        assert results["dr"].value == pytest.approx(results["dm"].value)
        assert results["mrdr"].value == pytest.approx(results["dm"].value)
        for name in ("dr", "mrdr"):
            # The value is real, so it keeps its status and stays in the table;
            # the flag is what stops it being read as a doubly-robust estimate.
            assert results[name].status == "ok"
            assert WARNING_DR_REDUCES_TO_DM in results[name].warnings
            assert WARNING_NO_OVERLAP in results[name].warnings
            assert "not doubly robust" in results[name].detail
        assert results["dm"].warnings == (WARNING_NO_OVERLAP,)
        assert WARNING_DR_REDUCES_TO_DM not in results["dm"].warnings
        assert results["dr"].as_row()["warnings"] == f"{WARNING_NO_OVERLAP};{WARNING_DR_REDUCES_TO_DM}"

    def test_dropped_confidence_interval_is_recorded_not_absorbed(self):
        """An estimate resting on 2 of 300 rounds loses its interval; that must show.

        The interval is the one signal that exposes such an estimate, so losing
        it silently while ``status`` stays ``"ok"`` is worse than having no
        interval at all.
        """
        rng = np.random.default_rng(13)
        n, n_actions = 300, 3
        actions = np.zeros(n, dtype=int)
        target = np.ones(n, dtype=int)
        target[:2] = 0  # the policies agree on 2 rounds out of 300
        results = evaluate_policy(
            actions, rng.random(n), rng.uniform(0.1, 0.9, n),
            _deterministic_dist(target, n_actions), estimators=["snipw"], n_bootstrap=100,
        )
        r = results["snipw"]
        assert np.isfinite(r.value)          # the point estimate survives
        assert np.isnan(r.lower) and np.isnan(r.upper)
        assert not r.has_confidence_interval
        assert WARNING_CI_UNAVAILABLE in r.warnings
        assert "confidence interval unavailable" in r.detail
        assert r.as_row()["warnings"] == WARNING_CI_UNAVAILABLE

    def test_unknown_estimator_raises(self):
        rng = np.random.default_rng(9)
        n = 20
        with pytest.raises(OPEError, match="unknown estimator"):
            evaluate_policy(
                rng.integers(0, 2, n), rng.random(n), np.full(n, 0.5),
                np.full((n, 2), 0.5), estimators=["not_an_estimator"],
            )
