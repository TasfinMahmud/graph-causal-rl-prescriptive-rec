"""Tests for the offline RL agents and bandit baselines.

These verify the properties whose absence made the previous implementations
different algorithms from the ones they were named after.
"""

import numpy as np
import pytest

from gcrl.config import RLConfig
from gcrl.models.rl import (
    DiscreteIQL,
    LinUCB,
    NeuralUCB,
    RandomPolicy,
    build_policy,
)


@pytest.fixture
def linear_bandit():
    rng = np.random.default_rng(0)
    n, dim, n_actions = 800, 6, 4
    context = rng.normal(size=(n, dim))
    weights = rng.normal(size=(n_actions, dim))
    actions = rng.integers(0, n_actions, n)
    rewards = (context * weights[actions]).sum(axis=1) + 0.1 * rng.normal(size=n)
    oracle = np.argmax(context @ weights.T, axis=1)
    return context, actions, rewards, oracle, n_actions, dim


class TestLinUCB:
    def test_recovers_the_oracle_policy(self, linear_bandit):
        context, actions, rewards, oracle, n_actions, dim = linear_bandit
        model = LinUCB(n_actions, dim, alpha=0.1)
        model.fit(context, actions, rewards)
        assert (model.predict(context) == oracle).mean() > 0.9

    def test_confidence_bound_changes_the_policy(self, linear_bandit):
        """Without the UCB term this is ridge regression, not LinUCB."""
        context, actions, rewards, _, n_actions, dim = linear_bandit
        greedy = LinUCB(n_actions, dim, alpha=0.0)
        optimistic = LinUCB(n_actions, dim, alpha=5.0)
        greedy.fit(context, actions, rewards)
        optimistic.fit(context, actions, rewards)
        assert not np.array_equal(greedy.predict(context), optimistic.predict(context))

    def test_chunked_fitting_equals_single_pass(self, linear_bandit):
        """Sufficient statistics must accumulate across chunks, not be overwritten."""
        context, actions, rewards, _, n_actions, dim = linear_bandit
        whole = LinUCB(n_actions, dim)
        whole.fit(context, actions, rewards)

        chunked = LinUCB(n_actions, dim)
        for start in range(0, len(actions), 200):
            s = slice(start, start + 200)
            chunked.fit(context[s], actions[s], rewards[s])

        np.testing.assert_allclose(whole.A, chunked.A, rtol=1e-10)
        np.testing.assert_allclose(whole.b, chunked.b, rtol=1e-10)

    def test_uncertainty_shrinks_with_more_data(self, linear_bandit):
        context, actions, rewards, _, n_actions, dim = linear_bandit
        model = LinUCB(n_actions, dim)
        model.fit(context[:50], actions[:50], rewards[:50])
        early = np.linalg.inv(model.A[0]).trace()
        model.fit(context[50:], actions[50:], rewards[50:])
        assert np.linalg.inv(model.A[0]).trace() < early

    def test_rejects_wrong_context_dimension(self, linear_bandit):
        _, actions, rewards, _, n_actions, dim = linear_bandit
        with pytest.raises(ValueError, match="context dimension mismatch"):
            LinUCB(n_actions, dim).fit(np.zeros((len(actions), dim + 1)), actions, rewards)

    def test_rejects_out_of_range_actions(self, linear_bandit):
        context, _, rewards, _, n_actions, dim = linear_bandit
        bad = np.full(len(rewards), n_actions + 5)
        with pytest.raises(ValueError, match="actions must lie in"):
            LinUCB(n_actions, dim).fit(context, bad, rewards)

    def test_roundtrips_through_disk(self, linear_bandit, tmp_path):
        context, actions, rewards, _, n_actions, dim = linear_bandit
        model = LinUCB(n_actions, dim)
        model.fit(context, actions, rewards)
        expected = model.predict(context[:20])
        model.save(tmp_path / "linucb")
        restored = LinUCB(n_actions, dim)
        restored.load(tmp_path / "linucb")
        np.testing.assert_array_equal(restored.predict(context[:20]), expected)


class TestNeuralUCB:
    def test_gradient_covariance_accumulates(self, linear_bandit):
        """Z growing beyond lambda is what produces the exploration bonus."""
        context, actions, rewards, _, n_actions, dim = linear_bandit
        model = NeuralUCB(n_actions, dim, hidden=16, epochs=2, lambda_=1.0)
        initial = model.Z_diag.clone()
        model.fit(context[:200], actions[:200], rewards[:200])
        assert (model.Z_diag >= initial).all()
        assert model.Z_diag.max() > 1.0

    def test_predicts_valid_actions(self, linear_bandit):
        context, actions, rewards, _, n_actions, dim = linear_bandit
        model = NeuralUCB(n_actions, dim, hidden=16, epochs=1)
        model.fit(context[:200], actions[:200], rewards[:200])
        predictions = model.predict(context[:50])
        assert predictions.shape == (50,)
        assert predictions.min() >= 0 and predictions.max() < n_actions

    @staticmethod
    def _trained(linear_bandit):
        import torch

        context, actions, rewards, _, n_actions, dim = linear_bandit
        model = NeuralUCB(n_actions, dim, hidden=16, epochs=4, nu=1.0)
        model.fit(context[:400], actions[:400], rewards[:400])
        model.network.eval()
        return model, torch.as_tensor(context[400:432], dtype=torch.float32)

    def test_confidence_width_is_per_context(self, linear_bandit):
        """The bonus is sqrt(g_a(x)^T Z^-1 g_a(x)) with g_a taken at each x.

        Differentiating the batched sum instead gives sum_i g_a(x_i), one
        vector for the whole chunk, so every row gets the same width and the
        bonus stops being a function of the context. That is the shape of the
        bug this guards against: it makes the widths below identical row to row.
        """
        model, obs = self._trained(linear_bandit)
        widths = model._ucb_widths(obs)

        assert widths.shape == (len(obs), model.n_actions)
        spread = widths.std(dim=0)
        assert (spread > 1e-3).all(), f"widths do not vary across contexts: {spread}"
        # and the variation is a real fraction of the width, not float noise
        assert (spread / widths.mean(dim=0)).max() > 0.01

    def test_batched_widths_equal_row_by_row_widths(self, linear_bandit):
        """A row's width must not depend on which rows share its chunk."""
        model, obs = self._trained(linear_bandit)
        batched = model._ucb_widths(obs).numpy()
        row_by_row = np.concatenate(
            [model._ucb_widths(obs[i : i + 1]).numpy() for i in range(len(obs))]
        )
        np.testing.assert_allclose(batched, row_by_row, rtol=1e-4, atol=1e-5)

        model.ucb_memory_budget = 1 << 20  # force several row and action blocks
        rechunked = model._ucb_widths(obs).numpy()
        np.testing.assert_allclose(rechunked, row_by_row, rtol=1e-4, atol=1e-5)

    def test_batched_prediction_equals_row_by_row_prediction(self, linear_bandit):
        model, obs = self._trained(linear_bandit)
        context = obs.numpy()
        batched = model.predict(context)
        row_by_row = np.concatenate(
            [model.predict(context[i : i + 1]) for i in range(len(context))]
        )
        np.testing.assert_array_equal(batched, row_by_row)

    def test_widths_match_one_backward_pass_per_context(self, linear_bandit):
        """The vectorised path must equal plain autograd, row by row, action by action."""
        model, obs = self._trained(linear_bandit)
        reference = model._ucb_widths_rowwise(obs[:8]).numpy()
        vectorised = model._ucb_widths(obs[:8]).numpy()
        np.testing.assert_allclose(vectorised, reference, rtol=1e-4, atol=1e-5)

    def test_exploration_bonus_changes_the_policy(self, linear_bandit):
        """A bonus that is constant per action cannot reorder actions per row."""
        model, obs = self._trained(linear_bandit)
        context = obs.numpy()
        optimistic = model.predict(context)
        model.nu = 0.0
        greedy = model.predict(context)
        assert not np.array_equal(optimistic, greedy)


class TestDiscreteIQL:
    def test_learns_a_better_than_random_policy(self):
        rng = np.random.default_rng(0)
        n, dim, n_actions = 2000, 4, 3
        context = rng.normal(size=(n, dim))
        best = (context[:, 0] > 0).astype(int)
        actions = rng.integers(0, n_actions, n)
        rewards = (actions == best).astype(float)

        model = DiscreteIQL(n_actions, dim, hidden=64, epochs=30, batch_size=256)
        model.fit(context, actions, rewards)
        assert (model.predict(context) == best).mean() > 1.0 / n_actions

    def test_action_distribution_is_a_proper_distribution(self):
        rng = np.random.default_rng(0)
        model = DiscreteIQL(3, 4, hidden=16, epochs=2)
        model.fit(rng.normal(size=(200, 4)), rng.integers(0, 3, 200), rng.random(200))
        dist = model.action_distribution(rng.normal(size=(10, 4)))
        assert dist.shape == (10, 3)
        np.testing.assert_allclose(dist.sum(axis=1), 1.0, rtol=1e-5)
        assert (dist >= 0).all()

    def test_rejects_invalid_expectile(self):
        with pytest.raises(ValueError, match="expectile must be in"):
            DiscreteIQL(3, 4, expectile=1.5)

    def test_expectile_loss_is_asymmetric(self):
        """The asymmetry is what makes this expectile regression rather than MSE."""
        import torch
        positive = DiscreteIQL._expectile_loss(torch.tensor([1.0]), 0.7)
        negative = DiscreteIQL._expectile_loss(torch.tensor([-1.0]), 0.7)
        assert positive > negative


class TestRandomPolicy:
    def test_distribution_is_uniform_not_one_hot(self):
        """A one-hot sample would misrepresent a stochastic policy to the estimator."""
        policy = RandomPolicy(n_actions=5, seed=0)
        dist = policy.action_distribution(np.zeros((10, 3)))
        # The sparse UniformPolicy has to convert to its dense distribution here:
        # it used to present itself to NumPy as a sequence of UniformPolicy, and
        # this line recursed until the interpreter was killed mid-suite.
        assert np.asarray(dist).shape == (10, 5)
        np.testing.assert_allclose(dist, 0.2)

    def test_is_reproducible_under_a_seed(self):
        a = RandomPolicy(5, seed=1).predict(np.zeros((20, 2)))
        b = RandomPolicy(5, seed=1).predict(np.zeros((20, 2)))
        np.testing.assert_array_equal(a, b)


class TestPolicyFactory:
    @pytest.mark.parametrize("name", ["Random", "LinUCB", "NeuralUCB", "IQL", "DQN", "CQL", "BCQ"])
    def test_every_configured_agent_can_be_built(self, name):
        policy = build_policy(name, 5, 8, RLConfig(), device="cpu", seed=0)
        assert hasattr(policy, "fit") and hasattr(policy, "predict")

    def test_unknown_agent_raises_rather_than_substituting(self):
        """The previous code silently returned Discrete SAC when asked for IQL."""
        with pytest.raises(ValueError, match="unknown agent"):
            build_policy("NotAnAgent", 5, 8, RLConfig(), device="cpu", seed=0)

    def test_iql_is_a_real_iql_not_a_sac_fallback(self):
        policy = build_policy("IQL", 5, 8, RLConfig(), device="cpu", seed=0)
        assert isinstance(policy, DiscreteIQL)
        assert hasattr(policy, "value_network")  # expectile regression target
        assert hasattr(policy, "expectile")


class TestIQLPredictIsChunked:
    """`predict` must never issue one forward pass over every round.

    The policy network emits one logit per action, so the activation is
    `n_rows x n_actions` however small the network is. At KuaiRec's 4,676,570
    evaluation rounds over 3,327 actions that is 58 GiB in a single tensor --
    which is exactly what killed a Phase 4 run after Phase 3 had completed.
    Every other agent chunked; this one did not.
    """

    @staticmethod
    def _iql(n_actions=7, context_dim=4):
        from gcrl.models.rl import DiscreteIQL

        return DiscreteIQL(n_actions=n_actions, context_dim=context_dim, epochs=1)

    def test_chunked_prediction_equals_a_single_pass(self):
        """Chunking is numerically inert: argmax is per row."""
        import torch

        rng = np.random.default_rng(0)
        policy = self._iql()
        obs = rng.normal(size=(500, 4)).astype(np.float32)

        policy.policy_network.eval()
        with torch.no_grad():
            reference = (
                policy.policy_network(
                    torch.as_tensor(obs, dtype=torch.float32, device=policy.device)
                ).argmax(dim=1).cpu().numpy()
            )
        np.testing.assert_array_equal(policy.predict(obs), reference)

    def test_chunk_size_does_not_change_the_answer(self):
        rng = np.random.default_rng(1)
        policy = self._iql()
        obs = rng.normal(size=(300, 4)).astype(np.float32)

        baseline = policy.predict(obs)
        for budget in (1, 7, 13, 10_000_000):
            policy.PREDICT_CELL_BUDGET = budget
            np.testing.assert_array_equal(policy.predict(obs), baseline)

    def test_no_forward_pass_exceeds_the_cell_budget(self):
        """The guard that actually pins the defect: watch every call's shape."""
        policy = self._iql(n_actions=1000)
        policy.PREDICT_CELL_BUDGET = 50_000
        rng = np.random.default_rng(2)
        obs = rng.normal(size=(400, 4)).astype(np.float32)

        seen = []
        original = policy.policy_network.forward

        def watched(x):
            seen.append(x.shape[0] * policy.n_actions)
            return original(x)

        policy.policy_network.forward = watched
        policy.predict(obs)
        assert seen, "the network was never called"
        assert max(seen) <= policy.PREDICT_CELL_BUDGET, (
            f"a forward pass covered {max(seen)} cells, above the "
            f"{policy.PREDICT_CELL_BUDGET}-cell budget"
        )

    def test_empty_input(self):
        assert self._iql().predict(np.empty((0, 4), dtype=np.float32)).shape == (0,)
