"""Phase 4 must never materialise a dense ``(n_rounds, n_actions)`` grid.

The defect these tests pin. A 6.7-hour KuaiRec run completed Phase 3 and died at
the top of Phase 4 with::

    phase4: Unable to allocate 14.5 GiB for an array with shape (4676570, 3327)
    and data type bool

4,676,570 evaluation rounds x 3,327 actions as ``bool`` is exactly 14.5 GiB. Two
separate places built that grid, and both are pinned here:

1. :func:`gcrl.data.kuairec.build_exact_reward_lookup` computed its coverage
   report as ``observed[rows].mean()``. That gather IS the 14.5 GiB boolean
   grid, and it is the allocation that actually killed the run -- the dtype in
   the message identifies it, because the other path would have asked for
   58.0 GiB of ``float32`` first. It was building the grid only to divide two
   integers.

2. ``np.sum(action_dist * q_estimates, axis=1)`` in
   :func:`gcrl.evaluation.ope._expected_q` -- the fallback for any policy that
   is not one of ``SPARSE_POLICIES`` -- multiplied a dense action distribution
   by the lookup, which reached ``ExactRewardLookup.__rmul__`` and materialised
   the same grid as ``float64`` plus its mask: 115.9 GiB plus 14.5 GiB. Of every
   agent the pipeline builds, ``DiscreteIQL`` is the one that takes this path,
   because it is the only one whose ``action_distribution`` can return a dense
   array.

The tests assert the property that matters -- peak memory bounded by a block
size rather than by the number of evaluation rounds -- as well as the mechanism,
by making the dense paths raise.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pandas as pd
import pytest

from gcrl.config import load_config
from gcrl.data.kuairec import (
    ExactRewardLookup,
    KuaiRecData,
    build_exact_reward_lookup,
    build_full_reward_matrix,
)
from gcrl.encoders import IdentifierIndexer
from gcrl.evaluation.ope import (
    DENSE_ACTION_DIST_CELL_BUDGET,
    DENSE_Q_MODEL_CELL_BUDGET,
    DenseDistributionError,
    DeterministicPolicy,
    UniformPolicy,
    assert_dense_distribution_affordable,
    assert_dense_q_model_affordable,
    bootstrap_interval,
    exact_value,
)
from gcrl.models.rl import D3RLPyPolicy, DiscreteIQL, LinUCB, NeuralUCB, RandomPolicy
from gcrl.phases.phase4_ope import run_phase4

N_USERS, N_ACTIONS, N_ROUNDS, CONTEXT_DIM = 9, 24, 200, 3


# -- fixtures -------------------------------------------------------------


def _block(seed: int = 5, n_users: int = N_USERS, n_actions: int = N_ACTIONS):
    """A ground-truth block with a deliberate hole and one wholly unseen user."""
    rng = np.random.default_rng(seed)
    rewards = rng.random((n_users, n_actions)).astype(np.float32)
    observed = rng.random((n_users, n_actions)) > 0.25
    observed[n_users - 1] = False        # a round this user appears in is unscored
    return rewards, observed


@pytest.fixture(scope="module")
def rounds():
    rng = np.random.default_rng(6)
    return {
        "rows": rng.integers(0, N_USERS - 1, N_ROUNDS),
        "context": rng.random((N_ROUNDS, CONTEXT_DIM)).astype(np.float32),
        "actions": rng.integers(0, N_ACTIONS, N_ROUNDS),
        "rewards": rng.random(N_ROUNDS),
        "propensities": np.full(N_ROUNDS, 1.0 / N_ACTIONS),
    }


@pytest.fixture(scope="module")
def policies(rounds):
    """Every agent ``gcrl.models.rl.build_policy`` can return, trained for a step."""
    context, actions = rounds["context"], rounds["actions"]
    rewards = rounds["rewards"]
    built: dict = {"Random": RandomPolicy(N_ACTIONS, seed=1)}

    linucb = LinUCB(N_ACTIONS, CONTEXT_DIM)
    linucb.fit(context, actions, rewards)
    built["LinUCB"] = linucb

    neural = NeuralUCB(N_ACTIONS, CONTEXT_DIM, hidden=8, epochs=1)
    neural.fit(context, actions, rewards)
    built["NeuralUCB"] = neural

    iql = DiscreteIQL(N_ACTIONS, CONTEXT_DIM, hidden=8, num_layers=1, epochs=1, batch_size=16)
    iql.fit(context, actions, rewards)
    built["IQL"] = iql

    for algorithm in D3RLPyPolicy.SUPPORTED:
        agent = D3RLPyPolicy(
            algorithm, N_ACTIONS, CONTEXT_DIM, hidden_units=8, num_layers=1,
            batch_size=16, epochs=1, device="cpu", seed=1,
        )
        agent.fit(context, actions, rewards.astype(np.float32))
        built[algorithm] = agent
    return built


@pytest.fixture
def lookup(rounds):
    rewards, observed = _block()
    return ExactRewardLookup(rewards, observed, rounds["rows"], {"duplicate_cells": 0})


@pytest.fixture
def no_dense_grid(monkeypatch):
    """Make the whole-grid materialisation raise, so any caller of it is caught."""

    def refuse(self):
        raise AssertionError(
            f"to_dense_masked() built the whole {self.shape} grid; at KuaiRec's scale "
            f"that is the allocation this test exists to prevent"
        )

    monkeypatch.setattr(ExactRewardLookup, "to_dense_masked", refuse)


def _phase4_config(tmp_path, estimators="[exact]"):
    path = tmp_path / "phase4.yaml"
    path.write_text(f"""
experiment_name: dense
seed: 3
device: cpu
paths:
  root: "{tmp_path.as_posix()}"
dataset:
  name: dense
  loader: obd
  num_actions: {N_ACTIONS}
causal:
  enabled: false
rl:
  agents: [Random]
  state_source: raw_features
ope:
  estimators: {estimators}
  cross_fitting_folds: 3
  n_bootstrap: 5
""")
    config = load_config(path)
    config.paths.ensure_output_dirs()
    return config


# -- 1. the allocation that killed the run --------------------------------


class TestTheCoverageReportDoesNotDensify:
    """``build_exact_reward_lookup`` counted observed cells by building the grid."""

    @staticmethod
    def _eval_block(n_users: int, n_items: int):
        users = np.repeat(np.arange(100, 100 + n_users), n_items)
        items = np.tile(np.arange(500, 500 + n_items), n_users)
        frame = pd.DataFrame({
            "user_id": users,
            "video_id": items,
            "watch_ratio": ((users * 7 + items * 13) % 11) / 4.0,
            "play_duration": 1.0,
            "video_duration": 1.0,
        })
        frame["reward"] = (frame["watch_ratio"] >= 2.0).astype(np.float64)
        data = KuaiRecData(frame, "small", n_users, n_items, 2.0)
        return (
            data,
            IdentifierIndexer("u").fit(frame["user_id"]),
            IdentifierIndexer("i").fit(frame["video_id"]),
        )

    def test_the_grid_is_never_gathered(self, monkeypatch):
        """``observed[rows]`` is the (n_rounds, n_actions) bool array from the log."""

        class NoDenseGather(np.ndarray):
            """An ``observed`` matrix that refuses a per-ROUND 2-D gather."""

            def __getitem__(self, key):
                out = super().__getitem__(key)
                if isinstance(out, np.ndarray) and out.ndim == 2 and out.size > 4 * N_USERS:
                    raise AssertionError(
                        f"gathered a dense {out.shape} {out.dtype} grid: at KuaiRec's "
                        f"4,676,570 rounds x 3,327 actions this is the 14.5 GiB "
                        f"allocation that aborted Phase 4"
                    )
                return out

        data, users, items = self._eval_block(6, 30)
        original = build_full_reward_matrix

        def wrapped(*args, **kwargs):
            rewards, observed = original(*args, **kwargs)
            return rewards, observed.view(NoDenseGather)

        monkeypatch.setattr("gcrl.data.kuairec.build_full_reward_matrix", wrapped)
        rng = np.random.default_rng(2)
        built = build_exact_reward_lookup(
            data, users, items, round_user_index=rng.integers(0, 6, 400),
        )
        assert built.report["coverage"] == pytest.approx(1.0)

    def test_peak_memory_does_not_scale_with_the_grid(self):
        """Peak allocation must follow the rounds, not rounds x actions."""
        n_users, n_items, n_rounds = 100, 300, 200_000
        data, users, items = self._eval_block(n_users, n_items)
        rng = np.random.default_rng(3)
        rows = rng.integers(0, n_users, n_rounds)
        grid_bytes = n_rounds * n_items          # a dense bool grid, for scale

        tracemalloc.start()
        tracemalloc.reset_peak()
        try:
            build_exact_reward_lookup(data, users, items, round_user_index=rows)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        # A dense implementation peaks at >= grid_bytes (60 MB here, 14.5 GiB
        # on KuaiRec).
        assert peak < grid_bytes / 8, f"peak {peak:,} B against a {grid_bytes:,} B grid"

    def test_the_reported_numbers_are_unchanged(self):
        """The cheap count must equal the count the dense gather produced."""
        data, users, items = self._eval_block(20, 45)
        rng = np.random.default_rng(4)
        rows = rng.integers(0, 20, 900)
        built = build_exact_reward_lookup(data, users, items, round_user_index=rows)

        _, observed = build_full_reward_matrix(data, users, items)
        consulted = observed[rows]               # the pre-fix computation
        assert built.report["cells_consulted"] == int(consulted.size)
        assert built.report["cells_observed"] == int(consulted.sum())
        assert built.report["coverage"].hex() == float(consulted.mean()).hex()


# -- 2. no policy may densify the lookup ----------------------------------


class TestNoPolicyDensifiesTheLookup:
    """Every agent the pipeline can build, scored without the grid existing."""

    @pytest.mark.parametrize(
        "agent", ["Random", "LinUCB", "NeuralUCB", "IQL", "DQN", "DoubleDQN", "CQL", "BCQ"]
    )
    def test_exact_value_never_builds_the_grid(
        self, agent, policies, rounds, lookup, no_dense_grid
    ):
        action_dist = policies[agent].action_distribution(rounds["context"])
        value = exact_value(action_dist, lookup)
        assert np.isfinite(value)

    @pytest.mark.parametrize(
        "agent", ["Random", "LinUCB", "NeuralUCB", "IQL", "DQN", "DoubleDQN", "CQL", "BCQ"]
    )
    def test_the_bootstrap_never_builds_the_grid(
        self, agent, policies, rounds, lookup, no_dense_grid
    ):
        action_dist = policies[agent].action_distribution(rounds["context"])
        lower, upper = bootstrap_interval(
            lambda index: exact_value(action_dist[index], lookup[index]),
            N_ROUNDS, n_bootstrap=4, seed=7,
        )
        assert lower <= upper

    def test_iql_is_the_agent_that_took_the_dense_path(self, policies, rounds):
        """The parametrised tests above have teeth only if IQL is really dense."""
        distributions = {
            name: policy.action_distribution(rounds["context"])
            for name, policy in policies.items()
        }
        dense = [name for name, d in distributions.items() if isinstance(d, np.ndarray)]
        assert dense == ["IQL"]
        assert distributions["IQL"].shape == (N_ROUNDS, N_ACTIONS)

    def test_phase4_scores_a_dense_policy_without_the_grid(
        self, policies, rounds, lookup, no_dense_grid, tmp_path
    ):
        """End to end: the agent that densified is evaluated by Phase 4 itself."""
        config = _phase4_config(tmp_path)
        [evaluation] = run_phase4(
            {"IQL": policies["IQL"]}, rounds["context"], rounds["actions"],
            rounds["rewards"], rounds["propensities"], N_ACTIONS, config, seed=3,
            true_rewards=lookup,
        )
        assert evaluation.estimates["exact"].is_identifiable

    @staticmethod
    def _dense_exact_value_peak(n_rounds: int, n_actions: int = 500) -> tuple[float, int]:
        n_users = 8
        rewards, observed = _block(seed=8, n_users=n_users, n_actions=n_actions)
        rows = np.random.default_rng(9).integers(0, n_users - 1, n_rounds)
        built = ExactRewardLookup(rewards, observed, rows, {})
        distribution = np.full((n_rounds, n_actions), 1.0 / n_actions)
        tracemalloc.start()
        tracemalloc.reset_peak()
        try:
            value = exact_value(distribution, built)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        return value, peak

    def test_peak_memory_is_bounded_by_the_block_not_the_rounds(self):
        """A dense distribution over four times the rounds costs the same block."""
        n_actions = 500
        value, small = self._dense_exact_value_peak(40_000, n_actions)
        bigger, large = self._dense_exact_value_peak(160_000, n_actions)
        assert np.isfinite(value) and np.isfinite(bigger)

        # A block is about 18 bytes per cell: the rewards and their product as
        # float64, each carrying a boolean mask.
        ceiling = 25 * ExactRewardLookup.EXPECTED_CELL_BLOCK
        grid_bytes = 160_000 * n_actions * 8     # the float64 grid: 640 MB
        assert large < ceiling, f"peak {large:,} B above the {ceiling:,} B block ceiling"
        assert large < grid_bytes / 8, f"peak {large:,} B against a {grid_bytes:,} B grid"
        # Quadrupling the rounds must not move the peak: it is a function of the
        # block size alone, not of the number of rounds.
        assert large < 1.5 * small, f"peak grew with the rounds: {small:,} -> {large:,}"


# -- 3. the numbers may not move ------------------------------------------


class TestBitIdenticalToTheDenseReference:
    """The blocked computation is a memory change, not a numerical one."""

    @staticmethod
    def _dense_reference(action_dist, rewards, observed, rows):
        """The dense reference for ``exact_value``, computed without blocking."""
        dense = np.ma.MaskedArray(rewards[rows].astype(np.float64), mask=~observed[rows])
        if isinstance(action_dist, DeterministicPolicy):
            per_round = dense[np.arange(len(rows)), action_dist.actions]
        elif isinstance(action_dist, UniformPolicy):
            per_round = dense.mean(axis=1)
        else:
            per_round = np.sum(action_dist * dense, axis=1)
        return float(np.mean(per_round))

    @pytest.mark.parametrize("n_actions,n_rounds", [(24, 200), (7, 3), (300, 5_000)])
    def test_every_policy_shape_is_bit_identical(self, n_actions, n_rounds):
        rng = np.random.default_rng(12)
        rewards, observed = _block(seed=13, n_users=N_USERS, n_actions=n_actions)
        rows = rng.integers(0, N_USERS, n_rounds)
        built = ExactRewardLookup(rewards, observed, rows, {})

        softmax = rng.random((n_rounds, n_actions))
        softmax /= softmax.sum(axis=1, keepdims=True)
        for label, action_dist in (
            ("dense", softmax),
            ("deterministic", DeterministicPolicy(rng.integers(0, n_actions, n_rounds), n_actions)),
            ("uniform", UniformPolicy(n_rounds, n_actions)),
        ):
            got = exact_value(action_dist, built)
            want = self._dense_reference(action_dist, rewards, observed, rows)
            assert got.hex() == want.hex(), f"{label} moved: {got!r} != {want!r}"

    @pytest.mark.parametrize("cells_per_block", [1, 17, 512, 10**9])
    def test_the_block_size_cannot_change_the_answer(self, cells_per_block):
        rng = np.random.default_rng(14)
        rewards, observed = _block(seed=15)
        rows = rng.integers(0, N_USERS, 137)
        built = ExactRewardLookup(rewards, observed, rows, {})
        softmax = rng.random((137, N_ACTIONS))
        softmax /= softmax.sum(axis=1, keepdims=True)

        blocked = built.expected_under(softmax, cells_per_block=cells_per_block)
        whole = np.sum(softmax * built.to_dense_masked(), axis=1)
        np.testing.assert_array_equal(np.ma.getdata(blocked), np.ma.getdata(whole))
        np.testing.assert_array_equal(np.ma.getmaskarray(blocked), np.ma.getmaskarray(whole))

    def test_a_round_with_nothing_observed_stays_masked(self):
        """Masking, not zero-imputation, survives the blocking."""
        rewards = np.ones((2, 3), dtype=np.float32)
        observed = np.ones((2, 3), dtype=bool)
        observed[1] = False
        built = ExactRewardLookup(rewards, observed, np.array([0, 1, 0]), {})
        expected = built.expected_under(np.full((3, 3), 1 / 3))
        assert np.ma.getmaskarray(expected).tolist() == [False, True, False]
        assert exact_value(np.full((3, 3), 1 / 3), built) == pytest.approx(1.0)


# -- 4. the guard ---------------------------------------------------------


class TestTheDenseDistributionGuard:
    """Refuse an unaffordable distribution in the first second, by name."""

    def test_the_message_names_the_policy_the_shape_and_the_memory(self):
        with pytest.raises(DenseDistributionError) as raised:
            assert_dense_distribution_affordable("IQL", 4_676_570, 3_327)
        message = str(raised.value)
        assert message.startswith("IQL needs a DENSE action distribution")
        assert "4,676,570 rounds x 3,327 actions" in message
        assert "15,558,948,390 cells" in message
        assert "115.9 GiB as float64" in message
        assert "50,000,000-cell budget" in message

    def test_an_affordable_distribution_is_allowed(self):
        assert_dense_distribution_affordable("IQL", 10_000, 100)

    def test_phase4_refuses_before_fitting_the_reward_model(self, monkeypatch, tmp_path, rounds):
        """The guard runs before the expensive part of Phase 4, not after it."""

        def never(*args, **kwargs):
            raise AssertionError("the reward model was fitted before the guard fired")

        monkeypatch.setattr("gcrl.phases.phase4_ope.cross_fitted_reward_model", never)

        class HugeDense:
            """An agent that answers densely over KuaiRec's 3,327 actions."""

            n_actions = 3_327

            def distribution_is_dense(self, n_rounds: int) -> bool:
                return True

            def action_distribution(self, observations):
                raise AssertionError("the distribution was built before the guard fired")

        # 20,000 rounds x 3,327 actions = 66.5e6 cells, above the 50e6 budget.
        n = 20_000
        rng = np.random.default_rng(18)
        config = _phase4_config(tmp_path, estimators="[dm]")
        with pytest.raises(DenseDistributionError, match="Greedy") as raised:
            run_phase4(
                {"Greedy": HugeDense()}, rng.random((n, CONTEXT_DIM)),
                rng.integers(0, 3_327, n), rng.random(n), np.full(n, 1 / 3_327),
                3_327, config, seed=3,
            )
        assert "20,000 rounds x 3,327 actions" in str(raised.value)

    def test_a_policy_that_misreports_itself_is_still_refused(self, tmp_path, rounds, lookup):
        """The backstop checks what came back, not what the agent claimed."""

        class Liar:
            n_actions = N_ACTIONS

            def distribution_is_dense(self, n_rounds: int) -> bool:
                return False             # untrue

            def action_distribution(self, observations):
                # Only its shape is read; the guard refuses before any element is.
                class Grid:
                    shape = (4_676_570, 3_327)

                return Grid()

        config = _phase4_config(tmp_path)
        with pytest.raises(RuntimeError, match="Liar") as raised:
            run_phase4(
                {"Liar": Liar()}, rounds["context"], rounds["actions"], rounds["rewards"],
                rounds["propensities"], N_ACTIONS, config, seed=3, true_rewards=lookup,
            )
        # Not merely "something went wrong": the refusal names the grid and its cost.
        assert isinstance(raised.value.__cause__, DenseDistributionError)
        assert "4,676,570 rounds x 3,327 actions" in str(raised.value)
        assert "115.9 GiB as float64" in str(raised.value)

    def test_a_sparse_policy_at_the_same_scale_is_not_refused(self, policies):
        """The guard is about density, not about the number of rounds."""
        huge = 4_676_570
        for agent in ("Random", "LinUCB", "NeuralUCB", "IQL", "DQN", "CQL", "BCQ"):
            assert policies[agent].distribution_is_dense(huge) is False

    def test_iql_can_never_produce_a_distribution_phase4_refuses(self):
        """IQL's own fallback threshold and Phase 4's budget are one number."""
        assert DiscreteIQL.DENSE_CELL_BUDGET == DENSE_ACTION_DIST_CELL_BUDGET
        iql = DiscreteIQL(3_327, CONTEXT_DIM, hidden=4, num_layers=1, epochs=0)
        rounds_at_budget = DENSE_ACTION_DIST_CELL_BUDGET // 3_327
        assert iql.distribution_is_dense(rounds_at_budget)
        assert_dense_distribution_affordable("IQL", rounds_at_budget, 3_327)
        assert not iql.distribution_is_dense(rounds_at_budget + 1)

    def test_the_reward_model_grid_is_refused_too(self, monkeypatch, tmp_path):
        """``q(x, a)`` has no sparse form; at KuaiRec's full scale it is 115.9 GiB."""

        def never(*args, **kwargs):
            raise AssertionError("the reward model was fitted before the guard fired")

        monkeypatch.setattr("gcrl.phases.phase4_ope.cross_fitted_reward_model", never)

        class Greedy:
            n_actions = 3_327

            def distribution_is_dense(self, n_rounds: int) -> bool:
                return False              # sparse: only the reward model is at fault

            def action_distribution(self, observations):
                return DeterministicPolicy(np.zeros(len(observations), dtype=np.int64), 3_327)

        n = 700_000                       # x 3,327 = 2.3e9 cells, 17.4 GiB as float64
        rng = np.random.default_rng(19)
        config = _phase4_config(tmp_path, estimators="[dm]")
        with pytest.raises(DenseDistributionError, match="reward model") as raised:
            run_phase4(
                {"Greedy": Greedy()}, rng.random((n, CONTEXT_DIM)),
                rng.integers(0, 3_327, n), rng.random(n), np.full(n, 1 / 3_327),
                3_327, config, seed=3,
            )
        assert "700,000 rounds x 3,327 actions" in str(raised.value)
        assert "17.4 GiB as float64" in str(raised.value)

    def test_every_shipped_config_stays_under_the_reward_model_budget(self):
        """The budget refuses the impossible, never something that runs today."""
        # The largest grid any config in configs/ asks a reward model for:
        # kuairec_validate_ope scores 100,000 rounds over KuaiRec's 3,327 actions.
        assert_dense_q_model_affordable(100_000, 3_327)
        assert DENSE_Q_MODEL_CELL_BUDGET / 5 > 100_000 * 3_327
        # And the grid that cannot be held is refused.
        with pytest.raises(DenseDistributionError):
            assert_dense_q_model_affordable(4_676_570, 3_327)

    def test_multiplying_the_whole_lookup_is_refused_with_arithmetic(self):
        """Any caller outside the estimators gets an explanation, not a MemoryError."""
        rewards, observed = _block(seed=16, n_users=4, n_actions=3_327)
        rows = np.random.default_rng(17).integers(0, 4, 4_676_570)
        built = ExactRewardLookup(rewards, observed, rows, {})
        with pytest.raises(ValueError, match="refusing to multiply") as raised:
            built * 1.0
        assert "4,676,570 rounds x 3,327 actions" in str(raised.value)
