"""Confidence intervals resample the INDEPENDENT UNIT, not the logged round.

The defect these tests pin. ``bootstrap_interval`` resampled rounds. On
KuaiRec's enumerated evaluation block the 4,676,570 rounds are 1,411 users'
contexts replicated ~3,314 times each, so resampling rounds asserts 4,676,570
independent observations where there are 1,411 -- and reports an interval
roughly ``sqrt(3,314) ~ 58x`` too narrow. Those intervals go into a published
table.

``TestCoverage`` is the evidence that this is a defect rather than a
preference: on data with a known true value, the round-resampled 95% interval
covers it 24% of the time. The unit-resampled one covers it 90%.

The change is deliberately visible -- it widens published intervals -- so
``TestRecordedInEveryRow`` pins that every row says what was resampled and how
many of them there were. And because it must not move anything where nothing is
replicated, ``TestReducesToTheRoundBootstrap`` pins that case to the bit.
"""

from __future__ import annotations

import numpy as np
import pytest

from gcrl.config import load_config
from gcrl.evaluation.ope import (
    BootstrapUnits,
    DeterministicPolicy,
    bootstrap_interval,
    evaluate_policy,
    first_appearance_index,
)
from gcrl.phases.phase4_ope import run_phase4

N_ACTIONS = 5


def logged_rounds(n_contexts: int, per_context: int, seed: int = 0):
    """Rounds that replicate contexts, as an enumerated evaluation block does."""
    rng = np.random.default_rng(seed)
    context_id = np.repeat(np.arange(n_contexts), per_context)
    rng.shuffle(context_id)
    contexts = rng.normal(size=(n_contexts, 4)).astype(np.float32)
    n = len(context_id)
    return {
        "context": contexts[context_id],
        "context_id": context_id,
        "actions": rng.integers(0, N_ACTIONS, n),
        "rewards": rng.random(n),
        "propensities": np.full(n, 1.0 / N_ACTIONS),
        "n_contexts": n_contexts,
    }


# -- 1. it reduces exactly where nothing is replicated --------------------


class TestReducesToTheRoundBootstrap:
    """One round per unit must be the identical computation, not merely equivalent."""

    @staticmethod
    def _estimator(values):
        return lambda index: float(values[index].mean())

    def test_the_draw_is_the_same_draw(self):
        units = BootstrapUnits.per_round(500)
        assert units.is_per_round
        got = units.draw(np.random.default_rng(4))
        want = np.random.default_rng(4).integers(0, 500, size=500)
        np.testing.assert_array_equal(got, want)

    def test_distinct_identifiers_are_recognised_as_one_round_each(self):
        rng = np.random.default_rng(5)
        # Identifiers that are distinct but neither sorted nor contiguous: the
        # unit is still the round, so the draw must not move.
        ids = rng.permutation(np.arange(1000, 1300) * 7)
        units = BootstrapUnits.from_ids(ids)
        assert units.is_per_round
        assert units.n_units == 300
        np.testing.assert_array_equal(
            units.draw(np.random.default_rng(9)),
            np.random.default_rng(9).integers(0, 300, size=300),
        )

    def test_the_interval_is_byte_identical(self):
        rng = np.random.default_rng(6)
        values = rng.normal(size=400)
        without = bootstrap_interval(self._estimator(values), 400, n_bootstrap=64, seed=42)
        with_units = bootstrap_interval(
            self._estimator(values), 400, n_bootstrap=64, seed=42,
            units=rng.permutation(400),
        )
        assert with_units[0].hex() == without[0].hex()
        assert with_units[1].hex() == without[1].hex()

    def test_every_estimator_keeps_its_interval(self):
        """End to end through ``evaluate_policy``, on unreplicated rounds."""
        rng = np.random.default_rng(7)
        n = 300
        actions = rng.integers(0, N_ACTIONS, n)
        rewards = rng.random(n)
        propensities = rng.uniform(0.2, 0.9, n)
        q = rng.random((n, N_ACTIONS))
        truth = rng.random((n, N_ACTIONS))
        policy = DeterministicPolicy(
            np.where(rng.random(n) < 0.5, actions, (actions + 1) % N_ACTIONS), N_ACTIONS
        )
        names = ["ipw", "snipw", "dm", "dr", "mrdr", "exact"]
        common = {
            "actions": actions, "rewards": rewards, "propensities": propensities,
            "action_dist": policy, "q_estimates": q, "true_rewards": truth,
            "estimators": names, "n_bootstrap": 32, "seed": 11,
        }
        without = evaluate_policy(**common)
        with_units = evaluate_policy(bootstrap_units=np.arange(n), **common)
        for name in names:
            assert with_units[name].value.hex() == without[name].value.hex(), name
            assert with_units[name].lower.hex() == without[name].lower.hex(), name
            assert with_units[name].upper.hex() == without[name].upper.hex(), name
            assert with_units[name].resampling_unit == "round"
            assert with_units[name].n_independent_units == n


# -- 2. it carries a unit's rounds together -------------------------------


class TestTheUnitIsResampledWhole:
    def test_a_draw_carries_every_round_of_each_chosen_unit(self):
        ids = np.array([2, 0, 2, 1, 0, 2])
        units = BootstrapUnits.from_ids(ids)
        assert units.n_units == 3
        drawn = units.draw(np.random.default_rng(3))
        chosen, counts = np.unique(ids[drawn], return_counts=True)
        # Whatever was drawn, each unit appears a whole number of times over.
        sizes = dict(zip(*np.unique(ids, return_counts=True), strict=True))
        for unit, count in zip(chosen, counts, strict=True):
            assert count % sizes[unit] == 0

    def test_the_replicate_has_as_many_rounds_as_the_sample(self):
        ids = np.repeat(np.arange(50), 20)
        units = BootstrapUnits.from_ids(ids)
        rng = np.random.default_rng(2)
        for _ in range(5):
            assert len(units.draw(rng)) == 1000     # balanced units, balanced replicate

    def test_unbalanced_units_are_still_whole(self):
        ids = np.concatenate([np.zeros(7), np.ones(2), np.full(11, 2)]).astype(int)
        units = BootstrapUnits.from_ids(ids)
        rng = np.random.default_rng(1)
        for _ in range(20):
            drawn = units.draw(rng)
            picked = np.bincount(ids[drawn], minlength=3)
            assert set(np.unique(picked / np.array([7, 2, 11]))) <= {0.0, 1.0, 2.0, 3.0}

    def test_the_interval_widens_by_about_the_square_root_of_the_replication(self):
        rng = np.random.default_rng(8)
        n_contexts, per_context = 60, 36
        ids = np.repeat(np.arange(n_contexts), per_context)
        values = rng.normal(size=n_contexts)[ids]
        estimator = lambda index: float(values[index].mean())  # noqa: E731

        by_round = bootstrap_interval(estimator, len(values), n_bootstrap=200, seed=3)
        by_context = bootstrap_interval(
            estimator, len(values), n_bootstrap=200, seed=3, units=ids
        )
        ratio = (by_context[1] - by_context[0]) / (by_round[1] - by_round[0])
        assert 0.5 * per_context**0.5 < ratio < 2.0 * per_context**0.5, ratio

    def test_the_point_estimate_does_not_move(self):
        """Only the interval changes; the estimate is not resampled."""
        data = logged_rounds(20, 30, seed=2)
        n = len(data["actions"])
        q = np.random.default_rng(1).random((n, N_ACTIONS))
        policy = DeterministicPolicy(data["actions"], N_ACTIONS)
        common = {
            "actions": data["actions"], "rewards": data["rewards"],
            "propensities": data["propensities"], "action_dist": policy,
            "q_estimates": q, "estimators": ["snipw", "dm", "dr"], "n_bootstrap": 16,
        }
        without = evaluate_policy(**common)
        clustered = evaluate_policy(bootstrap_units=data["context_id"], **common)
        for name in ("snipw", "dm", "dr"):
            assert clustered[name].value.hex() == without[name].value.hex(), name


# -- 3. the coverage simulation -------------------------------------------


class TestCoverage:
    """With a known true value, does a 95% interval contain it 95% of the time?

    The data is the structure this change exists for, in miniature: each context
    carries a value drawn from a population, and all of its rounds carry that
    same value -- exactly as KuaiRec's exact evaluation gives every round of a
    user the reward of the one cell the policy selects for that user. The
    estimand is the population mean, which the round mean estimates; the truth
    is known because the population is.
    """

    N_CONTEXTS, PER_CONTEXT, N_SIMULATIONS, N_BOOTSTRAP = 60, 40, 200, 200
    TRUE_VALUE = 0.0

    @classmethod
    def _coverage(cls) -> tuple[dict[str, float], float]:
        rng = np.random.default_rng(0)
        ids = np.repeat(np.arange(cls.N_CONTEXTS), cls.PER_CONTEXT)
        units = BootstrapUnits.from_ids(ids)
        covered = {"round": 0, "context": 0}
        widths = {"round": [], "context": []}
        for simulation in range(cls.N_SIMULATIONS):
            values = rng.normal(cls.TRUE_VALUE, 1.0, cls.N_CONTEXTS)[ids]

            def estimator(index, _values=values):
                return float(_values[index].mean())

            for name, unit in (("round", None), ("context", units)):
                lower, upper = bootstrap_interval(
                    estimator, len(values), n_bootstrap=cls.N_BOOTSTRAP,
                    seed=1000 + simulation, units=unit,
                )
                covered[name] += int(lower <= cls.TRUE_VALUE <= upper)
                widths[name].append(upper - lower)
        coverage = {k: v / cls.N_SIMULATIONS for k, v in covered.items()}
        ratio = float(np.mean(widths["context"]) / np.mean(widths["round"]))
        return coverage, ratio

    @pytest.fixture(scope="class")
    def measured(self):
        return self._coverage()

    def test_the_round_bootstrap_is_far_below_its_nominal_level(self, measured):
        coverage, _ = measured
        # Measured: 0.245 against a nominal 0.95. Three quarters of the
        # published intervals would exclude the true value.
        assert coverage["round"] < 0.45, coverage

    def test_the_unit_bootstrap_is_close_to_its_nominal_level(self, measured):
        coverage, _ = measured
        # Measured: 0.90. A percentile bootstrap over 60 units under-covers
        # slightly; that is a property of the percentile method at this many
        # units, not of the clustering.
        assert coverage["context"] > 0.82, coverage

    def test_the_gap_is_not_a_rounding_difference(self, measured):
        coverage, _ = measured
        assert coverage["context"] - coverage["round"] > 0.5, coverage

    def test_the_width_ratio_matches_the_replication_factor(self, measured):
        _, ratio = measured
        # sqrt(40) = 6.32; measured 6.40.
        assert 0.7 * self.PER_CONTEXT**0.5 < ratio < 1.4 * self.PER_CONTEXT**0.5, ratio


# -- 4. it is recorded, not silent ----------------------------------------


def _phase4_config(tmp_path, estimators="[snipw, dm, dr]"):
    path = tmp_path / "units.yaml"
    path.write_text(f"""
experiment_name: units
seed: 3
device: cpu
paths:
  root: "{tmp_path.as_posix()}"
dataset:
  name: units
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
  n_bootstrap: 20
  confidence_level: 0.95
""")
    config = load_config(path)
    config.paths.ensure_output_dirs()
    return config


class TestRecordedInEveryRow:
    """A changed method that is not recorded is a silently changed table."""

    @pytest.fixture(scope="class")
    def evaluated(self, tmp_path_factory):
        data = logged_rounds(25, 24, seed=3)
        config = _phase4_config(tmp_path_factory.mktemp("units"))
        chosen = data["actions"]

        class Fixed:
            n_actions = N_ACTIONS

            def action_distribution(self, observations):
                return DeterministicPolicy(chosen, N_ACTIONS)

        [evaluation] = run_phase4(
            {"Fixed": Fixed()}, data["context"], data["actions"], data["rewards"],
            data["propensities"], N_ACTIONS, config, seed=3,
            round_context_id=data["context_id"], independent_unit="context",
        )
        return data, evaluation

    def test_every_row_names_the_unit_and_counts_it(self, evaluated):
        data, evaluation = evaluated
        for row in evaluation.as_rows():
            assert row["resampling_unit"] == "context"
            assert row["n_independent_units"] == data["n_contexts"]
            assert row["bootstrap_method"] == "cluster_percentile"
            assert row["n_bootstrap"] == 20
            assert row["confidence_level"] == 0.95
            # The round count is still reported, and is still the round count.
            assert row["n_samples"] == len(data["actions"])
            assert row["n_distinct_contexts"] == data["n_contexts"]

    def test_the_paper_can_state_the_method_from_one_row(self, evaluated):
        data, evaluation = evaluated
        row = evaluation.as_rows()[0]
        statement = (
            f"{row['bootstrap_method']} bootstrap, {row['n_bootstrap']} replicates, "
            f"resampling {row['n_independent_units']} {row['resampling_unit']}s "
            f"({row['n_samples']} rounds) at {row['confidence_level']:.0%}"
        )
        assert statement == (
            f"cluster_percentile bootstrap, 20 replicates, resampling "
            f"{data['n_contexts']} contexts ({len(data['actions'])} rounds) at 95%"
        )

    def test_rounds_are_the_unit_when_nothing_is_replicated(self, tmp_path):
        rng = np.random.default_rng(12)
        n = 200
        config = _phase4_config(tmp_path, estimators="[snipw]")
        actions = rng.integers(0, N_ACTIONS, n)

        class Fixed:
            n_actions = N_ACTIONS

            def action_distribution(self, observations):
                return DeterministicPolicy(actions, N_ACTIONS)

        [evaluation] = run_phase4(
            {"Fixed": Fixed()}, rng.normal(size=(n, 4)).astype(np.float32), actions,
            rng.random(n), np.full(n, 1.0 / N_ACTIONS), N_ACTIONS, config, seed=3,
        )
        for row in evaluation.as_rows():
            assert row["resampling_unit"] == "round"
            assert row["n_independent_units"] == n
            assert row["bootstrap_method"] == "percentile"


class TestFirstAppearanceIndex:
    """The numbering that makes every reduction exact."""

    def test_groups_are_numbered_by_first_appearance(self):
        ids, representatives = first_appearance_index(np.array([9, 4, 9, 7, 4]))
        np.testing.assert_array_equal(ids, [0, 1, 0, 2, 1])
        np.testing.assert_array_equal(representatives, [0, 1, 3])

    def test_distinct_identifiers_map_to_themselves(self):
        values = np.random.default_rng(3).permutation(50) * 3
        ids, representatives = first_appearance_index(values)
        np.testing.assert_array_equal(ids, np.arange(50))
        np.testing.assert_array_equal(representatives, np.arange(50))

    def test_an_empty_log_is_not_a_crash(self):
        ids, representatives = first_appearance_index(np.array([], dtype=np.int64))
        assert len(ids) == 0 and len(representatives) == 0


# -- 5. which unit, and why -----------------------------------------------


def enumerated_block(n_contexts: int = 30, n_actions: int = N_ACTIONS, seed: int = 0):
    """A fully observed block: every context paired with every action exactly once."""
    rng = np.random.default_rng(seed)
    context_id = np.repeat(np.arange(n_contexts), n_actions)
    actions = np.tile(np.arange(n_actions), n_contexts)
    contexts = rng.normal(size=(n_contexts, 4)).astype(np.float32)
    return {
        "context": contexts[context_id], "context_id": context_id, "actions": actions,
        "rewards": rng.random(len(actions)),
        "propensities": np.full(len(actions), 1.0 / n_actions),
        "n_contexts": n_contexts,
    }


def repeated_draw_log(n_contexts: int = 30, per_context: int = 40, seed: int = 0):
    """A plain log: each context's rounds are many independent impressions.

    OBD in miniature -- every ``(persona, item)`` cell is logged several times,
    each with its own click, so the rounds are independent observations even
    though only ``n_contexts`` distinct contexts exist.
    """
    rng = np.random.default_rng(seed)
    context_id = np.repeat(np.arange(n_contexts), per_context)
    contexts = rng.normal(size=(n_contexts, 4)).astype(np.float32)
    n = len(context_id)
    return {
        "context": contexts[context_id], "context_id": context_id,
        "actions": rng.integers(0, N_ACTIONS, n), "rewards": rng.random(n),
        "propensities": np.full(n, 1.0 / N_ACTIONS), "n_contexts": n_contexts,
    }


class TestTheUnitFollowsTheSamplingDesign:
    """Replication alone does not make rounds dependent; a census does.

    KuaiRec's evaluation block observes each ``(user, item)`` cell ONCE: a
    user's rounds are that user's complete row, so the sample is of users. OBD's
    log observes each ``(persona, item)`` cell ~42 times, each with its own
    click: those rounds are independent draws and the sample really is the
    rounds. Clustering the second would be the same error as not clustering the
    first, in the opposite direction.
    """

    def test_an_enumeration_is_recognised(self):
        from gcrl.phases.phase4_ope import ContextIndex, resolve_independent_unit

        data = enumerated_block()
        index = ContextIndex.of(data["context"], data["context_id"])
        unit, per_cell = resolve_independent_unit("auto", index, data["actions"])
        assert per_cell == pytest.approx(1.0)
        assert unit == "context"

    def test_repeated_draws_are_recognised(self):
        from gcrl.phases.phase4_ope import ContextIndex, resolve_independent_unit

        data = repeated_draw_log()
        index = ContextIndex.of(data["context"], data["context_id"])
        unit, per_cell = resolve_independent_unit("auto", index, data["actions"])
        assert per_cell > 5.0
        assert unit == "round"

    def test_an_explicit_declaration_wins(self):
        from gcrl.phases.phase4_ope import ContextIndex, resolve_independent_unit

        data = repeated_draw_log()
        index = ContextIndex.of(data["context"], data["context_id"])
        assert resolve_independent_unit("context", index, data["actions"])[0] == "context"
        assert resolve_independent_unit("round", index, data["actions"])[0] == "round"

    def test_an_unknown_unit_raises(self):
        from gcrl.phases.phase4_ope import ContextIndex, resolve_independent_unit

        data = repeated_draw_log(4, 4)
        index = ContextIndex.of(data["context"], data["context_id"])
        with pytest.raises(ValueError, match="independent_unit must be"):
            resolve_independent_unit("user", index, data["actions"])

    def test_a_repeated_draw_log_keeps_todays_numbers_and_intervals(self, tmp_path):
        """OBD's shape: nothing moves -- not the q model, not the interval."""
        from gcrl.phases.phase4_ope import ContextIndex, cross_fitted_reward_model

        data = repeated_draw_log(n_contexts=20, per_context=30, seed=4)
        config = _phase4_config(tmp_path, estimators="[dm]")
        chosen = data["actions"]

        class Fixed:
            n_actions = N_ACTIONS

            def action_distribution(self, observations):
                return DeterministicPolicy(chosen, N_ACTIONS)

        [evaluation] = run_phase4(
            {"Fixed": Fixed()}, data["context"], data["actions"], data["rewards"],
            data["propensities"], N_ACTIONS, config, seed=3,
            round_context_id=data["context_id"],
        )
        # The reward model is the round-folded, one-row-per-round model it was.
        n = len(chosen)
        q = cross_fitted_reward_model(
            data["context"], data["actions"], data["rewards"], N_ACTIONS, 3, seed=3,
            context_index=ContextIndex.per_round(n),
        )
        assert isinstance(q, np.ndarray)
        expected = float(np.mean(q[np.arange(n), chosen]))
        assert evaluation.estimates["dm"].value.hex() == expected.hex()

        for row in evaluation.as_rows():
            assert row["resampling_unit"] == "round"
            assert row["n_independent_units"] == n
            assert row["bootstrap_method"] == "percentile"
            # The replication is still reported, so the reader can see it.
            assert row["n_distinct_contexts"] == data["n_contexts"]

    def test_an_enumerated_block_clusters_without_being_told(self, tmp_path):
        data = enumerated_block(n_contexts=40, n_actions=N_ACTIONS, seed=5)
        config = _phase4_config(tmp_path, estimators="[snipw]")
        chosen = np.zeros(len(data["actions"]), dtype=np.int64)

        class Fixed:
            n_actions = N_ACTIONS

            def action_distribution(self, observations):
                return DeterministicPolicy(chosen, N_ACTIONS)

        [evaluation] = run_phase4(
            {"Fixed": Fixed()}, data["context"], data["actions"], data["rewards"],
            data["propensities"], N_ACTIONS, config, seed=3,
            round_context_id=data["context_id"],
        )
        for row in evaluation.as_rows():
            assert row["resampling_unit"] == "context"
            assert row["n_independent_units"] == data["n_contexts"]
            assert row["n_samples"] == len(data["actions"])
