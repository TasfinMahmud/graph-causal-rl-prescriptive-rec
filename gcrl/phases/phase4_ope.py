"""Phase 4: off-policy evaluation on the held-out test split.

Every agent is scored by the same estimators on the same rounds. Four
properties are enforced:

**The test split is used, and only here.** The previous implementation trained
and evaluated on the same full file, so every published number was in-sample.

**The reward model is cross-fitted.** ``q(x, a)`` is fitted on rounds other than
the ones it scores. Fitting it on the evaluation rounds themselves lets a
gradient-boosted regressor memorise their noise: on a fixture whose context
carries no signal at all, an in-sample ``q`` correlated 0.90 with the realised
test reward and inflated the direct method by 25%, against 5.8% for an honest
one. The inflation is not a constant offset -- it is largest for the agents
whose chosen actions the model fits best -- so it reorders the results table
rather than merely shifting it. ``ope.cross_fitting_folds`` sets K; each fold's
``q-hat`` is fitted on the other K-1 folds, which is the standard construction.
MRDR refits per policy under its own variance-minimising weights, and gets the
same treatment.

**The folds partition contexts, not rounds.** On an enumerated evaluation block
each context is replicated thousands of times, and a fold boundary drawn through
a context leaves its identical twin in the training fold -- which returns the
in-sample inflation the cross-fitting exists to remove, while every formal check
still passes. ``q(x, a)`` is stored once per context for the same reason it is
fitted that way: the duplicate rows are bit-identical, and on KuaiRec's full
block there are 115.9 GiB of them.

**Confidence intervals resample the independent unit.** With 1,411 contexts
replicated ~3,314 times, a bootstrap over rounds reports an interval about
``sqrt(3,314) ~ 58x`` too narrow. Every row records the unit, how many of them
there were, and the method.

**Propensities are what they claim to be.** OBD ships the behaviour policy's
propensity per round. KuaiRec ships none, so one is estimated from the training
log and every row records ``propensity_source``; substituting a constant makes a
self-normalised estimator collapse to a mean reward under the name SNIPW.

**Failures raise.** There is no ``except Exception: return 0.0``. A crashed
evaluation previously became a ``0.0`` in the results table, indistinguishable
from a real measurement of zero -- which is how several published zeros arose.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import ExperimentConfig
from ..evaluation.ope import (
    DENSE_ACTION_DIST_CELL_BUDGET,
    DENSE_Q_MODEL_CELL_BUDGET,
    SPARSE_POLICIES,
    BootstrapUnits,
    DeterministicPolicy,
    OPEResult,
    assert_dense_distribution_affordable,
    assert_dense_q_model_affordable,
    assert_reward_model_fit_affordable,
    describe_dense_grid,
    evaluate_policy,
    first_appearance_index,
    mrdr_weights,
)
from ..logging_utils import get_logger
from ..models.rl import BasePolicy

logger = get_logger(__name__)

#: Used when a caller supplies propensities without saying where they came from.
#: The default is deliberately NOT "logged": a results table that says "logged"
#: beside numbers nobody vouched for is exactly the kind of unbacked claim this
#: codebase exists to prevent. The CLI always passes the real provenance.
PROPENSITY_UNSPECIFIED = "unspecified"

#: Row-block size, in cells, for multiplying a dense action distribution by a
#: deduplicated ``q``. Each round's expectation is a sum along its own row, so
#: blocking rows changes the peak memory and nothing about the arithmetic.
#: Matches :data:`gcrl.evaluation.ope._EXPECTED_Q_CELL_BLOCK`.
_Q_EXPECTED_CELL_BLOCK = 1_000_000


@dataclass
class PolicyEvaluation:
    agent: str
    dataset: str
    state_source: str
    cate_reward_weight: float
    seed: int
    n_test_rounds: int
    estimates: dict[str, OPEResult]
    #: Provenance carried into every row, so no number can be read without it.
    propensity_source: str = PROPENSITY_UNSPECIFIED
    q_model: str = "none"
    exact_ground_truth: dict = field(default_factory=dict)
    #: What the evaluation measures, given how the training log and the
    #: evaluation rounds overlap. On KuaiRec this is the cold-item regime and
    #: the table cannot be read as a warm-start recommendation result.
    evaluation_regime: str = "unspecified"
    #: Distinct contexts among the evaluation rounds. When far below
    #: ``n_test_rounds`` the rounds are replicates of a handful of decisions,
    #: and the interval must be -- and is -- resampled over those contexts;
    #: resampling rounds would make it too narrow by roughly the square root of
    #: their ratio. Each estimate additionally carries ``resampling_unit`` and
    #: ``n_independent_units``, which say what was actually resampled.
    n_distinct_contexts: int | None = None

    def as_rows(self) -> list[dict[str, object]]:
        rows = []
        for result in self.estimates.values():
            row = {
                "agent": self.agent, "dataset": self.dataset,
                "state_source": self.state_source,
                "cate_reward_weight": self.cate_reward_weight,
                "seed": self.seed, "n_test_rounds": self.n_test_rounds,
            }
            row.update(result.as_row())
            row["propensity_source"] = self.propensity_source
            row["q_model"] = self.q_model
            row["evaluation_regime"] = self.evaluation_regime
            row["n_distinct_contexts"] = self.n_distinct_contexts
            row.update(self.exact_ground_truth)
            rows.append(row)
        return rows


#: Rows compared at a time when checking that a supplied ``round_context_id``
#: really does identify one context. Bounds the check's memory at KuaiRec's
#: scale, where the whole comparison would be a (4,676,570 x 64) boolean array.
_CONTEXT_CHECK_ROW_BLOCK = 200_000


def _distinct_context_rows(context: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Group rounds by the exact bytes of their context row.

    Two rounds share a context only if their rows are bitwise identical, which
    is the condition under which a fitted model's prediction for them is
    identical too -- the property the deduplication rests on. Rows are compared
    as raw bytes through a ``void`` view, so this is exact: no tolerance, and no
    two rows are merged because they are merely close.
    """
    flat = np.ascontiguousarray(context)
    flat = flat.reshape(len(flat), -1) if flat.ndim > 1 else flat.reshape(len(flat), 1)
    if flat.shape[1] == 0:
        return np.zeros(len(flat), dtype=np.int64), np.zeros(1, dtype=np.int64)
    void = flat.view(np.dtype((np.void, flat.dtype.itemsize * flat.shape[1])))
    return first_appearance_index(void.reshape(len(flat)))


def _contexts_are_constant_within(
    context: np.ndarray, ids: np.ndarray, representatives: np.ndarray
) -> bool:
    """Does every round of a group carry its group representative's context row?

    A supplied ``round_context_id`` is a claim about the data -- "these rounds
    share a context" -- and the deduplication turns that claim into arithmetic.
    It is verified rather than trusted: with ``state_source: raw_features`` the
    rounds of one user carry DIFFERENT contexts, and deduplicating them would
    silently change every DM/DR/MRDR number. Checked in row blocks so the
    verification itself never allocates a grid.
    """
    rows = representatives[ids]
    for start in range(0, len(context), _CONTEXT_CHECK_ROW_BLOCK):
        stop = start + _CONTEXT_CHECK_ROW_BLOCK
        if not np.array_equal(context[start:stop], context[rows[start:stop]]):
            return False
    return True


class ContextIndex:
    """Which evaluation rounds share a context, and which are independent draws.

    KuaiRec's evaluation block is an enumeration: every one of 1,411 users was
    paired with every one of 3,327 items, so its 4,676,570 rounds carry only
    1,411 distinct contexts, each replicated ~3,314 times. That single fact
    drives two things that would otherwise both be wrong.

    **``q(x, a)`` is held once per context.** ``q`` depends on a round only
    through its context, so the ~3,314 rows of one user are bit-identical
    copies. Computing one and gathering it is not an approximation; it is the
    difference between 115.9 GiB and 37.6 MB.

    **The bootstrap resamples contexts, not rounds.** 4,676,570 rounds are not
    4,676,570 independent observations, and treating them as such reports an
    interval about ``sqrt(3,314) ~ 58x`` too narrow.

    Two groupings are kept because they are not always the same. ``ids`` groups
    rounds whose context row is bitwise identical -- what ``q`` may be
    deduplicated over. ``unit_ids`` groups rounds that were not drawn
    independently -- the user or persona the round belongs to. Under
    ``state_source: gnn_embeddings`` these coincide, because the state IS the
    user's embedding; under ``raw_features`` one user's rounds carry different
    contexts, and then ``q`` must not be deduplicated while the bootstrap must
    still treat the user as the unit. Taking the coarser grouping for the
    bootstrap is the conservative direction: it can only widen the interval.
    """

    __slots__ = (
        "ids", "representatives", "source", "unit_ids", "n_units", "unit_source", "_units"
    )

    def __init__(
        self,
        ids: np.ndarray,
        representatives: np.ndarray,
        source: str,
        unit_ids: np.ndarray,
        unit_source: str,
    ) -> None:
        self.ids = np.asarray(ids, dtype=np.int64)
        self.representatives = np.asarray(representatives, dtype=np.int64)
        self.source = source
        self.unit_ids = np.asarray(unit_ids, dtype=np.int64)
        self.n_units = int(self.unit_ids.max()) + 1 if self.unit_ids.size else 0
        self.unit_source = unit_source
        self._units: BootstrapUnits | None = None

    @classmethod
    def of(cls, context: np.ndarray, round_context_id: np.ndarray | None = None) -> ContextIndex:
        """Build the index, using ``round_context_id`` when it is available.

        The identifier is a shortcut, not an authority: it says which rounds
        *should* share a context, and the context array says whether they do.
        When it is not supplied -- a caller that never had one -- the grouping
        is derived from the context rows themselves, which is always correct and
        costs a sort of the rounds.
        """
        context = np.asarray(context)
        n_rounds = len(context)
        if round_context_id is not None:
            unit_ids, representatives = first_appearance_index(np.asarray(round_context_id))
            if len(unit_ids) != n_rounds:
                raise ValueError(
                    f"round_context_id has {len(unit_ids)} entries but there are {n_rounds} "
                    f"evaluation rounds; the two must be aligned round for round"
                )
            if _contexts_are_constant_within(context, unit_ids, representatives):
                return cls(unit_ids, representatives, "round_context_id",
                           unit_ids, "round_context_id")
            logger.warning(
                "round_context_id groups the %d evaluation rounds into %d contexts, but the "
                "rounds of at least one of those groups carry DIFFERENT context rows. q(x, a) "
                "is deduplicated over identical rows only, so the grouping is rebuilt from the "
                "context array; the bootstrap keeps the supplied grouping, which is the "
                "conservative choice for an interval.",
                n_rounds, int(unit_ids.max()) + 1 if unit_ids.size else 0,
            )
            ids, representatives = _distinct_context_rows(context)
            return cls(ids, representatives, "context_rows", unit_ids, "round_context_id")
        ids, representatives = _distinct_context_rows(context)
        return cls(ids, representatives, "context_rows", ids, "context_rows")

    @classmethod
    def per_round(cls, n_rounds: int) -> ContextIndex:
        """Every round is its own context, with no deduplication applied.

        Used when the rounds replicate a context but are nonetheless independent
        draws -- see :func:`resolve_independent_unit`. Folds, ``q`` and the
        bootstrap then all reduce to the per-round case, index for index.
        """
        rounds = np.arange(int(n_rounds), dtype=np.int64)
        return cls(rounds, rounds, "round", rounds, "round")

    @property
    def n_rounds(self) -> int:
        return len(self.ids)

    @property
    def n_contexts(self) -> int:
        return len(self.representatives)

    @property
    def is_trivial(self) -> bool:
        """True when no context is shared, so deduplication has nothing to do."""
        return self.n_contexts == self.n_rounds

    @property
    def bootstrap_units(self) -> BootstrapUnits:
        """The independent units a confidence interval must be resampled over."""
        if self._units is None:
            self._units = BootstrapUnits.from_ids(self.unit_ids)
        return self._units

    def describe(self) -> str:
        return (
            f"{self.n_rounds:,} rounds, {self.n_contexts:,} distinct contexts "
            f"(from {self.source}), {self.n_units:,} independent units "
            f"(from {self.unit_source})"
        )


#: Rounds per distinct ``(context, action)`` cell below which the evaluation
#: rounds are read as an ENUMERATION of those cells rather than as repeated
#: independent draws of them. A complete enumeration has exactly 1.0; KuaiRec's
#: evaluation block measures 1.0 and OBD's log measures about 42.
ENUMERATED_ROUNDS_PER_CELL = 1.5


def rounds_per_context_action_cell(ids: np.ndarray, actions: np.ndarray) -> float:
    """How many logged rounds there are per distinct ``(context, action)`` cell."""
    ids = np.asarray(ids, dtype=np.int64)
    actions = np.asarray(actions, dtype=np.int64)
    if len(ids) == 0:
        return float("nan")
    keys = ids * (int(actions.max(initial=0)) + 1) + actions
    return len(ids) / max(1, len(np.unique(keys)))


def resolve_independent_unit(
    independent_unit: str, index: ContextIndex, actions: np.ndarray
) -> tuple[str, float]:
    """Is the independent observation the CONTEXT or the ROUND?

    Rounds replicating a context does not by itself make them dependent, and
    this is the distinction that decides it: whether a ``(context, action)`` cell
    was observed ONCE or MANY times.

    **Observed once -- an enumeration.** KuaiRec's evaluation block pairs each
    of 1,411 users with each of 3,327 items exactly once. Nothing about a user's
    ~3,314 rounds is a fresh draw: they are that user's complete row, and the
    sample is of users. Every estimator collapses accordingly -- ``exact`` gives
    each of a user's rounds the identical ``R(u, pi(u))``; under the block's own
    propensities a deterministic user-level policy agrees with exactly one round
    per user, so SNIPW is a mean over 1,411 numbers; DM reads one deduplicated
    ``q`` row per user. Treating 4,676,570 rounds as 4,676,570 observations
    inflates the sample size by 3,314 and narrows the interval by ~58x. Worse,
    the ONE round whose logged action is ``pi(u)`` is the exact cell DM is asked
    to predict, so a round-partitioned cross-fit puts the answer in the training
    fold 80% of the time -- in a table whose purpose is to measure DM's error
    against that ground truth.

    **Observed many times -- repeated draws.** OBD's 1,599,990 logged rounds
    carry ~474 personas over 80 actions: about 42 independent impressions of
    each ``(persona, item)`` cell, each with its own click. Those rounds ARE
    independent observations; the sample size really is the round count.
    Clustering them would widen every OBD interval by ~58x -- the same error in
    the opposite direction -- and holding whole personas out of the reward
    model's folds would make it extrapolate to personas it has never seen, which
    changes DM's bias rather than removing a leak. So OBD keeps rounds, and its
    published numbers do not move.

    ``independent_unit`` overrides the rule with ``"context"`` or ``"round"``;
    ``"auto"`` applies it. The caller that knows -- the CLI, which knows whether
    the test partition is a fully observed evaluation block -- declares it, and
    the rule is the fallback for callers that do not.

    Returns:
        The unit, and the measured rounds per ``(context, action)`` cell.
    """
    if independent_unit not in {"auto", "context", "round"}:
        raise ValueError(
            f"independent_unit must be 'auto', 'context' or 'round', got "
            f"{independent_unit!r}"
        )
    if index.is_trivial:
        # Nothing is replicated; the two readings coincide.
        return "round", 1.0
    per_cell = rounds_per_context_action_cell(index.ids, actions)
    if independent_unit != "auto":
        return independent_unit, per_cell
    return ("context" if per_cell < ENUMERATED_ROUNDS_PER_CELL else "round"), per_cell


class ReplicatedQEstimates:
    """A ``(n_rounds, n_actions)`` view of ``q(x, a)`` stored once per context.

    ``q(x_i, .)`` is a function of round ``i``'s context alone, so on data whose
    rounds replicate contexts the dense grid is mostly bit-identical copies:
    KuaiRec's evaluation block asks for 4,676,570 x 3,327 = 115.9 GiB of
    float64 to hold 1,411 x 3,327 = 37.6 MB of distinct values. This class holds
    the distinct values and gathers a row when one is asked for, presenting the
    same surface ``gcrl.evaluation.ope`` uses of a dense ``q``: ``.shape``,
    ``q[rounds, actions]``, ``q[index]`` for the bootstrap, ``q.mean(axis=1)``
    for a uniform policy, and ``expected_under`` for a dense one.

    Every number it returns is the number the dense grid would have returned,
    bit for bit -- the gather is a copy, not a recomputation.
    """

    __slots__ = ("_q", "_rows", "_row_mean")

    def __init__(
        self,
        q_by_context: np.ndarray,
        rows: np.ndarray,
        _row_mean: np.ndarray | None = None,
    ) -> None:
        self._q = np.asarray(q_by_context)
        if self._q.ndim != 2:
            raise ValueError(f"q must be (n_contexts, n_actions), got shape {self._q.shape}")
        self._rows = np.atleast_1d(np.asarray(rows, dtype=np.int64))
        if self._rows.size and (
            self._rows.min() < 0 or self._rows.max() >= self._q.shape[0]
        ):
            raise ValueError(
                f"context index out of range [0, {self._q.shape[0]}); observed "
                f"[{self._rows.min()}, {self._rows.max()}]"
            )
        # A uniform policy's expectation is the row mean, which is a property of
        # the CONTEXT: computed once per context and gathered, it is identical
        # to the dense grid's row mean and costs nothing per round.
        self._row_mean = self._q.mean(axis=1) if _row_mean is None else _row_mean

    #: Refuse to be absorbed into a ufunc, so ``dense_action_dist * q`` reaches
    #: :meth:`__rmul__` instead of silently materialising the grid.
    __array_ufunc__ = None

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self._rows), self._q.shape[1])

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def n_distinct_contexts(self) -> int:
        return int(self._q.shape[0])

    def __getitem__(self, key):
        """``q[rounds, actions]`` gathers cells; ``q[index]`` reslices rounds."""
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError(f"expected (rounds, actions), got a {len(key)}-tuple")
            rounds, columns = key
            return self._q[self._rows[rounds], columns]
        return ReplicatedQEstimates(self._q, self._rows[key], self._row_mean)

    def mean(self, axis=None, **_):
        """Per-round mean over actions for ``axis=1`` -- a uniform policy's DM value."""
        if axis != 1:
            raise ValueError(
                f"ReplicatedQEstimates.mean supports axis=1 (the per-round mean over "
                f"actions), got {axis!r}"
            )
        return self._row_mean[self._rows]

    def expected_under(self, action_dist, cells_per_block: int | None = None) -> np.ndarray:
        """``E_{a ~ pi_e}[q(x_i, a)]`` per round for a DENSE action distribution.

        Exactly ``np.sum(action_dist * q_dense, axis=1)`` -- the same elements
        in the same order, hence the same floating-point result -- with the
        rounds walked in blocks so the peak is set by the block size rather than
        by the number of rounds.
        """
        shape = tuple(getattr(action_dist, "shape", ()))
        if shape != self.shape:
            raise ValueError(
                f"action distribution {shape} does not match q {self.shape}"
            )
        n_rounds, n_actions = self.shape
        block = int(cells_per_block or _Q_EXPECTED_CELL_BLOCK)
        rows_per_block = max(1, block // max(1, n_actions))
        if n_rounds <= rows_per_block:
            return np.sum(action_dist * self._q[self._rows], axis=1)
        return np.concatenate([
            np.sum(
                action_dist[start : start + rows_per_block]
                * self._q[self._rows[start : start + rows_per_block]],
                axis=1,
            )
            for start in range(0, n_rounds, rows_per_block)
        ])

    def to_dense(self) -> np.ndarray:
        """The WHOLE ``(n_rounds, n_actions)`` grid. Memory-hungry by design."""
        return self._q[self._rows]

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        if copy is False:
            raise ValueError(
                "ReplicatedQEstimates stores one row per context, so it cannot be "
                "converted with copy=False"
            )
        self._refuse_if_unaffordable("converted to a dense array")
        dense = self.to_dense()
        return dense if dtype is None else dense.astype(dtype, copy=False)

    def __mul__(self, other):
        self._refuse_if_unaffordable("multiplied as a dense grid")
        return self.to_dense() * other

    __rmul__ = __mul__

    def _refuse_if_unaffordable(self, what: str) -> None:
        n_rounds, n_actions = self.shape
        if n_rounds * n_actions <= DENSE_Q_MODEL_CELL_BUDGET:
            return
        raise ValueError(
            f"refusing to have q {what}: {describe_dense_grid(n_rounds, n_actions)}, against "
            f"a budget of {DENSE_Q_MODEL_CELL_BUDGET:,} cells, to hold "
            f"{self.n_distinct_contexts:,} distinct rows. The estimators index q per round "
            f"and per action instead, which never builds the grid."
        )


def _fit_q_model(
    context: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    n_actions: int,
    seed: int,
    sample_weight: np.ndarray | None,
):
    """Fit one gradient-boosted regressor over ``[context, one-hot(action)]``.

    One model across all arms shares statistical strength instead of fitting a
    separate model per arm, which at KuaiRec's catalogue size would leave most
    arms with almost no data.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    n = len(actions)
    one_hot = np.zeros((n, n_actions), dtype=np.float32)
    one_hot[np.arange(n), actions] = 1.0
    model = HistGradientBoostingRegressor(max_iter=100, random_state=seed)
    model.fit(np.hstack([context, one_hot]), rewards, sample_weight=sample_weight)
    return model


def _predict_all_actions(model, context: np.ndarray, n_actions: int) -> np.ndarray:
    """Query a fitted model for every action, giving ``q(x_i, a)`` for each round."""
    n = len(context)
    q = np.zeros((n, n_actions), dtype=np.float64)
    for action in range(n_actions):
        probe = np.zeros((n, n_actions), dtype=np.float32)
        probe[:, action] = 1.0
        q[:, action] = model.predict(np.hstack([context, probe]))
    return q


def fit_reward_model(
    context: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    n_actions: int,
    seed: int = 42,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Fit ``q(x, a)`` on these rounds and score these same rounds.

    This is the IN-SAMPLE fit. It is kept because the cross-fitted estimator is
    built from it fold by fold, and because the difference between the two is
    the measurement this phase's tests make. Nothing in the pipeline scores a
    round with a model that saw it: use :func:`cross_fitted_reward_model`.
    """
    return _predict_all_actions(
        _fit_q_model(context, actions, rewards, n_actions, seed, sample_weight),
        context, n_actions,
    )


def context_fold_indices(
    index: ContextIndex,
    n_folds: int,
    seed: int = 42,
    sample_weight: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Partition the CONTEXTS into folds, and report each fold as round indices.

    Folds have to be assigned by context rather than by round, for two reasons
    that point the same way.

    **Honesty.** Cross-fitting's guarantee is that the model scoring a round was
    fitted without it. When a context is replicated, partitioning ROUNDS puts
    some of a context's rounds in the training folds and others in the scored
    fold -- and the training rounds carry the IDENTICAL context vector. A
    gradient-boosted regressor asked for ``q(x, a)`` can then read the answer off
    a sibling round that shares ``x``, so a model that formally never saw round
    ``i`` has seen ``i``'s context with a realised reward attached. The guarantee
    is nominally intact and actually gone: on the replicated fixture in
    ``tests/test_context_deduplication.py``, where a context's reward level is
    drawn independently of its features and so cannot be predicted honestly at
    all, a round-partitioned q correlates 0.77 with that level against -0.05 for
    a context-partitioned one. Partitioning by context restores the guarantee --
    a scored context appears in no training fold at all.

    **Well-definedness.** ``q`` is held once per context, so exactly one model
    must produce that row. A context split across folds has no single model to
    attribute its row to; the question only has an answer when a context belongs
    to one fold.

    With one round per context the two partitions are the same partition. That
    is not approximate: contexts are numbered by first appearance, so context
    ``k`` is round ``k``, ``KFold`` sees the identical index array, and the round
    indices below come back in the identical (ascending) order -- which is why
    deduplication leaves every point estimate bit-identical.

    Returns:
        One ``(fit_rounds, score_rounds, score_contexts)`` triple per fold.

    Raises:
        ValueError: if there are fewer contexts than folds, or if MRDR's weights
            are too concentrated to stratify.
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    n_contexts, n_rounds = index.n_contexts, index.n_rounds
    if n_folds < 2:
        raise ValueError(
            f"cross-fitting needs at least 2 folds, got {n_folds}; with one fold the model "
            f"is fitted on the very rounds it scores"
        )
    if n_contexts < n_folds:
        detail = (
            f"{n_rounds} evaluation rounds" if index.is_trivial
            else f"the {n_contexts} distinct contexts of {n_rounds} evaluation rounds"
        )
        raise ValueError(
            f"cannot cross-fit {detail} over {n_folds} folds. Lower "
            f"ope.cross_fitting_folds or evaluate on more rounds."
        )

    contexts = np.arange(n_contexts)
    if sample_weight is None:
        splits = KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(contexts)
    else:
        weights = np.asarray(sample_weight)
        if len(weights) != n_rounds:
            raise ValueError(
                f"sample_weight has {len(weights)} entries but there are {n_rounds} rounds"
            )
        # A context supports MRDR if ANY of its rounds does: the weight is zero
        # wherever the evaluation policy disagrees with the log, and one
        # agreeing round is enough to give that context a say in the fit.
        positive = np.zeros(n_contexts, dtype=bool)
        positive[index.ids[weights > 0]] = True
        n_positive = int(positive.sum())
        if n_positive < n_folds:
            supported = (
                f"only {n_positive} of {n_rounds} rounds carry a non-zero MRDR weight"
                if index.is_trivial else
                f"only {n_positive} of {n_contexts} distinct contexts carry a non-zero MRDR "
                f"weight on any of their {n_rounds} rounds"
            )
            raise ValueError(
                f"{supported}, which cannot be spread over {n_folds} folds. The evaluation "
                f"policy agrees with the log too rarely for MRDR to be estimable here; "
                f"report that rather than fitting a reward model on weights that are zero "
                f"almost everywhere."
            )
        splits = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=seed
        ).split(contexts, positive)

    folds = []
    for _, score_contexts in splits:
        scored = np.zeros(n_contexts, dtype=bool)
        scored[score_contexts] = True
        per_round = scored[index.ids]
        # Ascending, exactly as KFold's own masks yield them, so the trivial
        # case reproduces the previous partition index for index.
        folds.append(
            (np.flatnonzero(~per_round), np.flatnonzero(per_round), np.flatnonzero(scored))
        )
    return folds


def cross_fitted_reward_model(
    context: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    n_actions: int,
    n_folds: int,
    seed: int = 42,
    sample_weight: np.ndarray | None = None,
    context_index: ContextIndex | None = None,
):
    """``q(x, a)`` for every round, each fitted without that round OR ITS CONTEXT.

    The evaluation CONTEXTS are partitioned into ``n_folds``; the model scoring a
    context is fitted on the rounds of the other folds' contexts only. The
    result indexes like the dense ``(n_rounds, n_actions)`` array
    :func:`fit_reward_model` returns and drops straight into DM, DR and MRDR,
    but its value for round ``i`` contains neither round ``i``'s own noise nor
    that of any other round sharing ``i``'s context.

    ``q`` itself is stored once per distinct context -- see
    :class:`ReplicatedQEstimates`; the duplicate rows a dense grid would hold are
    bit-identical, and on KuaiRec's evaluation block they are 115.9 GiB of them.
    When no context is shared there is nothing to deduplicate and a plain NumPy
    array is returned, identical to what this function returned before.

    When ``sample_weight`` is given -- the MRDR case, where the weight is zero on
    every round the evaluation policy disagrees with the log -- the folds are
    stratified on whether a context has any supported round, so no training fold
    can end up with zero total weight purely as an artefact of the partition.

    Raises:
        ValueError: if there are fewer contexts than folds, or if MRDR's weights
            are too concentrated to stratify. Neither is papered over: an
            unfittable reward model must stop the run, not quietly become an
            in-sample one.
    """
    index = ContextIndex.of(context) if context_index is None else context_index
    if index.n_rounds != len(actions):
        raise ValueError(
            f"the context index covers {index.n_rounds} rounds but there are {len(actions)}"
        )
    folds = context_fold_indices(index, n_folds, seed, sample_weight)

    q = np.zeros((index.n_contexts, n_actions), dtype=np.float64)
    for fold, (fit_index, _, score_contexts) in enumerate(folds, start=1):
        weights = None if sample_weight is None else np.asarray(sample_weight)[fit_index]
        model = _fit_q_model(
            context[fit_index], actions[fit_index], rewards[fit_index],
            n_actions, seed + fold, weights,
        )
        q[score_contexts] = _predict_all_actions(
            model, context[index.representatives[score_contexts]], n_actions
        )
    logger.info(
        "Cross-fitted the reward model over %d folds of %d distinct contexts (%d rounds); no "
        "round was scored by a model that saw it or any other round of its context.",
        n_folds, index.n_contexts, index.n_rounds,
    )
    if index.is_trivial:
        # One context per round: the deduplicated grid IS the dense grid, and
        # the rows are already in round order.
        return q
    logger.info(
        "q(x, a) is held once per context: %s instead of %s.",
        describe_dense_grid(index.n_contexts, n_actions, unit="distinct contexts"),
        describe_dense_grid(index.n_rounds, n_actions),
    )
    return ReplicatedQEstimates(q, index.ids)


def _exact_round_coverage(true_rewards, action_dist, n_rounds: int) -> tuple[int, int]:
    """How many evaluation rounds the ground-truth lookup can actually score.

    A round counts as scored when the reward of the cell the policy's
    distribution puts mass on was recorded. For a deterministic policy that is
    one cell; for a stochastic one the round is lost only if the lookup has
    nothing observed for it at all.
    """
    if not hasattr(true_rewards, "observed_for"):
        return n_rounds, 0
    if isinstance(action_dist, DeterministicPolicy):
        observed = true_rewards.observed_for(action_dist.actions)
    else:
        observed = ~np.ma.getmaskarray(true_rewards.mean(axis=1))
    scored = int(np.count_nonzero(observed))
    return scored, n_rounds - scored


def working_context_index(
    context: np.ndarray,
    actions: np.ndarray,
    round_context_id: np.ndarray | None = None,
    independent_unit: str = "auto",
    reported: ContextIndex | None = None,
) -> tuple[ContextIndex, ContextIndex, str, float]:
    """The index the folds, ``q`` and the bootstrap actually use.

    Returns ``(working, reported, unit, rounds_per_cell)``. ``reported`` always
    describes the data -- how many distinct contexts the rounds carry, which
    every result row records whatever the unit turns out to be. ``working`` is
    the one that drives behaviour: the same index when the context is the
    independent unit, and a one-round-per-context index when it is not, which
    makes folds, ``q`` and intervals identical to what they were.
    """
    reported = ContextIndex.of(context, round_context_id) if reported is None else reported
    unit, per_cell = resolve_independent_unit(independent_unit, reported, actions)
    working = reported if unit == "context" else ContextIndex.per_round(reported.n_rounds)
    return working, reported, unit, per_cell


def assert_policy_distribution_affordable(
    agent_name: str,
    policy,
    n_rounds: int,
    n_actions: int | None = None,
) -> None:
    """Refuse, before asking for it, a distribution too large to hold.

    The cost of a dense ``(n_rounds, n_actions)`` action distribution is known
    from two integers, so it is checked from those two integers -- not after the
    allocation has been attempted. A 6.7-hour run died in Phase 4 on an opaque
    NumPy ``MemoryError``; the shape that killed it was knowable in the first
    second, and this is the second.

    An agent says whether it will answer densely via
    :meth:`~gcrl.models.rl.BasePolicy.distribution_is_dense`. One that does not
    implement it (a test double, a policy from outside this package) cannot be
    pre-flighted and is left to the backstop in :func:`evaluate_single_policy`,
    which checks what it actually returned.

    Raises:
        DenseDistributionError: naming the agent, the grid and what it costs.
    """
    declares = getattr(policy, "distribution_is_dense", None)
    if declares is None or not declares(int(n_rounds)):
        return
    width = int(getattr(policy, "n_actions", 0) or n_actions or 0)
    assert_dense_distribution_affordable(agent_name, n_rounds, width)


def evaluate_single_policy(
    agent_name: str,
    policy: BasePolicy,
    context: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    propensities: np.ndarray,
    config: ExperimentConfig,
    seed: int,
    q_estimates: np.ndarray | None = None,
    true_rewards: np.ndarray | None = None,
    n_actions: int | None = None,
    propensity_source: str = PROPENSITY_UNSPECIFIED,
    exact_report: dict | None = None,
    evaluation_regime: str = "unspecified",
    n_distinct_contexts: int | None = None,
    context_index: ContextIndex | None = None,
) -> PolicyEvaluation:
    """Score one policy on the test rounds with every configured estimator.

    MRDR is given its own reward model. Its defining feature is that the outcome
    model is fitted under a loss that minimises the *variance* of the resulting
    DR estimator, and those weights depend on the evaluation policy -- so the
    model must be refitted per policy. Reusing the plain DR model would make the
    MRDR column a duplicate of the DR column. That refit is cross-fitted for the
    same reason the shared one is.

    ``context_index`` says which rounds share a context. It is what makes the
    confidence intervals resample the independent unit rather than the round,
    and what lets MRDR's refitted ``q`` be cross-fitted and stored by context.
    Built here when the caller has none -- by the same rule
    :func:`resolve_independent_unit` applies in :func:`run_phase4`, so a direct
    caller gets the same answer as the pipeline does.
    """
    if context_index is None:
        context_index, _, _, _ = working_context_index(context, actions)
    index = context_index
    units = index.bootstrap_units
    assert_policy_distribution_affordable(agent_name, policy, len(actions), n_actions)
    action_dist = policy.action_distribution(context)
    # Backstop: an agent that answered the question above wrongly -- or does not
    # answer it at all -- is still refused here, before the distribution reaches
    # an estimator and before it is multiplied by the ground-truth lookup.
    if not isinstance(action_dist, SPARSE_POLICIES) and len(getattr(action_dist, "shape", ())) == 2:
        assert_dense_distribution_affordable(
            agent_name, action_dist.shape[0], action_dist.shape[1]
        )

    if action_dist.shape[1] != int(actions.max(initial=0)) + 1 and action_dist.shape[1] < int(
        actions.max(initial=0)
    ) + 1:
        raise ValueError(
            f"{agent_name} produced a distribution over {action_dist.shape[1]} actions but the "
            f"logs contain action index {int(actions.max())}"
        )

    estimators = list(config.ope.estimators)
    estimates: dict = {}
    ground_truth: dict[str, object] = {}

    if "exact" in estimators and true_rewards is not None:
        scored, masked = _exact_round_coverage(true_rewards, action_dist, len(actions))
        if scored == 0:
            raise ValueError(
                f"none of the {len(actions)} evaluation rounds has a recorded reward for the "
                f"action {agent_name} selects, so its exact value is not defined. Reporting a "
                f"number here would mean averaging over an empty set."
            )
        if masked:
            logger.warning(
                "[%s] %d of %d rounds (%.3f%%) have no recorded reward for the selected "
                "action and are MASKED OUT of the exact expectation, not imputed as zero. "
                "The exact value is conditional on the cell being observed.",
                agent_name, masked, len(actions), 100 * masked / len(actions),
            )
        ground_truth = {
            "exact_rounds_scored": scored,
            "exact_rounds_masked": masked,
            **{f"exact_{k}": v for k, v in (exact_report or {}).items()},
        }

    plain = [e for e in estimators if e != "mrdr"]
    if plain:
        estimates.update(
            evaluate_policy(
                actions=actions, rewards=rewards, propensities=propensities,
                action_dist=action_dist, q_estimates=q_estimates, true_rewards=true_rewards,
                estimators=plain, n_bootstrap=config.ope.n_bootstrap,
                confidence_level=config.ope.confidence_level, seed=seed,
                bootstrap_units=units,
            )
        )

    if "mrdr" in estimators:
        if n_actions is None:
            raise ValueError("MRDR requires n_actions to refit its reward model")
        weights = mrdr_weights(actions, propensities, action_dist)
        folds = config.ope.cross_fitting_folds
        supported = int(np.count_nonzero(weights > 0))
        # What has to be spread over the folds is SUPPORTED CONTEXTS, because
        # that is what the folds partition. On replicated data the two counts
        # differ, and using the round count here would let an unfittable model
        # through to raise inside the cross-fit instead of being reported.
        supported_contexts = int(np.count_nonzero(np.bincount(
            index.ids[weights > 0], minlength=index.n_contexts
        )))
        if min(supported, supported_contexts) < folds:
            # MRDR's weight is zero on every round the evaluation policy
            # disagrees with the log. Too few supported rounds and there is no
            # variance-minimising model to fit -- which is a finding about the
            # estimator on this data, not a failure of the run. It is recorded
            # as such: NaN with a status, never a number.
            carriers = (
                f"only {supported} of {len(actions)} rounds carry a non-zero MRDR weight"
                if index.is_trivial else
                f"only {supported_contexts} of {index.n_contexts} distinct contexts carry a "
                f"non-zero MRDR weight on any of their {len(actions)} rounds"
            )
            detail = (
                f"{carriers}, fewer than the {folds} cross-fitting folds, so the "
                f"variance-minimising reward model is not estimable; no confidence interval "
                f"or effective sample size is reported"
            )
            logger.warning(
                "[%s] MRDR is not identifiable on this data: %s. Reported as "
                "not_identifiable rather than as a numeric value.", agent_name, detail,
            )
            estimates["mrdr"] = OPEResult(
                "mrdr", float("nan"), float("nan"), float("nan"), len(actions), None,
                status="not_identifiable", detail=detail,
                resampling_unit=units.unit_name, n_independent_units=units.n_units,
                bootstrap_method=units.method, n_bootstrap=config.ope.n_bootstrap,
                confidence_level=config.ope.confidence_level,
            )
        else:
            logger.info(
                "[%s] refitting the reward model under the MRDR objective, cross-fitted over "
                "%d folds (%d supported rounds, weight range [%.3g, %.3g])",
                agent_name, folds, supported, weights.min(), weights.max(),
            )
            q_mrdr = cross_fitted_reward_model(
                context, actions, rewards, n_actions, folds, seed, sample_weight=weights,
                context_index=index,
            )
            estimates.update(
                evaluate_policy(
                    actions=actions, rewards=rewards, propensities=propensities,
                    action_dist=action_dist, q_estimates=q_mrdr, estimators=["mrdr"],
                    n_bootstrap=config.ope.n_bootstrap,
                    confidence_level=config.ope.confidence_level, seed=seed,
                    bootstrap_units=units,
                )
            )

    for name, result in estimates.items():
        if not result.is_identifiable:
            logger.info(
                "[%s] %s %s = %s (%s)", config.dataset.name, agent_name, name.upper(),
                result.status, result.detail,
            )
            continue
        logger.info(
            "[%s] %s %s = %.6f  [%.6f, %.6f]  (ESS %.0f / %d)",
            config.dataset.name, agent_name, name.upper(), result.value,
            result.lower, result.upper, result.effective_sample_size or 0, result.n_samples,
        )

    return PolicyEvaluation(
        agent=agent_name,
        dataset=config.dataset.name,
        state_source=config.rl.state_source,
        cate_reward_weight=config.rl.cate_reward_weight,
        seed=seed,
        n_test_rounds=len(actions),
        estimates=estimates,
        propensity_source=propensity_source,
        q_model=(
            f"cross_fitted(k={config.ope.cross_fitting_folds}"
            + ("" if index.is_trivial else ", folds=contexts, q=per_context")
            + ")"
            if {"dm", "dr", "mrdr"} & set(estimators) else "none"
        ),
        exact_ground_truth=ground_truth,
        evaluation_regime=evaluation_regime,
        n_distinct_contexts=n_distinct_contexts,
    )


def run_phase4(
    policies: dict[str, BasePolicy],
    context: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    propensities: np.ndarray,
    n_actions: int,
    config: ExperimentConfig,
    seed: int | None = None,
    true_rewards: np.ndarray | None = None,
    propensity_source: str = PROPENSITY_UNSPECIFIED,
    evaluation_regime: str = "unspecified",
    round_context_id: np.ndarray | None = None,
    independent_unit: str = "auto",
) -> list[PolicyEvaluation]:
    """Evaluate every trained policy and write the results table.

    Args:
        true_rewards: the ground-truth reward of every (round, action) cell,
            required by the ``exact`` estimator and supplied by the caller that
            loaded the fully observed evaluation block. An
            :class:`~gcrl.data.kuairec.ExactRewardLookup` also carries a coverage
            report, which is copied into every result row.
        propensity_source: ``"logged"``, or a description of the estimator that
            produced the propensities. It is written into every row.
        evaluation_regime: what the evaluation measures given how the training
            log and the evaluation rounds overlap. Written into every row,
            because a cold-item table and a warm-start table are not the same
            result and must not be read as one.
        round_context_id: one identifier per round saying which context it
            belongs to. It sets how many DISTINCT decisions the rounds
            represent, and with that three things: the confidence intervals are
            resampled over contexts rather than rounds, the cross-fitting folds
            partition contexts rather than rounds, and ``q(x, a)`` is held once
            per context rather than once per round. It is verified against the
            context array rather than trusted, and derived from that array when
            it is not supplied.
        independent_unit: ``"context"``, ``"round"`` or ``"auto"``. Whether the
            rounds of one context are an enumeration of that context's cells --
            one sample, many rows -- or repeated independent draws of them. It
            decides the fold assignment, whether ``q`` is deduplicated, and what
            the confidence intervals resample; see
            :func:`resolve_independent_unit` for the rule ``"auto"`` applies and
            why the two datasets answer it differently.

    Raises:
        RuntimeError: if any agent fails to evaluate. One agent's failure stops
            the phase rather than contributing a placeholder row.
    """
    seed = seed if seed is not None else config.seed
    logger.info(
        "Phase 4: evaluating %d policies on %d held-out rounds with %s | propensities: %s",
        len(policies), len(actions), config.ope.estimators, propensity_source,
    )

    if "exact" in config.ope.estimators and true_rewards is None:
        raise ValueError(
            "ope.estimators requests 'exact' but no ground-truth reward matrix was supplied. "
            "Exact evaluation needs a fully observed evaluation block -- set "
            "dataset.eval_file (KuaiRec's small_matrix) so one is built and passed in, or "
            "drop 'exact' from the estimator list."
        )

    exact_report = dict(getattr(true_rewards, "report", {}) or {})
    if exact_report:
        logger.info("Exact ground truth: %s", json.dumps(exact_report, sort_keys=True))
    logger.info("Evaluation regime: %s", evaluation_regime)

    context_index, reported_index, unit, rounds_per_cell = working_context_index(
        context, actions, round_context_id, independent_unit
    )
    n_distinct_contexts = reported_index.n_contexts
    logger.info("Context structure: %s", reported_index.describe())
    if not reported_index.is_trivial:
        replication = len(actions) / n_distinct_contexts
        consequence = (
            "Each (context, action) cell carries %.1f rounds, so a context's rounds are an "
            "ENUMERATION of its cells rather than independent draws: the independent unit is "
            "the CONTEXT. The bootstrap resamples contexts, carrying all of each one's "
            "rounds -- resampling rounds would report an interval roughly %.0fx too narrow -- "
            "and the cross-fitting folds partition contexts, so no model scores a context it "
            "was fitted on."
            if unit == "context" else
            "Each (context, action) cell carries %.1f rounds, so those rounds are REPEATED "
            "INDEPENDENT DRAWS of the same cells and the independent unit remains the ROUND. "
            "The interval is resampled over rounds; clustering them would make it roughly "
            "%.0fx too wide, and holding whole contexts out of the reward model's folds would "
            "make it extrapolate to contexts it never saw."
        )
        logger.warning(
            "The %d evaluation rounds carry only %d distinct contexts (%.1f rounds each). A "
            "policy that depends on the context alone makes %d distinct decisions, not %d. "
            + consequence +
            " (unit %s; every row records 'n_distinct_contexts', 'n_independent_units' and "
            "'resampling_unit'.)",
            len(actions), n_distinct_contexts, replication,
            n_distinct_contexts, len(actions),
            rounds_per_cell, replication ** 0.5,
            "declared" if independent_unit != "auto" else "measured",
        )

    # Every agent is asked what shape its action distribution will be BEFORE
    # anything expensive happens -- before the reward model is cross-fitted and
    # before the first estimator runs. An agent that needs a dense grid larger
    # than the budget stops the phase here, in the first second, with the shape
    # and the memory named. The failure this replaces was a NumPy MemoryError
    # after 6.7 hours, which said nothing about which agent caused it.
    for agent_name, policy in policies.items():
        assert_policy_distribution_affordable(agent_name, policy, len(actions), n_actions)
    if {"dm", "dr", "mrdr"} & set(config.ope.estimators):
        # q(x, a) is a dense grid -- one row per DISTINCT CONTEXT, since rounds
        # that share a context share a bit-identical row -- and it is built
        # before the first agent is scored. Its cost is knowable now too, and so
        # is that of the matrix the model is FITTED on, which deduplication does
        # not shrink because every round is its own training example.
        assert_dense_q_model_affordable(
            len(actions), n_actions,
            n_contexts=context_index.n_contexts,      # the working index: what is built
        )
        assert_reward_model_fit_affordable(
            len(actions) - len(actions) // max(2, config.ope.cross_fitting_folds),
            context.shape[1] if context.ndim > 1 else 1,
            n_actions,
        )
    logger.info(
        "Action-distribution pre-flight passed for %d agents over %d rounds x %d actions "
        "(dense budget %s cells)",
        len(policies), len(actions), n_actions, f"{DENSE_ACTION_DIST_CELL_BUDGET:,}",
    )

    # The DM/DR reward model is policy-independent and fitted once, cross-fitted
    # over the evaluation rounds. MRDR's is policy-dependent and refitted per
    # agent inside evaluate_single_policy, cross-fitted the same way.
    needs_q = bool({"dm", "dr"} & set(config.ope.estimators))
    q_estimates = None
    if needs_q:
        logger.info(
            "Cross-fitting the reward model shared by DM and DR over %d folds",
            config.ope.cross_fitting_folds,
        )
        q_estimates = cross_fitted_reward_model(
            context, actions, rewards, n_actions, config.ope.cross_fitting_folds, seed,
            context_index=context_index,
        )

    evaluations: list[PolicyEvaluation] = []
    for agent_name, policy in policies.items():
        try:
            evaluations.append(
                evaluate_single_policy(
                    agent_name, policy, context, actions, rewards, propensities,
                    config, seed, q_estimates, true_rewards, n_actions,
                    propensity_source=propensity_source, exact_report=exact_report,
                    evaluation_regime=evaluation_regime,
                    n_distinct_contexts=n_distinct_contexts,
                    context_index=context_index,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"evaluation of agent {agent_name!r} failed: {exc}. The run is stopped rather "
                f"than recording a placeholder score for this agent."
            ) from exc

    rows = [row for evaluation in evaluations for row in evaluation.as_rows()]
    output = Path(config.paths.results) / f"phase4_ope_{config.experiment_name}_seed{seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    logger.info("Phase 4 results -> %s (%d rows)", output, len(rows))

    try:
        import pandas as pd

        csv_path = output.with_suffix(".csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        logger.info("Phase 4 results also written to %s", csv_path)
    except ImportError:  # pragma: no cover
        pass

    return evaluations
