"""Off-policy evaluation estimators.

These estimate the value of an evaluation policy from data logged under a
different behaviour policy. Every estimator here is implemented from its
definition and unit-tested against cases with known answers.

Implementing them natively rather than depending on the Open Bandit Pipeline is
deliberate: ``obp`` pins ``PyYAML<6``, which no longer builds on Python 3.11+,
so a dependency on it would make the benchmark unrunnable on current
interpreters. The estimator definitions follow Saito et al. (2021) and
Farajtabar et al. (2018).

Notation
--------
``n``            number of logged rounds
``a_i``          action taken by the behaviour policy in round ``i``
``r_i``          observed reward in round ``i``
``p_i``          behaviour-policy propensity of ``a_i``, i.e. ``pi_b(a_i | x_i)``
``pi_e(a | x_i)`` evaluation-policy probability of action ``a``
``q(x_i, a)``    reward model estimate
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..logging_utils import get_logger

logger = get_logger(__name__)

_EPSILON = 1e-12


class OPEError(ValueError):
    """Raised when inputs to an estimator are inconsistent or degenerate."""


class DenseDistributionError(OPEError):
    """Raised when an evaluation policy's action distribution is too large to hold.

    Refusing before the run starts is the entire point of this exception: a
    dense ``(n_rounds, n_actions)`` matrix is the one input to Phase 4 whose
    cost is knowable from two integers, and the alternative is a NumPy
    ``MemoryError`` raised hours later, after the policies it would have scored
    have already been trained.
    """


#: Above this many ``(n_rounds x n_actions)`` cells a dense action distribution
#: is refused rather than allocated. 5e7 cells is 381.5 MiB as float64 for the
#: distribution alone, and the same grid again for anything multiplied by it.
#: It is deliberately the single source of this number: ``DiscreteIQL`` reads it
#: to decide when to fall back to its greedy policy, so no agent can produce a
#: distribution that Phase 4 will then refuse.
DENSE_ACTION_DIST_CELL_BUDGET = 50_000_000

#: Above this many cells the DM/DR/MRDR reward model is refused. ``q(x, a)`` is
#: genuinely a dense grid -- the estimators index it per round and per action --
#: so this is a ceiling on what can be held, not a blocking threshold, and it is
#: set far above every configuration in ``configs/``: the largest of those asks
#: for 3.3e8 cells (2.7 GiB). Nothing that runs today is refused by it.
#:
#: The grid it measures is one row per DISTINCT CONTEXT, not one per round:
#: ``q(x_i, .)`` depends on round ``i`` only through its context, so rounds that
#: share a context share a row exactly (see
#: :class:`gcrl.phases.phase4_ope.ReplicatedQEstimates`). On KuaiRec's full
#: evaluation block that is the difference between 4,676,570 x 3,327 =
#: 1.6e10 cells (115.9 GiB) and 1,411 x 3,327 = 4.7e6 cells (37.6 MB).
DENSE_Q_MODEL_CELL_BUDGET = 2_000_000_000

#: Above this many cells the reward model's TRAINING matrix is refused.
#:
#: Deduplicating ``q`` removes the prediction grid, not the fit: every round is
#: its own training example, because rounds sharing a context still carry
#: different actions and different rewards. ``_fit_q_model`` hands
#: ``HistGradientBoostingRegressor`` an ``(n_fit_rounds, context_dim +
#: n_actions)`` design matrix -- the one-hot action block is what makes it wide
#: -- and scikit-learn converts it to ``float64`` (``X_DTYPE``) before binning,
#: so a cell costs 8 bytes there on top of the 4 the ``float32`` hstack already
#: cost. This is knowable from three integers, so it is checked from them.
#:
#: 1e9 cells is 7.5 GiB as float64 plus 3.7 GiB for the float32 matrix it is
#: converted from. The largest fit any config in ``configs/`` asks for is
#: kuairec_validate_ope's 100,000 evaluation rounds over 3,327 actions and a
#: 64-dimensional state: 3.4e8 cells, three times under the budget. KuaiRec's
#: FULL block asks for 1.3e10 cells (94.6 GiB as float64), which no
#: deduplication can shrink, and is refused.
REWARD_MODEL_FIT_CELL_BUDGET = 1_000_000_000

#: Row-block size, in cells, for multiplying a dense action distribution by
#: ``q``. Each round's expectation is a sum along its own row, so blocking rows
#: changes nothing about the arithmetic -- only the peak memory, which becomes a
#: function of this constant instead of of the number of evaluation rounds.
#: Matches :attr:`gcrl.data.kuairec.ExactRewardLookup.EXPECTED_CELL_BLOCK`, which
#: blocks the same product from the other side.
_EXPECTED_Q_CELL_BLOCK = 1_000_000


def describe_dense_grid(n_rounds: int, n_actions: int, unit: str = "rounds") -> str:
    """``"4,676,570 rounds x 3,327 actions = ... cells, 115.9 GiB as float64"``."""
    cells = int(n_rounds) * int(n_actions)
    size = cells * 8
    # MiB below a gibibyte, so a deduplicated grid does not report itself as
    # "0.0 GiB" in the log line that exists to show how much it saved.
    cost = f"{size / 1024**3:,.1f} GiB" if size >= 1024**3 else f"{size / 1024**2:,.1f} MiB"
    return (
        f"{int(n_rounds):,} {unit} x {int(n_actions):,} actions = {cells:,} cells, "
        f"{cost} as float64"
    )


#: The evaluation policy never selects a logged action, so every importance
#: weight is zero. Estimators that need overlap raise; those that do not (the
#: direct method and its doubly-robust wrappers) still return a number, and this
#: flag records that no logged round supports it.
WARNING_NO_OVERLAP = "no_overlap"

#: The DR correction term is identically zero, so the "doubly robust" value is
#: exactly the direct-method value and carries none of the double-robustness
#: guarantee that its name implies.
WARNING_DR_REDUCES_TO_DM = "dr_reduces_to_dm"

#: The bootstrap could not produce an interval; ``lower``/``upper`` are NaN.
WARNING_CI_UNAVAILABLE = "ci_unavailable"


@dataclass(frozen=True)
class OPEResult:
    """A point estimate with a bootstrap confidence interval.

    ``status`` distinguishes three outcomes that must never be confused:

    * ``"ok"``               -- a value was computed.
    * ``"not_identifiable"`` -- the quantity is not estimable from this data
      (for example, the evaluation policy never selects a logged action, so
      every importance weight is zero). ``value`` is ``NaN``. This is a
      *finding about the estimator*, not a failure of the run, and reporting it
      as ``0.0`` would make it indistinguishable from a real measurement of zero.

    A genuine bug still raises; it never lands here.

    ``warnings`` is a separate channel for a value that *was* computed but must
    not be read at face value: the confidence interval could not be built, or
    the estimator silently degenerated into a different one (see the
    ``WARNING_*`` constants). These stay out of ``status`` deliberately --
    downstream code treats any non-``"ok"`` status as "no number to report", and
    suppressing a real point estimate is the opposite of the intent. A warning
    keeps the number visible and says what has to be said about it.
    """

    estimator: str
    value: float
    lower: float
    upper: float
    n_samples: int
    effective_sample_size: float | None = None
    status: str = "ok"
    detail: str | None = None
    warnings: tuple[str, ...] = ()
    #: How the interval was built. ``n_samples`` counts ROUNDS, which is not the
    #: number of independent observations when rounds replicate a context, so
    #: the three fields below record what the bootstrap actually resampled:
    #: the unit (``"round"`` or ``"context"``), how many of them there were, and
    #: the method. An interval is not interpretable without them -- resampling
    #: 4,676,570 replicated rounds instead of 1,411 contexts narrows it by
    #: ``sqrt(3,314) ~ 58x`` -- so they travel in every result row.
    resampling_unit: str = "round"
    n_independent_units: int | None = None
    bootstrap_method: str = "percentile"
    n_bootstrap: int | None = None
    confidence_level: float | None = None

    @property
    def is_identifiable(self) -> bool:
        return self.status == "ok"

    @property
    def has_confidence_interval(self) -> bool:
        """False when the interval was dropped; ``lower``/``upper`` are then NaN."""
        return WARNING_CI_UNAVAILABLE not in self.warnings

    def as_row(self) -> dict[str, object]:
        return {
            "estimator": self.estimator,
            "value": self.value,
            "ci_lower": self.lower,
            "ci_upper": self.upper,
            "n_samples": self.n_samples,
            "effective_sample_size": self.effective_sample_size,
            "status": self.status,
            "detail": self.detail,
            # Joined rather than a list so the row survives a CSV round-trip as
            # readably as it does a JSON one. Empty string means no caveat.
            "warnings": ";".join(self.warnings),
            "resampling_unit": self.resampling_unit,
            "n_independent_units": self.n_independent_units,
            "bootstrap_method": self.bootstrap_method,
            "n_bootstrap": self.n_bootstrap,
            "confidence_level": self.confidence_level,
        }


class DeterministicPolicy:
    """A deterministic policy's action distribution, stored without materialising it.

    Every estimator here needs only two things from an evaluation policy:

    * ``pi_e(a_i | x_i)`` for the logged action -- a length-``n`` vector;
    * ``E_{a ~ pi_e}[q(x_i, a)]`` for the direct-method baseline.

    For a deterministic policy both are one-hot lookups, so the dense
    ``(n_rounds, n_actions)`` matrix is never needed. That matters at scale: with
    KuaiRec's 10,728 actions and a 100k-round test split, the dense form is
    8.6 GB of almost entirely zeros.

    Instances support ``[index]`` so the bootstrap can resample rounds.
    """

    __slots__ = ("actions", "n_actions")

    def __init__(self, actions: np.ndarray, n_actions: int):
        self.actions = np.asarray(actions, dtype=np.int64)
        self.n_actions = int(n_actions)
        if self.actions.size and (
            self.actions.max() >= self.n_actions or self.actions.min() < 0
        ):
            raise OPEError(
                f"policy selected action outside [0, {self.n_actions}); observed range "
                f"[{self.actions.min()}, {self.actions.max()}]"
            )

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, index) -> DeterministicPolicy:
        return DeterministicPolicy(self.actions[index], self.n_actions)

    @property
    def shape(self) -> tuple:
        return (len(self.actions), self.n_actions)

    def chosen_probability(self, logged_actions: np.ndarray) -> np.ndarray:
        """``pi_e(a_i | x_i)``: 1 where the policy agrees with the log, else 0."""
        return (self.actions == np.asarray(logged_actions)).astype(np.float64)

    def expected_q(self, q_estimates: np.ndarray) -> np.ndarray:
        """``E_{a ~ pi_e}[q(x_i, a)]`` = ``q(x_i, pi_e(x_i))``."""
        return q_estimates[np.arange(len(self.actions)), self.actions]

    def to_dense(self) -> np.ndarray:
        """Materialise the one-hot matrix. Only for small action spaces."""
        dense = np.zeros(self.shape, dtype=np.float64)
        dense[np.arange(len(self.actions)), self.actions] = 1.0
        return dense

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Convert to the dense ``(n_rounds, n_actions)`` matrix for NumPy.

        Without this, ``np.asarray`` sees ``__len__`` plus a ``__getitem__``
        that returns another policy and treats the object as a nested sequence,
        yielding a ragged object array instead of the distribution. Note this
        materialises the matrix the class exists to avoid, so it is for callers
        that genuinely want the dense form; the estimators dispatch on
        :data:`SPARSE_POLICIES` and never reach it.
        """
        return _as_dense_array(self, dtype, copy)


class UniformPolicy:
    """A uniform-random policy, stored as a scalar rather than a dense matrix."""

    __slots__ = ("n_rounds", "n_actions")

    def __init__(self, n_rounds: int, n_actions: int):
        self.n_rounds = int(n_rounds)
        self.n_actions = int(n_actions)

    def __len__(self) -> int:
        return self.n_rounds

    def __getitem__(self, index) -> UniformPolicy:
        length = len(np.asarray(index)) if np.ndim(index) else 1
        return UniformPolicy(length, self.n_actions)

    @property
    def shape(self) -> tuple:
        return (self.n_rounds, self.n_actions)

    def chosen_probability(self, logged_actions: np.ndarray) -> np.ndarray:
        return np.full(len(logged_actions), 1.0 / self.n_actions)

    def expected_q(self, q_estimates: np.ndarray) -> np.ndarray:
        return q_estimates.mean(axis=1)

    def to_dense(self) -> np.ndarray:
        return np.full(self.shape, 1.0 / self.n_actions)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Convert to the dense ``(n_rounds, n_actions)`` matrix for NumPy.

        Without this, ``np.asarray`` treats the object as a sequence whose
        elements are themselves sequences -- ``__getitem__`` returns another
        ``UniformPolicy`` -- and recurses until the process is killed.
        """
        return _as_dense_array(self, dtype, copy)


def _as_dense_array(policy, dtype, copy) -> np.ndarray:
    """Shared ``__array__`` body for the sparse policy representations."""
    if copy is False:  # NumPy 2 asks for a no-copy view; there is nothing to view.
        raise ValueError(
            f"{type(policy).__name__} stores no dense array, so it cannot be "
            "converted with copy=False"
        )
    dense = policy.to_dense()
    return dense if dtype is None else dense.astype(dtype, copy=False)


#: Policy representations that avoid materialising a dense (n_rounds, n_actions) array.
SPARSE_POLICIES = (DeterministicPolicy, UniformPolicy)


def _chosen_probability(action_dist, actions: np.ndarray) -> np.ndarray:
    if isinstance(action_dist, SPARSE_POLICIES):
        return action_dist.chosen_probability(actions)
    return action_dist[np.arange(len(actions)), actions]


def _expected_q(action_dist, q_estimates: np.ndarray) -> np.ndarray:
    if isinstance(action_dist, SPARSE_POLICIES):
        return action_dist.expected_q(q_estimates)
    return _dense_expected_q(action_dist, q_estimates)


def _dense_expected_q(action_dist, q_estimates) -> np.ndarray:
    """``E_{a ~ pi_e}[q(x_i, a)]`` for a DENSE action distribution, in row blocks.

    ``np.sum(action_dist * q_estimates, axis=1)`` is the definition, and it is
    what this computes -- but writing it in one expression materialises the
    whole ``(n_rounds, n_actions)`` product, and, when ``q_estimates`` is an
    :class:`~gcrl.data.kuairec.ExactRewardLookup`, forces the lookup to
    materialise the grid it exists to avoid: 58.0 GiB of rewards plus 14.5 GiB
    of mask on the full KuaiRec split.

    Round ``i``'s expectation depends only on row ``i`` of each operand, so the
    rows are processed in blocks of about :data:`_EXPECTED_Q_CELL_BLOCK` cells.
    Every row is summed over exactly the same elements in exactly the same
    order as the single-expression form, so the result is bit-identical; only
    the peak memory changes, from O(n_rounds x n_actions) to O(block).
    """
    shape = getattr(action_dist, "shape", ())
    if len(shape) != 2:  # not a (rounds, actions) matrix; nothing to block over
        return np.sum(action_dist * q_estimates, axis=1)

    # A ``q`` that knows how to answer this without being materialised answers
    # it itself -- ExactRewardLookup walks its own rounds in blocks and never
    # builds the (n_rounds, n_actions) grid.
    expected_under = getattr(q_estimates, "expected_under", None)
    if expected_under is not None:
        return expected_under(action_dist)

    n_rounds, n_actions = int(shape[0]), int(shape[1])
    rows_per_block = max(1, _EXPECTED_Q_CELL_BLOCK // max(1, n_actions))
    if n_rounds <= rows_per_block:
        return np.sum(action_dist * q_estimates, axis=1)

    blocks = [
        np.sum(action_dist[start : start + rows_per_block]
               * q_estimates[start : start + rows_per_block], axis=1)
        for start in range(0, n_rounds, rows_per_block)
    ]
    # np.ma.concatenate keeps the per-round mask that records which rounds had
    # no observed cell at all; plain concatenate keeps the unmasked dtype.
    if any(np.ma.isMaskedArray(block) for block in blocks):
        return np.ma.concatenate(blocks)
    return np.concatenate(blocks)


def assert_dense_distribution_affordable(
    agent_name: str,
    n_rounds: int,
    n_actions: int,
    budget: int = DENSE_ACTION_DIST_CELL_BUDGET,
) -> None:
    """Refuse a dense ``(n_rounds, n_actions)`` action distribution above ``budget``.

    Called at the top of Phase 4, before any estimator runs and before the
    reward model is fitted, so an agent whose distribution cannot be held is
    reported in the first second rather than as a NumPy allocation failure
    after hours of work.

    Raises:
        DenseDistributionError: naming the agent, the grid and what it costs.
    """
    cells = int(n_rounds) * int(n_actions)
    if cells <= int(budget):
        return
    raise DenseDistributionError(
        f"{agent_name} needs a DENSE action distribution over "
        f"{describe_dense_grid(n_rounds, n_actions)}, which is above the "
        f"{int(budget):,}-cell budget ({int(budget) * 8 / 1024**3:.1f} GiB). Every estimator "
        f"that consumes it -- and the ground-truth lookup it is multiplied by -- needs a "
        f"grid of the same shape again. Return a sparse action distribution "
        f"(DeterministicPolicy or UniformPolicy, which are exact, not approximations) for "
        f"this many rounds, or evaluate fewer rounds. Refused before any estimator ran."
    )


def assert_dense_q_model_affordable(
    n_rounds: int,
    n_actions: int,
    budget: int = DENSE_Q_MODEL_CELL_BUDGET,
    n_contexts: int | None = None,
) -> None:
    """Refuse a reward model whose ``q(x, a)`` grid cannot be held.

    Unlike an action distribution, ``q`` has no sparse form: DM, DR and MRDR
    index it per round and per action, so a grid above the budget is not a
    representation problem but a request that cannot be served. Refusing it at
    the top of Phase 4 turns a 115.9 GiB NumPy allocation failure -- hours in,
    after the policies have been trained -- into a message in the first second.

    It does, however, have a REDUNDANT form. ``q(x_i, .)`` depends on round
    ``i`` only through its context, so the grid that is actually built holds one
    row per distinct context and gathers it per round; ``n_contexts`` is the
    number of rows that costs. On KuaiRec's enumerated evaluation block that is
    1,411 rows rather than 4,676,570 -- 37.6 MB rather than 115.9 GiB -- and it
    is the deduplicated size that this budget is spent on. Every duplicate row
    would be bit-identical, so nothing is approximated by not holding it.

    Raises:
        DenseDistributionError: naming the grid and what it costs.
    """
    rows = int(n_rounds) if n_contexts is None else int(n_contexts)
    cells = rows * int(n_actions)
    if cells <= int(budget):
        return
    deduplicated = rows < int(n_rounds)
    grid = describe_dense_grid(
        rows, n_actions, unit="distinct contexts" if deduplicated else "rounds"
    )
    preamble = (
        f"the DM/DR/MRDR reward model needs q(x, a) for every DISTINCT CONTEXT and every "
        f"action. The {int(n_rounds):,} evaluation rounds carry {rows:,} distinct contexts, "
        f"so the deduplicated grid is {grid}"
        if deduplicated else
        f"the DM/DR/MRDR reward model needs q(x, a) for every round and every action: a "
        f"dense {grid}"
    )
    raise DenseDistributionError(
        f"{preamble}, above the {int(budget):,}-cell "
        f"budget ({int(budget) * 8 / 1024**3:.1f} GiB). Unlike an action distribution this "
        f"grid has no sparse form -- the estimators index it per round and per action. "
        f"Either drop 'dm', 'dr' and 'mrdr' from ope.estimators ('exact' needs no reward "
        f"model and carries no estimator error at all), or evaluate fewer rounds with "
        f"dataset.eval_subsample_rows. Refused before the model was fitted."
    )


def assert_reward_model_fit_affordable(
    n_fit_rounds: int,
    n_context_features: int,
    n_actions: int,
    budget: int = REWARD_MODEL_FIT_CELL_BUDGET,
) -> None:
    """Refuse a reward model whose TRAINING matrix cannot be held.

    The companion to :func:`assert_dense_q_model_affordable`, and the one that
    survives deduplication. ``q``'s prediction grid collapses to one row per
    distinct context because those rows are identical; the training matrix does
    not, because rounds sharing a context still carry different actions and
    different rewards, so each is its own example. ``_fit_q_model`` builds
    ``[context, one-hot(action)]`` -- ``n_fit_rounds x (context_dim +
    n_actions)`` -- as ``float32``, and scikit-learn's histogram booster then
    converts it to ``float64`` before binning it.

    Checking it here keeps the guarantee the other two guards give: a shape that
    cannot be held stops the phase in its first second, with the arithmetic
    spelled out, rather than as a ``MemoryError`` after the policies have been
    trained.

    Raises:
        DenseDistributionError: naming the matrix and what it costs.
    """
    rows, width = int(n_fit_rounds), int(n_context_features) + int(n_actions)
    cells = rows * width
    if cells <= int(budget):
        return
    raise DenseDistributionError(
        f"the DM/DR/MRDR reward model must be FITTED on {rows:,} rounds x {width:,} features "
        f"({int(n_context_features):,} context dimensions + {int(n_actions):,} one-hot action "
        f"columns) = {cells:,} cells, {cells * 8 / 1024**3:,.1f} GiB once scikit-learn has "
        f"converted the matrix to float64 (plus {cells * 4 / 1024**3:,.1f} GiB for the float32 "
        f"matrix it converts), above the {int(budget):,}-cell budget "
        f"({int(budget) * 8 / 1024**3:.1f} GiB). Deduplicating q(x, a) by context does not "
        f"shrink this: rounds that share a context still carry different actions and rewards, "
        f"so each one is its own training example. Either drop 'dm', 'dr' and 'mrdr' from "
        f"ope.estimators ('exact' needs no reward model and carries no estimator error at "
        f"all), or fit on fewer rounds with dataset.eval_subsample_rows. Refused before the "
        f"model was fitted."
    )


def _validate(
    actions: np.ndarray,
    rewards: np.ndarray,
    propensities: np.ndarray,
    action_dist: np.ndarray,
) -> None:
    n = len(actions)
    if not (len(rewards) == len(propensities) == len(action_dist) == n):
        raise OPEError(
            f"length mismatch: actions={len(actions)}, rewards={len(rewards)}, "
            f"propensities={len(propensities)}, action_dist={len(action_dist)}"
        )
    if n == 0:
        raise OPEError("cannot evaluate on zero logged rounds")
    if np.any(propensities <= 0):
        raise OPEError(
            "propensities must be strictly positive; "
            f"{int((propensities <= 0).sum())} rounds have propensity <= 0. "
            "A zero propensity means the behaviour policy could not have taken "
            "the logged action, which violates the overlap assumption."
        )
    if np.any(propensities > 1):
        raise OPEError("propensities must be <= 1")
    if isinstance(action_dist, SPARSE_POLICIES):
        if len(action_dist) != n:
            raise OPEError(
                f"action_dist covers {len(action_dist)} rounds but actions has {n}"
            )
    else:
        if action_dist.ndim != 2:
            raise OPEError(
                f"action_dist must be 2-D (n_rounds, n_actions), got shape {action_dist.shape}"
            )
        if not np.allclose(action_dist.sum(axis=1), 1.0, atol=1e-4):
            raise OPEError("each row of action_dist must sum to 1")
    if actions.max() >= action_dist.shape[1]:
        raise OPEError(
            f"logged action {int(actions.max())} is outside the action space "
            f"({action_dist.shape[1]} actions)"
        )


def _importance_weights(
    actions: np.ndarray, propensities: np.ndarray, action_dist: np.ndarray
) -> np.ndarray:
    """``pi_e(a_i | x_i) / pi_b(a_i | x_i)`` for each logged round."""
    chosen = _chosen_probability(action_dist, actions)
    return chosen / np.maximum(propensities, _EPSILON)


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size, ``(sum w)^2 / sum w^2``.

    A value far below ``n`` means a handful of rounds dominate the estimate, so
    the point estimate is unstable no matter what its confidence interval says.
    """
    denominator = np.sum(weights**2)
    if denominator <= _EPSILON:
        return 0.0
    return float(np.sum(weights) ** 2 / denominator)


def ipw(
    actions: np.ndarray, rewards: np.ndarray, propensities: np.ndarray, action_dist: np.ndarray
) -> float:
    """Inverse propensity weighting: ``(1/n) sum w_i r_i``. Unbiased, high variance.

    As with :func:`snipw`, all-zero weights mean the evaluation policy never
    selects a logged action. The arithmetic would happily return ``0.0``, but
    that zero is the absence of evidence, not a measured value of zero, so it
    raises instead.
    """
    _validate(actions, rewards, propensities, action_dist)
    weights = _importance_weights(actions, propensities, action_dist)
    if np.sum(weights) <= _EPSILON:
        raise OPEError(
            "sum of importance weights is zero: the evaluation policy never selects "
            "any logged action, so its value is not identifiable from this data"
        )
    return float(np.mean(weights * rewards))


def snipw(
    actions: np.ndarray, rewards: np.ndarray, propensities: np.ndarray, action_dist: np.ndarray
) -> float:
    """Self-normalised IPW: ``sum w_i r_i / sum w_i``.

    Self-normalisation trades a small bias for a large variance reduction and
    bounds the estimate within the observed reward range. Note the weights must
    genuinely vary: with a constant propensity the normalisation cancels and
    this degenerates to the mean reward over rounds the policies agree on,
    which is *not* an off-policy estimate.
    """
    _validate(actions, rewards, propensities, action_dist)
    weights = _importance_weights(actions, propensities, action_dist)
    weight_sum = np.sum(weights)
    if weight_sum <= _EPSILON:
        raise OPEError(
            "sum of importance weights is zero: the evaluation policy never selects "
            "any logged action, so its value is not identifiable from this data"
        )
    return float(np.sum(weights * rewards) / weight_sum)


def direct_method(action_dist, q_estimates: np.ndarray) -> float:
    """``(1/n) sum_i sum_a pi_e(a|x_i) q(x_i,a)``. Low variance, biased by model error."""
    if action_dist.shape != q_estimates.shape:
        raise OPEError(
            f"action_dist {action_dist.shape} and q_estimates {q_estimates.shape} must match"
        )
    return float(np.mean(_expected_q(action_dist, q_estimates)))


def doubly_robust(
    actions: np.ndarray,
    rewards: np.ndarray,
    propensities: np.ndarray,
    action_dist,
    q_estimates: np.ndarray,
) -> float:
    """Doubly robust estimator.

    ``DM + (1/n) sum_i w_i (r_i - q(x_i, a_i))``

    Consistent if *either* the reward model or the propensities are correct --
    but only while the correction term carries information. If the evaluation
    policy never selects a logged action every ``w_i`` is zero, the correction
    vanishes identically and this returns the direct-method value, which is
    consistent only if the reward model is right. :func:`evaluate_policy`
    detects that case and flags the result with
    :data:`WARNING_DR_REDUCES_TO_DM`; a caller using this function directly must
    check the weights itself.
    """
    _validate(actions, rewards, propensities, action_dist)
    if q_estimates.shape != action_dist.shape:
        raise OPEError(
            f"q_estimates {q_estimates.shape} must match action_dist {action_dist.shape}"
        )
    weights = _importance_weights(actions, propensities, action_dist)
    q_logged = q_estimates[np.arange(len(actions)), actions]
    baseline = direct_method(action_dist, q_estimates)
    correction = np.mean(weights * (rewards - q_logged))
    return float(baseline + correction)


def mrdr(
    actions: np.ndarray,
    rewards: np.ndarray,
    propensities: np.ndarray,
    action_dist: np.ndarray,
    q_estimates: np.ndarray,
) -> float:
    """More Robust Doubly Robust (Farajtabar et al., 2018).

    MRDR keeps the DR functional form but requires the reward model to have been
    fitted under the MRDR loss, which minimises the *variance* of the resulting
    DR estimator rather than the reward prediction error. Fitting that model is
    the job of :func:`fit_mrdr_reward_model`; passing a plain regression model
    here yields ordinary DR, so the two must be used together.
    """
    return doubly_robust(actions, rewards, propensities, action_dist, q_estimates)


def mrdr_weights(
    actions: np.ndarray, propensities: np.ndarray, action_dist: np.ndarray
) -> np.ndarray:
    """Per-round sample weights for fitting an MRDR reward model.

    The MRDR objective weights each logged round by
    ``pi_e(a_i|x_i) * (1 - pi_b(a_i|x_i)) / pi_b(a_i|x_i)^2``, which is the term
    that drives the variance of the DR estimator.
    """
    _validate(actions, np.zeros(len(actions)), propensities, action_dist)
    chosen_eval = _chosen_probability(action_dist, actions)
    return chosen_eval * (1.0 - propensities) / np.maximum(propensities**2, _EPSILON)


def exact_value(
    action_dist, true_rewards: np.ndarray
) -> float:
    """Exact policy value on a fully observed reward matrix.

    When every counterfactual reward is known -- as in KuaiRec's fully observed
    matrix -- the policy value is computed directly with no estimator error.
    ``true_rewards[i, a]`` is the reward for taking action ``a`` in round ``i``.
    """
    if action_dist.shape != true_rewards.shape:
        raise OPEError(
            f"action_dist {action_dist.shape} and true_rewards {true_rewards.shape} must match"
        )
    return float(np.mean(_expected_q(action_dist, true_rewards)))


def first_appearance_index(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Group a 1-D array of identifiers, numbering the groups by first appearance.

    Returns ``(group_of_element, first_element_of_group)``: the group each
    element belongs to, and one representative index per group.

    The ordering is the whole point. ``np.unique`` numbers groups by SORTED key,
    which for data with no repeats permutes the elements; numbering by first
    appearance instead makes the k-th group the k-th element whenever every key
    is distinct. That is what lets everything built on top of this -- the
    cross-fitting folds and the bootstrap -- reduce EXACTLY, index for index, to
    the per-element version it replaces when there is nothing to group.
    """
    values = np.asarray(values)
    if values.ndim != 1:
        raise OPEError(f"expected a 1-D array of identifiers, got shape {values.shape}")
    if values.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    _, first, inverse = np.unique(values, return_index=True, return_inverse=True)
    inverse = np.reshape(inverse, -1)
    order = np.argsort(first, kind="stable")          # unique keys, by first appearance
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order), dtype=np.int64)
    return rank[inverse], first[order].astype(np.int64)


class BootstrapUnits:
    """The INDEPENDENT units a bootstrap resamples, and the rounds each carries.

    A percentile bootstrap estimates the sampling distribution of an estimator
    by resampling the unit that was drawn independently. Resampling a unit that
    was *not* drawn independently understates the variance by roughly the square
    root of the replication factor: on KuaiRec's enumerated evaluation block all
    ~3,314 rounds of a user share that user's context, so resampling rounds
    treats 4,676,570 independent draws where there are 1,411, and reports an
    interval about ``sqrt(3,314) ~ 58x`` too narrow.

    This class carries the block structure: a draw picks ``n_units`` units with
    replacement and takes ALL of each chosen unit's rounds, which is the
    standard cluster (block) bootstrap.

    When every unit holds exactly one round the class is inert:
    :meth:`draw` then performs the identical ``rng.integers(0, n, size=n)`` call
    on the identical generator state and returns its result unchanged, so an
    interval computed through it is byte-for-byte the one computed without it.
    """

    __slots__ = ("n_units", "n_rounds", "unit_name", "_order", "_starts", "_counts")

    def __init__(
        self,
        n_units: int,
        n_rounds: int,
        order: np.ndarray | None,
        starts: np.ndarray | None,
        counts: np.ndarray | None,
        unit_name: str = "round",
    ) -> None:
        self.n_units = int(n_units)
        self.n_rounds = int(n_rounds)
        self.unit_name = unit_name
        self._order = order
        self._starts = starts
        self._counts = counts

    @classmethod
    def per_round(cls, n_rounds: int) -> BootstrapUnits:
        """Every round is its own unit: the classical i.i.d. bootstrap."""
        return cls(n_rounds, n_rounds, None, None, None, "round")

    @classmethod
    def from_ids(cls, unit_ids: np.ndarray, unit_name: str = "context") -> BootstrapUnits:
        """Units named by a per-round identifier -- a user, a persona, a context."""
        ids, _ = first_appearance_index(np.asarray(unit_ids))
        n_rounds = len(ids)
        n_units = int(ids.max()) + 1 if n_rounds else 0
        if n_units == n_rounds:
            # Nothing is replicated. Numbering by first appearance makes unit k
            # exactly round k, so the general path below would reproduce the
            # per-round bootstrap; taking it literally keeps that exact.
            return cls.per_round(n_rounds)
        counts = np.bincount(ids, minlength=n_units).astype(np.int64)
        order = np.argsort(ids, kind="stable").astype(np.int64)
        starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int64)
        return cls(n_units, n_rounds, order, starts, counts, unit_name)

    @classmethod
    def of(cls, units, n_rounds: int) -> BootstrapUnits:
        """Normalise ``None`` / an id array / an instance into an instance."""
        if units is None:
            return cls.per_round(n_rounds)
        if isinstance(units, BootstrapUnits):
            if units.n_rounds != int(n_rounds):
                raise OPEError(
                    f"bootstrap units cover {units.n_rounds} rounds but the estimator has "
                    f"{int(n_rounds)}"
                )
            return units
        return cls.from_ids(units)

    @property
    def is_per_round(self) -> bool:
        """True when the unit is the round itself, i.e. nothing is replicated."""
        return self._order is None

    @property
    def rounds_per_unit(self) -> float:
        return self.n_rounds / self.n_units if self.n_units else float("nan")

    @property
    def method(self) -> str:
        return "percentile" if self.is_per_round else "cluster_percentile"

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        """One bootstrap replicate: ``n_units`` units with replacement, all their rounds."""
        chosen = rng.integers(0, self.n_units, size=self.n_units)
        if self._order is None:
            return chosen
        counts = self._counts[chosen]
        total = int(counts.sum())
        # Expand each chosen unit into its rounds without a Python-level loop:
        # position j of the output reads order[start_of_its_unit + offset_in_unit].
        out_starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        within = np.arange(total, dtype=np.int64) - np.repeat(out_starts, counts)
        return self._order[np.repeat(self._starts[chosen], counts) + within]


def bootstrap_interval(
    estimator_fn,
    n_samples: int,
    n_bootstrap: int = 100,
    confidence_level: float = 0.95,
    seed: int = 42,
    units: BootstrapUnits | np.ndarray | None = None,
) -> tuple:
    """Percentile bootstrap interval for any estimator.

    ``estimator_fn`` receives an index array and returns a scalar. Resampling
    indices rather than recomputing the policy keeps the bootstrap over the
    logged data.

    ``units`` names the INDEPENDENT unit. Without it every round is treated as
    an independent draw, which is only true when no two rounds share a context;
    with it, a replicate draws units with replacement and carries all of each
    unit's rounds (a cluster bootstrap). See :class:`BootstrapUnits` for why the
    difference is a factor of ``sqrt(rounds per unit)`` in the interval's width,
    and note that passing units with one round each is a no-op down to the bit.
    """
    if n_bootstrap < 2:
        raise OPEError("n_bootstrap must be >= 2")
    units = BootstrapUnits.of(units, n_samples)
    rng = np.random.default_rng(seed)
    values = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        indices = units.draw(rng)
        values[b] = estimator_fn(indices)
    alpha = (1.0 - confidence_level) / 2.0
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


def evaluate_policy(
    actions: np.ndarray,
    rewards: np.ndarray,
    propensities: np.ndarray,
    action_dist: np.ndarray,
    q_estimates: np.ndarray | None = None,
    true_rewards: np.ndarray | None = None,
    estimators: list | None = None,
    n_bootstrap: int = 100,
    confidence_level: float = 0.95,
    seed: int = 42,
    bootstrap_units: BootstrapUnits | np.ndarray | None = None,
) -> dict[str, OPEResult]:
    """Run the requested estimators and return point estimates with intervals.

    ``bootstrap_units`` names the independent unit the confidence intervals are
    resampled over -- a :class:`BootstrapUnits`, or one identifier per round.
    Omitting it resamples rounds, which is correct only when no two rounds share
    a context. Whatever is used is recorded on every :class:`OPEResult`, because
    the width of an interval is not interpretable without it.

    Raises:
        OPEError: if a requested estimator's inputs are unavailable. Estimators
            never silently fall back to a different quantity or return a
            placeholder value.
    """
    estimators = estimators or ["snipw", "dr", "mrdr"]
    actions = np.asarray(actions, dtype=np.int64)
    rewards = np.asarray(rewards, dtype=np.float64)
    propensities = np.asarray(propensities, dtype=np.float64)
    if not isinstance(action_dist, SPARSE_POLICIES):
        action_dist = np.asarray(action_dist, dtype=np.float64)

    _validate(actions, rewards, propensities, action_dist)
    weights = _importance_weights(actions, propensities, action_dist)
    ess = effective_sample_size(weights)
    n = len(actions)

    if ess < 0.01 * n:
        logger.warning(
            "Effective sample size is %.1f out of %d rounds (%.2f%%). Importance-weighted "
            "estimates are dominated by very few rounds and should be reported with this "
            "caveat.", ess, n, 100.0 * ess / n,
        )

    units = BootstrapUnits.of(bootstrap_units, n)
    if not units.is_per_round:
        logger.info(
            "Bootstrap resamples %d independent %s units (%.1f rounds each), carrying every "
            "round of each drawn unit. Resampling the %d rounds instead would report an "
            "interval about %.0fx too narrow.",
            units.n_units, units.unit_name, units.rounds_per_unit, n,
            units.rounds_per_unit ** 0.5,
        )

    no_overlap = float(np.sum(weights)) <= _EPSILON
    if no_overlap:
        logger.warning(
            "Every importance weight is zero: the evaluation policy selects no logged action. "
            "IPW and SNIPW are not identifiable here. DM still returns a value, but it is the "
            "reward model extrapolating with no support from the logged rounds, and DR/MRDR "
            "reduce to exactly that value with no double-robustness guarantee. These are "
            "flagged in the result records."
        )

    if np.allclose(propensities, propensities[0]):
        logger.warning(
            "All propensities are identical (%.6g). Self-normalised estimators degenerate "
            "to the mean reward over agreed-upon actions under a constant propensity; this "
            "is not an off-policy estimate. Verify that logged propensities were loaded.",
            float(propensities[0]),
        )

    results: dict[str, OPEResult] = {}
    #: Written onto every result, identifiable or not, so no interval can be
    #: read without the unit it was resampled over.
    provenance = {
        "resampling_unit": units.unit_name,
        "n_independent_units": units.n_units,
        "bootstrap_method": units.method,
        "n_bootstrap": int(n_bootstrap),
        "confidence_level": float(confidence_level),
    }

    def _register(
        name: str,
        compute,
        fn,
        warnings: tuple[str, ...] = (),
        detail: str | None = None,
    ) -> None:
        """Compute an estimate, recording non-identifiability rather than raising.

        Only the specific, meaningful condition is caught: an estimator that is
        mathematically undefined on this data. Every other exception propagates,
        because it indicates a bug and must not become a table entry.

        ``warnings``/``detail`` carry caveats about a value that *was* computed.
        A bootstrap that cannot produce an interval adds one here rather than
        leaving ``status="ok"`` beside a pair of NaNs: the interval is often the
        only signal that a point estimate rests on a handful of rounds, so its
        disappearance has to be recorded, not absorbed.
        """
        try:
            point = compute()
        except OPEError as exc:
            if "not identifiable" not in str(exc):
                raise
            logger.warning(
                "%s is not identifiable on this data: %s. Reported as not_identifiable "
                "rather than as a numeric value.", name.upper(), exc,
            )
            results[name] = OPEResult(
                name, float("nan"), float("nan"), float("nan"), n, ess,
                status="not_identifiable", detail=str(exc), **provenance,
            )
            return

        try:
            lower, upper = bootstrap_interval(
                fn, n, n_bootstrap, confidence_level, seed, units=units
            )
        except OPEError as exc:
            logger.warning(
                "%s: the bootstrap could not produce a confidence interval (%s). A resampled "
                "replicate was not identifiable, which usually means the estimate rests on very "
                "few rounds (ESS %.1f of %d). The point estimate is kept but flagged %r; it must "
                "not be presented as though an interval had been computed.",
                name.upper(), exc, ess, n, WARNING_CI_UNAVAILABLE,
            )
            lower = upper = float("nan")
            warnings = (*warnings, WARNING_CI_UNAVAILABLE)
            detail = "; ".join(
                part for part in (detail, f"confidence interval unavailable: {exc}") if part
            )
        results[name] = OPEResult(
            name, point, lower, upper, n, ess, detail=detail or None, warnings=warnings,
            **provenance,
        )

    for name in estimators:
        if name == "ipw":
            _register("ipw", lambda: ipw(actions, rewards, propensities, action_dist),
                      lambda idx: ipw(actions[idx], rewards[idx], propensities[idx], action_dist[idx]))
        elif name == "snipw":
            _register("snipw", lambda: snipw(actions, rewards, propensities, action_dist),
                      lambda idx: snipw(actions[idx], rewards[idx], propensities[idx], action_dist[idx]))
        elif name == "dm":
            if q_estimates is None:
                raise OPEError("estimator 'dm' requires q_estimates from a fitted reward model")
            _register(
                "dm",
                lambda: direct_method(action_dist, q_estimates),
                lambda idx: direct_method(action_dist[idx], q_estimates[idx]),
                warnings=(WARNING_NO_OVERLAP,) if no_overlap else (),
                detail=(
                    "the evaluation policy selects no logged action, so this value is the "
                    "reward model extrapolating to actions no logged round observed"
                ) if no_overlap else None,
            )
        elif name in {"dr", "mrdr"}:
            if q_estimates is None:
                raise OPEError(
                    f"estimator {name!r} requires q_estimates from a fitted reward model; "
                    "fit one in Phase 4 rather than omitting the estimator silently"
                )
            fn = doubly_robust if name == "dr" else mrdr
            _register(
                name,
                lambda _fn=fn: _fn(actions, rewards, propensities, action_dist, q_estimates),
                lambda idx, _fn=fn: _fn(actions[idx], rewards[idx], propensities[idx],
                                        action_dist[idx], q_estimates[idx]),
                warnings=(WARNING_NO_OVERLAP, WARNING_DR_REDUCES_TO_DM) if no_overlap else (),
                detail=(
                    "every importance weight is zero, so the correction term is identically "
                    "zero and this equals the direct-method estimate; it is not doubly robust "
                    "on this data"
                ) if no_overlap else None,
            )
        elif name == "exact":
            if true_rewards is None:
                raise OPEError(
                    "estimator 'exact' requires a fully observed reward matrix; it is only "
                    "valid for datasets such as KuaiRec's small_matrix"
                )
            _register("exact", lambda: exact_value(action_dist, true_rewards),
                      lambda idx: exact_value(action_dist[idx], true_rewards[idx]))
        else:
            raise OPEError(f"unknown estimator {name!r}")

    return results
