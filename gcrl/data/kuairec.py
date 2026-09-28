"""KuaiRec loading.

KuaiRec ships two interaction matrices with different roles:

* ``big_matrix.csv``  -- 7,176 users x 10,728 items, sparse observational log.
  This is the *training* log.
* ``small_matrix.csv`` -- 1,411 users x 3,327 items, **fully observed**. Every
  user-item pair in this block has a recorded outcome.

The fully observed block is the reason to use this dataset: a policy's value on
it can be computed by lookup instead of estimated, so there is no estimator
error at all. The correct protocol is therefore train on ``big_matrix`` and
evaluate exactly on ``small_matrix``, which is what this module supports.

The previous pipeline mixed the two -- describing ``big_matrix`` statistics in
the paper while parts of the code loaded ``small_matrix``, and evaluating on the
same file it trained on. Each loader here states which matrix it returns.

Reward definition
-----------------
``reward = 1[watch_ratio >= threshold]`` with ``threshold`` defaulting to 2.0,
the convention used in the KuaiRec paper. Because every reported number depends
on it, the threshold is an explicit argument and is returned in the metadata so
it can be quoted rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_WATCH_RATIO_THRESHOLD = 2.0

#: Column lineage, consumed by :func:`gcrl.models.causal.check_treatment_provenance`.
#: ``watch_ratio`` is derived from ``play_duration``, so any treatment defined
#: from ``watch_ratio`` while ``play_duration`` is the outcome is circular.
KUAIREC_PROVENANCE: dict[str, list] = {
    "watch_ratio": ["play_duration", "video_duration"],
    "reward": ["watch_ratio"],
}


@dataclass
class KuaiRecData:
    interactions: pd.DataFrame
    matrix_name: str
    n_users: int
    n_items: int
    reward_threshold: float
    social_edges: np.ndarray | None = None
    user_features: pd.DataFrame | None = None

    def describe(self) -> str:
        return (
            f"KuaiRec[{self.matrix_name}] {len(self.interactions):,} interactions | "
            f"{self.n_users:,} users x {self.n_items:,} items | "
            f"positive rate {self.interactions['reward'].mean():.4f} "
            f"(watch_ratio >= {self.reward_threshold})"
        )


def _resolve_data_dir(root: Path) -> Path:
    """Locate the directory holding the CSVs, tolerating the archive's nesting."""
    root = Path(root)
    for candidate in (root, root / "data", root / "KuaiRec 2.0" / "data"):
        if (candidate / "small_matrix.csv").exists():
            return candidate
    raise FileNotFoundError(
        f"could not locate KuaiRec CSVs under {root}. Expected 'small_matrix.csv' in "
        f"{root}, {root / 'data'}, or {root / 'KuaiRec 2.0' / 'data'}."
    )


def load_kuairec(
    root: Path,
    matrix: str = "big",
    reward_threshold: float = DEFAULT_WATCH_RATIO_THRESHOLD,
    subsample_rows: int | None = None,
    seed: int = 42,
    load_social_graph: bool = True,
    load_user_features: bool = True,
) -> KuaiRecData:
    """Load one KuaiRec matrix plus optional side information.

    Args:
        matrix: ``"big"`` (sparse training log) or ``"small"`` (fully observed
            evaluation block). These are different user and item universes;
            statistics from one must never be reported for the other.

    Raises:
        FileNotFoundError: if the CSVs cannot be located.
        ValueError: for an unknown matrix name or a missing required column.
    """
    if matrix not in {"big", "small"}:
        raise ValueError(f"matrix must be 'big' or 'small', got {matrix!r}")

    data_dir = _resolve_data_dir(root)
    path = data_dir / f"{matrix}_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    logger.info("Loading KuaiRec %s_matrix from %s", matrix, path)
    frame = pd.read_csv(path)

    required = {"user_id", "video_id", "watch_ratio", "play_duration", "video_duration"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns {sorted(missing)}")

    if subsample_rows is not None and subsample_rows < len(frame):
        rng = np.random.default_rng(seed)
        index = np.sort(rng.choice(len(frame), size=subsample_rows, replace=False))
        frame = frame.iloc[index].reset_index(drop=True)
        logger.info("Subsampled %d interactions (seed=%d)", subsample_rows, seed)

    frame["reward"] = (frame["watch_ratio"] >= reward_threshold).astype(np.float64)

    social_edges = load_social_edges(data_dir) if load_social_graph else None
    user_features = None
    if load_user_features and (data_dir / "user_features.csv").exists():
        user_features = pd.read_csv(data_dir / "user_features.csv")

    data = KuaiRecData(
        interactions=frame,
        matrix_name=matrix,
        n_users=int(frame["user_id"].nunique()),
        n_items=int(frame["video_id"].nunique()),
        reward_threshold=reward_threshold,
        social_edges=social_edges,
        user_features=user_features,
    )
    logger.info(data.describe())
    return data


def subsample_interactions(
    frame: pd.DataFrame, n_rows: int | None, seed: int, label: str
) -> pd.DataFrame:
    """Keep ``n_rows`` rows, chosen reproducibly and in their original order."""
    if n_rows is None or n_rows >= len(frame):
        return frame.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    index = np.sort(rng.choice(len(frame), size=n_rows, replace=False))
    logger.info("Subsampled %s to %d of %d rows (seed=%d)", label, n_rows, len(frame), seed)
    return frame.iloc[index].reset_index(drop=True)


@dataclass
class BlockOverlap:
    """How the sparse log and the fully observed block relate to each other.

    Measured, never assumed. The first version of this protocol assumed the two
    files overlap and restricted the training log to the evaluation block's item
    universe; on the published KuaiRec files that removes precisely the 1,411
    users the block exists to evaluate, because the dataset's authors deleted
    the block's (user, item) pairs from the log to prevent leakage. The numbers
    below are computed on whatever files are actually loaded, and are reported
    beside every result.
    """

    eval_users: int
    eval_users_in_log: int
    eval_user_log_rows: int
    eval_items: int
    eval_item_log_rows: int
    eval_item_log_users: int
    #: Rows of the log that are an (evaluation user, evaluation item) pair --
    #: the cells the evaluation will score. Zero on KuaiRec, by construction.
    overlap_rows: int
    overlap_cells: int

    @property
    def is_cold_item(self) -> bool:
        """True when the log contains none of the pairs the evaluation scores."""
        return self.overlap_rows == 0

    @property
    def regime(self) -> str:
        if self.is_cold_item:
            return "cold_item: the log contains none of the (evaluation user, action) pairs"
        return (
            f"warm: {self.overlap_cells} (evaluation user, action) cells appear in the log"
        )

    def describe(self) -> str:
        return (
            f"overlap: {self.eval_users_in_log}/{self.eval_users} evaluation users appear in "
            f"the log ({self.eval_user_log_rows:,} rows, over items outside the action space "
            f"where cold); the {self.eval_items} evaluation actions appear in "
            f"{self.eval_item_log_rows:,} log rows from {self.eval_item_log_users} other users; "
            f"{self.overlap_rows:,} log rows are an (evaluation user, evaluation action) pair. "
            f"Regime -- {self.regime}."
        )

    def as_dict(self) -> dict:
        import dataclasses as _dc

        payload = {f.name: getattr(self, f.name) for f in _dc.fields(self)}
        payload["regime"] = self.regime
        return payload


def measure_block_overlap(
    log: pd.DataFrame, eval_block: pd.DataFrame, action_universe: np.ndarray
) -> BlockOverlap:
    """Compute :class:`BlockOverlap` from the two loaded frames."""
    eval_users = np.sort(eval_block["user_id"].unique())
    in_log = log["user_id"].isin(eval_users)
    in_universe = log["video_id"].isin(action_universe)
    item_rows = log[in_universe]
    overlap = log[in_log & in_universe]
    return BlockOverlap(
        eval_users=int(len(eval_users)),
        eval_users_in_log=int(log.loc[in_log, "user_id"].nunique()),
        eval_user_log_rows=int(in_log.sum()),
        eval_items=int(len(action_universe)),
        eval_item_log_rows=int(len(item_rows)),
        eval_item_log_users=int(item_rows["user_id"].nunique()),
        overlap_rows=int(len(overlap)),
        overlap_cells=int(overlap.drop_duplicates(subset=["user_id", "video_id"]).shape[0]),
    )


@dataclass
class KuaiRecTrainEval:
    """The two-matrix protocol, assembled.

    ``train`` is the WHOLE sparse log. It is deliberately not restricted to the
    evaluation block's item universe: on KuaiRec that restriction deletes every
    evaluation user, because the block's (user, item) pairs were removed from
    the log by the dataset's authors. The log is what the graph representation
    is trained on, and it must cover both populations -- the evaluation users
    (through their own history, over other items) and the evaluation actions
    (through other users' history).

    ``eval_rounds`` are the rounds Phase 4 scores. ``eval_block`` is the WHOLE
    evaluation matrix and is what the ground-truth lookup is built from:
    subsampling the rounds must not reduce what is known about the cells.
    ``action_universe`` is the evaluation block's full item universe, which is
    the action space of the whole experiment. ``overlap`` records how the two
    files relate, which determines what the evaluation actually measures.
    """

    train: KuaiRecData
    eval_rounds: KuaiRecData
    eval_block: KuaiRecData
    action_universe: np.ndarray
    overlap: BlockOverlap

    def describe(self) -> str:
        return (
            f"train {self.train.describe()}\n"
            f"eval rounds {self.eval_rounds.describe()}\n"
            f"ground truth from {len(self.eval_block.interactions):,} rows of "
            f"{self.eval_block.matrix_name}_matrix over {len(self.action_universe):,} actions\n"
            f"{self.overlap.describe()}"
        )


def load_kuairec_train_eval(
    root: Path,
    train_matrix: str = "big",
    eval_matrix: str = "small",
    reward_threshold: float = DEFAULT_WATCH_RATIO_THRESHOLD,
    subsample_rows: int | None = None,
    eval_subsample_rows: int | None = None,
    seed: int = 42,
    load_social_graph: bool = True,
    load_user_features: bool = True,
) -> KuaiRecTrainEval:
    """Train on the whole sparse log, evaluate exactly on the fully observed block.

    The action space is the EVALUATION block's item universe, because those are
    the only actions whose counterfactual reward is known. The training log is
    **not** filtered to it. On the published KuaiRec files that filter would
    remove all 1,411 evaluation users: their 571,061 log rows cover 6,334 items,
    none of which is one of the block's 3,327 -- the authors removed the block's
    pairs from the log to prevent leakage. The whole log covers both populations
    and only their intersection is empty.

    What each part of the log is used for is therefore different, and stated
    rather than implied:

    * the **graph representation** is trained on every row, so the evaluation
      users are embedded from their own history and the evaluation actions from
      other users' history;
    * the **policy** can only be trained on rows whose action lies in the action
      space, because an agent cannot take an action outside its own action set.
      :func:`gcrl.pipeline.prepare_dataset` makes that split and reports it.

    Raises:
        ValueError: if the two matrices are the same, or if the evaluation
            block's actions never appear in the log at all -- there would then
            be nothing from which to learn a policy over them.
    """
    if train_matrix == eval_matrix:
        raise ValueError(
            f"the training log and the evaluation block must be different matrices; both are "
            f"{train_matrix!r}. Evaluating on the file a policy was trained on is the defect "
            f"this protocol exists to remove."
        )

    eval_block = load_kuairec(
        root, matrix=eval_matrix, reward_threshold=reward_threshold, subsample_rows=None,
        seed=seed, load_social_graph=False, load_user_features=False,
    )
    action_universe = np.sort(eval_block.interactions["video_id"].unique())

    train = load_kuairec(
        root, matrix=train_matrix, reward_threshold=reward_threshold, subsample_rows=None,
        seed=seed, load_social_graph=load_social_graph, load_user_features=load_user_features,
    )
    overlap = measure_block_overlap(
        train.interactions, eval_block.interactions, action_universe
    )
    logger.info(overlap.describe())

    if overlap.eval_users_in_log == 0:
        raise ValueError(
            f"none of the {overlap.eval_users} evaluation users appears in the "
            f"{train_matrix}_matrix, so no policy could be given a trained representation "
            f"for them."
        )
    if overlap.eval_item_log_rows == 0:
        raise ValueError(
            f"none of the {overlap.eval_items} evaluation actions appears in the "
            f"{train_matrix}_matrix, so there is nothing from which to learn a policy over "
            f"the action space."
        )
    if overlap.is_cold_item:
        logger.warning(
            "COLD-ITEM REGIME: not one of the %d log rows for the %d evaluation users uses "
            "an action from the evaluation block. Every (user, action) pair Phase 4 scores is "
            "one the log has never seen. This is how KuaiRec is built -- the block's pairs "
            "were removed from the log to prevent leakage -- and it is recorded in every "
            "result row, because it changes what the evaluation measures.",
            overlap.eval_user_log_rows, overlap.eval_users,
        )

    rows = subsample_interactions(
        train.interactions, subsample_rows, seed, f"the {train_matrix} log"
    )
    rounds = subsample_interactions(
        eval_block.interactions, eval_subsample_rows, seed, f"the {eval_matrix} eval rounds"
    )

    bundle = KuaiRecTrainEval(
        train=KuaiRecData(
            interactions=rows, matrix_name=train_matrix,
            n_users=int(rows["user_id"].nunique()), n_items=int(rows["video_id"].nunique()),
            reward_threshold=reward_threshold, social_edges=train.social_edges,
            user_features=train.user_features,
        ),
        eval_rounds=KuaiRecData(
            interactions=rounds, matrix_name=eval_matrix,
            n_users=int(rounds["user_id"].nunique()), n_items=int(rounds["video_id"].nunique()),
            reward_threshold=reward_threshold,
        ),
        eval_block=eval_block,
        action_universe=action_universe,
        overlap=overlap,
    )
    logger.info(bundle.describe())
    return bundle


def load_social_edges(data_dir: Path) -> np.ndarray | None:
    """Parse ``social_network.csv`` into a ``(2, n_edges)`` edge array.

    Parse failures are counted and reported rather than swallowed. The previous
    implementation used a bare ``except: continue``, so a format change could
    silently reduce the social graph to nothing while the pipeline reported
    success.
    """
    path = Path(data_dir) / "social_network.csv"
    if not path.exists():
        logger.warning("social_network.csv not found in %s; continuing without social edges", data_dir)
        return None

    frame = pd.read_csv(path)
    if not {"user_id", "friend_list"}.issubset(frame.columns):
        raise ValueError(f"{path} must contain 'user_id' and 'friend_list' columns")

    sources, targets, failures = [], [], 0
    for user_id, friend_list in zip(frame["user_id"], frame["friend_list"], strict=True):
        try:
            if isinstance(friend_list, str):
                cleaned = friend_list.strip().strip("[]")
                friends = [int(part) for part in cleaned.split(",") if part.strip()] if cleaned else []
            elif isinstance(friend_list, (list, tuple)):
                friends = [int(f) for f in friend_list]
            else:
                friends = []
        except (ValueError, TypeError):
            failures += 1
            continue
        sources.extend([int(user_id)] * len(friends))
        targets.extend(friends)

    if failures:
        failure_rate = failures / max(1, len(frame))
        logger.warning(
            "Failed to parse %d of %d friend lists (%.2f%%)", failures, len(frame), 100 * failure_rate
        )
        if failure_rate > 0.10:
            raise ValueError(
                f"{failure_rate:.1%} of friend lists failed to parse; the social graph would be "
                f"substantially incomplete. Check the file format rather than proceeding."
            )

    if not sources:
        logger.warning("social_network.csv contained no parseable edges")
        return None

    edges = np.array([sources, targets], dtype=np.int64)
    logger.info("Parsed %d directed friendship edges from %d users", edges.shape[1], len(frame))
    return edges


def build_full_reward_matrix(
    data: KuaiRecData,
    user_indexer,
    item_indexer,
    n_users: int | None = None,
    n_actions: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialise the fully observed reward block for exact policy evaluation.

    Args:
        data: the ``small_matrix`` block. Pass it *whole*: the ground-truth table
            must be built from every recorded cell even when the evaluation
            rounds themselves are subsampled, otherwise the coverage reported
            beside the numbers understates what is actually known.
        user_indexer / item_indexer: the indexers SHARED with the training log.
            Building them separately per matrix is the one mistake here that
            fails silently -- see :func:`build_exact_reward_lookup`.
        n_users / n_actions: the matrix shape. Defaults to the indexer sizes.
            ``n_actions`` must be the pipeline's declared action space, so a
            column exists for every action a policy may select; columns beyond
            the eval block's item universe stay unobserved rather than
            pretending a reward of zero.

    Returns:
        ``(rewards, observed)`` each of shape ``(n_users, n_actions)``.
        ``observed`` marks which cells were actually recorded. Nothing here
        interprets an unobserved cell; :class:`ExactRewardLookup` masks them out
        of the expectation instead of imputing a value.

    Raises:
        ValueError: if called on the sparse ``big_matrix``, where exact
            evaluation is not defined, or if a requested dimension is smaller
            than the indexer it must address.
    """
    if data.matrix_name != "small":
        raise ValueError(
            "exact evaluation requires the fully observed small_matrix; "
            f"got '{data.matrix_name}_matrix'. Estimating a policy value on the sparse "
            f"big_matrix requires an off-policy estimator, not a lookup."
        )

    n_users = len(user_indexer) if n_users is None else int(n_users)
    n_actions = len(item_indexer) if n_actions is None else int(n_actions)
    if n_users < len(user_indexer) or n_actions < len(item_indexer):
        raise ValueError(
            f"requested a {n_users} x {n_actions} reward matrix, but the shared indexers "
            f"address {len(user_indexer)} users and {len(item_indexer)} items. Widening the "
            f"matrix is safe; narrowing it would drop cells whose reward is known."
        )

    rewards = np.zeros((n_users, n_actions), dtype=np.float32)
    observed = np.zeros((n_users, n_actions), dtype=bool)

    users = user_indexer.transform(data.interactions["user_id"].to_numpy(), strict=False)
    items = item_indexer.transform(data.interactions["video_id"].to_numpy(), strict=False)
    valid = (users >= 0) & (items >= 0)

    rewards[users[valid], items[valid]] = data.interactions["reward"].to_numpy()[valid]
    observed[users[valid], items[valid]] = True

    dropped = int((~valid).sum())
    if dropped:
        logger.info(
            "%d of %d eval-block rows (%.2f%%) reference a user or item outside the shared "
            "index and cannot contribute a ground-truth cell.",
            dropped, len(valid), 100 * dropped / max(1, len(valid)),
        )
    logger.info(
        "Ground-truth reward block %d x %d | %d cells observed (%.2f%% of the grid)",
        n_users, n_actions, int(observed.sum()), 100 * observed.mean(),
    )
    return rewards, observed


def count_duplicate_cells(data: KuaiRecData) -> int:
    """How many rows of the eval block repeat a ``(user, item)`` pair.

    :func:`build_full_reward_matrix` is a last-write-wins scatter, so a repeated
    pair means one of its outcomes is discarded. On a block advertised as fully
    observed this should be zero; it is counted and reported rather than assumed,
    and :func:`ExactRewardLookup.assert_reproduces_logged_rewards` uses it as the
    only admissible explanation for a lookup/log disagreement.
    """
    return int(data.interactions.duplicated(subset=["user_id", "video_id"]).sum())


class ExactRewardLookup:
    """A ``(n_rounds, n_actions)`` view of the ground-truth reward block.

    Exact evaluation needs ``R(x_i, a)`` for every evaluation round ``i`` and
    every action ``a``. Materialising that densely is what this class exists to
    avoid: KuaiRec's eval block over a 400k-round split and 3,327 actions is
    5 GB of float32 that is entirely redundant, because a round's rewards are
    fully determined by its user. The class keeps the ``(n_users, n_actions)``
    block once and presents a row-gathered view, duck-typed to the tiny surface
    ``gcrl.evaluation.ope`` actually uses: ``.shape``, ``q[rows, cols]``,
    ``q.mean(axis=1)`` and ``q[index]`` for the bootstrap.

    **Unobserved cells are masked, not imputed.** Every gather returns a
    :class:`numpy.ma.MaskedArray`, so ``np.mean`` -- which is what
    :func:`gcrl.evaluation.ope.exact_value` applies -- averages over the cells
    whose reward was genuinely recorded. The alternative, filling an unobserved
    cell with ``0.0``, puts a fabricated number inside a column captioned
    "ground truth" and biases every policy that selects such an action downward.
    Masking instead reports ``E[R(x, pi(x)) | that cell was observed]`` and
    records, per policy, how many rounds were excluded, so the conditioning is
    visible in the results file rather than hidden in a default value.
    """

    __slots__ = ("_rewards", "_observed", "_rows", "_row_sum", "_row_count", "report")

    def __init__(
        self,
        rewards: np.ndarray,
        observed: np.ndarray,
        rows: np.ndarray,
        report: dict | None = None,
        _row_sum: np.ndarray | None = None,
        _row_count: np.ndarray | None = None,
    ) -> None:
        self._rewards = rewards
        self._observed = observed
        self._rows = np.atleast_1d(np.asarray(rows, dtype=np.int64))
        if self._rows.size and (
            self._rows.min() < 0 or self._rows.max() >= rewards.shape[0]
        ):
            raise ValueError(
                f"round user index out of range [0, {rewards.shape[0]}); observed "
                f"[{self._rows.min()}, {self._rows.max()}]"
            )
        # Per-user totals let mean(axis=1) -- the uniform policy's expectation --
        # be answered in O(n_rounds) without touching the dense block again.
        self._row_sum = (
            (rewards * observed).sum(axis=1, dtype=np.float64) if _row_sum is None else _row_sum
        )
        self._row_count = (
            observed.sum(axis=1, dtype=np.int64) if _row_count is None else _row_count
        )
        self.report = report or {}

    # -- numpy interop ----------------------------------------------------
    #: Refuse to be absorbed into a ufunc, so ``dense_action_dist * lookup``
    #: falls back to :meth:`__rmul__` instead of producing a ragged object array.
    __array_ufunc__ = None

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self._rows), self._rewards.shape[1])

    def __len__(self) -> int:
        return len(self._rows)

    def _view(self, rows) -> ExactRewardLookup:
        return ExactRewardLookup(
            self._rewards, self._observed, rows, self.report, self._row_sum, self._row_count
        )

    def __getitem__(self, key):
        """``lookup[rows, cols]`` gathers cells; ``lookup[index]`` reslices rounds."""
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError(f"expected (rounds, actions), got a {len(key)}-tuple")
            rounds, columns = key
            rows = self._rows[rounds]
            values = self._rewards[rows, columns]
            return np.ma.MaskedArray(
                values.astype(np.float64), mask=~self._observed[rows, columns]
            )
        return self._view(self._rows[key])

    def mean(self, axis=None, **_):
        """Mean over observed cells -- per round for ``axis=1``, overall otherwise."""
        counts = self._row_count[self._rows]
        totals = self._row_sum[self._rows]
        if axis == 1:
            return np.ma.MaskedArray(
                np.divide(totals, np.maximum(counts, 1)), mask=counts == 0
            )
        if axis is None:
            total_count = int(counts.sum())
            if total_count == 0:
                return np.ma.masked
            return float(totals.sum() / total_count)
        raise ValueError(f"ExactRewardLookup.mean supports axis in (None, 1), got {axis!r}")

    @property
    def n_cells(self) -> int:
        """``n_rounds * n_actions``: the size of the grid this class never builds."""
        return len(self._rows) * self._rewards.shape[1]

    @property
    def n_observed_cells(self) -> int:
        """Recorded cells among the gathered rounds, counted in O(n_rounds).

        The obvious spelling -- ``observed[rows].sum()`` -- gathers the whole
        ``(n_rounds, n_actions)`` boolean grid to count it, which for KuaiRec's
        4,676,570 rounds over 3,327 actions is 14.5 GiB and is what killed a
        6.7-hour run at the top of Phase 4. A round's observed count is a
        property of its USER, so summing the per-user counts over the rounds
        gives the identical integer for the size of the rounds alone.
        """
        return int(self._row_count[self._rows].sum())

    #: Row blocks are sized to about this many cells in :meth:`expected_under`.
    #: Each round's expectation is a sum along its own row, so this changes the
    #: peak memory and nothing else about the arithmetic. A block costs about
    #: 18 bytes per cell (rewards and their product as float64, each with a
    #: boolean mask), so 1e6 cells is an ~18 MB working set whatever the number
    #: of evaluation rounds.
    EXPECTED_CELL_BLOCK = 1_000_000

    def _dense_block(self, rows: np.ndarray) -> np.ma.MaskedArray:
        """The masked ``(len(rows), n_actions)`` sub-block for ``rows``.

        The bounded primitive the estimators are built on. Callers pass a slice
        of :attr:`_rows`, so the allocation is theirs to size;
        :meth:`to_dense_masked` is the one caller that passes all of them.
        """
        return np.ma.MaskedArray(
            self._rewards[rows].astype(np.float64), mask=~self._observed[rows]
        )

    def to_dense_masked(self) -> np.ma.MaskedArray:
        """The WHOLE dense ``(n_rounds, n_actions)`` masked grid. Memory-hungry by design.

        This is the explicit escape hatch, for a caller that has checked the
        size and genuinely wants the grid. Nothing in the evaluation path calls
        it: :meth:`expected_under` walks the rounds in blocks instead, so the
        peak is set by :attr:`EXPECTED_CELL_BLOCK` rather than by the number of
        evaluation rounds. On the full KuaiRec split this method is 58.0 GiB of
        rewards plus 14.5 GiB of mask.
        """
        return self._dense_block(self._rows)

    def expected_under(self, action_dist, cells_per_block: int | None = None):
        """``E_{a ~ pi_e}[R(x_i, a)]`` per round for a DENSE action distribution.

        This is exactly ``np.sum(action_dist * self, axis=1)`` -- the same
        elements, in the same order, so the same floating-point result -- with
        the rounds walked in blocks of about ``cells_per_block`` cells. The
        single-expression form materialises the whole grid this class exists to
        avoid; that is the 58.0 GiB allocation a dense policy would otherwise
        force on the full KuaiRec split.

        Unobserved cells stay masked, so a round's expectation is over the
        actions whose reward was recorded, and a round with nothing recorded at
        all comes back masked rather than as a fabricated zero -- identical to
        the dense path's behaviour.

        Sparse policies never reach this: they select cells directly through
        ``lookup[rounds, actions]`` or :meth:`mean`.
        """
        weights = action_dist
        shape = getattr(weights, "shape", ())
        if tuple(shape) != self.shape:
            raise ValueError(
                f"action distribution {tuple(shape)} does not match the ground-truth "
                f"lookup {self.shape}"
            )
        n_rounds, n_actions = self.shape
        block = int(cells_per_block or self.EXPECTED_CELL_BLOCK)
        rows_per_block = max(1, block // max(1, n_actions))
        starts = range(0, n_rounds, rows_per_block) if n_rounds else (0,)
        blocks = [
            np.sum(
                self._dense_block(self._rows[start : start + rows_per_block])
                * weights[start : start + rows_per_block],
                axis=1,
            )
            for start in starts
        ]
        return blocks[0] if len(blocks) == 1 else np.ma.concatenate(blocks)

    def __mul__(self, other):
        """``action_dist * lookup``, refused when the grid is above the cell budget.

        A dense action distribution multiplied by the lookup materialises the
        very grid this class exists to avoid, and NumPy's failure mode for that
        is an opaque ``MemoryError`` naming a shape and a dtype -- after however
        many hours of training produced the policy. Above
        :data:`~gcrl.evaluation.ope.DENSE_ACTION_DIST_CELL_BUDGET` the
        multiplication is refused with the arithmetic spelled out instead.
        """
        from ..evaluation.ope import DENSE_ACTION_DIST_CELL_BUDGET, describe_dense_grid

        if self.n_cells > DENSE_ACTION_DIST_CELL_BUDGET:
            n_rounds, n_actions = self.shape
            raise ValueError(
                f"refusing to multiply the ground-truth lookup as a dense grid: "
                f"{describe_dense_grid(n_rounds, n_actions)}, against a budget of "
                f"{DENSE_ACTION_DIST_CELL_BUDGET:,} cells. This happens when a DENSE "
                f"action distribution is multiplied by the lookup; the estimators "
                f"evaluate that expectation in row blocks instead. Call "
                f"to_dense_masked() explicitly if the dense grid is genuinely wanted."
            )
        return self.to_dense_masked() * other

    __rmul__ = __mul__

    # -- diagnostics ------------------------------------------------------
    def observed_for(self, actions: np.ndarray) -> np.ndarray:
        """Which rounds have a recorded reward for the action the policy picks."""
        actions = np.asarray(actions, dtype=np.int64)
        if len(actions) != len(self._rows):
            raise ValueError(
                f"{len(actions)} actions for {len(self._rows)} rounds; the ground-truth "
                f"lookup must be built from the same rounds that are being scored"
            )
        return self._observed[self._rows, actions]

    def assert_reproduces_logged_rewards(
        self, actions: np.ndarray, rewards: np.ndarray
    ) -> None:
        """Verify the lookup agrees with the log on the rounds' own logged actions.

        Every evaluation round is itself a recorded cell of the ground-truth
        block, so ``lookup[i, a_i]`` must equal the round's logged reward. If the
        user or item indexers were fitted per matrix instead of shared, this
        check is what catches it: a shifted index still produces a
        well-shaped matrix full of plausible numbers, and nothing else in the
        pipeline would notice.

        Raises:
            ValueError: on any disagreement that repeated ``(user, item)`` rows
                in the eval block cannot account for.
        """
        actions = np.asarray(actions, dtype=np.int64)
        rewards = np.asarray(rewards, dtype=np.float64)
        observed = self.observed_for(actions)
        if not observed.all():
            raise ValueError(
                f"{int((~observed).sum())} of {len(observed)} evaluation rounds have no "
                f"recorded reward for their OWN logged action. Every round is a row of the "
                f"ground-truth block, so this means the rounds and the block were built "
                f"from different data or with unshared indexers."
            )
        looked_up = np.asarray(self[np.arange(len(actions)), actions])
        mismatches = int(np.count_nonzero(looked_up != rewards))
        allowed = int(self.report.get("duplicate_cells", 0))
        if mismatches > allowed:
            raise ValueError(
                f"the ground-truth lookup disagrees with the logged reward on {mismatches} "
                f"of {len(rewards)} evaluation rounds, but only {allowed} repeated "
                f"(user, item) rows could explain a disagreement. The user or item indexer "
                f"is not shared between the training log and the evaluation block."
            )
        if mismatches:
            logger.warning(
                "%d rounds disagree with the lookup, matching the %d repeated (user, item) "
                "rows in the evaluation block (last write wins).", mismatches, allowed,
            )
        logger.info(
            "Index alignment verified: the ground-truth block reproduces the logged reward "
            "of all %d evaluation rounds.", len(rewards),
        )


def build_exact_reward_lookup(
    data: KuaiRecData,
    user_indexer,
    item_indexer,
    round_user_index: np.ndarray,
    n_actions: int | None = None,
    n_users: int | None = None,
) -> ExactRewardLookup:
    """Ground truth for the evaluation rounds, with its coverage recorded.

    Args:
        data: the whole ``small_matrix`` block.
        user_indexer / item_indexer: the indexers shared with the training log.
        round_user_index: the user index of each evaluation round, in the exact
            order Phase 4 scores them. Alignment between this array and the
            rounds is by construction; :meth:`ExactRewardLookup.assert_reproduces_logged_rewards`
            checks it anyway.

    The returned lookup's ``report`` carries the coverage of the cells exact
    evaluation can actually consult -- not the coverage of the whole grid, which
    on a shared index is dominated by users the eval block never saw and would
    understate what is known by a factor of five.
    """
    rewards, observed = build_full_reward_matrix(
        data, user_indexer, item_indexer, n_users=n_users, n_actions=n_actions
    )
    rows = np.asarray(round_user_index, dtype=np.int64)
    lookup = ExactRewardLookup(rewards, observed, rows)
    # Coverage is counted from the per-user observed counts, NOT from
    # ``observed[rows]``: that gather is the dense (n_rounds, n_actions) boolean
    # grid -- 14.5 GiB on the full KuaiRec split -- and building it to compute a
    # single fraction is what aborted Phase 4 after Phase 3 had already run.
    # The counts give the identical integers; see ExactRewardLookup.n_observed_cells.
    cells_consulted = lookup.n_cells
    cells_observed = lookup.n_observed_cells
    coverage = cells_observed / cells_consulted if cells_consulted else 0.0
    report = {
        "eval_matrix": f"{data.matrix_name}_matrix",
        "eval_rows": int(len(data.interactions)),
        "reward_threshold": float(data.reward_threshold),
        "n_rounds": int(len(rows)),
        "n_actions": int(rewards.shape[1]),
        "cells_consulted": int(cells_consulted),
        "cells_observed": int(cells_observed),
        "coverage": coverage,
        "unobserved_policy": "masked_out_of_expectation",
        "duplicate_cells": count_duplicate_cells(data),
    }
    logger.info(
        "Exact ground truth: %.4f%% of the %d round x %d action cells are observed; the "
        "remainder are MASKED OUT of the expectation, not imputed as zero.",
        100 * coverage, report["n_rounds"], report["n_actions"],
    )
    if report["duplicate_cells"]:
        logger.warning(
            "%d rows of %s repeat a (user, item) pair; the reward block keeps the last one.",
            report["duplicate_cells"], report["eval_matrix"],
        )
    if coverage < 0.99:
        logger.warning(
            "Only %.2f%% of the consulted cells are observed. Exact values are then "
            "conditional on the cell being recorded, and the per-policy masked-round counts "
            "in the Phase 4 results must be quoted alongside them.", 100 * coverage,
        )
    lookup.report = report
    return lookup
