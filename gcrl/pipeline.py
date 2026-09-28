"""Dataset preparation and pipeline orchestration.

Turns a loaded dataset into the arrays every phase consumes, keeping the split
in one place so the test partition is created once and touched only by Phase 4.

Two evaluation protocols are supported, and which one is in force is a property
of the config, not of a code path chosen at runtime:

**One log, split temporally.** The default. Train, validation and test are three
disjoint slices of a single file, ordered in time.

**A separate, fully observed evaluation block.** When ``dataset.eval_file`` is
set, the test partition is that file in its entirety and train/validation come
from the training log alone. This is what KuaiRec's ``big_matrix`` /
``small_matrix`` pair is for: the log is sparse, the evaluation block is
essentially complete, and a policy's value on it can be looked up instead of
estimated. The two files are indexed with ONE pair of indexers, because a
per-file index would line the same raw id up against two different rows and
produce a table of plausible, meaningless numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data.graphs import InteractionGraph, build_bipartite_graph
from .data.splits import DataSplit, split_frame
from .encoders import DeterministicCategoricalEncoder, IdentifierIndexer
from .logging_utils import get_logger

logger = get_logger(__name__)

#: Value of ``PreparedDataset.propensity_source`` when the dataset shipped its
#: own logged propensities. Anything else means the number was produced here and
#: must be reported as such.
PROPENSITY_LOGGED = "logged"


@dataclass
class BehaviourPolicy:
    """An estimated ``pi_b(a | x)``, plus what is needed to report it honestly.

    KuaiRec logs no propensity. The choice is therefore between not running the
    importance-weighted estimators at all, and estimating the behaviour policy
    from the log. A constant ``1 / n_actions`` is not a third option: it makes
    every importance weight identical, which turns SNIPW into a plain mean over
    the rounds the policies agree on and DR into the direct method, while the
    results table still says "SNIPW" and "DR".

    The estimator is a per-user action distribution shrunk toward the marginal
    one::

        pi_b(a | u) = (n(u, a) + alpha * m(a)) / (n(u) + alpha)

    with ``m`` the Laplace-smoothed marginal over actions and ``alpha`` the
    shrinkage in pseudo-counts. It is conditioned on the user because the user
    *is* the context every agent in this pipeline observes -- Phase 3's state is
    that user's graph embedding -- so this conditions on exactly the information
    the evaluation policies see. It is a proper distribution (it sums to one
    over actions for every user) and is strictly positive everywhere, so overlap
    holds by construction rather than by luck. ``alpha -> inf`` recovers the
    marginal popularity policy; ``alpha = 0`` would trust one user's empirical
    frequencies completely.

    It is fitted on the TRAINING rounds only. Fitting it on the rounds it then
    weights would be the same in-sample error that Phase 4 cross-fits the reward
    model to avoid.
    """

    marginal: np.ndarray
    user_totals: np.ndarray
    pair_keys: np.ndarray
    pair_counts: np.ndarray
    n_actions: int
    shrinkage: float

    def score(self, user_index: np.ndarray, actions: np.ndarray) -> np.ndarray:
        user_index = np.asarray(user_index, dtype=np.int64)
        actions = np.asarray(actions, dtype=np.int64)
        if user_index.shape != actions.shape:
            raise ValueError(
                f"user_index has {user_index.shape} entries but actions has {actions.shape}"
            )
        if actions.size and (actions.min() < 0 or actions.max() >= self.n_actions):
            raise ValueError(
                f"action outside [0, {self.n_actions}); observed "
                f"[{actions.min()}, {actions.max()}]"
            )
        if user_index.size and (
            user_index.min() < 0 or user_index.max() >= len(self.user_totals)
        ):
            raise ValueError(
                f"user index outside [0, {len(self.user_totals)}); the behaviour policy must "
                f"be fitted with the same shared indexers the rounds were built with"
            )

        # n(u, a) by lookup into the sorted (user, action) key table. Pairs the
        # training log never saw contribute 0, which leaves only the shrinkage
        # term -- small, but strictly positive, so overlap never breaks.
        query = user_index * self.n_actions + actions
        pair = np.zeros(len(query), dtype=np.float64)
        if len(self.pair_keys):
            position = np.clip(np.searchsorted(self.pair_keys, query), 0, len(self.pair_keys) - 1)
            found = self.pair_keys[position] == query
            pair[found] = self.pair_counts[position[found]]

        totals = self.user_totals[user_index].astype(np.float64)
        propensity = (pair + self.shrinkage * self.marginal[actions]) / (totals + self.shrinkage)
        if not np.all((propensity > 0) & (propensity <= 1)):
            raise ValueError(
                "the estimated behaviour policy produced a propensity outside (0, 1]; "
                "this is a bug in the estimator, not a property of the data"
            )
        return propensity

    def describe(self) -> str:
        return f"estimated:user_conditional_shrinkage(alpha={self.shrinkage:g})"


def fit_behaviour_policy(
    user_index: np.ndarray, actions: np.ndarray, n_users: int, n_actions: int, shrinkage: float
) -> BehaviourPolicy:
    """Fit :class:`BehaviourPolicy` on the logged training rounds."""
    user_index = np.asarray(user_index, dtype=np.int64)
    actions = np.asarray(actions, dtype=np.int64)
    if len(user_index) == 0:
        raise ValueError("cannot estimate a behaviour policy from zero logged rounds")

    action_counts = np.bincount(actions, minlength=n_actions).astype(np.float64)
    marginal = (action_counts + 1.0) / (action_counts.sum() + n_actions)

    user_totals = np.bincount(user_index, minlength=n_users).astype(np.int64)
    pair_keys, pair_counts = np.unique(user_index * n_actions + actions, return_counts=True)

    logger.info(
        "Estimated behaviour policy from %d training rounds: %d users, %d actions, "
        "%d observed (user, action) pairs, shrinkage alpha=%g",
        len(actions), n_users, n_actions, len(pair_keys), shrinkage,
    )
    return BehaviourPolicy(
        marginal=marginal, user_totals=user_totals, pair_keys=pair_keys,
        pair_counts=pair_counts.astype(np.float64), n_actions=n_actions, shrinkage=shrinkage,
    )


def behaviour_policy_diagnostics(
    policy: BehaviourPolicy, user_index: np.ndarray, actions: np.ndarray, label: str
) -> dict[str, float]:
    """How well the estimated policy explains the actions it is asked to weight.

    Mean log ``pi_b(a_i | x_i)`` is a proper scoring rule, and the uniform policy
    is the reference: a fitted policy that does not beat ``log(1/A)`` has learned
    nothing, and the importance weights it produces are noise.
    """
    propensities = policy.score(user_index, actions)
    log_likelihood = float(np.mean(np.log(propensities)))
    uniform = float(np.log(1.0 / policy.n_actions))
    unseen = float(np.mean(policy.user_totals[user_index] == 0)) if len(user_index) else 0.0
    diagnostics = {
        f"{label}_mean_log_likelihood": log_likelihood,
        f"{label}_uniform_log_likelihood": uniform,
        f"{label}_propensity_min": float(propensities.min()),
        f"{label}_propensity_max": float(propensities.max()),
        f"{label}_unseen_user_fraction": unseen,
    }
    logger.info(
        "Behaviour policy on %s rounds: mean log pi_b = %.4f (uniform would be %.4f), "
        "propensity in [%.3g, %.3g], %.2f%% of rounds have a user unseen in training",
        label, log_likelihood, uniform, propensities.min(), propensities.max(), 100 * unseen,
    )
    if log_likelihood <= uniform:
        logger.warning(
            "The estimated behaviour policy on %s rounds fits no better than a uniform one "
            "(%.4f vs %.4f). Importance weights derived from it carry no information about "
            "the logging policy and the SNIPW/DR/MRDR columns must say so.",
            label, log_likelihood, uniform,
        )
    return diagnostics


@dataclass
class PreparedDataset:
    """A dataset in the form every phase expects."""

    frame: pd.DataFrame
    split: DataSplit
    graph: InteractionGraph
    user_indexer: IdentifierIndexer
    item_indexer: IdentifierIndexer
    feature_columns: list[str]
    n_actions: int
    provenance: dict[str, list[str]]
    reward_definition: str
    #: Where ``__propensity`` came from. ``"logged"``, an ``"estimated:..."``
    #: description, or ``"constant:..."``. Phase 4 copies this into every result
    #: row so an importance-weighted number can never be read as though the
    #: dataset had shipped its own propensities.
    propensity_source: str = PROPENSITY_LOGGED
    propensity_diagnostics: dict[str, float] = field(default_factory=dict)
    #: Set when the test partition is a separate, fully observed evaluation file.
    evaluation_block: str | None = None
    #: Item node index used to BUILD THE GRAPH. It spans the whole training log,
    #: which is wider than the action space whenever the log contains items the
    #: evaluation block cannot score. Only user embeddings leave Phase 1, so the
    #: two indexes never have to agree -- but they are different, and keeping
    #: the graph's one here is what makes that auditable rather than implicit.
    graph_item_indexer: IdentifierIndexer | None = None
    #: Rows the graph representation was trained on, and the subset of them that
    #: could also train a policy (action inside the action space).
    representation_rows: int = 0
    policy_rows: int = 0
    #: What the evaluation actually measures, given how the two files overlap.
    evaluation_regime: str = "single_log"

    def arrays(self, subset: str = "all") -> dict[str, np.ndarray]:
        """Return the arrays for one partition.

        Args:
            subset: ``"train"``, ``"validation"``, ``"test"`` or ``"all"``.
        """
        frame = {
            "train": self.split.train,
            "validation": self.split.validation,
            "test": self.split.test,
            "all": self.frame,
        }[subset]

        return {
            "user_index": frame["__user_index"].to_numpy(dtype=np.int64),
            "actions": frame["__action_index"].to_numpy(dtype=np.int64),
            "rewards": frame["__reward"].to_numpy(dtype=np.float64),
            "propensities": frame["__propensity"].to_numpy(dtype=np.float64),
            "raw_features": frame[self.feature_columns].to_numpy(dtype=np.float32),
        }


def resolve_feature_columns(frame: pd.DataFrame, config: ExperimentConfig) -> list[str]:
    """Resolve the numeric feature columns from explicit names and patterns."""
    columns: list[str] = [c for c in config.dataset.feature_columns if c in frame.columns]
    for pattern in config.dataset.feature_column_patterns:
        columns.extend(c for c in frame.columns if pattern in c and c not in columns)

    numeric = [c for c in columns if pd.api.types.is_numeric_dtype(frame[c])]
    dropped = set(columns) - set(numeric)
    if dropped:
        logger.warning(
            "Dropping %d non-numeric feature columns: %s. Encode them explicitly via "
            "dataset.categorical_columns rather than relying on an implicit cast.",
            len(dropped), sorted(dropped)[:5],
        )
    if not numeric:
        raise ValueError(
            "no numeric feature columns resolved. Set dataset.feature_columns or "
            "dataset.feature_column_patterns to match this dataset's schema."
        )
    logger.info("Resolved %d numeric feature columns", len(numeric))
    return numeric


def _split_training_log(
    frame: pd.DataFrame, config: ExperimentConfig, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, str, tuple[float, float]]:
    """Cut the training log into train and validation only.

    Under the evaluation-block protocol the log contributes no test rows -- the
    evaluation block is the test set -- so the configured ``train:val`` ratio is
    renormalised over the log and every logged row lands in one of the two.
    ``split_frame`` still does the cutting, so the temporal-leakage assertion and
    the tie warning apply here exactly as they do elsewhere; its third partition
    is folded back into validation.
    """
    import dataclasses as _dc

    total = config.split.train_ratio + config.split.val_ratio
    held_out = config.split.val_ratio / total
    inner = _dc.replace(
        config.split,
        train_ratio=config.split.train_ratio / total,
        val_ratio=held_out / 2.0,
        test_ratio=held_out / 2.0,
    )
    split = split_frame(frame, inner, seed=seed)
    validation = pd.concat([split.validation, split.test], ignore_index=True)
    return split.train, validation, split.strategy, split.boundaries


def prepare_dataset(
    frame: pd.DataFrame,
    config: ExperimentConfig,
    user_column: str,
    item_column: str,
    reward_column: str,
    propensity_column: str | None = None,
    social_edges: np.ndarray | None = None,
    provenance: dict[str, list[str]] | None = None,
    seed: int | None = None,
    eval_frame: pd.DataFrame | None = None,
    action_universe: np.ndarray | None = None,
    evaluation_block: str | None = None,
) -> PreparedDataset:
    """Index, split and graph-ify a raw interaction frame.

    Args:
        eval_frame: an independent evaluation block. When given it becomes the
            whole test partition, train/validation come from ``frame`` alone,
            and both frames are indexed with the same indexers.
        action_universe: the raw item ids that define the action space. Pass it
            when the evaluation rounds are a subsample, so the action space is
            the evaluation block's full item universe rather than whatever
            happened to survive subsampling.

    Raises:
        ValueError: if a required column is missing, or a propensity column is
            required by the OPE config but absent and ``ope.behaviour_policy``
            does not authorise estimating one. Substituting a constant
            propensity is not offered, because it silently converts a
            self-normalised estimator into a plain mean.
    """
    seed = seed if seed is not None else config.seed
    frame = frame.copy()
    eval_frame = None if eval_frame is None else eval_frame.copy()

    for part in (frame, eval_frame):
        if part is None:
            continue
        for column in (user_column, item_column, reward_column):
            if column not in part.columns:
                raise ValueError(f"required column {column!r} not present in the data")

    def _both() -> list[pd.DataFrame]:
        return [frame] if eval_frame is None else [frame, eval_frame]

    if config.dataset.categorical_columns:
        present = [c for c in config.dataset.categorical_columns if c in frame.columns]
        if present:
            # Fitted on both blocks so a category means the same code in each.
            encoder = DeterministicCategoricalEncoder(present).fit(
                pd.concat(_both(), ignore_index=True)
            )
            frame = encoder.transform(frame)
            if eval_frame is not None:
                eval_frame = encoder.transform(eval_frame)
            logger.info(
                "Encoded %d categorical columns deterministically (cardinalities: %s)",
                len(present), {c: encoder.cardinality(c) for c in present},
            )

    # ONE pair of indexers across both blocks. Fitting them per file is the
    # failure this protocol is most exposed to: the shapes still line up, every
    # downstream check still passes, and the ground-truth lookup silently reads
    # the wrong row.
    user_indexer = IdentifierIndexer("user").fit(
        pd.concat([part[user_column] for part in _both()], ignore_index=True)
    )
    item_indexer = IdentifierIndexer("item").fit(
        pd.concat([part[item_column] for part in _both()], ignore_index=True)
        if action_universe is None
        else pd.Series(np.asarray(action_universe))
    )

    for part in _both():
        part["__user_index"] = user_indexer.transform(part[user_column])
        part["__reward"] = part[reward_column].to_numpy(dtype=np.float64)

    # The evaluation block must lie entirely inside the action space; the
    # training log need not, and on KuaiRec it does not overlap it at all.
    if eval_frame is not None:
        eval_frame["__action_index"] = item_indexer.transform(eval_frame[item_column])
    frame["__action_index"] = item_indexer.transform(
        frame[item_column], strict=eval_frame is None
    )

    n_actions = config.dataset.num_actions or len(item_indexer)
    if len(item_indexer) > n_actions:
        raise ValueError(
            f"the data contains {len(item_indexer)} distinct items but dataset.num_actions "
            f"is {n_actions}. Widen num_actions rather than clipping."
        )

    graph_item_indexer: IdentifierIndexer | None = None
    if eval_frame is None:
        split = split_frame(frame, config.split, seed=seed)
        combined = frame
        graph_frame = split.train
        graph_item_index = graph_frame["__action_index"].to_numpy(dtype=np.int64)
        graph_num_items = max(len(item_indexer), n_actions)
        regime = "single_log"
    else:
        log_train, log_validation, strategy, boundaries = _split_training_log(frame, config, seed)

        # ---------------------------------------------------------------
        # The representation and the policy are trained on different rows,
        # and the difference is the whole story on KuaiRec.
        #
        # The GRAPH is trained on every logged row of the training period. It
        # has to be: the evaluation users' entire history lies outside the
        # action space, so filtering the log to the action space would delete
        # them and leave their embeddings untrained. The guard below rejects
        # any configuration in which that happens.
        #
        # A POLICY can only be trained on rows whose action is inside its own
        # action set, so those rows -- and only those -- become the train and
        # validation partitions.
        # ---------------------------------------------------------------
        graph_frame = log_train
        graph_item_indexer = IdentifierIndexer("graph_item").fit(
            pd.concat(
                [frame[item_column], pd.Series(np.asarray(action_universe))]
                if action_universe is not None
                else [frame[item_column], eval_frame[item_column]],
                ignore_index=True,
            )
        )
        graph_item_index = graph_item_indexer.transform(graph_frame[item_column])
        graph_num_items = len(graph_item_indexer)

        split = DataSplit(
            train=log_train[log_train["__action_index"] >= 0].reset_index(drop=True),
            validation=log_validation[log_validation["__action_index"] >= 0].reset_index(drop=True),
            test=eval_frame.reset_index(drop=True),
            strategy=f"{strategy}+eval_block", boundaries=boundaries,
        )
        combined = pd.concat([split.train, split.validation, split.test], ignore_index=True)
        logger.info(
            "Evaluation block protocol: the graph is trained on %d logged rows over %d items; "
            "of those, %d rows (%.1f%%) have an action inside the %d-action evaluation space "
            "and can also train a policy. Test = %d rounds from %s, never seen in training.",
            len(graph_frame), graph_num_items, len(split.train),
            100 * len(split.train) / max(1, len(graph_frame)), n_actions,
            len(split.test), evaluation_block,
        )
        regime = _check_evaluation_reachability(split, graph_frame)

    source, diagnostics = _assign_propensities(
        combined, split, config, propensity_column, n_actions, len(user_indexer),
        evaluation_block=evaluation_block,
    )

    # The graph is built from TRAINING-PERIOD interactions only. Including
    # validation or test edges would leak held-out interactions into the
    # representation that is later used to score them.
    graph = build_bipartite_graph(
        user_index=graph_frame["__user_index"].to_numpy(dtype=np.int64),
        item_index=graph_item_index,
        num_users=len(user_indexer),
        num_items=graph_num_items,
        social_edges=social_edges,
        user_id_to_index=user_indexer.mapping,
        embedding_dim=config.gnn.embedding_dim,
        seed=seed,
    )
    logger.info(
        "Graph built from %d training interactions only (no held-out edges)", len(graph_frame)
    )

    return PreparedDataset(
        frame=combined,
        split=split,
        graph=graph,
        user_indexer=user_indexer,
        item_indexer=item_indexer,
        feature_columns=resolve_feature_columns(combined, config),
        n_actions=n_actions,
        provenance=provenance or {},
        reward_definition=config.dataset.reward_definition or f"{reward_column} (as logged)",
        propensity_source=source,
        propensity_diagnostics=diagnostics,
        evaluation_block=evaluation_block,
        graph_item_indexer=graph_item_indexer,
        representation_rows=len(graph_frame),
        policy_rows=len(split.train),
        evaluation_regime=regime,
    )


def _check_evaluation_reachability(split: DataSplit, graph_frame: pd.DataFrame) -> str:
    """Verify the evaluation users are reachable, and name the regime.

    This is the guard that caught the first version of this protocol, which
    filtered the training log to the evaluation block's action space. On KuaiRec
    that filter removes every evaluation user, because the dataset's authors
    deleted the block's (user, item) pairs from the log. It still fires, but on
    the population it should always have checked: the rows the GRAPH is trained
    on, not the subset that can also train a policy.

    The distinction matters, because "the evaluation users contribute no
    policy-training rows" is not an error -- on KuaiRec it is the dataset's
    design -- while "the evaluation users contribute nothing at all" is.

    Returns:
        A description of the evaluation regime, carried into every result row.

    Raises:
        ValueError: if no evaluation user appears in the rows the graph is
            trained on. Every policy would then be scored on users whose
            representation was never trained.
    """
    represented = set(graph_frame["__user_index"].to_numpy().tolist())
    evaluated = set(split.test["__user_index"].to_numpy().tolist())
    reachable = evaluated & represented
    if not reachable:
        raise ValueError(
            f"none of the {len(evaluated)} evaluation users appears in the "
            f"{len(represented)} users the graph is trained on. Every policy would be scored "
            f"on users whose representation was never trained. Do not filter the training log "
            f"to the evaluation action space: on KuaiRec that removes exactly these users."
        )
    if len(reachable) < len(evaluated):
        cold = len(evaluated) - len(reachable)
        logger.warning(
            "%d of %d evaluation users (%.1f%%) never appear in the training log; their "
            "graph embedding is the untrained node feature. Report this alongside the "
            "evaluation numbers.", cold, len(evaluated), 100 * cold / len(evaluated),
        )

    # How much history each evaluation user actually contributes. This is the
    # budget their embedding is learned from, and `dataset.subsample_rows` cuts
    # the log uniformly -- so a small sample leaves the evaluation users, who
    # are a minority of it, very thin. On KuaiRec they are 4.6% of the log, so a
    # 300,000-row sample leaves roughly ten interactions each.
    per_user = graph_frame.loc[
        graph_frame["__user_index"].isin(reachable), "__user_index"
    ].value_counts()
    median_history = float(per_user.median()) if len(per_user) else 0.0
    logger.info(
        "Evaluation users contribute %d of the %d rows the graph is trained on; median "
        "history %.0f interactions per evaluation user.",
        int(per_user.sum()), len(graph_frame), median_history,
    )
    if median_history < 20:
        logger.warning(
            "The median evaluation user has only %.0f logged interactions to be embedded "
            "from. Their representation -- the ONLY route from an evaluation user to an "
            "evaluation action -- is thin, and results from this run describe that budget, "
            "not the full dataset. Raise dataset.subsample_rows or report the number.",
            median_history,
        )

    # Does the policy-training data contain any of the (user, action) pairs the
    # evaluation will score?
    policy_users = set(split.train["__user_index"].to_numpy().tolist())
    shared_users = evaluated & policy_users
    overlap_cells = 0
    if shared_users:
        train_pairs = set(
            map(
                tuple,
                split.train.loc[
                    split.train["__user_index"].isin(shared_users),
                    ["__user_index", "__action_index"],
                ].drop_duplicates().to_numpy().tolist(),
            )
        )
        test_pairs = set(
            map(
                tuple,
                split.test[["__user_index", "__action_index"]]
                .drop_duplicates().to_numpy().tolist(),
            )
        )
        overlap_cells = len(train_pairs & test_pairs)

    if overlap_cells == 0:
        logger.warning(
            "COLD-ITEM EVALUATION: not one of the (evaluation user, action) pairs Phase 4 "
            "scores appears in the training log. The evaluation users are represented through "
            "their history over OTHER items (%d of them appear in the log), and the actions "
            "are represented through OTHER users' history (%d rows). No policy can reach an "
            "evaluation action for an evaluation user except through the graph. Recorded in "
            "every result row as the evaluation regime.",
            len(reachable), len(split.train),
        )
        return (
            "cold_item: no (evaluation user, action) pair appears in the training log; "
            "every scored decision is a genuine counterfactual"
        )
    logger.info(
        "%d (evaluation user, action) cells also appear in the training log.", overlap_cells
    )
    return f"warm: {overlap_cells} (evaluation user, action) cells appear in the training log"


def enumerated_block_propensities(
    user_index: np.ndarray, actions: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    """``pi_b(a | u)`` for a fully observed evaluation block -- known, not estimated.

    A fully observed block is not a log produced by some unknown recommender: it
    is an enumeration. Its authors played every catalogue item for every user in
    the block, so the probability that a given row carries action ``a`` for user
    ``u`` is a property of how the block was constructed, ``n(u, a) / n(u)``,
    which for a complete enumeration is ``1 / k(u)`` with ``k(u)`` the number of
    actions enumerated for that user.

    This is computed from ``(user, action)`` pairs only. No outcome enters it, so
    there is nothing to overfit and no leakage; it is a design parameter of the
    data collection, read off rather than modelled. That makes it the right
    propensity for the evaluation rounds, and it replaces -- for those rounds --
    the behaviour policy estimated from the sparse log, which describes a
    different data-generating process entirely and would attach importance
    weights of order ``10^5`` to an enumeration.

    One consequence is worth stating rather than discovering: under an
    enumerated block a deterministic, user-level policy agrees with exactly one
    round per user, so SNIPW reduces to the mean true reward of the selected
    action over users -- essentially the exact value. That is a property of
    evaluating on a complete block, not a bug, and it means the error the
    validation table reports for DM, DR and MRDR is attributable to the reward
    model alone.
    """
    user_index = np.asarray(user_index, dtype=np.int64)
    actions = np.asarray(actions, dtype=np.int64)
    if len(user_index) == 0:
        return np.zeros(0, dtype=np.float64), {}

    per_user = np.bincount(user_index).astype(np.float64)
    keys, inverse, counts = np.unique(
        np.stack([user_index, actions]), axis=1, return_inverse=True, return_counts=True
    )
    pair_counts = counts[inverse].astype(np.float64)
    propensities = pair_counts / per_user[user_index]

    repeat_rate = float(counts.mean())
    diagnostics = {
        "block_rows": float(len(user_index)),
        "block_distinct_cells": float(keys.shape[1]),
        "block_rows_per_cell": repeat_rate,
        "block_propensity_min": float(propensities.min()),
        "block_propensity_max": float(propensities.max()),
    }
    logger.info(
        "Evaluation block propensities read off the enumeration: %d rows over %d distinct "
        "(user, action) cells (%.3f rows per cell), pi_b in [%.3g, %.3g]",
        len(user_index), keys.shape[1], repeat_rate, propensities.min(), propensities.max(),
    )
    if repeat_rate > 1.5:
        logger.warning(
            "The evaluation block repeats each (user, action) cell %.2f times on average, so "
            "it is not a clean enumeration. pi_b is still its exact empirical action "
            "distribution, but describe the block as a log rather than a full enumeration.",
            repeat_rate,
        )
    return propensities, diagnostics


def _assign_propensities(
    combined: pd.DataFrame,
    split: DataSplit,
    config: ExperimentConfig,
    propensity_column: str | None,
    n_actions: int,
    n_users: int,
    evaluation_block: str | None = None,
) -> tuple[str, dict[str, float]]:
    """Write ``__propensity`` into every partition and say where it came from.

    Four outcomes, never silently interchangeable:

    * the test partition is a fully observed evaluation block -- its propensity
      is read off the enumeration and is exactly known, and the sparse log's
      partitions keep whatever the rules below give them;
    * the dataset logged its propensities -- they are used;
    * ``ope.behaviour_policy == "estimated"`` -- one is fitted on the TRAINING
      rounds and every downstream row is stamped ``estimated:...``;
    * none of the above, while an importance-weighted estimator is requested --
      this raises. A constant is written only when nothing consumes it, and even
      then it is labelled ``constant:...`` rather than passed off as a policy.
    """
    parts = [combined, split.train, split.validation, split.test]
    needs_propensity = bool(set(config.ope.estimators) & {"ipw", "snipw", "dr", "mrdr"})

    block_propensities, block_diagnostics = (np.zeros(0), {})
    if evaluation_block and len(split.test):
        block_propensities, block_diagnostics = enumerated_block_propensities(
            split.test["__user_index"].to_numpy(dtype=np.int64),
            split.test["__action_index"].to_numpy(dtype=np.int64),
        )

    def _apply_block(source: str, diagnostics: dict) -> tuple[str, dict]:
        """Overwrite the test rounds with the block's own known propensities."""
        if not evaluation_block or not len(split.test):
            return source, diagnostics
        split.test["__propensity"] = block_propensities
        combined.loc[combined.index[-len(split.test):], "__propensity"] = block_propensities
        merged = {**diagnostics, **block_diagnostics}
        logger.info(
            "Evaluation rounds use the block's OWN propensities (enumerated), not %s; the "
            "sparse log's partitions keep theirs.", source,
        )
        return (
            f"eval_block:enumerated_empirical | log:{source}",
            merged,
        )

    if propensity_column and propensity_column in combined.columns:
        for part in parts:
            part["__propensity"] = part[propensity_column].to_numpy(dtype=np.float64)
        logger.info(
            "Using logged propensities from %r: range [%.6f, %.6f]",
            propensity_column, combined["__propensity"].min(), combined["__propensity"].max(),
        )
        return _apply_block(PROPENSITY_LOGGED, {})

    if config.ope.behaviour_policy == "estimated":
        policy = fit_behaviour_policy(
            split.train["__user_index"].to_numpy(dtype=np.int64),
            split.train["__action_index"].to_numpy(dtype=np.int64),
            n_users=n_users, n_actions=n_actions,
            shrinkage=config.ope.behaviour_policy_shrinkage,
        )
        for part in parts:
            if len(part) == 0:
                part["__propensity"] = np.zeros(0, dtype=np.float64)
                continue
            part["__propensity"] = policy.score(
                part["__user_index"].to_numpy(dtype=np.int64),
                part["__action_index"].to_numpy(dtype=np.int64),
            )
        diagnostics: dict[str, float] = {}
        # Only score the partitions this policy actually weights. With an
        # evaluation block the test rounds use the block's own propensities, so
        # reporting the log policy's fit on them would describe a model that is
        # not used and warn about a degeneracy that does not apply.
        scored = [("train", split.train)]
        if not evaluation_block:
            scored.append(("test", split.test))
        for label, part in scored:
            if len(part):
                diagnostics.update(behaviour_policy_diagnostics(
                    policy,
                    part["__user_index"].to_numpy(dtype=np.int64),
                    part["__action_index"].to_numpy(dtype=np.int64),
                    label,
                ))
        logger.warning(
            "Propensities are ESTIMATED, not logged (%s). Every importance-weighted result "
            "measures OPE error under an estimated behaviour policy, which is the realistic "
            "recommender setting but is NOT the same quantity as OPE with true propensities. "
            "Each Phase 4 row records this in 'propensity_source'.",
            policy.describe(),
        )
        return _apply_block(policy.describe(), diagnostics)

    # The evaluation block supplies the only propensities Phase 4 consumes, so
    # an importance-weighted estimator is already supported when one is present.
    scored_rounds_are_covered = bool(evaluation_block) and bool(len(split.test))

    if needs_propensity and not scored_rounds_are_covered:
        if config.ope.use_logged_propensity:
            raise ValueError(
                f"OPE estimators {config.ope.estimators} require logged propensities, but "
                f"column {propensity_column!r} is not present. Either supply it, set "
                f"ope.behaviour_policy: estimated to fit a behaviour policy from the log, or "
                f"restrict ope.estimators to ones that do not need it (e.g. 'exact' or 'dm')."
            )
        raise ValueError(
            f"OPE estimators {config.ope.estimators} consume propensities, but none are "
            f"logged and ope.behaviour_policy is {config.ope.behaviour_policy!r}. A constant "
            f"1/n_actions would make every importance weight identical, reducing SNIPW to a "
            f"mean and DR to the direct method under their own names."
        )

    for part in parts:
        part["__propensity"] = 1.0 / n_actions
    logger.warning(
        "No logged propensity for the sparse log, and none is consumed there (%s); writing "
        "the uniform constant 1/%d. It is labelled 'constant:uniform' in every result row "
        "so it can never be read as a behaviour policy.",
        config.ope.estimators, n_actions,
    )
    return _apply_block(f"constant:uniform(1/{n_actions})", {})


def summarise_dataset(prepared: PreparedDataset, config: ExperimentConfig) -> dict[str, object]:
    """Facts for the paper's dataset table, derived from what was actually loaded.

    Generating the table from the loaded data rather than transcribing it by
    hand is what keeps the reported statistics and the executed run in
    agreement.
    """
    frame = prepared.frame
    summary: dict[str, object] = {
        "dataset": config.dataset.name,
        "n_interactions": int(len(frame)),
        "n_users": len(prepared.user_indexer),
        "n_items": len(prepared.item_indexer),
        "n_actions": prepared.n_actions,
        "reward_definition": prepared.reward_definition,
        "mean_reward": float(frame["__reward"].mean()),
        "propensity_source": prepared.propensity_source,
        "propensity_min": float(frame["__propensity"].min()),
        "propensity_max": float(frame["__propensity"].max()),
        "propensity_is_constant": bool(frame["__propensity"].nunique() == 1),
        "n_train": len(prepared.split.train),
        "n_validation": len(prepared.split.validation),
        "n_test": len(prepared.split.test),
        "split_strategy": prepared.split.strategy,
        "evaluation_block": prepared.evaluation_block,
        "evaluation_regime": prepared.evaluation_regime,
        # The representation and the policy are trained on different row sets
        # whenever the log reaches outside the action space, so both are
        # reported. Quoting only one of them misdescribes the experiment.
        "representation_rows": prepared.representation_rows,
        "policy_rows": prepared.policy_rows,
        "graph_item_universe": (
            len(prepared.graph_item_indexer) if prepared.graph_item_indexer is not None
            else len(prepared.item_indexer)
        ),
        "subsampled_to": config.dataset.subsample_rows,
        "eval_subsampled_to": config.dataset.eval_subsample_rows,
        "graph_nodes": prepared.graph.num_nodes,
        "graph_edges": int(prepared.graph.edge_index.size(1)),
    }
    summary.update(prepared.propensity_diagnostics)
    return summary
