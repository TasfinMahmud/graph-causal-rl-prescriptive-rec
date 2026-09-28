"""Open Bandit Dataset loading.

OBD logs a fashion e-commerce recommender under two known behaviour policies,
``random`` (uniform) and ``bts`` (Bernoulli Thompson Sampling), across three
campaigns. Critically it ships the *logged propensity* of each recorded action,
which is what makes unbiased off-policy evaluation possible on this data.

Two things the previous pipeline did are corrected here.

First, the logged propensity column was ignored and replaced by the constant
``1 / n_actions``. A constant propensity cancels out of a self-normalised
estimator, so the reported "SNIPW" reduced to the mean reward over rounds where
the policies happened to agree. This loader reads the real column and refuses to
proceed if it is absent.

Second, the behaviour policy was never pinned: the paper described the BTS
subset while the code path pointed at ``random/``. Policy and campaign are now
required arguments and are recorded in the returned metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

logger = get_logger(__name__)

VALID_POLICIES = ("random", "bts")
VALID_CAMPAIGNS = ("all", "men", "women")


@dataclass
class BanditFeedback:
    """Logged bandit feedback in the standard OPE format."""

    context: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    propensities: np.ndarray
    positions: np.ndarray | None
    n_actions: int
    timestamps: np.ndarray | None = None
    user_index: np.ndarray | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.actions)

    def validate(self) -> None:
        n = len(self.actions)
        for name, array in (
            ("context", self.context), ("rewards", self.rewards), ("propensities", self.propensities)
        ):
            if len(array) != n:
                raise ValueError(f"{name} has {len(array)} rows but actions has {n}")
        if self.actions.min(initial=0) < 0 or self.actions.max(initial=-1) >= self.n_actions:
            raise ValueError(
                f"actions must lie in [0, {self.n_actions}); observed "
                f"[{self.actions.min(initial=0)}, {self.actions.max(initial=-1)}]"
            )
        if np.any(self.propensities <= 0) or np.any(self.propensities > 1):
            raise ValueError(
                "propensities must lie in (0, 1]; a non-positive propensity violates overlap"
            )

    def subset(self, index: np.ndarray) -> BanditFeedback:
        return BanditFeedback(
            context=self.context[index],
            actions=self.actions[index],
            rewards=self.rewards[index],
            propensities=self.propensities[index],
            positions=None if self.positions is None else self.positions[index],
            n_actions=self.n_actions,
            timestamps=None if self.timestamps is None else self.timestamps[index],
            user_index=None if self.user_index is None else self.user_index[index],
            metadata=dict(self.metadata),
        )


def load_obd(
    root: Path,
    policy: str = "bts",
    campaign: str = "all",
    subsample_rows: int | None = None,
    seed: int = 42,
    chunk_size: int = 500_000,
) -> pd.DataFrame:
    """Load one OBD (policy, campaign) log as a DataFrame.

    Args:
        root: Directory containing ``{policy}/{campaign}/{campaign}.csv``.
        policy: ``"random"`` or ``"bts"``. Determines the behaviour policy whose
            propensities are logged, so it changes what the OPE estimates mean.
        subsample_rows: If set, draw this many rows uniformly without
            replacement. Subsampling is a legitimate way to keep runtimes
            manageable; it is recorded in the metadata and must be reported.

    Raises:
        FileNotFoundError: if the log is absent.
        ValueError: for an unknown policy/campaign, or a missing required column.
    """
    if policy not in VALID_POLICIES:
        raise ValueError(f"policy must be one of {VALID_POLICIES}, got {policy!r}")
    if campaign not in VALID_CAMPAIGNS:
        raise ValueError(f"campaign must be one of {VALID_CAMPAIGNS}, got {campaign!r}")

    path = Path(root) / policy / campaign / f"{campaign}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"OBD log not found at {path}. Expected layout: "
            f"<root>/{{random,bts}}/{{all,men,women}}/<campaign>.csv"
        )

    logger.info("Loading OBD policy=%s campaign=%s from %s", policy, campaign, path)

    size_gb = path.stat().st_size / 1e9
    if subsample_rows is not None and size_gb > 1.0:
        frame = _chunked_subsample(path, subsample_rows, seed, chunk_size)
    else:
        if size_gb > 2.0:
            logger.warning(
                "Reading %.1f GB into memory. Set dataset.subsample_rows to sample it "
                "in bounded memory instead.", size_gb,
            )
        frame = pd.read_csv(path, index_col=0)

    required = {"item_id", "click", "propensity_score", "position", "timestamp"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"OBD log at {path} is missing required columns {sorted(missing)}. "
            f"The propensity column in particular cannot be substituted with a constant "
            f"without invalidating every off-policy estimate."
        )

    if subsample_rows is not None and subsample_rows < len(frame):
        rng = np.random.default_rng(seed)
        index = np.sort(rng.choice(len(frame), size=subsample_rows, replace=False))
        frame = frame.iloc[index].reset_index(drop=True)
        logger.info("Subsampled %d of the logged rounds (seed=%d)", subsample_rows, seed)

    logger.info(
        "Loaded %d rounds | CTR=%.5f | %d distinct items | propensity range [%.5f, %.5f]",
        len(frame), frame["click"].mean(), frame["item_id"].nunique(),
        frame["propensity_score"].min(), frame["propensity_score"].max(),
    )
    return frame


def build_user_personas(frame: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
    """Group rows into user personas by their categorical feature tuple.

    OBD ships no user identifier, only hashed categorical context. Grouping
    identical tuples into a persona is a modelling choice made *by this
    pipeline*, not a property of the dataset, and the resulting persona count
    must be described that way wherever it is reported.

    The persona key is the joined feature tuple rather than a hash, so it is
    stable across processes -- unlike Python's randomised string ``hash()``.
    """
    missing = set(feature_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"persona feature columns not present: {sorted(missing)}")
    return frame[feature_columns].astype(str).agg("|".join, axis=1)


def to_bandit_feedback(
    frame: pd.DataFrame,
    n_actions: int,
    feature_columns: list[str],
    policy: str = "bts",
    campaign: str = "all",
    embedding_lookup: np.ndarray | None = None,
    user_index: np.ndarray | None = None,
) -> BanditFeedback:
    """Convert an OBD frame into :class:`BanditFeedback`.

    Args:
        embedding_lookup: Optional ``(n_users, dim)`` matrix. When supplied with
            ``user_index``, the context becomes the graph embedding of each
            round's user rather than its tabular features. This is the switch
            that makes the GNN state actually reach the agent.
    """
    actions = frame["item_id"].to_numpy(dtype=np.int64)
    if actions.max(initial=-1) >= n_actions:
        raise ValueError(
            f"item_id {int(actions.max())} exceeds the declared action space of {n_actions}. "
            f"Widen n_actions rather than clipping, which silently merges distinct items."
        )

    if embedding_lookup is not None:
        if user_index is None:
            raise ValueError("embedding_lookup requires user_index")
        context = embedding_lookup[user_index].astype(np.float32)
    else:
        available = [c for c in feature_columns if c in frame.columns]
        if not available:
            raise ValueError(
                f"none of the requested feature columns are present. "
                f"Requested: {feature_columns[:10]}"
            )
        context = frame[available].to_numpy(dtype=np.float32)

    feedback = BanditFeedback(
        context=np.nan_to_num(context),
        actions=actions,
        rewards=frame["click"].to_numpy(dtype=np.float64),
        propensities=frame["propensity_score"].to_numpy(dtype=np.float64),
        positions=frame["position"].to_numpy(dtype=np.int64) if "position" in frame else None,
        n_actions=n_actions,
        timestamps=frame["timestamp"].to_numpy() if "timestamp" in frame else None,
        user_index=user_index,
        metadata={
            "dataset": "OBD", "policy": policy, "campaign": campaign,
            "n_rounds": len(frame), "context_source":
                "gnn_embeddings" if embedding_lookup is not None else "raw_features",
        },
    )
    feedback.validate()
    return feedback


def _chunked_subsample(
    path: Path, n_target: int, seed: int, chunk_size: int
) -> pd.DataFrame:
    """Draw a uniform sample from a large CSV without loading it into memory.

    Two passes: the first counts rows, the second keeps a proportional share of
    each chunk. The result is a uniform sample of the whole file, which a
    ``nrows=N`` prefix would not be -- OBD is time-ordered, so a prefix is one
    contiguous period rather than a sample of the campaign.

    The peak memory is one chunk plus the accumulated sample, so a 6 GB log can
    be sampled inside a few hundred megabytes.
    """
    logger.info("Counting rows in %s (%.1f GB) for proportional sampling", path.name, path.stat().st_size / 1e9)
    total = 0
    for chunk in pd.read_csv(path, index_col=0, usecols=[0], chunksize=chunk_size):
        total += len(chunk)

    if n_target >= total:
        logger.info("Requested %d rows but the log has %d; loading all of it", n_target, total)
        return pd.read_csv(path, index_col=0)

    fraction = n_target / total
    rng = np.random.default_rng(seed)
    pieces = []
    kept = 0
    for i, chunk in enumerate(pd.read_csv(path, index_col=0, chunksize=chunk_size)):
        take = int(round(len(chunk) * fraction))
        if take:
            index = np.sort(rng.choice(len(chunk), size=min(take, len(chunk)), replace=False))
            pieces.append(chunk.iloc[index])
            kept += len(index)
        if (i + 1) % 5 == 0:
            logger.info("  sampled %d rows so far", kept)

    frame = pd.concat(pieces, ignore_index=True)
    logger.info(
        "Sampled %d of %d logged rounds (%.3f%%, seed=%d) in bounded memory",
        len(frame), total, 100 * len(frame) / total, seed,
    )
    return frame
