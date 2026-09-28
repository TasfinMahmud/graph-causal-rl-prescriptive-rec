"""Train / validation / test partitioning.

Every reported metric must come from data the model did not train on. The
pipeline therefore produces three disjoint splits and the test split is read
exactly once, in Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import SplitConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DataSplit:
    """Three disjoint frames plus the boundaries used to produce them."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    strategy: str
    boundaries: tuple[float, float]

    def describe(self) -> str:
        total = len(self.train) + len(self.validation) + len(self.test)
        return (
            f"{self.strategy} split -> train={len(self.train):,} "
            f"({len(self.train)/total:.1%}), val={len(self.validation):,} "
            f"({len(self.validation)/total:.1%}), test={len(self.test):,} "
            f"({len(self.test)/total:.1%})"
        )


def split_frame(frame: pd.DataFrame, config: SplitConfig, seed: int = 42) -> DataSplit:
    """Partition ``frame`` according to ``config``.

    For the temporal strategy the frame is sorted by timestamp and cut at the
    ratio boundaries, so every training row precedes every validation row and
    every validation row precedes every test row. The ordering is asserted
    afterwards rather than assumed.

    Raises:
        ValueError: if the frame is too small to produce three non-empty splits,
            or if the timestamp column is missing for a temporal split.
    """
    config.validate()
    n_rows = len(frame)
    if n_rows < 3:
        raise ValueError(f"cannot split {n_rows} rows into three non-empty partitions")

    if config.strategy == "temporal":
        if config.timestamp_column not in frame.columns:
            raise ValueError(
                f"temporal split requires column {config.timestamp_column!r}; "
                f"available columns: {list(frame.columns)[:20]}"
            )
        ordered = frame.sort_values(config.timestamp_column, kind="mergesort").reset_index(drop=True)
    else:
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(n_rows)
        ordered = frame.iloc[permutation].reset_index(drop=True)

    train_end = int(n_rows * config.train_ratio)
    val_end = train_end + int(n_rows * config.val_ratio)

    train_end = max(1, min(train_end, n_rows - 2))
    val_end = max(train_end + 1, min(val_end, n_rows - 1))

    train = ordered.iloc[:train_end].copy()
    validation = ordered.iloc[train_end:val_end].copy()
    test = ordered.iloc[val_end:].copy()

    if config.strategy == "temporal":
        _assert_temporal_ordering(train, validation, test, config.timestamp_column)

    split = DataSplit(
        train=train,
        validation=validation,
        test=test,
        strategy=config.strategy,
        boundaries=(train_end / n_rows, val_end / n_rows),
    )
    logger.info(split.describe())
    return split


def _assert_temporal_ordering(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, column: str
) -> None:
    """Verify no split contains a timestamp from a later split.

    Ties on the boundary timestamp are reported rather than tolerated silently,
    because a heavily tied timestamp column means the temporal split is not
    actually separating the periods.
    """
    if train.empty or validation.empty or test.empty:
        raise ValueError("temporal split produced an empty partition; increase the dataset size")

    train_max, val_min = train[column].max(), validation[column].min()
    val_max, test_min = validation[column].max(), test[column].min()

    if train_max > val_min:
        raise AssertionError(
            f"temporal leakage: max train {column} ({train_max}) exceeds min validation ({val_min})"
        )
    if val_max > test_min:
        raise AssertionError(
            f"temporal leakage: max validation {column} ({val_max}) exceeds min test ({test_min})"
        )
    if train_max == val_min or val_max == test_min:
        logger.warning(
            "Split boundaries fall on tied %s values; the partitions are disjoint by row "
            "but not strictly separated in time.", column
        )
