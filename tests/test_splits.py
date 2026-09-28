"""Tests for train/validation/test partitioning."""

import numpy as np
import pandas as pd
import pytest

from gcrl.config import SplitConfig
from gcrl.data.splits import split_frame


@pytest.fixture
def frame():
    n = 1000
    return pd.DataFrame({"timestamp": np.arange(n), "value": np.random.default_rng(0).random(n)})


class TestTemporalSplit:
    def test_partitions_are_disjoint_and_complete(self, frame):
        split = split_frame(frame, SplitConfig())
        total = len(split.train) + len(split.validation) + len(split.test)
        assert total == len(frame)

    def test_no_future_leaks_into_the_past(self, frame):
        split = split_frame(frame, SplitConfig())
        assert split.train["timestamp"].max() <= split.validation["timestamp"].min()
        assert split.validation["timestamp"].max() <= split.test["timestamp"].min()

    def test_respects_configured_ratios(self, frame):
        split = split_frame(frame, SplitConfig(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2))
        assert len(split.train) == pytest.approx(600, abs=2)
        assert len(split.validation) == pytest.approx(200, abs=2)

    def test_unsorted_input_is_sorted_first(self):
        frame = pd.DataFrame({"timestamp": [5, 1, 3, 2, 4, 0], "v": range(6)})
        split = split_frame(frame, SplitConfig(train_ratio=0.5, val_ratio=0.25, test_ratio=0.25))
        assert split.train["timestamp"].max() <= split.test["timestamp"].min()

    def test_missing_timestamp_column_raises(self):
        with pytest.raises(ValueError, match="requires column"):
            split_frame(pd.DataFrame({"v": range(10)}), SplitConfig(strategy="temporal"))

    def test_tiny_frame_raises_rather_than_emitting_empty_splits(self):
        with pytest.raises(ValueError, match="three non-empty"):
            split_frame(pd.DataFrame({"timestamp": [1, 2]}), SplitConfig())


class TestRandomSplit:
    def test_is_reproducible(self, frame):
        config = SplitConfig(strategy="random")
        a = split_frame(frame, config, seed=1)
        b = split_frame(frame, config, seed=1)
        pd.testing.assert_frame_equal(a.train, b.train)

    def test_different_seeds_differ(self, frame):
        config = SplitConfig(strategy="random")
        a = split_frame(frame, config, seed=1)
        b = split_frame(frame, config, seed=2)
        assert not a.train.equals(b.train)
