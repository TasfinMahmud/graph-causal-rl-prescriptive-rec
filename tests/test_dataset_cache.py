"""Tests for the prepared-dataset disk cache.

The cache exists because each phase loads the dataset independently, so a
four-phase run over three variants and three seeds reads the source CSV 36
times. On OBD that source is 6.3 GB and the reads dominate total runtime.

The risk a cache introduces is worse than the cost it removes: a key that
misses a relevant field serves a dataset built under different settings, and
every downstream number becomes quietly wrong. These tests pin the key's
behaviour in both directions -- what must share an entry, and what must not.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from gcrl.cli import _dataset_cache_key
from gcrl.config import load_config

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


@pytest.fixture
def obd():
    return load_config(CONFIGS / "obd.yaml")


def test_key_is_deterministic(obd):
    assert _dataset_cache_key(obd, 42)[0] == _dataset_cache_key(obd, 42)[0]


def test_key_changes_with_seed(obd):
    assert _dataset_cache_key(obd, 42)[0] != _dataset_cache_key(obd, 43)[0]


def test_key_changes_with_dataset(obd):
    other = load_config(CONFIGS / "kuairec.yaml")
    assert _dataset_cache_key(obd, 42)[0] != _dataset_cache_key(other, 42)[0]


@pytest.mark.parametrize(
    "field,value",
    [
        ("subsample_rows", 12345),
        ("chunk_size", 1234),
        ("num_actions", 7),
        ("train_file", "random/all/all.csv"),
        ("reward_column", "something_else"),
    ],
)
def test_key_changes_with_any_dataset_field(obd, field, value):
    """A dataset field that alters the output must alter the key."""
    changed = copy.deepcopy(obd)
    setattr(changed.dataset, field, value)
    assert _dataset_cache_key(obd, 42)[0] != _dataset_cache_key(changed, 42)[0], (
        f"changing dataset.{field} did not change the cache key; "
        "the cache would serve a dataset built under different settings"
    )


def test_key_is_shared_across_ablation_variants(obd):
    """The ablation arms differ only in rl.*, so they must reuse one entry.

    This is the entire point of the cache: three variants x three seeds should
    load the CSV three times (once per seed), not thirty-six times.
    """
    for attr, value in (("state_source", "raw_features"), ("cate_reward_weight", 0.5)):
        variant = copy.deepcopy(obd)
        setattr(variant.rl, attr, value)
        assert _dataset_cache_key(obd, 42)[0] == _dataset_cache_key(variant, 42)[0], (
            f"rl.{attr} changed the dataset cache key; the ablation arms would "
            "each re-read the source CSV for no reason"
        )


def test_payload_is_json_serialisable_and_complete(obd):
    import dataclasses
    import json

    _, payload = _dataset_cache_key(obd, 42)
    json.dumps(payload)  # must not raise; it is written next to the entry
    recorded = set(payload["dataset"])
    expected = {f.name for f in dataclasses.fields(obd.dataset)}
    assert recorded == expected, f"cache key omits dataset fields: {expected - recorded}"


class TestCateArtefactPath:
    """Phase 2's output location must not depend on which ablation arm runs.

    Phase 2 never reads rl.*, so keying its artefacts by experiment_name made
    all three arms refit identical forests over millions of rows -- on OBD the
    single most expensive thing in the pipeline. These tests pin the fix.
    """

    def test_path_is_keyed_by_dataset_not_experiment_name(self, obd):
        from gcrl.cli import cate_artefact_path

        path = cate_artefact_path(obd, "XLearner", 42)
        assert obd.dataset.name in path.name
        assert "seed42" in path.name

    def test_path_is_identical_across_ablation_arms(self, obd):
        """main, ablation_raw and ablation_cate must share one CATE file."""
        from gcrl.cli import cate_artefact_path

        base = cate_artefact_path(obd, "XLearner", 42)
        for attr, value in (("state_source", "raw_features"), ("cate_reward_weight", 0.5)):
            arm = copy.deepcopy(obd)
            setattr(arm.rl, attr, value)
            arm.experiment_name = f"obd_{attr}"
            assert cate_artefact_path(arm, "XLearner", 42) == base, (
                f"rl.{attr} changed the CATE path; the arms would refit identical forests"
            )

    def test_path_changes_with_dataset_estimator_and_seed(self, obd):
        from gcrl.cli import cate_artefact_path

        base = cate_artefact_path(obd, "XLearner", 42)
        assert cate_artefact_path(obd, "SLearner", 42) != base
        assert cate_artefact_path(obd, "XLearner", 43) != base
        other = load_config(CONFIGS / "kuairec.yaml")
        assert cate_artefact_path(other, "XLearner", 42) != base
