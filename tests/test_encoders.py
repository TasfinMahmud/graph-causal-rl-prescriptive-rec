"""Tests for deterministic encoding.

The central property: encoding must depend only on the data, never on process
state. The previous implementation used Python's ``hash()``, which is randomised
per process, so feature matrices differed on every run regardless of any seed.
"""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from gcrl.encoders import UNKNOWN_INDEX, DeterministicCategoricalEncoder, IdentifierIndexer


@pytest.fixture
def frame():
    return pd.DataFrame({"colour": ["red", "blue", "green", "red", "blue"], "n": [1, 2, 3, 4, 5]})


class TestDeterministicCategoricalEncoder:
    def test_is_stable_across_processes(self):
        """The property Python's hash() cannot provide."""
        script = (
            "import pandas as pd;"
            "from gcrl.encoders import DeterministicCategoricalEncoder as E;"
            "f=pd.DataFrame({'c':['red','blue','green']});"
            "print(list(E(['c']).fit_transform(f)['c']))"
        )
        runs = {
            subprocess.run(
                [sys.executable, "-c", script], capture_output=True, text=True, check=True
            ).stdout.strip()
            for _ in range(3)
        }
        assert len(runs) == 1, f"encoding differed across processes: {runs}"

    def test_baseline_python_hash_is_not_stable(self):
        """Demonstrates why the replacement was necessary."""
        script = "print(hash('red') % 100000)"
        runs = {
            subprocess.run(
                [sys.executable, "-c", script], capture_output=True, text=True, check=True
            ).stdout.strip()
            for _ in range(5)
        }
        assert len(runs) > 1, "expected PYTHONHASHSEED randomisation across processes"

    def test_encoding_is_order_independent(self, frame):
        a = DeterministicCategoricalEncoder(["colour"]).fit(frame)
        b = DeterministicCategoricalEncoder(["colour"]).fit(frame.iloc[::-1])
        assert a.vocabularies == b.vocabularies

    def test_unknown_values_map_to_sentinel(self, frame):
        encoder = DeterministicCategoricalEncoder(["colour"]).fit(frame)
        out = encoder.transform(pd.DataFrame({"colour": ["purple"], "n": [0]}))
        assert out["colour"].iloc[0] == UNKNOWN_INDEX

    def test_unknown_rate_is_measurable(self, frame):
        encoder = DeterministicCategoricalEncoder(["colour"]).fit(frame)
        held_out = pd.DataFrame({"colour": ["red", "purple", "cyan", "blue"]})
        assert encoder.unknown_rate(held_out, "colour") == pytest.approx(0.5)

    def test_partial_fit_preserves_existing_codes(self, frame):
        encoder = DeterministicCategoricalEncoder(["colour"]).fit(frame)
        before = dict(encoder.vocabularies["colour"])
        encoder.partial_fit(pd.DataFrame({"colour": ["cyan", "red"]}))
        for value, code in before.items():
            assert encoder.vocabularies["colour"][value] == code
        assert "cyan" in encoder.vocabularies["colour"]

    def test_roundtrips_through_disk(self, frame, tmp_path):
        encoder = DeterministicCategoricalEncoder(["colour"]).fit(frame)
        path = tmp_path / "enc.json"
        encoder.save(path)
        restored = DeterministicCategoricalEncoder.load(path)
        pd.testing.assert_frame_equal(encoder.transform(frame), restored.transform(frame))

    def test_missing_column_raises(self, frame):
        with pytest.raises(KeyError):
            DeterministicCategoricalEncoder(["absent"]).fit(frame)


class TestIdentifierIndexer:
    def test_produces_contiguous_indices(self):
        indexer = IdentifierIndexer("item").fit([10, 5, 99, 5])
        assert sorted(indexer.mapping.values()) == [0, 1, 2]

    def test_out_of_range_raises_instead_of_clipping(self):
        """np.clip(actions, 0, n-1) silently merges distinct items into one."""
        indexer = IdentifierIndexer("item").fit([0, 1, 2])
        with pytest.raises(KeyError, match="outside the fitted index"):
            indexer.transform([0, 1, 99])

    def test_non_strict_mode_marks_unknowns(self):
        indexer = IdentifierIndexer("item").fit([0, 1, 2])
        codes = indexer.transform([0, 99], strict=False)
        assert codes[1] == UNKNOWN_INDEX

    def test_inverse_transform_recovers_ids(self):
        indexer = IdentifierIndexer("user").fit([100, 200, 300])
        codes = indexer.transform([300, 100])
        np.testing.assert_array_equal(indexer.inverse_transform(codes), [300, 100])

    def test_roundtrips_through_disk(self, tmp_path):
        indexer = IdentifierIndexer("user").fit([7, 3, 9])
        path = tmp_path / "idx.json"
        indexer.save(path)
        restored = IdentifierIndexer.load(path)
        np.testing.assert_array_equal(indexer.transform([3, 9]), restored.transform([3, 9]))
