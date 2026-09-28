"""Tests that seeding actually makes runs reproducible."""

import subprocess
import sys
from pathlib import Path

import numpy as np

from gcrl.seeding import new_rng, seed_everything


class TestSeedEverything:
    def test_numpy_is_reproducible(self):
        seed_everything(42)
        first = np.random.rand(5)
        seed_everything(42)
        np.testing.assert_array_equal(first, np.random.rand(5))

    def test_torch_is_reproducible(self):
        import torch
        seed_everything(42)
        first = torch.randn(5)
        seed_everything(42)
        assert torch.equal(first, torch.randn(5))

    def test_determinism_survives_a_different_string_hash_seed(self):
        """The property that actually matters, tested the way it must be.

        ``PYTHONHASHSEED`` is read by the interpreter at start-up, so no call
        made from inside a running process can pin it -- and this project does
        not try to. What it relies on instead is that nothing in ``gcrl``
        derives a value from ``hash()``.

        Asserting that ``seed_everything`` sets ``os.environ["PYTHONHASHSEED"]``
        and then reading the variable back would be a tautology: it would pass
        even if the setting had no effect, and would prove nothing about
        reproducibility.

        This test proves the real thing. It launches two interpreters with
        *genuinely different* hash seeds set in the environment before start-up,
        and requires the encoder vocabulary, the identifier index and a named
        RNG stream to come out identical. If any of them ever started depending
        on ``hash()``, this fails.
        """
        import json
        import os

        script = (
            "import json; import pandas as pd;"
            "from gcrl.encoders import DeterministicCategoricalEncoder, IdentifierIndexer;"
            "from gcrl.seeding import new_rng;"
            "frame = pd.DataFrame({'c': ['zeta', 'alpha', 'mu', 'alpha', 'beta']});"
            "enc = DeterministicCategoricalEncoder(['c']).fit(frame);"
            "idx = IdentifierIndexer('user').fit([7, 3, 11, 3]);"
            "print(json.dumps({"
            "  'vocab': enc.vocabularies['c'],"
            "  'index': [int(i) for i in idx.transform([11, 7, 3])],"
            "  'draw': new_rng(42, 'split').random(4).tolist(),"
            "}))"
        )
        root = str(Path(__file__).resolve().parent.parent)
        outputs = []
        for hash_seed in ("1", "999999"):
            env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": root}
            done = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, check=True, cwd=root, env=env,
            )
            outputs.append(json.loads(done.stdout))

        assert outputs[0] == outputs[1], (
            "output changed with the interpreter's string-hash seed, so something "
            f"in gcrl now depends on hash(): {outputs}"
        )


class TestNamedStreams:
    def test_streams_are_independent(self):
        assert not np.array_equal(
            new_rng(42, "split").random(5), new_rng(42, "negatives").random(5)
        )

    def test_same_stream_is_reproducible(self):
        np.testing.assert_array_equal(
            new_rng(42, "split").random(5), new_rng(42, "split").random(5)
        )
