"""Deterministic seeding for every source of randomness used in this project.

Reproducibility is a stated goal of the benchmark, so seeding is centralised
here rather than scattered across entry points.

**What is guaranteed.** :func:`seed_everything` pins the three generators this
project actually draws from -- ``random``, ``numpy.random`` and ``torch`` (plus
CUDA and deterministic cuDNN kernels) -- and :func:`new_rng` derives per-stream
generators reproducibly.

**What is not, and why that is sufficient.** Python's per-process randomisation
of ``str`` hashing is *not* pinned here, and cannot be: ``PYTHONHASHSEED`` is
read by the interpreter at start-up, so assigning it to ``os.environ`` from
inside a running process has no effect on that process's hashing. An earlier
version did exactly that and this docstring claimed the opposite. It is
sufficient because nothing in ``gcrl`` derives a value from ``hash()``:
:func:`new_rng` mixes stream names with BLAKE2b,
``gcrl.encoders.DeterministicCategoricalEncoder`` uses a fitted, sorted
vocabulary, and ``gcrl.data.obd`` hashes with ``hashlib``. Callers who do want
pinned string hashing must set ``PYTHONHASHSEED`` in the environment *before*
launching the interpreter; no call made from Python can do it for them.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np


def seed_everything(seed: int = 42, deterministic_torch: bool = True) -> int:
    """Seed Python, NumPy and (if installed) PyTorch.

    Args:
        seed: Seed applied to every generator.
        deterministic_torch: Request deterministic cuDNN kernels. This costs
            throughput and is not supported by every operator, so it can be
            disabled when exact GPU determinism is not required.

    Returns:
        The seed that was applied, so callers can log it.

    Note:
        This does **not** touch ``PYTHONHASHSEED``. Setting it here would be
        inert for this process (the interpreter reads it at start-up) while
        still leaking into any subprocess, which is worse than doing nothing:
        it would look like a guarantee and behave like a side effect. See the
        module docstring for what is pinned and why it covers this project.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


def new_rng(seed: int, stream: str | None = None) -> np.random.Generator:
    """Return an independent NumPy generator for a named stream.

    Deriving a child generator per stream (data splitting, negative sampling,
    the random policy baseline) keeps those streams from consuming each other's
    draws, so adding a call site does not perturb unrelated results.

    The stream name is mixed in with BLAKE2b rather than :func:`hash`, because
    the built-in hash of a string is randomised per process and would make the
    derived generator differ between runs.
    """
    if stream is None:
        return np.random.default_rng(seed)
    digest = hashlib.blake2b(stream.encode("utf-8"), digest_size=8).digest()
    offset = int.from_bytes(digest, "little")
    return np.random.default_rng((seed + offset) % (2**32))
