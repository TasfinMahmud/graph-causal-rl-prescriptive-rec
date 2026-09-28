"""Deterministic encoding of categorical columns.

The previous implementation encoded non-numeric features with
``hash(str(x)) % 100000``. Python randomises string hashing per process, so
that mapping changed on every run and no seed could stabilise it: two runs of
the same script on the same data produced different feature matrices. This
module replaces it with a fitted, serialisable vocabulary.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

UNKNOWN_INDEX = -1


class DeterministicCategoricalEncoder:
    """Maps categorical values to stable integer codes.

    The vocabulary is built from sorted unique values, so it depends only on the
    data and never on process state or row order. Values unseen at fit time
    encode to :data:`UNKNOWN_INDEX` rather than raising, which keeps evaluation
    on a held-out split from crashing on a rare category, while still making the
    unknowns countable via :meth:`unknown_rate`.
    """

    def __init__(self, columns: Iterable[str]) -> None:
        self.columns: list[str] = list(columns)
        self.vocabularies: dict[str, dict[str, int]] = {}
        self._fitted = False

    def fit(self, frame: pd.DataFrame) -> DeterministicCategoricalEncoder:
        for column in self.columns:
            if column not in frame.columns:
                raise KeyError(f"column {column!r} not present in frame")
            values = frame[column].astype(str).unique()
            self.vocabularies[column] = {
                value: index for index, value in enumerate(sorted(values))
            }
        self._fitted = True
        return self

    def partial_fit(self, frame: pd.DataFrame) -> DeterministicCategoricalEncoder:
        """Extend the vocabulary from an additional chunk.

        Codes already assigned are never reassigned, so encoding a chunk before
        and after a ``partial_fit`` yields the same codes for known values.
        """
        for column in self.columns:
            if column not in frame.columns:
                raise KeyError(f"column {column!r} not present in frame")
            vocabulary = self.vocabularies.setdefault(column, {})
            new_values = sorted(set(frame[column].astype(str).unique()) - set(vocabulary))
            next_index = len(vocabulary)
            for offset, value in enumerate(new_values):
                vocabulary[value] = next_index + offset
        self._fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("encoder must be fitted before transform()")
        out = frame.copy()
        for column in self.columns:
            vocabulary = self.vocabularies[column]
            out[column] = (
                frame[column].astype(str).map(vocabulary).fillna(UNKNOWN_INDEX).astype(np.int64)
            )
        return out

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)

    def unknown_rate(self, frame: pd.DataFrame, column: str) -> float:
        """Fraction of rows in ``column`` whose value is outside the vocabulary."""
        if column not in self.vocabularies:
            raise KeyError(f"column {column!r} was not fitted")
        vocabulary = self.vocabularies[column]
        known = frame[column].astype(str).isin(vocabulary)
        return float(1.0 - known.mean()) if len(frame) else 0.0

    def cardinality(self, column: str) -> int:
        return len(self.vocabularies[column])

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"columns": self.columns, "vocabularies": self.vocabularies}
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: Path) -> DeterministicCategoricalEncoder:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        encoder = cls(payload["columns"])
        encoder.vocabularies = {
            column: {str(k): int(v) for k, v in vocab.items()}
            for column, vocab in payload["vocabularies"].items()
        }
        encoder._fitted = True
        return encoder


class IdentifierIndexer:
    """Maps dataset identifiers (user ids, item ids) to contiguous indices.

    Graph construction and embedding lookup both need a stable, contiguous index
    per entity. Clipping raw identifiers into a range -- as
    ``np.clip(actions, 0, n_actions - 1)`` previously did -- silently collapses
    every out-of-range identifier onto one action and corrupts the action space.
    This class fails loudly instead.
    """

    def __init__(self, name: str = "id") -> None:
        self.name = name
        self.mapping: dict[int, int] = {}
        self._inverse: np.ndarray | None = None

    def fit(self, identifiers: Iterable) -> IdentifierIndexer:
        unique = sorted({int(value) for value in identifiers})
        self.mapping = {value: index for index, value in enumerate(unique)}
        self._inverse = np.asarray(unique, dtype=np.int64)
        return self

    def transform(self, identifiers: Iterable, strict: bool = True) -> np.ndarray:
        values = np.asarray(list(identifiers), dtype=np.int64)
        codes = np.fromiter(
            (self.mapping.get(int(value), UNKNOWN_INDEX) for value in values),
            dtype=np.int64,
            count=len(values),
        )
        if strict and (codes == UNKNOWN_INDEX).any():
            missing = np.unique(values[codes == UNKNOWN_INDEX])[:10]
            raise KeyError(
                f"{self.name}: {int((codes == UNKNOWN_INDEX).sum())} identifiers are outside the "
                f"fitted index (examples: {missing.tolist()}). Refit the indexer on the full "
                f"identifier set rather than clipping out-of-range values."
            )
        return codes

    def fit_transform(self, identifiers: Iterable) -> np.ndarray:
        return self.fit(identifiers).transform(identifiers)

    def inverse_transform(self, codes: np.ndarray) -> np.ndarray:
        if self._inverse is None:
            raise RuntimeError("indexer must be fitted before inverse_transform()")
        return self._inverse[np.asarray(codes, dtype=np.int64)]

    def __len__(self) -> int:
        return len(self.mapping)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump({"name": self.name, "mapping": {str(k): v for k, v in self.mapping.items()}},
                      handle, indent=2)

    @classmethod
    def load(cls, path: Path) -> IdentifierIndexer:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        indexer = cls(payload["name"])
        indexer.mapping = {int(k): int(v) for k, v in payload["mapping"].items()}
        inverse = np.zeros(len(indexer.mapping), dtype=np.int64)
        for raw, code in indexer.mapping.items():
            inverse[code] = raw
        indexer._inverse = inverse
        return indexer
