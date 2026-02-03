from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

try:
    import hnswlib
except Exception as e:  # pragma: no cover
    raise RuntimeError("hnswlib is required for vector retrieval") from e


Space = Literal["cosine", "l2"]


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom


@dataclass
class VectorIndex:
    dim: int
    space: Space = "cosine"
    M: int = 16
    ef_construction: int = 200
    _index: "hnswlib.Index" | None = None

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be > 0")
        if self._index is None:
            self._index = hnswlib.Index(space=self.space, dim=int(self.dim))

    @property
    def index(self) -> "hnswlib.Index":
        assert self._index is not None
        return self._index

    def init(self, max_elements: int) -> None:
        self.index.init_index(
            max_elements=int(max_elements),
            ef_construction=int(self.ef_construction),
            M=int(self.M),
        )
        # determinism-ish: keep single thread in tests/CI
        if hasattr(self.index, "set_num_threads"):
            self.index.set_num_threads(1)

    def set_ef(self, ef: int) -> None:
        self.index.set_ef(int(ef))

    def add_items(self, vectors: np.ndarray, ids: list[int] | np.ndarray) -> None:
        vec = np.asarray(vectors, dtype=np.float32)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        if vec.ndim != 2 or vec.shape[1] != self.dim:
            raise ValueError(f"Expected vectors shape (N,{self.dim}); got {vec.shape}")
        if self.space == "cosine":
            vec = _l2_normalize_rows(vec)
        vec = np.ascontiguousarray(vec)

        labels = np.asarray(ids, dtype=np.int32)
        if labels.ndim != 1 or labels.shape[0] != vec.shape[0]:
            raise ValueError("ids must be 1D and match vectors rows")
        labels = np.ascontiguousarray(labels)
        self.index.add_items(vec, labels)

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(query_vec, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.ndim != 2 or q.shape[1] != self.dim:
            raise ValueError(f"Expected query shape (N,{self.dim}); got {q.shape}")
        if self.space == "cosine":
            q = _l2_normalize_rows(q)
        q = np.ascontiguousarray(q)

        n = int(getattr(self.index, "get_current_count", lambda: 0)())
        k = min(int(top_k), n)
        if k <= 0:
            return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.float32)

        # hnswlib requires ef >= k
        self.index.set_ef(max(k, 50))

        labels, dists = self.index.knn_query(q, k=k)
        return labels[0].astype(np.int32, copy=False), dists[0].astype(np.float32, copy=False)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.index.save_index(str(path))

    @classmethod
    def load(cls, path: Path, dim: int, space: Space = "cosine") -> "VectorIndex":
        idx = cls(dim=int(dim), space=space)
        idx.index.load_index(str(path), max_elements=1)
        if hasattr(idx.index, "set_num_threads"):
            idx.index.set_num_threads(1)
        return idx

