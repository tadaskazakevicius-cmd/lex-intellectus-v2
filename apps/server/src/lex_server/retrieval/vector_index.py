from __future__ import annotations

import json
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


def _meta_path(index_path: Path) -> Path:
    # index.bin -> index.meta.json
    if index_path.suffix.lower() == ".bin":
        return index_path.with_suffix(".meta.json")
    return index_path.with_suffix(index_path.suffix + ".meta.json")


def _resolve_index_file(path: Path) -> Path:
    """
    Accepts:
    - a directory -> use directory/index.bin
    - a file path:
        - if endswith .bin -> use it
        - else -> treat as "base" and append .bin
    """
    if path.exists() and path.is_dir():
        return path / "index.bin"
    if path.suffix.lower() == ".bin":
        return path
    # allow passing ".../index" or ".../index.hnsw" etc -> normalize to ".../index.bin"
    return path.with_suffix(".bin")


@dataclass
class VectorIndex:
    dim: int
    space: Space = "cosine"
    M: int = 16
    ef_construction: int = 200
    _index: "hnswlib.Index" | None = None

    def __post_init__(self) -> None:
        if int(self.dim) <= 0:
            raise ValueError("dim must be > 0")
        if self.space not in ("cosine", "l2"):
            raise ValueError(f"space must be 'cosine' or 'l2', got {self.space!r}")
        if self._index is None:
            self._index = hnswlib.Index(space=self.space, dim=int(self.dim))

    @property
    def index(self) -> "hnswlib.Index":
        assert self._index is not None
        return self._index

    def _set_single_thread(self) -> None:
        # determinism-ish: keep single thread in tests/CI
        if hasattr(self.index, "set_num_threads"):
            self.index.set_num_threads(1)

    def init(self, max_elements: int) -> None:
        me = int(max_elements)
        if me < 1:
            raise ValueError("max_elements must be >= 1")
        self.index.init_index(
            max_elements=me,
            ef_construction=int(self.ef_construction),
            M=int(self.M),
        )
        self._set_single_thread()

    def count(self) -> int:
        return int(getattr(self.index, "get_current_count", lambda: 0)())

    def is_empty(self) -> bool:
        return self.count() <= 0

    def set_ef(self, ef: int) -> None:
        self.index.set_ef(int(ef))

    def add_items(self, vectors: np.ndarray, ids: list[int] | np.ndarray) -> None:
        vec = np.asarray(vectors, dtype=np.float32)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        if vec.ndim != 2 or vec.shape[1] != int(self.dim):
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
        if q.ndim != 2 or q.shape[1] != int(self.dim):
            raise ValueError(f"Expected query shape (N,{self.dim}); got {q.shape}")
        if self.space == "cosine":
            q = _l2_normalize_rows(q)
        q = np.ascontiguousarray(q)

        n = self.count()
        k = min(int(top_k), n)
        if k <= 0:
            return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.float32)

        # hnswlib requires ef >= k
        self.index.set_ef(max(k, 50))

        labels, dists = self.index.knn_query(q, k=k)
        return labels[0].astype(np.int32, copy=False), dists[0].astype(np.float32, copy=False)

    def save(self, path: Path) -> Path:
        """
        Saves:
          - <path>.bin (hnswlib index)
          - <path>.meta.json (dim/space/count)
        Returns the resolved index file path.
        """
        index_file = _resolve_index_file(Path(path))
        index_file.parent.mkdir(parents=True, exist_ok=True)

        # write index
        self.index.save_index(str(index_file))

        # write metadata (handy for safe load)
        meta = {
            "dim": int(self.dim),
            "space": self.space,
            "count": self.count(),
            "M": int(self.M),
            "ef_construction": int(self.ef_construction),
        }
        _meta_path(index_file).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return index_file

    @classmethod
    def load(cls, path: Path, dim: int | None = None, space: Space | None = None) -> "VectorIndex":
        """
        Loads index from:
          - directory -> directory/index.bin
          - file path -> normalized to .bin

        If meta exists, dim/space default from it unless explicitly provided.
        """
        index_file = _resolve_index_file(Path(path))
        meta_file = _meta_path(index_file)

        meta: dict | None = None
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                meta = None

        final_dim = int(dim if dim is not None else (meta.get("dim") if isinstance(meta, dict) else 0))
        final_space = (space if space is not None else (meta.get("space") if isinstance(meta, dict) else "cosine"))

        if final_dim <= 0:
            raise ValueError(
                f"VectorIndex.load requires dim>0. "
                f"Pass dim explicitly or ensure meta exists at {str(meta_file)}."
            )
        if final_space not in ("cosine", "l2"):
            raise ValueError(f"Invalid space: {final_space!r}")

        if not index_file.exists() or not index_file.is_file():
            raise FileNotFoundError(f"Index file not found: {str(index_file)}")

        idx = cls(dim=final_dim, space=final_space)

        # hnswlib needs max_elements param; use meta count if present, else 1
        max_el = 1
        if isinstance(meta, dict):
            try:
                max_el = max(1, int(meta.get("count") or 1))
            except Exception:
                max_el = 1

        idx.index.load_index(str(index_file), max_elements=max_el)
        idx._set_single_thread()
        return idx
