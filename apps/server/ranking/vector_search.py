from __future__ import annotations

from pathlib import Path

import numpy as np

from vector_index.hnsw_index import HNSWPackIndex


def hnsw_topn(pack_dir: Path, query_vec: np.ndarray, k: int) -> list[tuple[str, float]]:
    idx = HNSWPackIndex.load(pack_dir)
    return idx.query(query_vec.astype(np.float32, copy=False), k=int(k))

