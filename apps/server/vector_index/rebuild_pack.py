from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

try:
    from .hnsw_index import HNSWPackIndex
    from .paths import pack_index_dir
except ImportError:  # pragma: no cover
    # Support running as plain scripts (no package context).
    from hnsw_index import HNSWPackIndex  # type: ignore[no-redef]
    from paths import pack_index_dir  # type: ignore[no-redef]


def rebuild_pack_index(
    pack_id: str,
    dim: int,
    vectors_by_chunk_id: dict[str, "np.ndarray"],
    indices_root: Path,
) -> Path:
    """
    B3 MVP strategy: rebuild per pack apply.

    This intentionally does NOT do incremental updates. Whenever a pack is (re)applied,
    we wipe the pack's index directory and build a fresh index from the vectors provided
    by the caller.
    """
    pack_dir = pack_index_dir(indices_root, pack_id)
    if pack_dir.exists():
        shutil.rmtree(pack_dir)
    pack_dir.mkdir(parents=True, exist_ok=True)

    HNSWPackIndex.build(
        pack_id=pack_id,
        dim=dim,
        vectors=vectors_by_chunk_id,
        chunk_ids=None,
        out_dir=pack_dir,
        space="cosine",
    )
    return pack_dir

