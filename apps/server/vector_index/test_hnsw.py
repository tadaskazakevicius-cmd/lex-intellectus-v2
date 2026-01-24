from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow `python apps/server/vector_index/test_hnsw.py` to import sibling modules.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from hnsw_index import HNSWPackIndex  # noqa: E402


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def main() -> None:
    pack_id = "pack_test_vec"
    dim = 8
    n = 50

    rng = np.random.default_rng(0)
    base = rng.normal(size=(n, dim)).astype(np.float32)
    base = _normalize_rows(base)

    chunk_ids = [f"c{i:03d}" for i in range(n)]
    vectors_by_chunk_id = {cid: base[i] for i, cid in enumerate(chunk_ids)}

    out_dir = Path(".localdata") / "indices" / pack_id

    # Build & reload from disk.
    HNSWPackIndex.build(
        pack_id=pack_id,
        dim=dim,
        vectors=vectors_by_chunk_id,
        chunk_ids=None,
        out_dir=out_dir,
        space="cosine",
    )
    idx = HNSWPackIndex.load(out_dir)

    # 5 queries equal to existing vectors + small noise.
    picks = [0, 7, 13, 23, 49]
    ok = 0
    for p in picks:
        expected = chunk_ids[p]
        q = base[p] + rng.normal(scale=0.001, size=(dim,)).astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)

        top = idx.query(q, k=1)
        got = top[0][0]
        if got != expected:
            raise AssertionError(f"Top1 mismatch: expected={expected}, got={got}, res={top}")
        ok += 1

    print("HNSW B3 test OK")
    print(f"- pack_id={pack_id}")
    print(f"- dim={dim}")
    print(f"- vectors={n}")
    print(f"- queries_ok={ok}/{len(picks)}")


if __name__ == "__main__":
    main()

