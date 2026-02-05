from __future__ import annotations

"""
E3b: Offline HNSW vector index build helper.

Builds an hnswlib index for chunks stored in SQLite.

Design notes:
- We intentionally do NOT build the index on server startup.
- Production DB: we index `document_chunks` using SQLite `rowid` as the int label.
- Optional filter: case_id (by joining document_chunks -> case_documents).
"""

import argparse
import json
import os
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .vector_index import VectorIndex

Space = Literal["cosine", "l2"]


def _default_db_path() -> Path:
    # Must match the app's convention: prefer legacy .localdata/app.db if it exists, else A2 data_dir/app.db
    local = Path.cwd() / ".localdata" / "app.db"
    if local.exists():
        return local
    from ..paths import get_paths

    return get_paths().data_dir / "app.db"


def _default_indices_dir() -> Path:
    from ..paths import get_paths

    return get_paths().data_dir / "indices"


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    r = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?;",
        (name,),
    ).fetchone()
    return r is not None


def _has_columns(conn: sqlite3.Connection, table: str, cols: set[str]) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    got = {str(r[1]) for r in rows}
    return cols.issubset(got)


def iter_chunks(
    conn: sqlite3.Connection,
    *,
    case_id: str | None = None,
) -> Iterable[tuple[int, str, str]]:
    """
    Yield (chunk_int_id, chunk_id_str, content_text).

    Production schema:
    - document_chunks(id TEXT, text TEXT, document_id INTEGER, ...)
    - We use document_chunks.rowid (int) as the HNSW label
    - We store document_chunks.id as the external chunk_id_str
    - We store document_chunks.text as content_text

    Optional filter:
    - by case_id via JOIN document_chunks.document_id -> case_documents.id and case_documents.case_id
    """
    if _table_exists(conn, "document_chunks") and _has_columns(conn, "document_chunks", {"id", "text"}):
        if case_id:
            if not _table_exists(conn, "case_documents") or not _has_columns(conn, "case_documents", {"id", "case_id"}):
                raise ValueError("case_documents table missing or does not have required columns (id, case_id)")
            sql = """
            SELECT dc.rowid AS int_id, dc.id AS chunk_id, dc.text AS content
            FROM document_chunks dc
            JOIN case_documents cd ON cd.id = dc.document_id
            WHERE cd.case_id = ?
            ORDER BY dc.rowid ASC
            """
            for int_id, chunk_id, content in conn.execute(sql, (case_id,)):
                yield int(int_id), str(chunk_id), str(content)
            return

        sql = """
        SELECT dc.rowid AS int_id, dc.id AS chunk_id, dc.text AS content
        FROM document_chunks dc
        ORDER BY dc.rowid ASC
        """
        for int_id, chunk_id, content in conn.execute(sql):
            yield int(int_id), str(chunk_id), str(content)
        return

    # Minimal fallback schema (tests/dev)
    if _table_exists(conn, "chunks") and _has_columns(conn, "chunks", {"id", "chunk_id", "content"}):
        sql = "SELECT id, chunk_id, content FROM chunks ORDER BY id ASC"
        for int_id, chunk_id, content in conn.execute(sql):
            yield int(int_id), str(chunk_id), str(content)
        return

    raise ValueError("No supported chunk table found (expected document_chunks or chunks).")


def _count_chunks(conn: sqlite3.Connection, *, case_id: str | None = None) -> int:
    if _table_exists(conn, "document_chunks") and _has_columns(conn, "document_chunks", {"id", "text"}):
        if case_id:
            sql = """
            SELECT COUNT(*)
            FROM document_chunks dc
            JOIN case_documents cd ON cd.id = dc.document_id
            WHERE cd.case_id = ?
            """
            return int(conn.execute(sql, (case_id,)).fetchone()[0])
        return int(conn.execute("SELECT COUNT(*) FROM document_chunks;").fetchone()[0])

    if _table_exists(conn, "chunks") and _has_columns(conn, "chunks", {"id", "chunk_id", "content"}):
        return int(conn.execute("SELECT COUNT(*) FROM chunks;").fetchone()[0])

    raise ValueError("No supported chunk table found (expected document_chunks or chunks).")


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom


def build_vector_index(
    conn: sqlite3.Connection,
    embedder: Any,
    out_index_path: Path,
    out_idmap_path: Path,
    *,
    space: Space = "cosine",
    batch_size: int = 128,
    M: int = 16,
    ef_construction: int = 200,
    case_id: str | None = None,
) -> int:
    """
    Build and write an HNSW index + idmap.

    `embedder` must provide:
    - embed_texts(texts: list[str]) -> np.ndarray float32 shape (n,d)

    Returns number of indexed chunks.
    """
    n = _count_chunks(conn, case_id=case_id)
    if n <= 0:
        raise ValueError("No chunks to index")

    it = iter(iter_chunks(conn, case_id=case_id))

    # prime first batch to get dim
    first_batch: list[tuple[int, str, str]] = []
    for _ in range(min(int(batch_size), n)):
        try:
            first_batch.append(next(it))
        except StopIteration:
            break
    if not first_batch:
        raise ValueError("No chunks to index")

    texts0 = [t for _i, _cid, t in first_batch]
    vec0 = np.asarray(embedder.embed_texts(texts0), dtype=np.float32)
    if vec0.ndim != 2 or vec0.shape[0] != len(texts0):
        raise ValueError(f"Unexpected embedder output shape: {vec0.shape}")
    dim = int(vec0.shape[1])
    if space == "cosine":
        vec0 = _l2_normalize_rows(vec0)

    idx = VectorIndex(dim=dim, space=space)
    idx.M = int(M)
    idx.ef_construction = int(ef_construction)
    idx.init(max_elements=n)

    idmap: dict[int, str] = {}

    def add_batch(batch: list[tuple[int, str, str]], vecs: np.ndarray) -> None:
        int_ids = [int(x[0]) for x in batch]
        for int_id, chunk_id, _t in batch:
            idmap[int(int_id)] = str(chunk_id)
        idx.add_items(vecs, np.asarray(int_ids, dtype=np.int32))

    add_batch(first_batch, vec0)
    processed = len(first_batch)

    pending: list[tuple[int, str, str]] = []
    for row in it:
        pending.append(row)
        if len(pending) >= int(batch_size):
            texts = [t for _i, _cid, t in pending]
            vec = np.asarray(embedder.embed_texts(texts), dtype=np.float32)
            if vec.shape != (len(texts), dim):
                raise ValueError(f"Embedder output shape mismatch: got {vec.shape}, expected ({len(texts)},{dim})")
            if space == "cosine":
                vec = _l2_normalize_rows(vec)
            add_batch(pending, vec)
            processed += len(pending)
            if processed % 500 == 0:
                print(f"Indexed {processed}/{n} chunks...")
            pending = []

    if pending:
        texts = [t for _i, _cid, t in pending]
        vec = np.asarray(embedder.embed_texts(texts), dtype=np.float32)
        if vec.shape != (len(texts), dim):
            raise ValueError(f"Embedder output shape mismatch: got {vec.shape}, expected ({len(texts)},{dim})")
        if space == "cosine":
            vec = _l2_normalize_rows(vec)
        add_batch(pending, vec)
        processed += len(pending)

    out_index_path.parent.mkdir(parents=True, exist_ok=True)
    out_idmap_path.parent.mkdir(parents=True, exist_ok=True)

    idx.save(out_index_path)
    out_idmap_path.write_text(
        json.dumps({str(k): v for k, v in idmap.items()}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Index built. chunks={processed}, dim={dim}, index={out_index_path}, idmap={out_idmap_path}")
    return processed


def main() -> None:
    p = argparse.ArgumentParser(description="Build HNSW vector index from SQLite chunks (E3b).")
    p.add_argument("--db", type=Path, default=None, help="Path to sqlite db (default: auto)")
    p.add_argument("--onnx", type=Path, default=None, help="Path to ONNX embedding model")
    p.add_argument("--out", type=Path, default=None, help="Output index path")
    p.add_argument("--idmap", type=Path, default=None, help="Output idmap json path")
    p.add_argument("--space", type=str, default=os.environ.get("LEX_VECTOR_SPACE", "cosine"))
    p.add_argument("--batch-size", type=int, default=int(os.environ.get("LEX_VECTOR_BATCH_SIZE", "128")))
    p.add_argument("--M", type=int, default=int(os.environ.get("LEX_VECTOR_M", "16")))
    p.add_argument("--ef-construction", type=int, default=int(os.environ.get("LEX_VECTOR_EF_CONSTRUCTION", "200")))
    p.add_argument("--case-id", type=str, default=None, help="Only index chunks belonging to this case_id")
    args = p.parse_args()

    dbp = args.db or Path(os.environ.get("LEX_DB_PATH", str(_default_db_path())))
    indices_dir = _default_indices_dir()

    # IMPORTANT: align with retrieval endpoint env names
    out_index = args.out or Path(os.environ.get("LEX_VECTOR_INDEX_PATH", str(indices_dir / "chunks_hnsw.bin")))
    out_idmap = args.idmap or Path(os.environ.get("LEX_VECTOR_IDMAP_PATH", str(indices_dir / "chunks_hnsw_idmap.json")))

    # Align with endpoint env name
    onnx_path = args.onnx or (Path(os.environ["LEX_EMBED_ONNX_MODEL"]) if "LEX_EMBED_ONNX_MODEL" in os.environ else None)
    if onnx_path is None:
        raise SystemExit("Missing --onnx or LEX_EMBED_ONNX_MODEL")

    from .embedder_onnx import OnnxEmbedder

    embedder = OnnxEmbedder(onnx_path)

    conn = sqlite3.connect(dbp)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        build_vector_index(
            conn,
            embedder,
            out_index,
            out_idmap,
            space=str(args.space),  # type: ignore[arg-type]
            batch_size=int(args.batch_size),
            M=int(args.M),
            ef_construction=int(args.ef_construction),
            case_id=(str(args.case_id) if args.case_id else None),
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
