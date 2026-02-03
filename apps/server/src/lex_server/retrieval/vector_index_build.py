from __future__ import annotations

"""
E3b: Offline HNSW vector index build helper.

Reads chunk text from SQLite, embeds via ONNX (CPU) or any compatible embedder,
builds an hnswlib index, and writes:
- index binary
- idmap JSON: { "<int_id>": "<chunk_id_str>" }

Design notes:
- We intentionally do NOT build the index on server startup.
- For production DB, we index `document_chunks` using its SQLite `rowid` as the stable int label.
- For tests/CI, we also support a minimal table schema: (id INTEGER PRIMARY KEY, chunk_id TEXT, content TEXT, practice_doc_id TEXT).
"""

import argparse
import json
import os
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .vector_index import VectorIndex


def _default_db_path() -> Path:
    local = Path.cwd() / ".localdata" / "app.db"
    if local.exists():
        return local
    from ..paths import get_paths

    return get_paths().data_dir / "app.db"


def _default_indices_dir() -> Path:
    try:
        from ..paths import get_paths

        return get_paths().data_dir / "indices"
    except Exception:
        return Path.cwd() / ".localdata" / "indices"


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
    where_sql: str | None = None,
    params: tuple = (),
) -> Iterable[tuple[int, str, str]]:
    """
    Yield (chunk_int_id, chunk_id_str, content_text).

    Production path:
    - table: document_chunks(id TEXT, text TEXT, document_id INTEGER, ...)
    - chunk_int_id: document_chunks.rowid (int)
    - chunk_id_str: document_chunks.id (TEXT)
    - content_text: document_chunks.text

    Test path (minimal schema):
    - table: chunks(id INTEGER PRIMARY KEY, chunk_id TEXT, content TEXT, practice_doc_id TEXT)
    - chunk_int_id: chunks.id
    - chunk_id_str: chunks.chunk_id
    - content_text: chunks.content
    """
    where_sql = (where_sql or "").strip()
    params = params or ()

    if _table_exists(conn, "document_chunks") and _has_columns(conn, "document_chunks", {"id", "text"}):
        sql = """
        SELECT dc.rowid AS int_id, dc.id AS chunk_id, dc.text AS content
        FROM document_chunks dc
        """
        if where_sql:
            sql += f" WHERE {where_sql}"
        sql += " ORDER BY dc.rowid ASC"
        for int_id, chunk_id, content in conn.execute(sql, params):
            yield int(int_id), str(chunk_id), str(content)
        return

    # Minimal test table
    if _table_exists(conn, "chunks") and _has_columns(conn, "chunks", {"id", "chunk_id", "content"}):
        sql = "SELECT id, chunk_id, content FROM chunks"
        if where_sql:
            sql += f" WHERE {where_sql}"
        sql += " ORDER BY id ASC"
        for int_id, chunk_id, content in conn.execute(sql, params):
            yield int(int_id), str(chunk_id), str(content)
        return

    raise ValueError("No supported chunk table found (expected document_chunks or chunks).")


def _count_chunks(conn: sqlite3.Connection, where_sql: str | None, params: tuple) -> int:
    where_sql = (where_sql or "").strip()
    params = params or ()
    if _table_exists(conn, "document_chunks") and _has_columns(conn, "document_chunks", {"id", "text"}):
        sql = "SELECT COUNT(*) FROM document_chunks dc"
        if where_sql:
            sql += f" WHERE {where_sql}"
        return int(conn.execute(sql, params).fetchone()[0])
    if _table_exists(conn, "chunks") and _has_columns(conn, "chunks", {"id", "chunk_id", "content"}):
        sql = "SELECT COUNT(*) FROM chunks"
        if where_sql:
            sql += f" WHERE {where_sql}"
        return int(conn.execute(sql, params).fetchone()[0])
    raise ValueError("No supported chunk table found (expected document_chunks or chunks).")


def load_idmap(path: Path) -> dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("idmap json must be an object")
    out: dict[int, str] = {}
    for k, v in raw.items():
        out[int(k)] = str(v)
    return out


def build_vector_index(
    conn: sqlite3.Connection,
    embedder: Any,
    out_index_path: Path,
    out_idmap_path: Path,
    *,
    space: str = "cosine",
    batch_size: int = 128,
    M: int = 16,
    ef_construction: int = 200,
    where_sql: str | None = None,
    params: tuple = (),
) -> None:
    """
    Build and write an HNSW index + idmap for all chunks (optionally filtered by where_sql).

    `embedder` must provide:
    - embed_texts(texts: list[str]) -> np.ndarray float32 shape (n,d)
    """
    n = _count_chunks(conn, where_sql, params)
    if n <= 0:
        raise ValueError("No chunks to index")

    # Take one batch to determine embedding dim
    it = iter(iter_chunks(conn, where_sql=where_sql, params=params))
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
    # Note: VectorIndex.add_items() handles L2 normalization for cosine space

    idx = VectorIndex(dim=dim, space=space)  # type: ignore[arg-type]
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

    # Continue with remaining rows, embedding in batches.
    pending: list[tuple[int, str, str]] = []
    for row in it:
        pending.append(row)
        if len(pending) >= int(batch_size):
            texts = [t for _i, _cid, t in pending]
            vec = np.asarray(embedder.embed_texts(texts), dtype=np.float32)
            if vec.shape != (len(texts), dim):
                raise ValueError(f"Embedder output shape mismatch: got {vec.shape}, expected ({len(texts)},{dim})")
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
        add_batch(pending, vec)
        processed += len(pending)

    idx.save(out_index_path)
    out_idmap_path.parent.mkdir(parents=True, exist_ok=True)
    out_idmap_path.write_text(json.dumps({str(k): v for k, v in idmap.items()}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Index built. chunks={processed}, dim={dim}, index={out_index_path}, idmap={out_idmap_path}")


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
    p.add_argument("--practice-doc-id", type=str, default=None, help="Filter: only one practice doc (case_documents.id)")
    args = p.parse_args()

    dbp = args.db or Path(os.environ.get("LEX_DB_PATH", str(_default_db_path())))
    indices_dir = _default_indices_dir()
    out_index = args.out or Path(os.environ.get("LEX_VECTOR_INDEX_PATH", str(indices_dir / "chunks_hnsw.bin")))
    out_idmap = args.idmap or Path(os.environ.get("LEX_VECTOR_IDMAP_PATH", str(indices_dir / "chunks_hnsw_idmap.json")))

    onnx_path = args.onnx or (Path(os.environ["LEX_EMBEDDING_MODEL_PATH"]) if "LEX_EMBEDDING_MODEL_PATH" in os.environ else None)
    if onnx_path is None:
        raise SystemExit("Missing --onnx or LEX_EMBEDDING_MODEL_PATH")

    from .embedder_onnx import OnnxEmbedder

    embedder = OnnxEmbedder(onnx_path)

    conn = sqlite3.connect(dbp)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        where_sql = None
        params: tuple = ()
        if args.practice_doc_id:
            # Production schema: document_chunks.document_id -> case_documents.id
            # We filter by joining in WHERE for document_chunks path.
            if _table_exists(conn, "document_chunks"):
                where_sql = "dc.document_id = ?"
                params = (int(args.practice_doc_id),)
            else:
                where_sql = "practice_doc_id = ?"
                params = (str(args.practice_doc_id),)

        build_vector_index(
            conn,
            embedder,
            out_index,
            out_idmap,
            space=str(args.space),
            batch_size=int(args.batch_size),
            M=int(args.M),
            ef_construction=int(args.ef_construction),
            where_sql=where_sql,
            params=params,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()

