from __future__ import annotations

import hashlib
import sqlite3

import numpy as np
import pytest

from lex_server.retrieval.vector_index import VectorIndex
from lex_server.retrieval.vector_retrieval import VectorFilter, vector_retrieve, vector_search


class FakeEmbedder:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def embed_text(self, text: str) -> np.ndarray:
        # Deterministic embedding from sha256(text) -> float32 vector in [-1,1], then L2 normalize.
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vals = np.frombuffer(h[: self.dim], dtype=np.uint8).astype(np.float32)
        v = (vals / 127.5) - 1.0
        denom = np.linalg.norm(v) + 1e-12
        return (v / denom).astype(np.float32)


def _setup_db() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.execute("PRAGMA foreign_keys = ON;")
    con.executescript(
        """
        CREATE TABLE case_documents (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          case_id TEXT NOT NULL,
          original_name TEXT NOT NULL,
          mime TEXT NOT NULL,
          size_bytes INTEGER NOT NULL,
          sha256_hex TEXT NOT NULL,
          storage_relpath TEXT NOT NULL,
          created_at_utc TEXT NOT NULL
        );

        CREATE TABLE document_chunks (
          id TEXT PRIMARY KEY,
          document_id INTEGER NOT NULL REFERENCES case_documents(id) ON DELETE CASCADE,
          chunk_index INTEGER NOT NULL,
          start_offset INTEGER NOT NULL,
          end_offset INTEGER NOT NULL,
          word_count INTEGER NOT NULL,
          text TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        """
    )
    return con


def _insert_doc(con: sqlite3.Connection) -> int:
    cur = con.execute(
        """
        INSERT INTO case_documents(case_id, original_name, mime, size_bytes, sha256_hex, storage_relpath, created_at_utc)
        VALUES ('case', 'a', 'text/plain', 1, '00', 'x', '2026-01-01 00:00:00');
        """
    )
    return int(cur.lastrowid)


def _insert_chunk(con: sqlite3.Connection, *, doc_id: int, idx: int, text: str) -> int:
    chunk_id = f"{doc_id}:{idx}"
    con.execute(
        """
        INSERT INTO document_chunks(id, document_id, chunk_index, start_offset, end_offset, word_count, text, created_at)
        VALUES (?, ?, ?, 0, ?, 1, ?, '2026-01-01 00:00:00');
        """,
        (chunk_id, doc_id, idx, len(text), text),
    )
    # Return rowid (this is what we index as hnsw label in this MVP).
    rowid = int(con.execute("SELECT rowid FROM document_chunks WHERE id = ?;", (chunk_id,)).fetchone()[0])
    return rowid


def test_vector_search_deterministic() -> None:
    dim = 8
    idx = VectorIndex(dim=dim, space="cosine")
    idx.init(max_elements=5)

    # 5 fixed vectors (orthonormal-ish)
    vecs = np.eye(dim, dtype=np.float32)[:5]
    ids = np.arange(1, 6, dtype=np.int32)
    idx.add_items(vecs, ids)

    q = vecs[2]
    r1 = vector_search(idx, q, top_k=3)
    r2 = vector_search(idx, q, top_k=3)
    assert r1 == r2
    assert r1[0][0] == int(ids[2])


def test_vector_retrieve_maps_to_chunk() -> None:
    dim = 8
    embedder = FakeEmbedder(dim)
    con = _setup_db()
    try:
        d1 = _insert_doc(con)
        rid0 = _insert_chunk(con, doc_id=d1, idx=0, text="alpha content")
        rid1 = _insert_chunk(con, doc_id=d1, idx=1, text="beta content")

        idx = VectorIndex(dim=dim, space="cosine")
        idx.init(max_elements=2)

        # Map rowid -> vectors deterministically
        v0 = embedder.embed_text("alpha content")
        v1 = embedder.embed_text("beta content")
        idx.add_items(np.vstack([v0, v1]), np.asarray([rid0, rid1], dtype=np.int32))

        hits = vector_retrieve(con, idx, embedder, query="alpha content", top_k=1)
        assert len(hits) == 1
        assert hits[0].chunk_id.endswith(":0")
        assert hits[0].practice_doc_id == str(d1)
    finally:
        con.close()


def test_vector_filter_practice_doc_id() -> None:
    dim = 8
    embedder = FakeEmbedder(dim)
    con = _setup_db()
    try:
        d1 = _insert_doc(con)
        d2 = _insert_doc(con)
        rid1 = _insert_chunk(con, doc_id=d1, idx=0, text="pvm deklaracija fr0600")
        rid2 = _insert_chunk(con, doc_id=d2, idx=0, text="pvm deklaracija fr0600")

        idx = VectorIndex(dim=dim, space="cosine")
        idx.init(max_elements=2)
        v = embedder.embed_text("pvm deklaracija fr0600")
        idx.add_items(np.vstack([v, v]), np.asarray([rid1, rid2], dtype=np.int32))

        hits = vector_retrieve(
            con,
            idx,
            embedder,
            query="pvm deklaracija fr0600",
            top_k=1,
            flt=VectorFilter(practice_doc_id=str(d1)),
        )
        assert len(hits) == 1
        assert hits[0].practice_doc_id == str(d1)
    finally:
        con.close()

