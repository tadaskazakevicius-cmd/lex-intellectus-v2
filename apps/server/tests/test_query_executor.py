from __future__ import annotations

import sqlite3

from lex_server.retrieval.fts_retrieval import FtsFilter
from lex_server.retrieval.query_builder import QueryAtom, QueryPlan
from lex_server.retrieval.query_executor import execute_fts_plan


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

        CREATE VIRTUAL TABLE document_chunks_fts USING fts5(
          chunk_id UNINDEXED,
          text,
          tokenize = 'unicode61'
        );
        """
    )
    return con


def _insert_doc(con: sqlite3.Connection, *, mime: str = "text/plain", created_at_utc: str = "2026-01-01 00:00:00") -> int:
    cur = con.execute(
        """
        INSERT INTO case_documents(case_id, original_name, mime, size_bytes, sha256_hex, storage_relpath, created_at_utc)
        VALUES ('case', 'a', ?, 1, '00', 'x', ?);
        """,
        (mime, created_at_utc),
    )
    return int(cur.lastrowid)


def _insert_chunk(con: sqlite3.Connection, *, doc_id: int, idx: int, text: str) -> str:
    chunk_id = f"{doc_id}:{idx}"
    con.execute(
        """
        INSERT INTO document_chunks(id, document_id, chunk_index, start_offset, end_offset, word_count, text, created_at)
        VALUES (?, ?, ?, 0, ?, 1, ?, '2026-01-01 00:00:00');
        """,
        (chunk_id, doc_id, idx, len(text), text),
    )
    # Manual FTS insert (stable, no triggers in this in-memory setup)
    con.execute(
        "INSERT INTO document_chunks_fts(rowid, chunk_id, text) VALUES (last_insert_rowid(), ?, ?);",
        (chunk_id, text),
    )
    return chunk_id


def test_executor_multi_atom_merge_and_dedup() -> None:
    con = _setup_db()
    try:
        d1 = _insert_doc(con)
        shared = _insert_chunk(con, doc_id=d1, idx=0, text="alpha beta shared content")
        only_alpha = _insert_chunk(con, doc_id=d1, idx=1, text="alpha unique content")
        only_beta = _insert_chunk(con, doc_id=d1, idx=2, text="beta unique content")

        plan = QueryPlan(
            case_id=None,
            atoms=[
                QueryAtom(text="alpha", kind="keywords", weight=1.4),
                QueryAtom(text="beta", kind="keywords", weight=1.0),
            ],
            k=2,
        )
        hits = execute_fts_plan(con, plan, top_n=10, per_atom=10)

        ids = [h.chunk_id for h in hits]
        assert shared in ids
        assert only_alpha in ids
        assert only_beta in ids
        # shared should appear only once
        assert ids.count(shared) == 1

        shared_hit = next(h for h in hits if h.chunk_id == shared)
        assert len(shared_hit.matches) == 2  # matched by both atoms
    finally:
        con.close()


def test_executor_weight_ranking_prefers_higher_weight() -> None:
    con = _setup_db()
    try:
        d1 = _insert_doc(con)
        # Keep docs similar length/structure so bm25 is similar; weight should drive order.
        hi = _insert_chunk(con, doc_id=d1, idx=0, text="common filler alpha common filler")
        lo = _insert_chunk(con, doc_id=d1, idx=1, text="common filler beta common filler")

        plan = QueryPlan(
            case_id=None,
            atoms=[
                QueryAtom(text="alpha", kind="keywords", weight=1.4),
                QueryAtom(text="beta", kind="keywords", weight=1.0),
            ],
            k=2,
        )
        hits = execute_fts_plan(con, plan, top_n=2, per_atom=5)
        assert len(hits) == 2
        assert hits[0].chunk_id == hi
        assert hits[0].score > hits[1].score
    finally:
        con.close()


def test_executor_filters_passthrough_practice_doc_id() -> None:
    con = _setup_db()
    try:
        d1 = _insert_doc(con)
        d2 = _insert_doc(con)
        c1 = _insert_chunk(con, doc_id=d1, idx=0, text="pvm deklaracija fr0600")
        _insert_chunk(con, doc_id=d2, idx=0, text="pvm deklaracija fr0600")

        plan = QueryPlan(
            case_id=None,
            atoms=[QueryAtom(text="pvm", kind="keywords", weight=1.0)],
            k=1,
        )
        hits = execute_fts_plan(con, plan, top_n=10, per_atom=10, flt=FtsFilter(practice_doc_id=str(d1)))
        assert len(hits) >= 1
        assert all(h.practice_doc_id == str(d1) for h in hits)
        assert hits[0].chunk_id == c1
    finally:
        con.close()

