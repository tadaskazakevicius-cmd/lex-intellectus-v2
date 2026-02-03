from __future__ import annotations

import sqlite3

import pytest

from lex_server.retrieval.fts_retrieval import FtsFilter, fts_search


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


def _insert_doc(con: sqlite3.Connection, *, mime: str, created_at_utc: str) -> int:
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
    # manual FTS insert for deterministic test setup
    con.execute(
        "INSERT INTO document_chunks_fts(rowid, chunk_id, text) VALUES (last_insert_rowid(), ?, ?);",
        (chunk_id, text),
    )
    return chunk_id


def test_fts_basic_search_returns_hits() -> None:
    con = _setup_db()
    try:
        d1 = _insert_doc(con, mime="text/plain", created_at_utc="2026-01-10 10:00:00")
        d2 = _insert_doc(con, mime="application/pdf", created_at_utc="2026-01-12 10:00:00")

        c1 = _insert_chunk(con, doc_id=d1, idx=0, text="PVM deklaracija FR0600 pateikimas VMI.")
        _insert_chunk(con, doc_id=d1, idx=1, text="Kitas tekstas apie sutartį.")
        c3 = _insert_chunk(con, doc_id=d2, idx=0, text="FR0600 PVM deklaracija terminas iki 25 d.")
        _insert_chunk(con, doc_id=d2, idx=1, text="Darbo užmokestis ir GPM.")
        _insert_chunk(con, doc_id=d2, idx=2, text="Visai nesusijęs tekstas.")

        hits = fts_search(con, 'PVM deklaracija "FR0600"', top_n=10)
        assert len(hits) > 0
        assert all(isinstance(h.bm25_score, float) for h in hits)
        got_ids = [h.chunk_id for h in hits]
        assert c1 in got_ids or c3 in got_ids
    finally:
        con.close()


def test_fts_filter_by_practice_doc_id() -> None:
    con = _setup_db()
    try:
        d1 = _insert_doc(con, mime="text/plain", created_at_utc="2026-01-10 10:00:00")
        d2 = _insert_doc(con, mime="text/plain", created_at_utc="2026-01-10 10:00:00")
        c1 = _insert_chunk(con, doc_id=d1, idx=0, text="PVM deklaracija FR0600.")
        _insert_chunk(con, doc_id=d2, idx=0, text="PVM deklaracija FR0600.")

        hits = fts_search(con, "PVM", top_n=10, flt=FtsFilter(practice_doc_id=str(d1)))
        assert len(hits) >= 1
        assert all(h.practice_doc_id == str(d1) for h in hits)
        assert hits[0].chunk_id == c1
    finally:
        con.close()


def test_fts_filter_by_doc_type_and_date_range() -> None:
    con = _setup_db()
    try:
        d_txt = _insert_doc(con, mime="text/plain", created_at_utc="2026-01-05 10:00:00")
        d_pdf = _insert_doc(con, mime="application/pdf", created_at_utc="2026-02-01 10:00:00")
        _insert_chunk(con, doc_id=d_txt, idx=0, text="PVM deklaracija FR0600.")
        _insert_chunk(con, doc_id=d_pdf, idx=0, text="PVM deklaracija FR0600.")

        hits = fts_search(
            con,
            "FR0600",
            top_n=10,
            flt=FtsFilter(doc_type="text/plain", date_from="2026-01-01", date_to="2026-01-31"),
        )
        assert len(hits) >= 1
        assert all(h.practice_doc_id == str(d_txt) for h in hits)
    finally:
        con.close()

