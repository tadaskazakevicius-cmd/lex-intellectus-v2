from __future__ import annotations

import sqlite3

import pytest

from lex_server.retrieval.hybrid_retrieval import (
    extract_citations,
    merge_and_rank,
    hybrid_retrieve,
)
from lex_server.retrieval.fts_retrieval import FtsHit
from lex_server.retrieval.vector_retrieval import VectorHit


def test_merge_and_dedup_sources() -> None:
    fts = [FtsHit(chunk_id="c1", practice_doc_id="d1", bm25_score=0.5)]
    vec = [VectorHit(chunk_id="c1", practice_doc_id="d1", distance=0.2), VectorHit(chunk_id="c2", practice_doc_id="d2", distance=0.1)]
    merged = merge_and_rank(fts, vec, top_n=10)
    ids = [cid for cid, _ in merged]
    assert ids.count("c1") == 1
    info_c1 = dict(merged)["c1"]
    assert info_c1["fts_bm25"] is not None
    assert info_c1["vector_distance"] is not None


def test_extract_citations_match_window_offsets() -> None:
    text = "Pradzia. PVM deklaracija FR0600 pateikiama laiku. Pabaiga."
    cits = extract_citations(text, ["FR0600", "PVM deklaracija"], None)
    assert len(cits) >= 1
    c = cits[0]
    assert 0 <= c.start < c.end <= len(text)
    assert c.quote == text[c.start : c.end]
    assert "FR0600" in c.quote or "PVM" in c.quote


def test_extract_citations_fallback_when_no_match() -> None:
    text = "Visai kitas tekstas be termino."
    cits = extract_citations(text, ["neras"], None)
    assert len(cits) >= 1
    c = cits[0]
    assert c.quote == text[c.start : c.end]
    assert c.start == 0


def test_merge_and_rank_deterministic_order() -> None:
    # Score ties broken by bm25 then chunk_id
    fts = [
        FtsHit(chunk_id="b", practice_doc_id="d", bm25_score=1.0),
        FtsHit(chunk_id="a", practice_doc_id="d", bm25_score=1.0),
    ]
    merged = merge_and_rank(fts, [], top_n=10)
    assert [cid for cid, _ in merged] == ["a", "b"]


def test_hybrid_retrieve_monkeypatch_merge_and_citations(monkeypatch) -> None:
    # Minimal DB with required tables for text loading.
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
    with con:
        cur = con.execute(
            "INSERT INTO case_documents(case_id, original_name, mime, size_bytes, sha256_hex, storage_relpath, created_at_utc) VALUES ('case','a','text/plain',1,'00','x','2026-01-01 00:00:00');"
        )
        doc_id = int(cur.lastrowid)
        con.execute(
            "INSERT INTO document_chunks(id, document_id, chunk_index, start_offset, end_offset, word_count, text, created_at) VALUES ('c1', ?, 0, 0, 10, 2, 'PVM FR0600', '2026');",
            (doc_id,),
        )

    # Monkeypatch fts_search/vector retrieval to return fixed hits.
    import lex_server.retrieval.hybrid_retrieval as hr

    monkeypatch.setattr(hr, "fts_search", lambda _c, _q, top_n, flt: [FtsHit("c1", str(doc_id), 0.2)])
    monkeypatch.setattr(hr, "vector_retrieve", lambda *a, **k: [VectorHit("c1", str(doc_id), 0.3)])

    hits = hr.hybrid_retrieve(con, "PVM", top_n=5, use_vector=False)
    assert len(hits) == 1
    assert hits[0].citations and hits[0].citations[0].quote != ""

    con.close()

