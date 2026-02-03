from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from lex_server.retrieval.hybrid_retrieval import Citation, HybridHit
from lex_server.retrieval.persistence import create_run, load_run_hits, persist_run_results


def _apply_migration(conn: sqlite3.Connection) -> None:
    # Apply only the E5 tables; include schema_migrations for compatibility.
    sql = (Path(__file__).resolve().parents[1] / "db" / "migrations" / "0007_retrieval_runs.sql").read_text(
        encoding="utf-8"
    )
    # Also ensure schema_migrations exists in this isolated DB.
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE IF NOT EXISTS schema_migrations (
          version INTEGER PRIMARY KEY,
          applied_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )
    conn.executescript(sql)


def _make_hits() -> list[HybridHit]:
    return [
        HybridHit(
            chunk_id="c1",
            practice_doc_id="d1",
            score=0.9,
            sources={"fts_bm25": 0.2, "vector_distance": 0.3},
            citations=[Citation(quote="Q1", start=0, end=2, source_url=None)],
        ),
        HybridHit(
            chunk_id="c2",
            practice_doc_id="d1",
            score=0.8,
            sources={"fts_bm25": None, "vector_distance": 0.1},
            citations=[
                Citation(quote="Q2a", start=1, end=3, source_url="s"),
                Citation(quote="Q2b", start=3, end=5, source_url=None),
            ],
        ),
        HybridHit(
            chunk_id="c3",
            practice_doc_id="d2",
            score=0.7,
            sources={"fts_bm25": 1.2, "vector_distance": None},
            citations=[Citation(quote="Q3", start=2, end=4, source_url=None)],
        ),
    ]


def test_persist_and_reload_same_hits(tmp_path: Path) -> None:
    dbp = tmp_path / "t.db"
    con = sqlite3.connect(dbp)
    try:
        con.execute("PRAGMA foreign_keys = ON;")
        _apply_migration(con)

        hits = _make_hits()
        run_id = create_run(con, "q", 3, filters={"practice_doc_id": "d1"}, use_fts=True, use_vector=False)
        persist_run_results(con, run_id, hits)

        loaded = load_run_hits(con, run_id)
        assert loaded == hits
    finally:
        con.close()


def test_rank_order_preserved(tmp_path: Path) -> None:
    dbp = tmp_path / "t.db"
    con = sqlite3.connect(dbp)
    try:
        con.execute("PRAGMA foreign_keys = ON;")
        _apply_migration(con)

        hits = _make_hits()
        run_id = create_run(con, "q", 3, filters=None, use_fts=True, use_vector=True)
        persist_run_results(con, run_id, hits)
        loaded = load_run_hits(con, run_id)
        assert [h.chunk_id for h in loaded] == ["c1", "c2", "c3"]
    finally:
        con.close()


def test_api_hybrid_run_and_get(tmp_path: Path, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    # Create isolated DB file and patch _db_path in router module.
    dbp = tmp_path / "api.db"
    con = sqlite3.connect(dbp)
    try:
        con.execute("PRAGMA foreign_keys = ON;")
        _apply_migration(con)
    finally:
        con.close()

    import lex_server.retrieval.api as api_mod

    monkeypatch.setattr(api_mod, "_db_path", lambda: dbp)

    # Monkeypatch hybrid_retrieve to be deterministic (no ONNX/hnsw required)
    import lex_server.retrieval.hybrid_retrieval as hr

    hits = _make_hits()[:2]
    monkeypatch.setattr(api_mod, "hybrid_retrieve", lambda *_a, **_k: hits)

    from lex_server.main import app

    client = TestClient(app)
    r = client.post(
        "/api/retrieval/hybrid_run",
        json={"query": "q", "top_n": 2, "filters": None, "use_fts": True, "use_vector": False},
    )
    assert r.status_code == 200
    body = r.json()
    assert "run_id" in body
    run_id = body["run_id"]
    assert body["hits"][0]["chunk_id"] == "c1"

    g = client.get(f"/api/retrieval/runs/{run_id}")
    assert g.status_code == 200
    body2 = g.json()
    assert body2["run"]["id"] == run_id
    assert body2["hits"] == body["hits"]


def test_get_run_404(tmp_path: Path) -> None:
    dbp = tmp_path / "t.db"
    con = sqlite3.connect(dbp)
    try:
        con.execute("PRAGMA foreign_keys = ON;")
        _apply_migration(con)
    finally:
        con.close()

    from fastapi.testclient import TestClient
    import lex_server.retrieval.api as api_mod

    api_mod._db_path = lambda: dbp  # type: ignore[assignment]
    from lex_server.main import app

    client = TestClient(app)
    g = client.get("/api/retrieval/runs/00000000-0000-0000-0000-000000000000")
    assert g.status_code == 404

