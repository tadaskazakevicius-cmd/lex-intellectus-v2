from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from .hybrid_retrieval import Citation, HybridHit


def _utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def create_run(
    conn: sqlite3.Connection,
    query: str,
    top_n: int,
    filters: dict | None,
    use_fts: bool,
    use_vector: bool,
    *,
    algo_version: str = "hybrid_v1",
    meta: dict | None = None,
) -> str:
    run_id = str(uuid.uuid4())
    created_at = _utc_now_iso_z()
    filters_json = _stable_json(filters) if filters is not None else None
    meta_json = _stable_json(meta) if meta is not None else None

    with conn:
        conn.execute(
            """
            INSERT INTO retrieval_runs(
              id, created_at, query, top_n, filters_json, use_fts, use_vector, algo_version, meta_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                run_id,
                created_at,
                str(query),
                int(top_n),
                filters_json,
                1 if use_fts else 0,
                1 if use_vector else 0,
                str(algo_version),
                meta_json,
            ),
        )

    return run_id


def persist_run_results(conn: sqlite3.Connection, run_id: str, hits: list[HybridHit]) -> None:
    """
    Persist hits + citations in a single transaction.

    Ordering:
    - rank is persisted as 0..N-1 based on list order
    - citations persisted with idx 0.. based on list order
    """
    with conn:
        for rank, h in enumerate(hits):
            cur = conn.execute(
                """
                INSERT INTO retrieval_run_hits(
                  run_id, rank, chunk_id, practice_doc_id, score, fts_bm25, vector_distance
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    str(run_id),
                    int(rank),
                    h.chunk_id,
                    h.practice_doc_id,
                    float(h.score),
                    h.sources.get("fts_bm25"),
                    h.sources.get("vector_distance"),
                ),
            )
            hit_id = int(cur.lastrowid)
            for idx, c in enumerate(h.citations):
                conn.execute(
                    """
                    INSERT INTO retrieval_run_citations(hit_id, idx, quote, start, end, source_url)
                    VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    (hit_id, int(idx), c.quote, int(c.start), int(c.end), c.source_url),
                )


def load_run(conn: sqlite3.Connection, run_id: str) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT id, created_at, query, top_n, filters_json, use_fts, use_vector, algo_version, meta_json
        FROM retrieval_runs
        WHERE id = ?;
        """,
        (str(run_id),),
    ).fetchone()
    if not row:
        raise KeyError("run not found")

    filters = json.loads(row[4]) if row[4] else None
    meta = json.loads(row[8]) if row[8] else None
    return {
        "id": str(row[0]),
        "created_at": str(row[1]),
        "query": str(row[2]),
        "top_n": int(row[3]),
        "filters": filters,
        "use_fts": bool(int(row[5])),
        "use_vector": bool(int(row[6])),
        "algo_version": str(row[7]),
        "meta": meta,
    }


def load_run_hits(conn: sqlite3.Connection, run_id: str) -> list[HybridHit]:
    rows = conn.execute(
        """
        SELECT id, rank, chunk_id, practice_doc_id, score, fts_bm25, vector_distance
        FROM retrieval_run_hits
        WHERE run_id = ?
        ORDER BY rank ASC;
        """,
        (str(run_id),),
    ).fetchall()

    out: list[HybridHit] = []
    for hit_id, _rank, chunk_id, practice_doc_id, score, fts_bm25, vector_distance in rows:
        cit_rows = conn.execute(
            """
            SELECT idx, quote, start, end, source_url
            FROM retrieval_run_citations
            WHERE hit_id = ?
            ORDER BY idx ASC;
            """,
            (int(hit_id),),
        ).fetchall()
        citations = [
            Citation(quote=str(q), start=int(s), end=int(e), source_url=(str(u) if u is not None else None))
            for _idx, q, s, e, u in cit_rows
        ]
        out.append(
            HybridHit(
                chunk_id=str(chunk_id),
                practice_doc_id=str(practice_doc_id),
                score=float(score),
                sources={"fts_bm25": (float(fts_bm25) if fts_bm25 is not None else None),
                         "vector_distance": (float(vector_distance) if vector_distance is not None else None)},
                citations=citations,
            )
        )
    return out

