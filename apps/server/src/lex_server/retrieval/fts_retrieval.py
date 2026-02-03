from __future__ import annotations

"""
E2: SQLite FTS5 retrieval for chunk text.

How this is used with E1 QueryPlan:
- For each QueryAtom in `QueryPlan.atoms`, call `fts_search(conn, atom.text, top_n=..., flt=...)`.
- The caller can then fuse / rerank results using `atom.weight` (hybrid scoring in E3).
"""

import sqlite3
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


class FtsFilter(BaseModel):
    """
    Filtering support is limited to fields we can map to existing DB columns.

    - practice_doc_id maps to case_documents.id (stringified integer).
    - doc_type maps to case_documents.mime (exact match).
    - date_from/date_to filter case_documents.created_at_utc by date part (YYYY-MM-DD).

    Other fields are accepted for API compatibility but are not implemented in MVP:
    - court, tags
    """

    practice_doc_id: str | None = None
    doc_type: str | None = None
    court: str | None = None
    date_from: str | None = Field(default=None, description="YYYY-MM-DD")
    date_to: str | None = Field(default=None, description="YYYY-MM-DD")
    tags: list[str] | None = None


@dataclass(frozen=True)
class FtsHit:
    chunk_id: str
    practice_doc_id: str
    bm25_score: float


def fts_search(
    conn: sqlite3.Connection,
    query: str,
    top_n: int = 10,
    flt: FtsFilter | None = None,
) -> list[FtsHit]:
    """
    Run a parameterized FTS5 query against document_chunks_fts.

    Scoring:
    - Uses `bm25(document_chunks_fts)` where lower is better.
    - Returned `bm25_score` is that raw value.
    """
    q = (query or "").strip()
    if not q or top_n <= 0:
        return []

    flt = flt or FtsFilter()
    if flt.court is not None:
        raise ValueError("court filter not supported in MVP")
    if flt.tags:
        raise ValueError("tags filter not supported in MVP")

    where = ["document_chunks_fts MATCH ?"]
    params: list[Any] = [q]

    if flt.practice_doc_id:
        where.append("CAST(cd.id AS TEXT) = ?")
        params.append(str(flt.practice_doc_id))

    if flt.doc_type:
        where.append("cd.mime = ?")
        params.append(str(flt.doc_type))

    if flt.date_from:
        where.append("substr(cd.created_at_utc, 1, 10) >= ?")
        params.append(str(flt.date_from))
    if flt.date_to:
        where.append("substr(cd.created_at_utc, 1, 10) <= ?")
        params.append(str(flt.date_to))

    sql = f"""
    SELECT
      dc.id AS chunk_id,
      CAST(cd.id AS TEXT) AS practice_doc_id,
      bm25(document_chunks_fts) AS bm25_score
    FROM document_chunks_fts
    JOIN document_chunks dc ON document_chunks_fts.rowid = dc.rowid
    JOIN case_documents cd ON dc.document_id = cd.id
    WHERE {" AND ".join(where)}
    ORDER BY bm25_score ASC
    LIMIT ?;
    """
    params.append(int(top_n))

    rows = conn.execute(sql, tuple(params)).fetchall()
    return [FtsHit(chunk_id=str(r[0]), practice_doc_id=str(r[1]), bm25_score=float(r[2])) for r in rows]

