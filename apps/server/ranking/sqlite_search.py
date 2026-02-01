from __future__ import annotations

import sqlite3


def fts_topn(
    con: sqlite3.Connection,
    query: str,
    limit: int,
) -> list[tuple[str, float, str]]:
    rows = con.execute(
        """
        SELECT
          chunks.id,
          bm25(chunks_fts) AS bm25_raw,
          chunks.text
        FROM chunks_fts
        JOIN chunks ON chunks_fts.rowid = chunks.rowid
        WHERE chunks_fts MATCH ?
        ORDER BY bm25_raw ASC
        LIMIT ?;
        """,
        (query, int(limit)),
    ).fetchall()
    return [(str(cid), float(bm25_raw), str(text)) for cid, bm25_raw, text in rows]


def load_chunks_text(con: sqlite3.Connection, chunk_ids: list[str]) -> dict[str, str]:
    if not chunk_ids:
        return {}
    placeholders = ",".join(["?"] * len(chunk_ids))
    rows = con.execute(
        f"SELECT id, text FROM chunks WHERE id IN ({placeholders});",
        tuple(chunk_ids),
    ).fetchall()
    return {str(cid): str(text) for cid, text in rows}

