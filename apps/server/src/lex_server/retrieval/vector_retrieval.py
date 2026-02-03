from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .vector_index import VectorIndex


class EmbedderLike(Protocol):
    def embed_text(self, text: str) -> np.ndarray: ...


@dataclass(frozen=True)
class VectorHit:
    chunk_id: str
    practice_doc_id: str
    distance: float


@dataclass(frozen=True)
class VectorFilter:
    practice_doc_id: str | None = None


def vector_search(index: VectorIndex, query_vec: np.ndarray, top_k: int = 10) -> list[tuple[int, float]]:
    ids, dists = index.search(query_vec, top_k=top_k)
    return [(int(i), float(d)) for i, d in zip(ids.tolist(), dists.tolist(), strict=True)]


def _fetch_chunk_meta(conn: sqlite3.Connection, rowids: list[int]) -> dict[int, tuple[str, str]]:
    """
    Map document_chunks.rowid -> (chunk_id, practice_doc_id).
    """
    if not rowids:
        return {}
    placeholders = ",".join(["?"] * len(rowids))
    rows = conn.execute(
        f"""
        SELECT dc.rowid, dc.id, CAST(cd.id AS TEXT) AS practice_doc_id
        FROM document_chunks dc
        JOIN case_documents cd ON dc.document_id = cd.id
        WHERE dc.rowid IN ({placeholders});
        """,
        tuple(int(x) for x in rowids),
    ).fetchall()
    return {int(r[0]): (str(r[1]), str(r[2])) for r in rows}


def vector_retrieve(
    conn: sqlite3.Connection,
    index: VectorIndex,
    embedder: EmbedderLike,
    query: str,
    *,
    top_k: int = 10,
    flt: VectorFilter | None = None,
) -> list[VectorHit]:
    """
    Embed query -> ANN search -> map ids to chunk_id/practice_doc_id via SQLite.

    Filtering (MVP):
    - practice_doc_id filter is applied post-retrieval, fetching extra candidates to backfill.
    """
    q = (query or "").strip()
    if not q or top_k <= 0:
        return []
    flt = flt or VectorFilter()

    qv = embedder.embed_text(q)

    # Overfetch to allow filtering backfill.
    overfetch = max(int(top_k) * 5, int(top_k))
    pairs = vector_search(index, qv, top_k=overfetch)
    if not pairs:
        return []

    rowids = [rid for rid, _d in pairs]
    meta = _fetch_chunk_meta(conn, rowids)

    out: list[VectorHit] = []
    for rid, dist in pairs:
        m = meta.get(rid)
        if not m:
            continue
        chunk_id, practice_doc_id = m
        if flt.practice_doc_id and str(practice_doc_id) != str(flt.practice_doc_id):
            continue
        out.append(VectorHit(chunk_id=chunk_id, practice_doc_id=practice_doc_id, distance=float(dist)))
        if len(out) >= int(top_k):
            break
    return out

