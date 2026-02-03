from __future__ import annotations

"""
E4: Hybrid merge + dedup + citation extraction.

This layer merges lexical (FTS5 BM25) and vector (HNSW cosine distance) signals into a single ranked
result list, and extracts short citations (quotes) with offsets for UI.

Scoring choice (MVP, stable):
- fts_score = 1 / (1 + bm25)           # bm25: lower is better
- vec_score = 1 / (1 + distance)       # cosine distance: lower is better
- final score = 0.6 * fts_score + 0.4 * vec_score (missing signal => 0)

Rationale:
- Lexical precision is usually critical for legal text, but vector improves recall.
"""

import re
import sqlite3
from dataclasses import dataclass
from typing import Any

from .fts_retrieval import FtsFilter, FtsHit, fts_search
from .vector_retrieval import VectorFilter, VectorHit, vector_retrieve


@dataclass(frozen=True)
class Citation:
    quote: str
    start: int
    end: int
    source_url: str | None


@dataclass(frozen=True)
class HybridHit:
    chunk_id: str
    practice_doc_id: str
    score: float
    sources: dict[str, float | None]
    citations: list[Citation]


def merge_and_rank(
    fts_hits: list[FtsHit] | None,
    vec_hits: list[VectorHit] | None,
    *,
    top_n: int = 10,
) -> list[tuple[str, dict[str, Any]]]:
    """
    Dedup by chunk_id and rank by combined score.

    Returns list of (chunk_id, merged_info dict) where merged_info contains:
    - practice_doc_id
    - fts_bm25 (float|None)
    - vector_distance (float|None)
    - score (float)
    """
    fts_hits = fts_hits or []
    vec_hits = vec_hits or []
    if top_n <= 0:
        return []

    merged: dict[str, dict[str, Any]] = {}

    for h in fts_hits:
        merged.setdefault(
            h.chunk_id,
            {
                "chunk_id": h.chunk_id,
                "practice_doc_id": h.practice_doc_id,
                "fts_bm25": None,
                "vector_distance": None,
                "score": 0.0,
            },
        )
        m = merged[h.chunk_id]
        # Keep best (lowest) bm25
        if m["fts_bm25"] is None or float(h.bm25_score) < float(m["fts_bm25"]):
            m["fts_bm25"] = float(h.bm25_score)

    for h in vec_hits:
        merged.setdefault(
            h.chunk_id,
            {
                "chunk_id": h.chunk_id,
                "practice_doc_id": h.practice_doc_id,
                "fts_bm25": None,
                "vector_distance": None,
                "score": 0.0,
            },
        )
        m = merged[h.chunk_id]
        # Keep best (lowest) distance
        if m["vector_distance"] is None or float(h.distance) < float(m["vector_distance"]):
            m["vector_distance"] = float(h.distance)

    for m in merged.values():
        fts_bm25 = m["fts_bm25"]
        vec_dist = m["vector_distance"]
        fts_score = (1.0 / (1.0 + float(fts_bm25))) if fts_bm25 is not None else 0.0
        vec_score = (1.0 / (1.0 + float(vec_dist))) if vec_dist is not None else 0.0
        m["score"] = 0.6 * fts_score + 0.4 * vec_score

    items = list(merged.items())
    items.sort(
        key=lambda kv: (
            -float(kv[1]["score"]),
            float(kv[1]["fts_bm25"]) if kv[1]["fts_bm25"] is not None else 1e9,
            str(kv[0]),
        )
    )
    return items[: int(top_n)]


_WS_RE = re.compile(r"\s+")


def _collapse_ws(s: str) -> str:
    return _WS_RE.sub(" ", s).strip()


def _find_first_match(text: str, terms: list[str]) -> tuple[int, int] | None:
    """
    Return (start,end) for the earliest case-insensitive match of any term.
    """
    low = text.lower()
    best: tuple[int, int] | None = None
    for t in terms:
        t2 = t.strip()
        if not t2:
            continue
        pos = low.find(t2.lower())
        if pos < 0:
            continue
        span = (pos, pos + len(t2))
        if best is None or span[0] < best[0]:
            best = span
    return best


def _snap_to_word_boundary(text: str, start: int, end: int) -> tuple[int, int]:
    s = max(0, int(start))
    e = min(len(text), int(end))
    # expand start left to whitespace boundary
    while s > 0 and not text[s - 1].isspace():
        s -= 1
    # expand end right to whitespace boundary
    while e < len(text) and not text[e].isspace():
        e += 1
    return s, e


def extract_citations(
    chunk_text: str,
    query_terms: list[str],
    source_url: str | None,
    *,
    max_citations: int = 2,
) -> list[Citation]:
    """
    Extract short citations from chunk_text with offsets.

    Strategy:
    - If any query term matches: take a ~200 char window around the first match.
    - Else: fallback to first 200 chars.
    - Always returns at least 1 citation.
    """
    text = chunk_text or ""
    if text == "":
        return [Citation(quote="", start=0, end=0, source_url=source_url)]

    match = _find_first_match(text, query_terms)
    citations: list[Citation] = []

    if match:
        m_start, m_end = match
        center = (m_start + m_end) // 2
        win = 220
        s0 = max(0, center - win // 2)
        e0 = min(len(text), s0 + win)
        s0, e0 = _snap_to_word_boundary(text, s0, e0)
        quote = text[s0:e0]
        citations.append(Citation(quote=quote, start=s0, end=e0, source_url=source_url))
    else:
        e0 = min(len(text), 200)
        s0, e0 = _snap_to_word_boundary(text, 0, e0)
        quote = text[s0:e0]
        citations.append(Citation(quote=quote, start=s0, end=e0, source_url=source_url))

    # MVP: max 1 citation unless expanded later
    return citations[: max(1, int(max_citations))]


def _extract_query_terms(query: str) -> list[str]:
    """
    Extract a small list of terms/phrases from raw query string.
    """
    q = query or ""
    phrases = re.findall(r'"([^"]+)"', q)
    # remove phrases to avoid duplication
    q2 = re.sub(r'"[^"]+"', " ", q)
    words = [w for w in re.split(r"\s+", q2) if w]
    terms = [t.strip() for t in phrases + words if t.strip()]
    # stable dedup (case-insensitive)
    out: list[str] = []
    seen: set[str] = set()
    for t in terms:
        k = _collapse_ws(t).casefold()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(_collapse_ws(t))
    return out[:20]


def _load_chunk_texts(conn: sqlite3.Connection, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
    """
    Load chunk text and (optional) source URL for citation extraction.
    Returns: chunk_id -> {text, practice_doc_id, source_url}
    """
    if not chunk_ids:
        return {}
    placeholders = ",".join(["?"] * len(chunk_ids))
    rows = conn.execute(
        f"""
        SELECT dc.id AS chunk_id, dc.text, CAST(cd.id AS TEXT) AS practice_doc_id
        FROM document_chunks dc
        JOIN case_documents cd ON dc.document_id = cd.id
        WHERE dc.id IN ({placeholders});
        """,
        tuple(chunk_ids),
    ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for chunk_id, text, practice_doc_id in rows:
        out[str(chunk_id)] = {"text": str(text), "practice_doc_id": str(practice_doc_id), "source_url": None}
    return out


def hybrid_retrieve(
    conn: sqlite3.Connection,
    query: str,
    *,
    top_n: int = 10,
    filters: FtsFilter | None = None,
    use_fts: bool = True,
    use_vector: bool = True,
) -> list[HybridHit]:
    """
    Hybrid retrieval (FTS + vector) with citations.

    Vector retrieval is optional; if not configured (missing env vars), it will be skipped gracefully.
    """
    q = (query or "").strip()
    if not q or top_n <= 0:
        return []

    fts_hits: list[FtsHit] = []
    if use_fts:
        fts_hits = fts_search(conn, q, top_n=max(int(top_n) * 3, int(top_n)), flt=filters)

    vec_hits: list[VectorHit] = []
    if use_vector:
        # Runtime configuration via env (same as vector endpoint)
        import os
        from pathlib import Path

        model_path = os.environ.get("LEX_EMBED_ONNX_MODEL")
        index_path = os.environ.get("LEX_VECTOR_INDEX_PATH")
        dim = os.environ.get("LEX_VECTOR_DIM")
        if model_path and index_path and dim:
            from .embedder_onnx import OnnxEmbedder
            from .vector_index import VectorIndex

            embedder = OnnxEmbedder(Path(model_path))
            index = VectorIndex.load(Path(index_path), dim=int(dim), space="cosine")
            vflt = VectorFilter(practice_doc_id=filters.practice_doc_id if filters else None)
            vec_hits = vector_retrieve(
                conn, index, embedder, q, top_k=max(int(top_n) * 3, int(top_n)), flt=vflt
            )

    merged = merge_and_rank(fts_hits, vec_hits, top_n=int(top_n))
    chunk_ids = [cid for cid, _m in merged]
    texts = _load_chunk_texts(conn, chunk_ids)
    terms = _extract_query_terms(q)

    out: list[HybridHit] = []
    for cid, m in merged:
        t = texts.get(cid, {"text": "", "practice_doc_id": m["practice_doc_id"], "source_url": None})
        citations = extract_citations(
            t["text"],
            terms,
            t.get("source_url"),
            max_citations=2,
        )
        out.append(
            HybridHit(
                chunk_id=cid,
                practice_doc_id=str(m["practice_doc_id"]),
                score=float(m["score"]),
                sources={"fts_bm25": m["fts_bm25"], "vector_distance": m["vector_distance"]},
                citations=citations,
            )
        )
    return out

