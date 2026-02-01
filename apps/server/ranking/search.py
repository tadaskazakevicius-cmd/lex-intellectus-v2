from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from .hybrid_ranker import (
    SearchResult,
    compute_quote_match,
    compute_token_match,
    detect_exact_quote,
    detect_strong_lexical_token,
    hybrid_fuse,
)
from .sqlite_search import fts_topn, load_chunks_text
from .vector_search import hnsw_topn


def hybrid_search(
    con: sqlite3.Connection,
    pack_id: str,
    query: str,
    query_vec: np.ndarray,
    *,
    fts_k: int = 20,
    vec_k: int = 20,
    out_k: int = 10,
    # Manual weights (used only when auto=False)
    w_bm25: float = 0.45,
    w_vec: float = 0.55,
    quote_boost: float = 0.0,
    quote_miss_penalty: float = 0.0,
    token_boost: float = 0.0,
    auto: bool = True,
    indices_root: Path | None = None,
    debug: dict[str, object] | None = None,
) -> list[SearchResult]:
    indices_root = indices_root or (Path(".localdata") / "indices")
    pack_dir = indices_root / pack_id

    fts_rows = fts_topn(con, query=query, limit=int(fts_k))
    vec_rows = hnsw_topn(pack_dir=pack_dir, query_vec=query_vec, k=int(vec_k))

    bm25_raw_by_id: dict[str, float] = {cid: raw for cid, raw, _text in fts_rows}
    vec_dist_by_id: dict[str, float] = {cid: dist for cid, dist in vec_rows}

    candidates = sorted(set(bm25_raw_by_id.keys()) | set(vec_dist_by_id.keys()))
    if not candidates:
        return []

    quoted = detect_exact_quote(query)
    strong_token = detect_strong_lexical_token(query)
    is_quote = quoted is not None
    has_strong_lexical = (not is_quote) and (strong_token is not None)

    if auto:
        if is_quote:
            w_bm25 = 0.80
            w_vec = 0.20
            quote_boost = 0.35
            quote_miss_penalty = 0.15
            token_boost = 0.0
        else:
            w_bm25 = 0.45
            w_vec = 0.55
            quote_boost = 0.0
            quote_miss_penalty = 0.0
            token_boost = 0.0
            if has_strong_lexical:
                w_bm25 = 0.55
                w_vec = 0.45
                token_boost = 0.10

    if debug is not None:
        debug.clear()
        debug.update(
            {
                "is_quote": is_quote,
                "quoted": quoted,
                "has_strong_lexical": has_strong_lexical,
                "strong_token": strong_token,
                "w_bm25": w_bm25,
                "w_vec": w_vec,
                "quote_boost": quote_boost,
                "quote_miss_penalty": quote_miss_penalty,
                "token_boost": token_boost,
                "auto": auto,
            }
        )

    quote_match_by_id: dict[str, bool] | None = None
    token_match_by_id: dict[str, bool] | None = None

    need_texts = (quoted is not None and (quote_boost != 0.0 or quote_miss_penalty != 0.0)) or (
        strong_token is not None and token_boost != 0.0
    )
    texts: dict[str, str] = {}
    if need_texts:
        texts = load_chunks_text(con, candidates)

    if quoted is not None and (quote_boost != 0.0 or quote_miss_penalty != 0.0):
        quote_match_by_id = {cid: compute_quote_match(texts.get(cid, ""), quoted) for cid in candidates}

    if strong_token is not None and token_boost != 0.0 and not is_quote:
        token_match_by_id = {cid: compute_token_match(texts.get(cid, ""), strong_token) for cid in candidates}

    fused = hybrid_fuse(
        bm25_raw_by_id=bm25_raw_by_id,
        vec_dist_by_id=vec_dist_by_id,
        quote_match_by_id=quote_match_by_id,
        token_match_by_id=token_match_by_id,
        w_bm25=w_bm25,
        w_vec=w_vec,
        quote_boost=quote_boost,
        quote_miss_penalty=quote_miss_penalty,
        token_boost=token_boost,
    )

    results: list[SearchResult] = []
    for cid in candidates:
        final, bm25_norm, vec_norm = fused[cid]
        reasons: list[str] = []
        if is_quote:
            reasons.append("quote_intent")
            if quote_match_by_id and quote_match_by_id.get(cid, False):
                reasons.append("quote_boost")
            else:
                if quote_miss_penalty != 0.0:
                    reasons.append("quote_miss_penalty")
        else:
            if has_strong_lexical and strong_token is not None:
                reasons.append("strong_lexical_intent")
                if token_match_by_id and token_match_by_id.get(cid, False) and token_boost != 0.0:
                    reasons.append("strong_token_boost")

        results.append(
            SearchResult(
                chunk_id=cid,
                score=float(final),
                bm25_score=float(bm25_norm),
                vec_score=float(vec_norm),
                bm25_raw=bm25_raw_by_id.get(cid),
                vec_dist=vec_dist_by_id.get(cid),
                reasons=reasons,
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results[: int(out_k)]

