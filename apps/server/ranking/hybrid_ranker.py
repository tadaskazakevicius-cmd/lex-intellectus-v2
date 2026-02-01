from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    score: float
    bm25_score: float
    vec_score: float
    bm25_raw: float | None
    vec_dist: float | None
    reasons: list[str]


def normalize_minmax_lower_is_better(
    values: dict[str, float],
    *,
    eps: float = 1e-9,
) -> dict[str, float]:
    """
    Min-max normalize a "lower is better" signal into 0..1 "higher is better" scores.

    Example:
    - values: {a: 2.0, b: 4.0}
    - scores: {a: 1.0, b: 0.0}
    """
    if not values:
        return {}

    vmin = min(values.values())
    vmax = max(values.values())
    denom = vmax - vmin
    if denom < eps:
        return {k: 1.0 for k in values}

    return {k: float((vmax - v) / (denom + eps)) for k, v in values.items()}


_QUOTE_RE = re.compile(r'"([^"]+)"')


def detect_exact_quote(query: str) -> str | None:
    """
    Returns the first substring inside double quotes, or None if no quotes present.
    """
    m = _QUOTE_RE.search(query)
    if not m:
        return None
    quoted = m.group(1).strip()
    return quoted or None


def compute_quote_match(text: str, quoted: str) -> bool:
    """
    Simple case-insensitive substring match.
    """
    return quoted.lower() in text.lower()


_TOKEN_RE = re.compile(r"[0-9A-Za-zĄČĘĖĮŠŲŪŽąčęėįšųūž]+", re.UNICODE)


def detect_strong_lexical_token(query: str) -> str | None:
    """
    Detects a "strong lexical token" useful for intent-aware ranking.

    Heuristics (any token that satisfies one of these):
    - contains digits and length >= 6  (e.g. FR0600)
    - contains digits and has uppercase+digit mix
    """
    tokens = _TOKEN_RE.findall(query)
    best: tuple[int, str] | None = None
    for t in tokens:
        has_digit = any(ch.isdigit() for ch in t)
        if not has_digit:
            continue
        has_upper = any(ch.isalpha() and ch.isupper() for ch in t)

        is_len6_digits = len(t) >= 6
        is_upper_digit_mix = has_upper

        if not (is_len6_digits or is_upper_digit_mix):
            continue

        # Pick the strongest-looking token deterministically:
        # prefer longer, then more digits, then lexicographic.
        digit_count = sum(1 for ch in t if ch.isdigit())
        score = len(t) * 100 + digit_count * 10 + (1 if has_upper else 0)
        if best is None or score > best[0] or (score == best[0] and t < best[1]):
            best = (score, t)
    return best[1] if best else None


def compute_token_match(text: str, token: str) -> bool:
    return token.lower() in text.lower()


def hybrid_fuse(
    bm25_raw_by_id: dict[str, float],
    vec_dist_by_id: dict[str, float],
    quote_match_by_id: Optional[dict[str, bool]] = None,
    token_match_by_id: Optional[dict[str, bool]] = None,
    *,
    w_bm25: float = 0.5,
    w_vec: float = 0.5,
    quote_boost: float = 0.15,
    quote_miss_penalty: float = 0.0,
    token_boost: float = 0.0,
) -> dict[str, tuple[float, float, float]]:
    """
    Fuse BM25 (FTS5) + vector cosine distance (HNSW) + optional boosts/penalties.

    - bm25_raw: lower is better (FTS5 bm25 returns smaller=better)
    - vec_dist: lower is better (cosine distance)
    - normalization: both converted to 0..1 where higher is better
    - missing ids: union of keys; missing signal => 0 for that normalized score
    """
    bm25_norm = normalize_minmax_lower_is_better(bm25_raw_by_id)
    vec_norm = normalize_minmax_lower_is_better(vec_dist_by_id)

    out: dict[str, tuple[float, float, float]] = {}
    all_ids = set(bm25_raw_by_id.keys()) | set(vec_dist_by_id.keys())
    for cid in all_ids:
        b = float(bm25_norm.get(cid, 0.0))
        v = float(vec_norm.get(cid, 0.0))
        boosts = 0.0
        penalties = 0.0

        if quote_match_by_id is not None:
            if quote_match_by_id.get(cid, False):
                boosts += float(quote_boost)
            else:
                penalties += float(quote_miss_penalty)

        if token_match_by_id and token_match_by_id.get(cid, False):
            boosts += float(token_boost)

        final = float(w_bm25) * b + float(w_vec) * v + boosts - penalties
        out[cid] = (final, b, v)
    return out

