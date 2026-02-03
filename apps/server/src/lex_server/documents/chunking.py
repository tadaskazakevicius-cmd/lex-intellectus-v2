from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentChunk:
    chunk_index: int
    start_offset: int
    end_offset: int
    word_count: int
    text: str


_WORD_RE = re.compile(r"\S+")


def normalize_text(raw: str) -> str:
    """
    D3 normalization (stable, no lowercasing/diacritics changes):
    - replace CRLF/CR -> LF
    - tabs -> spaces
    - strip trailing whitespace per line
    - collapse 3+ newlines to exactly 2

    Note: we do NOT aggressively trim; output stays as-is aside from the above.
    """
    s = raw.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\t", " ")
    # strip trailing whitespace per line
    s = "\n".join([ln.rstrip() for ln in s.split("\n")])
    # collapse 3+ newlines to exactly 2
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


def word_spans(norm_text: str) -> list[tuple[int, int]]:
    """
    Returns [(start_char, end_char), ...] for each word in norm_text.
    Word definition: regex \\S+.
    """
    return [(m.start(), m.end()) for m in _WORD_RE.finditer(norm_text)]


def _has_paragraph_boundary(norm_text: str, spans: list[tuple[int, int]], end_idx: int) -> bool:
    """
    end_idx is the word index AFTER the last word in the chunk (i.e. chunk ends at spans[end_idx-1][1]).
    We detect paragraph boundary by checking for \\n\\n in the whitespace after the last word.
    """
    if end_idx <= 0 or end_idx > len(spans):
        return False
    last_end = spans[end_idx - 1][1]
    next_start = spans[end_idx][0] if end_idx < len(spans) else len(norm_text)
    ws = norm_text[last_end:next_start]
    return "\n\n" in ws


def _has_sentence_boundary(norm_text: str, spans: list[tuple[int, int]], end_idx: int) -> bool:
    if end_idx <= 0 or end_idx > len(spans):
        return False
    last_end = spans[end_idx - 1][1]
    if last_end <= 0:
        return False
    ch = norm_text[last_end - 1]
    return ch in ".!?"


def chunk_text(
    raw_text: str,
    *,
    min_words: int = 600,
    target_words: int = 900,
    max_words: int = 1200,
    overlap_words: int = 0,
    boundary_scan_paragraph: int = 80,
    boundary_scan_sentence: int = 40,
) -> tuple[str, list[DocumentChunk]]:
    """
    Chunk normalized text into 600–1200 word chunks with stable char offsets.

    Returns (norm_text, chunks).
    """
    if overlap_words < 0:
        raise ValueError("overlap_words must be >= 0")
    if not (0 < min_words <= target_words <= max_words):
        raise ValueError("Expected 0 < min_words <= target_words <= max_words")

    norm = normalize_text(raw_text)
    spans = word_spans(norm)
    n = len(spans)
    if n == 0:
        return norm, []

    if n <= max_words:
        start = spans[0][0]
        end = spans[-1][1]
        txt = norm[start:end]
        return norm, [
            DocumentChunk(
                chunk_index=0,
                start_offset=start,
                end_offset=end,
                word_count=n,
                text=txt,
            )
        ]

    chunks: list[DocumentChunk] = []
    i = 0
    ci = 0
    while i < n:
        remaining = n - i
        if remaining <= max_words:
            end_idx = n
        else:
            end0 = min(i + target_words, n)
            lo = min(i + min_words, n)
            hi = min(i + max_words, n)

            # Prefer paragraph boundary near end0 within ±boundary_scan_paragraph (clamped to [lo, hi])
            p_lo = max(lo, end0 - boundary_scan_paragraph)
            p_hi = min(hi, end0 + boundary_scan_paragraph)
            para_candidates = [j for j in range(p_lo, p_hi + 1) if _has_paragraph_boundary(norm, spans, j)]
            if para_candidates:
                end_idx = min(para_candidates, key=lambda j: (abs(j - end0), j))
            else:
                # Try sentence boundary near end0 within ±boundary_scan_sentence
                s_lo = max(lo, end0 - boundary_scan_sentence)
                s_hi = min(hi, end0 + boundary_scan_sentence)
                sent_candidates = [j for j in range(s_lo, s_hi + 1) if _has_sentence_boundary(norm, spans, j)]
                if sent_candidates:
                    end_idx = min(sent_candidates, key=lambda j: (abs(j - end0), j))
                else:
                    end_idx = max(lo, min(end0, hi))

            # If we got stuck (shouldn't), force progress.
            if end_idx <= i:
                end_idx = min(n, i + max(min_words, 1))

        start_off = spans[i][0]
        end_off = spans[end_idx - 1][1]
        txt = norm[start_off:end_off]
        wc = end_idx - i
        chunks.append(
            DocumentChunk(
                chunk_index=ci,
                start_offset=start_off,
                end_offset=end_off,
                word_count=wc,
                text=txt,
            )
        )

        ci += 1
        next_i = end_idx - overlap_words
        if next_i <= i:
            next_i = end_idx
        i = next_i

    # Re-index chunk_index to be 0..N-1 (stable)
    chunks = [
        DocumentChunk(
            chunk_index=idx,
            start_offset=c.start_offset,
            end_offset=c.end_offset,
            word_count=c.word_count,
            text=c.text,
        )
        for idx, c in enumerate(chunks)
    ]
    return norm, chunks

