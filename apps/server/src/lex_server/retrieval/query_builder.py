from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal


Kind = Literal["keywords", "phrase", "norm"]


@dataclass(frozen=True)
class QueryAtom:
    """
    One query element to be sent to retrieval backends.

    Notes for E2/E3:
    - **phrase** atoms (quoted) are typically good for FTS (BM25) precision.
    - **keywords** atoms are a broad net for BM25 and can help recall.
    - **norm** atoms (law references) are high-signal lexical anchors and often benefit BM25 heavily.
    - `weight` is intended for score fusion / boosting (hybrid ranking).
    """

    text: str
    kind: Kind
    weight: float
    filters: dict[str, Any] | None = None


@dataclass(frozen=True)
class QueryPlan:
    case_id: str | None
    atoms: list[QueryAtom]
    k: int


_WS_RE = re.compile(r"\s+")


def _collapse_ws(s: str) -> str:
    return _WS_RE.sub(" ", s).strip()


def _dedup_key(text: str) -> str:
    t = text.strip()
    if len(t) >= 2 and t[0] == '"' and t[-1] == '"':
        t = t[1:-1]
    return _collapse_ws(t).casefold()


def _quote_phrase(s: str) -> str:
    # Keep it simple: avoid embedding raw double quotes inside quotes.
    s = s.replace('"', "'")
    return f'"{s}"'


def _truncate_phrase(s: str, max_len: int = 160) -> str:
    if len(s) <= max_len:
        return s
    cut = s[:max_len]
    # Prefer not cutting in the middle of a word.
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
        cut = cut.rstrip()
        if cut:
            return cut
    return s[:max_len].rstrip()


def _standardize_norm(s: str) -> str:
    # Minimal standardization: collapse whitespace and remove spaces around dots.
    s = _collapse_ws(s)
    s = re.sub(r"\s*\.\s*", ".", s)  # "6. 248" -> "6.248"
    # Keep "str." readable (optional): ensure a space before "str." if missing
    s = re.sub(r"(?i)(\d)(str\.)", r"\1 \2", s)
    return _collapse_ws(s)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_path(obj: Any, *keys: str) -> Any:
    cur: Any = obj
    for k in keys:
        cur = _get(cur, k, None)
        if cur is None:
            return None
    return cur


def build_query_plan(case_frame: dict | Any, k: int = 6) -> QueryPlan:
    """
    Build a deterministic query plan from a CaseFrame-like object (dict or model).

    Candidate sources (if missing => skipped):
    - facts.summary -> 1 phrase atom (highest priority)
    - norms/legal_basis -> norm atoms (each separately)
    - claims/issues/questions -> up to 2 phrase atoms
    - facts.keywords -> 1 keywords atom

    Final atoms are capped at K with priority:
    summary phrase, norms, issues/claims phrases, keywords.
    """
    if k <= 0:
        return QueryPlan(case_id=_get(case_frame, "case_id", None), atoms=[], k=int(k))

    case_id = _get(case_frame, "case_id", None)

    atoms: list[QueryAtom] = []
    seen: set[str] = set()

    def add_atom(text: str, kind: Kind, weight: float) -> None:
        text = _collapse_ws(text)
        if not text:
            return
        key = _dedup_key(text)
        if key in seen:
            return
        seen.add(key)
        atoms.append(QueryAtom(text=text, kind=kind, weight=float(weight), filters=None))

    # 1) Summary -> phrase (quoted), weight 1.4
    summary = _get_path(case_frame, "facts", "summary")
    if isinstance(summary, str):
        s = _truncate_phrase(_collapse_ws(summary), 160)
        if s:
            add_atom(_quote_phrase(s), "phrase", 1.4)

    # 2) Norms / legal_basis -> norm atoms, weight 1.3
    norms = _get(case_frame, "norms", None)
    if norms is None:
        norms = _get(case_frame, "legal_basis", None)
    if isinstance(norms, list):
        for n in norms:
            if isinstance(n, str):
                txt = _standardize_norm(n)
            elif isinstance(n, dict):
                title = n.get("title") or n.get("name") or ""
                article = n.get("article") or n.get("ref") or ""
                if title and article:
                    txt = _standardize_norm(f"{title} {article}")
                elif title:
                    txt = _standardize_norm(str(title))
                elif article:
                    txt = _standardize_norm(str(article))
                else:
                    continue
            else:
                continue
            if txt:
                add_atom(txt, "norm", 1.3)

    # 3) Issues / claims / questions -> up to 2 phrase atoms, weight 1.2
    phrase_sources: list[str] = []
    for key in ("claims", "issues", "questions"):
        v = _get(case_frame, key, None)
        if isinstance(v, list):
            phrase_sources.extend([x for x in v if isinstance(x, str)])
    taken = 0
    for s in phrase_sources:
        if taken >= 2:
            break
        t = _truncate_phrase(_collapse_ws(s), 160)
        if not t:
            continue
        before = len(atoms)
        add_atom(_quote_phrase(t), "phrase", 1.2)
        if len(atoms) != before:
            taken += 1

    # 4) Keywords -> one keywords atom, weight 1.0
    keywords = _get_path(case_frame, "facts", "keywords")
    if keywords is None:
        # Backwards-compat / robustness: allow top-level keywords too.
        keywords = _get(case_frame, "keywords", None)
    if isinstance(keywords, list):
        kws = [_collapse_ws(x) for x in keywords if isinstance(x, str) and _collapse_ws(x)]
        if kws:
            add_atom(" ".join(kws), "keywords", 1.0)

    # Enforce K with required priority already respected by construction.
    atoms = atoms[: int(k)]
    return QueryPlan(case_id=str(case_id) if case_id is not None else None, atoms=atoms, k=int(k))

