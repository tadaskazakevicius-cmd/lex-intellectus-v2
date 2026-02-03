from __future__ import annotations

from dataclasses import dataclass

import sqlite3

from .fts_retrieval import FtsFilter, fts_search
from .query_builder import QueryAtom, QueryPlan


@dataclass(frozen=True)
class AggregatedHit:
    chunk_id: str
    practice_doc_id: str
    bm25_score: float  # best (lowest) bm25 among matches
    score: float  # aggregated score using atom weights
    matches: list[dict]  # debug info: {kind,text,weight,bm25_score}


def execute_fts_plan(
    conn: sqlite3.Connection,
    plan: QueryPlan,
    *,
    top_n: int = 10,
    per_atom: int = 10,
    flt: FtsFilter | None = None,
) -> list[AggregatedHit]:
    """
    Execute multiple FTS queries from an E1 QueryPlan and aggregate results.

    Aggregation:
    - base = -bm25_score              (bm25: lower/more negative is better)
    - atom_score = atom.weight * base
    - per chunk:
        score = max(atom_score) across matches (not sum)
        bm25_score = min(bm25_score) across matches

    Sorting:
    - score DESC, bm25_score ASC, chunk_id ASC
    """
    if top_n <= 0 or per_atom <= 0 or not plan.atoms:
        return []

    agg: dict[str, dict] = {}

    for atom in plan.atoms:
        hits = fts_search(conn, atom.text, top_n=per_atom, flt=flt)
        for h in hits:
            base = -float(h.bm25_score)
            atom_score = float(atom.weight) * base
            entry = agg.get(h.chunk_id)
            match_info = {
                "kind": atom.kind,
                "text": atom.text,
                "weight": float(atom.weight),
                "bm25_score": float(h.bm25_score),
            }
            if entry is None:
                agg[h.chunk_id] = {
                    "chunk_id": h.chunk_id,
                    "practice_doc_id": h.practice_doc_id,
                    "bm25_score": float(h.bm25_score),
                    "score": float(atom_score),
                    "matches": [match_info],
                }
            else:
                entry["bm25_score"] = min(float(entry["bm25_score"]), float(h.bm25_score))
                entry["score"] = max(float(entry["score"]), float(atom_score))
                entry["matches"].append(match_info)

    out = [
        AggregatedHit(
            chunk_id=str(v["chunk_id"]),
            practice_doc_id=str(v["practice_doc_id"]),
            bm25_score=float(v["bm25_score"]),
            score=float(v["score"]),
            matches=list(v["matches"]),
        )
        for v in agg.values()
    ]
    out.sort(key=lambda x: (-x.score, x.bm25_score, x.chunk_id))
    return out[: int(top_n)]

