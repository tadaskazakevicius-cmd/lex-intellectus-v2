from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .fts_retrieval import FtsFilter, fts_search
from .query_builder import QueryAtom, QueryPlan
from .query_executor import execute_fts_plan


router = APIRouter()


def _db_path() -> Path:
    local = Path.cwd() / ".localdata" / "app.db"
    if local.exists():
        return local
    from ..paths import get_paths

    return get_paths().data_dir / "app.db"


class FtsRequest(BaseModel):
    query: str
    top_n: int = Field(default=10, ge=1, le=100)
    filters: FtsFilter | None = None


class FtsResponse(BaseModel):
    hits: list[dict]


@router.post("/retrieval/fts")
def retrieval_fts(req: FtsRequest) -> JSONResponse:
    dbp = _db_path()
    if not dbp.exists():
        raise HTTPException(status_code=404, detail="DB not found")

    con = sqlite3.connect(dbp)
    try:
        con.execute("PRAGMA foreign_keys = ON;")
        try:
            hits = fts_search(con, req.query, top_n=req.top_n, flt=req.filters)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        return JSONResponse(
            {
                "hits": [
                    {
                        "chunk_id": h.chunk_id,
                        "practice_doc_id": h.practice_doc_id,
                        "bm25_score": h.bm25_score,
                    }
                    for h in hits
                ]
            }
        )
    finally:
        con.close()


class PlanAtomIn(BaseModel):
    text: str
    kind: str
    weight: float
    filters: dict | None = None


class PlanIn(BaseModel):
    case_id: str | None = None
    atoms: list[PlanAtomIn]
    k: int = 6


class FtsPlanRequest(BaseModel):
    plan: PlanIn
    top_n: int = Field(default=10, ge=1, le=100)
    per_atom: int = Field(default=10, ge=1, le=100)
    filters: FtsFilter | None = None


@router.post("/retrieval/fts_plan")
def retrieval_fts_plan(req: FtsPlanRequest) -> JSONResponse:
    dbp = _db_path()
    if not dbp.exists():
        raise HTTPException(status_code=404, detail="DB not found")

    # Minimal conversion to internal QueryPlan/QueryAtom
    atoms: list[QueryAtom] = []
    for a in req.plan.atoms:
        if a.kind not in ("keywords", "phrase", "norm"):
            raise HTTPException(status_code=400, detail=f"Invalid atom.kind: {a.kind}")
        atoms.append(QueryAtom(text=a.text, kind=a.kind, weight=float(a.weight), filters=None))
    plan = QueryPlan(case_id=req.plan.case_id, atoms=atoms[: int(req.plan.k)], k=int(req.plan.k))

    con = sqlite3.connect(dbp)
    try:
        con.execute("PRAGMA foreign_keys = ON;")
        try:
            hits = execute_fts_plan(
                con,
                plan,
                top_n=req.top_n,
                per_atom=req.per_atom,
                flt=req.filters,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        return JSONResponse(
            {
                "hits": [
                    {
                        "chunk_id": h.chunk_id,
                        "practice_doc_id": h.practice_doc_id,
                        "bm25_score": h.bm25_score,
                    }
                    for h in hits
                ]
            }
        )
    finally:
        con.close()

