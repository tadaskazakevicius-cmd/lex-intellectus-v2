from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .fts_retrieval import FtsFilter, fts_search
from .query_builder import QueryAtom, QueryPlan
from .query_executor import execute_fts_plan
from .vector_retrieval import VectorFilter, vector_retrieve
from .vector_index import VectorIndex
from .hybrid_retrieval import hybrid_retrieve
from .persistence import create_run, load_run, load_run_hits, persist_run_results


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


class VectorRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    filters: dict | None = None


@router.post("/retrieval/vector")
def retrieval_vector(req: VectorRequest) -> JSONResponse:
    """
    MVP vector endpoint.

    NOTE:
    This requires runtime configuration (model + index path). Tests use FakeEmbedder directly and do not
    depend on this endpoint.
    """
    import os
    from pathlib import Path

    model_path = os.environ.get("LEX_EMBED_ONNX_MODEL")
    index_path = os.environ.get("LEX_VECTOR_INDEX_PATH")
    dim = os.environ.get("LEX_VECTOR_DIM")
    if not model_path or not index_path or not dim:
        raise HTTPException(
            status_code=501,
            detail="Vector retrieval not configured. Set LEX_EMBED_ONNX_MODEL, LEX_VECTOR_INDEX_PATH, LEX_VECTOR_DIM.",
        )

    from .embedder_onnx import OnnxEmbedder

    embedder = OnnxEmbedder(Path(model_path))
    index = VectorIndex.load(Path(index_path), dim=int(dim), space="cosine")

    flt = VectorFilter(practice_doc_id=(req.filters or {}).get("practice_doc_id") if req.filters else None)

    dbp = _db_path()
    if not dbp.exists():
        raise HTTPException(status_code=404, detail="DB not found")

    con = sqlite3.connect(dbp)
    try:
        con.execute("PRAGMA foreign_keys = ON;")
        hits = vector_retrieve(con, index, embedder, req.query, top_k=req.top_k, flt=flt)
        return JSONResponse(
            {
                "hits": [
                    {"chunk_id": h.chunk_id, "practice_doc_id": h.practice_doc_id, "distance": h.distance}
                    for h in hits
                ]
            }
        )
    finally:
        con.close()


class HybridRequest(BaseModel):
    query: str
    top_n: int = Field(default=10, ge=1, le=100)
    filters: FtsFilter | None = None
    use_fts: bool = True
    use_vector: bool = True


@router.post("/retrieval/hybrid")
def retrieval_hybrid(req: HybridRequest) -> JSONResponse:
    dbp = _db_path()
    if not dbp.exists():
        raise HTTPException(status_code=404, detail="DB not found")

    con = sqlite3.connect(dbp)
    try:
        con.execute("PRAGMA foreign_keys = ON;")
        hits = hybrid_retrieve(
            con,
            req.query,
            top_n=req.top_n,
            filters=req.filters,
            use_fts=req.use_fts,
            use_vector=req.use_vector,
        )
        return JSONResponse(
            {
                "hits": [
                    {
                        "chunk_id": h.chunk_id,
                        "practice_doc_id": h.practice_doc_id,
                        "score": h.score,
                        "sources": h.sources,
                        "citations": [
                            {
                                "quote": c.quote,
                                "start": c.start,
                                "end": c.end,
                                "source_url": c.source_url,
                            }
                            for c in h.citations
                        ],
                    }
                    for h in hits
                ]
            }
        )
    finally:
        con.close()


@router.post("/retrieval/hybrid_run")
def retrieval_hybrid_run(req: HybridRequest) -> JSONResponse:
    dbp = _db_path()
    if not dbp.exists():
        raise HTTPException(status_code=404, detail="DB not found")

    con = sqlite3.connect(dbp)
    try:
        con.execute("PRAGMA foreign_keys = ON;")
        hits = hybrid_retrieve(
            con,
            req.query,
            top_n=req.top_n,
            filters=req.filters,
            use_fts=req.use_fts,
            use_vector=req.use_vector,
        )

        run_id = create_run(
            con,
            query=req.query,
            top_n=req.top_n,
            filters=req.filters.model_dump() if req.filters is not None else None,
            use_fts=req.use_fts,
            use_vector=req.use_vector,
            algo_version="hybrid_v1",
        )
        persist_run_results(con, run_id, hits)

        return JSONResponse(
            {
                "run_id": run_id,
                "hits": [
                    {
                        "chunk_id": h.chunk_id,
                        "practice_doc_id": h.practice_doc_id,
                        "score": h.score,
                        "sources": h.sources,
                        "citations": [
                            {
                                "quote": c.quote,
                                "start": c.start,
                                "end": c.end,
                                "source_url": c.source_url,
                            }
                            for c in h.citations
                        ],
                    }
                    for h in hits
                ],
            }
        )
    finally:
        con.close()


@router.get("/retrieval/runs/{run_id}")
def retrieval_get_run(run_id: str) -> JSONResponse:
    dbp = _db_path()
    if not dbp.exists():
        raise HTTPException(status_code=404, detail="DB not found")

    con = sqlite3.connect(dbp)
    try:
        con.execute("PRAGMA foreign_keys = ON;")
        try:
            run = load_run(con, run_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found")

        hits = load_run_hits(con, run_id)
        return JSONResponse(
            {
                "run": run,
                "hits": [
                    {
                        "chunk_id": h.chunk_id,
                        "practice_doc_id": h.practice_doc_id,
                        "score": h.score,
                        "sources": h.sources,
                        "citations": [
                            {
                                "quote": c.quote,
                                "start": c.start,
                                "end": c.end,
                                "source_url": c.source_url,
                            }
                            for c in h.citations
                        ],
                    }
                    for h in hits
                ],
            }
        )
    finally:
        con.close()

