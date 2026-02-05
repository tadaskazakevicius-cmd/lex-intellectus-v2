from __future__ import annotations

import os
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
    """
    DB path resolution:
    1) Optional override: LEX_DB_PATH
    2) Primary: get_paths().data_dir / "app.db"
    3) Legacy: if .localdata/app.db exists and primary is missing, copy once to primary
    """
    override = os.environ.get("LEX_DB_PATH", "").strip()
    if override:
        p = Path(override)
        return p if p.is_absolute() else (Path.cwd() / p)

    from ..paths import get_paths

    primary = get_paths().data_dir / "app.db"
    primary.parent.mkdir(parents=True, exist_ok=True)

    legacy = Path.cwd() / ".localdata" / "app.db"
    if not primary.exists() and legacy.exists():
        primary.write_bytes(legacy.read_bytes())

    return primary


def _require_existing_file(p: Path, *, env_name: str) -> Path:
    if not p.exists():
        raise HTTPException(
            status_code=501,
            detail=f"{env_name} points to missing path: {str(p)}",
        )
    if not p.is_file():
        raise HTTPException(
            status_code=501,
            detail=f"{env_name} must be a FILE path, got directory: {str(p)}",
        )
    return p


def _resolve_vector_index_path(raw: str) -> Path:
    """
    LEX_VECTOR_INDEX_PATH can be either:
    - a direct file path to the HNSW index
    - OR a directory, in which case we try common filenames inside it
    """
    p = Path(raw)

    # If direct file path
    if p.exists() and p.is_file():
        return p

    # If directory: try common index filenames
    if p.exists() and p.is_dir():
        candidates = [
            "index.bin",
            "hnsw.index",
            "vector.index",
            "vectors.index",
            "index.hnsw",
            "hnswlib.index",
        ]
        for name in candidates:
            cand = p / name
            if cand.exists() and cand.is_file():
                return cand

        raise HTTPException(
            status_code=501,
            detail=(
                "Vector index directory provided but no index file found. "
                f"Directory: {str(p)}. "
                "Expected one of: " + ", ".join(candidates) + ". "
                "Set LEX_VECTOR_INDEX_PATH to the actual index FILE path."
            ),
        )

    # Doesn't exist at all
    raise HTTPException(
        status_code=501,
        detail=f"LEX_VECTOR_INDEX_PATH points to missing path: {str(p)}",
    )


class FtsRequest(BaseModel):
    query: str
    top_n: int = Field(default=10, ge=1, le=100)
    filters: FtsFilter | None = None


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
    This requires runtime configuration (model + index path).
    """
    model_raw = os.environ.get("LEX_EMBED_ONNX_MODEL", "").strip()
    index_raw = os.environ.get("LEX_VECTOR_INDEX_PATH", "").strip()
    dim_raw = os.environ.get("LEX_VECTOR_DIM", "").strip()

    if not model_raw or not index_raw or not dim_raw:
        raise HTTPException(
            status_code=501,
            detail="Vector retrieval not configured. Set LEX_EMBED_ONNX_MODEL, LEX_VECTOR_INDEX_PATH, LEX_VECTOR_DIM.",
        )

    # Validate dim
    try:
        dim = int(dim_raw)
    except ValueError as e:
        raise HTTPException(status_code=501, detail=f"LEX_VECTOR_DIM must be int, got: {dim_raw!r}") from e

    model_path = Path(model_raw)
    model_path = _require_existing_file(model_path, env_name="LEX_EMBED_ONNX_MODEL")

    index_file = _resolve_vector_index_path(index_raw)

    from .embedder_onnx import OnnxEmbedder

    embedder = OnnxEmbedder(model_path)

    try:
        index = VectorIndex.load(index_file, dim=dim, space="cosine")
    except RuntimeError as e:
        # This is typically: "Cannot open file" or corrupted index
        raise HTTPException(
            status_code=501,
            detail=f"Failed to load vector index from {str(index_file)}: {str(e)}",
        ) from e

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
