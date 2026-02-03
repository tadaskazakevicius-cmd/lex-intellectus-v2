from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from . import __version__
from .audit import log_event
from .paths import ensure_dirs, get_paths

# ✅ FIX: relative import inside the lex_server package
from .documents.api import router as documents_router
from .caseframe.api import router as caseframe_router
from .retrieval.api import router as retrieval_router

app = FastAPI(title="Lex Intellectus Server", version=__version__)

# Documents / cases API
app.include_router(documents_router, prefix="/api")
app.include_router(caseframe_router, prefix="/api")
app.include_router(retrieval_router, prefix="/api")


@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "version": __version__})


def _spa_dir() -> Path:
    # Vite build output is copied here (A3).
    return Path(__file__).resolve().parent / "static" / "spa"


@app.on_event("startup")
def on_startup() -> None:
    paths = get_paths()
    ensure_dirs(paths)

    audit_log = paths.data_dir / "audit_log.jsonl"
    first_start = not audit_log.exists()
    if first_start:
        log_event(
            audit_log,
            event_type="first_start",
            details={
                "app_dir": str(paths.app_dir),
                "data_dir": str(paths.data_dir),
                "model_dir": str(paths.model_dir),
                "temp_dir": str(paths.temp_dir),
            },
        )


@app.get("/")
def spa_root():
    spa_dir = _spa_dir()
    index = spa_dir / "index.html"
    if not index.exists():
        return PlainTextResponse(
            "UI not built yet. Build the SPA to enable the web UI.",
            status_code=200,
        )
    return FileResponse(index)


@app.get("/{full_path:path}")
def spa_fallback(full_path: str):
    # If the request didn't match an API route, serve SPA assets or fall back
    # to index.html for client-side routing.
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not Found")

    spa_dir = _spa_dir()
    index = spa_dir / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="UI not built")

    candidate = (spa_dir / full_path).resolve()
    # Prevent path traversal.
    if spa_dir.resolve() not in candidate.parents and candidate != spa_dir.resolve():
        raise HTTPException(status_code=404, detail="Not Found")

    if candidate.is_file():
        return FileResponse(candidate)

    # Client-side routes (no matching file) fall back to index.html
    return FileResponse(index)
