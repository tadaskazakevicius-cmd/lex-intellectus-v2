from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from .generator import generate_case_frames
from .validate import validate_case_frames


router = APIRouter()


def _default_out_path(case_id: str) -> Path:
    return Path.cwd() / ".localdata" / "cases" / case_id / "case_frames.json"


def _db_path() -> Path:
    # Prefer dev-local DB, else fall back to platform data dir if configured.
    local = Path.cwd() / ".localdata" / "app.db"
    if local.exists():
        return local
    from ..paths import get_paths

    return get_paths().data_dir / "app.db"


@router.get("/cases/{case_id}/caseframe")
def get_caseframe(case_id: str):
    out_path = _default_out_path(case_id)
    if out_path.exists():
        return FileResponse(out_path, media_type="application/json")

    dbp = _db_path()
    if not dbp.exists():
        raise HTTPException(status_code=404, detail="DB not found")

    con = sqlite3.connect(dbp)
    try:
        con.execute("PRAGMA foreign_keys = ON;")
        data = generate_case_frames(con, case_id, out_path=out_path)
        validate_case_frames(data)
        return JSONResponse(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    finally:
        con.close()

