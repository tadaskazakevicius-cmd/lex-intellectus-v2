from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..audit import log_event
from ..paths import get_paths

router = APIRouter()


def _db_path() -> Path:
    """
    Single source of truth:
    - default: A2 data_dir/app.db
    - optional override: LEX_DB_PATH (absolute or relative to CWD)
    """
    override = os.environ.get("LEX_DB_PATH", "").strip()
    if override:
        p = Path(override)
        return p if p.is_absolute() else (Path.cwd() / p)

    return get_paths().data_dir / "app.db"


class CreateCaseIn(BaseModel):
    title: str = Field(min_length=1)
    description: str | None = None
    category: str | None = None


@router.post("/cases")
def create_case(body: CreateCaseIn) -> JSONResponse:
    case_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    dbp = _db_path()
    dbp.parent.mkdir(parents=True, exist_ok=True)
    if not dbp.exists():
        raise HTTPException(status_code=404, detail=f"DB not found at {dbp}")

    con = sqlite3.connect(dbp)
    try:
        con.execute("PRAGMA foreign_keys = ON;")
        with con:
            con.execute(
                """
                INSERT INTO cases(id, title, description, category, created_at_utc)
                VALUES (?, ?, ?, ?, ?);
                """,
                (case_id, body.title, body.description, body.category, now),
            )

        # Audit (JSONL under data_dir)
        paths = get_paths()
        log_event(
            paths.data_dir / "audit_log.jsonl",
            event_type="case_created",
            details={"case_id": case_id, "title": body.title, "category": body.category},
        )

        return JSONResponse(
            {
                "case_id": case_id,
                "title": body.title,
                "description": body.description,
                "category": body.category,
                "created_at_utc": now,
            }
        )
    finally:
        con.close()


@router.get("/cases/{case_id}")
def get_case(case_id: str) -> JSONResponse:
    dbp = _db_path()
    if not dbp.exists():
        raise HTTPException(status_code=404, detail=f"DB not found at {dbp}")

    con = sqlite3.connect(dbp)
    try:
        con.execute("PRAGMA foreign_keys = ON;")
        row = con.execute(
            """
            SELECT id, title, description, category, created_at_utc
            FROM cases
            WHERE id = ?;
            """,
            (case_id,),
        ).fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Case not found")

        return JSONResponse(
            {
                "case_id": str(row[0]),
                "title": str(row[1]),
                "description": (str(row[2]) if row[2] is not None else None),
                "category": (str(row[3]) if row[3] is not None else None),
                "created_at_utc": str(row[4]),
            }
        )
    finally:
        con.close()
