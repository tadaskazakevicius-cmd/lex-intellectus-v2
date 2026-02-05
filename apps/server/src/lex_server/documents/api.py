from __future__ import annotations

import sqlite3

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from ..audit import log_event
from ..paths import get_paths
from .pipeline import process_document
from .storage import IngestResult, connect_db, ingest_uploadfile, list_case_documents, set_document_status

# ✅ NO "/api" here – prefix is added in main.py
router = APIRouter()


@router.post("/cases/{case_id}/documents")
async def upload_case_document(
    case_id: str,
    background: BackgroundTasks,
    files: list[UploadFile] | None = File(default=None),
    file: UploadFile | None = File(default=None),
) -> JSONResponse:
    to_ingest = list(files or [])
    if file is not None:
        to_ingest.append(file)
    if not to_ingest:
        raise HTTPException(status_code=400, detail="No files provided")

    out: list[dict] = []
    for f in to_ingest:
        try:
            res: IngestResult = await ingest_uploadfile(case_id, f)
        except ValueError as e:
            raise HTTPException(status_code=415, detail=str(e)) from e

        out.append(
            {
                "id": res.id,
                "case_id": res.case_id,
                "original_name": res.original_name,
                "mime": res.mime,
                "size_bytes": res.size_bytes,
                "sha256": res.sha256_hex,
                "storage_relpath": res.storage_relpath,
                "deduped": res.deduped,
                "uploaded_at_utc": res.uploaded_at_utc,
                "status": res.status,
                "error": res.error,
            }
        )

        # Audit (JSONL under data_dir)
        paths = get_paths()
        log_event(
            paths.data_dir / "audit_log.jsonl",
            event_type="document_uploaded",
            details={
                "case_id": case_id,
                "doc_id": res.id,
                "filename": res.original_name,
                "size_bytes": res.size_bytes,
                "deduped": res.deduped,
            },
        )

        # Background processing: queued -> processing -> done/failed
        background.add_task(_process_doc_background, res.id)

    return JSONResponse(out)


def _process_doc_background(document_id: int) -> None:
    con = connect_db()
    try:
        con.execute("PRAGMA foreign_keys = ON;")
        set_document_status(con, document_id, status="processing", error=None)
        try:
            _ = process_document(con, document_id, base_data_dir=get_paths().data_dir)
            set_document_status(con, document_id, status="done", error=None)
        except Exception as e:
            set_document_status(con, document_id, status="failed", error=str(e)[:1000])
    finally:
        con.close()


@router.get("/cases/{case_id}/documents")
def list_documents(case_id: str) -> JSONResponse:
    con = connect_db()
    try:
        con.execute("PRAGMA foreign_keys = ON;")
        docs = list_case_documents(con, case_id)
        return JSONResponse({"documents": docs})
    finally:
        con.close()
