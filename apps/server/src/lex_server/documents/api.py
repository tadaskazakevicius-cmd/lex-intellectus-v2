from __future__ import annotations

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from .storage import IngestResult, ingest_uploadfile

# ✅ NO "/api" here – prefix is added in main.py
router = APIRouter()


@router.post("/cases/{case_id}/documents")
async def upload_case_document(
    case_id: str,
    file: UploadFile = File(...),
) -> JSONResponse:
    try:
        res: IngestResult = await ingest_uploadfile(case_id, file)
    except ValueError as e:
        raise HTTPException(status_code=415, detail=str(e)) from e

    return JSONResponse(
        {
            "id": res.id,
            "case_id": res.case_id,
            "original_name": res.original_name,
            "mime": res.mime,
            "size_bytes": res.size_bytes,
            "sha256": res.sha256_hex,
            "storage_relpath": res.storage_relpath,
            "deduped": res.deduped,
        }
    )
