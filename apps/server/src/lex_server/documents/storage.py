from __future__ import annotations

import hashlib
import mimetypes
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ALLOWED_MIMES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
}


def data_dir() -> Path:
    """
    A2: Always store app data under OS-agnostic data_dir.
    """
    from ..paths import get_paths

    return get_paths().data_dir


def db_path() -> Path:
    """
    A2: Always use data_dir/app.db as the primary DB location.

    Optional override:
      - LEX_DB_PATH: absolute or relative path (relative to current working dir)
    """
    override = os.environ.get("LEX_DB_PATH", "").strip()
    if override:
        p = Path(override)
        return p if p.is_absolute() else (Path.cwd() / p)

    return data_dir() / "app.db"


def connect_db() -> sqlite3.Connection:
    p = db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.execute("PRAGMA foreign_keys = ON;")
    return con


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(name: str) -> str:
    name = (name or "upload.bin").strip()
    name = name.replace("\\", "_").replace("/", "_")
    name = _SAFE_NAME_RE.sub("_", name)
    return name or "upload.bin"


def detect_mime(filename: str, content_type_header: str | None) -> str:
    """
    Rules:
    - If content_type_header is present and not application/octet-stream, prefer it.
    - Else use mimetypes.guess_type(filename)[0] if available.
    - Else fallback application/octet-stream.
    """
    if content_type_header:
        ct = content_type_header.split(";")[0].strip().lower()
        if ct and ct != "application/octet-stream":
            return ct
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


def docs_dir(case_id: str) -> Path:
    # G1: store uploads under data_dir/cases/{case_id}/uploads
    return data_dir() / "cases" / case_id / "uploads"


def final_relpath(case_id: str, sha256_hex: str, original_name: str) -> str:
    safe = sanitize_filename(original_name)
    rel = Path("cases") / case_id / "uploads" / f"{sha256_hex}__{safe}"
    return rel.as_posix()


def ensure_allowed_or_415(mime: str, filename: str) -> None:
    ext = Path(filename).suffix.lower()
    if mime in ALLOWED_MIMES:
        return
    if ext in {".pdf", ".docx", ".txt"}:
        # Allow if extension suggests supported type but header/guess was generic.
        return
    raise ValueError(
        f"Unsupported document type. mime={mime!r}, filename={filename!r}. Allowed: PDF/DOCX/TXT."
    )


@dataclass(frozen=True)
class IngestResult:
    id: int
    case_id: str
    original_name: str
    mime: str
    size_bytes: int
    sha256_hex: str
    storage_relpath: str
    deduped: bool
    status: str
    uploaded_at_utc: str
    error: str | None


def _fetch_existing(con: sqlite3.Connection, case_id: str, sha256_hex: str) -> IngestResult | None:
    row = con.execute(
        """
        SELECT
          id, case_id, original_name, mime, size_bytes, sha256_hex, storage_relpath,
          COALESCE(status, 'queued') AS status,
          created_at_utc,
          error
        FROM case_documents
        WHERE case_id = ? AND sha256_hex = ?
        """,
        (case_id, sha256_hex),
    ).fetchone()

    if not row:
        return None

    return IngestResult(
        id=int(row[0]),
        case_id=str(row[1]),
        original_name=str(row[2]),
        mime=str(row[3]),
        size_bytes=int(row[4]),
        sha256_hex=str(row[5]),
        storage_relpath=str(row[6]),
        deduped=True,
        status=str(row[7]),
        uploaded_at_utc=str(row[8]),
        error=(str(row[9]) if row[9] is not None else None),
    )


def _insert_row(
    con: sqlite3.Connection,
    *,
    case_id: str,
    original_name: str,
    mime: str,
    size_bytes: int,
    sha256_hex: str,
    storage_relpath: str,
) -> IngestResult:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    cur = con.execute(
        """
        INSERT INTO case_documents(
          case_id, original_name, mime, size_bytes, sha256_hex, storage_relpath,
          status, created_at_utc, updated_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?);
        """,
        (case_id, original_name, mime, int(size_bytes), sha256_hex, storage_relpath, now, now),
    )

    return IngestResult(
        id=int(cur.lastrowid),
        case_id=case_id,
        original_name=original_name,
        mime=mime,
        size_bytes=int(size_bytes),
        sha256_hex=sha256_hex,
        storage_relpath=storage_relpath,
        deduped=False,
        status="queued",
        uploaded_at_utc=now,
        error=None,
    )


async def ingest_uploadfile(case_id: str, upload) -> IngestResult:
    """
    Ingest UploadFile into data_dir/cases/{case_id}/uploads and insert into DB.
    Dedupes by UNIQUE(case_id, sha256_hex) (if the DB has that unique index/constraint).
    """
    original_name = upload.filename or "upload.bin"
    mime = detect_mime(original_name, getattr(upload, "content_type", None))
    ensure_allowed_or_415(mime, original_name)

    ddir = docs_dir(case_id)
    ddir.mkdir(parents=True, exist_ok=True)

    safe = sanitize_filename(original_name)
    tmp_name = f"tmp_{os.getpid()}_{safe}.uploading"
    tmp_path = ddir / tmp_name

    h = hashlib.sha256()
    size = 0

    with tmp_path.open("wb") as f:
        while True:
            b = await upload.read(1024 * 1024)
            if not b:
                break
            h.update(b)
            size += len(b)
            f.write(b)

    sha = h.hexdigest()
    rel = final_relpath(case_id, sha, original_name)

    final_path = data_dir() / rel
    final_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp_path, final_path)

    con = connect_db()
    try:
        with con:
            try:
                return _insert_row(
                    con,
                    case_id=case_id,
                    original_name=original_name,
                    mime=mime,
                    size_bytes=size,
                    sha256_hex=sha,
                    storage_relpath=rel,
                )
            except sqlite3.IntegrityError:
                existing = _fetch_existing(con, case_id, sha)

                # Remove the newly created file if this was a dup.
                try:
                    final_path.unlink()
                except FileNotFoundError:
                    pass

                if existing:
                    return existing
                raise
    finally:
        con.close()


def list_case_documents(con: sqlite3.Connection, case_id: str) -> list[dict]:
    rows = con.execute(
        """
        SELECT
          id, case_id, original_name, mime, size_bytes, sha256_hex, storage_relpath,
          COALESCE(status, 'queued') AS status,
          created_at_utc,
          error
        FROM case_documents
        WHERE case_id = ?
        ORDER BY id DESC;
        """,
        (case_id,),
    ).fetchall()

    return [
        {
            "id": int(r[0]),
            "case_id": str(r[1]),
            "original_name": str(r[2]),
            "mime": str(r[3]),
            "size_bytes": int(r[4]),
            "sha256": str(r[5]),
            "storage_relpath": str(r[6]),
            "status": str(r[7]),
            "uploaded_at_utc": str(r[8]),
            "error": (str(r[9]) if r[9] is not None else None),
        }
        for r in rows
    ]


def set_document_status(
    con: sqlite3.Connection,
    document_id: int,
    *,
    status: str,
    error: str | None = None,
) -> None:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    with con:
        con.execute(
            """
            UPDATE case_documents
            SET status = ?, error = ?, updated_at_utc = ?
            WHERE id = ?;
            """,
            (status, error, now, int(document_id)),
        )
