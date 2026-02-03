from __future__ import annotations

import hashlib
import mimetypes
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path


ALLOWED_MIMES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
}


def root_dir() -> Path:
    return Path.cwd()


def data_dir() -> Path:
    return root_dir() / ".localdata"


def db_path() -> Path:
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
    return data_dir() / "user_docs" / case_id


def final_relpath(case_id: str, sha256_hex: str, original_name: str) -> str:
    safe = sanitize_filename(original_name)
    rel = Path("user_docs") / case_id / f"{sha256_hex}__{safe}"
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


def _fetch_existing(con: sqlite3.Connection, case_id: str, sha256_hex: str) -> IngestResult | None:
    row = con.execute(
        """
        SELECT id, case_id, original_name, mime, size_bytes, sha256_hex, storage_relpath
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
    cur = con.execute(
        """
        INSERT INTO case_documents(case_id, original_name, mime, size_bytes, sha256_hex, storage_relpath)
        VALUES (?, ?, ?, ?, ?, ?);
        """,
        (case_id, original_name, mime, int(size_bytes), sha256_hex, storage_relpath),
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
    )


async def ingest_uploadfile(case_id: str, upload) -> IngestResult:
    """
    Ingest UploadFile into `.localdata/user_docs/<case_id>/...` and insert into DB.
    Dedupes by UNIQUE(case_id, sha256_hex).
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
                res = _insert_row(
                    con,
                    case_id=case_id,
                    original_name=original_name,
                    mime=mime,
                    size_bytes=size,
                    sha256_hex=sha,
                    storage_relpath=rel,
                )
                return res
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

