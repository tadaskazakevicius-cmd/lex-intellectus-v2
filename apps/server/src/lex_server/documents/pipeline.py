from __future__ import annotations

import sqlite3
from pathlib import Path

from .chunking import chunk_text, normalize_text
from .text_extract import extract_text
from .storage import data_dir as _default_data_dir


def _abs_storage_path(storage_relpath: str, *, base_data_dir: Path) -> Path:
    # storage_relpath is stored as POSIX (forward slashes) in DB.
    return base_data_dir / Path(*storage_relpath.split("/"))


def process_document(con: sqlite3.Connection, document_id: int, *, base_data_dir: Path | None = None) -> int:
    """
    Rebuild derived chunk data for one uploaded document.

    Flow:
    - read case_documents row
    - read file from disk
    - extract_text (D2)
    - normalize_text (D3)
    - chunk_text (D3)
    - DELETE existing chunks (idempotent)
    - INSERT new chunks with deterministic IDs: f"{document_id}:{chunk_index}"

    Returns: number of chunks inserted.
    """
    base_data_dir = base_data_dir or _default_data_dir()
    row = con.execute(
        """
        SELECT storage_relpath, mime
        FROM case_documents
        WHERE id = ?;
        """,
        (int(document_id),),
    ).fetchone()
    if not row:
        raise ValueError(f"case_documents not found: id={document_id}")

    storage_relpath, mime = str(row[0]), str(row[1])
    abs_path = _abs_storage_path(storage_relpath, base_data_dir=base_data_dir)
    if not abs_path.exists():
        raise ValueError(f"document file missing on disk: {abs_path}")

    raw_text = extract_text(abs_path, mime=mime)
    norm = normalize_text(raw_text)
    norm2, chunks = chunk_text(norm)
    if norm2 != norm:
        # normalize_text should be idempotent; chunk_text re-normalizes internally.
        raise ValueError("normalize_text is not idempotent with chunk_text()")

    with con:
        con.execute("DELETE FROM document_chunks WHERE document_id = ?;", (int(document_id),))
        for c in chunks:
            chunk_id = f"{int(document_id)}:{c.chunk_index}"
            con.execute(
                """
                INSERT INTO document_chunks(
                  id, document_id, chunk_index,
                  start_offset, end_offset, word_count, text
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    chunk_id,
                    int(document_id),
                    int(c.chunk_index),
                    int(c.start_offset),
                    int(c.end_offset),
                    int(c.word_count),
                    c.text,
                ),
            )

    return len(chunks)

