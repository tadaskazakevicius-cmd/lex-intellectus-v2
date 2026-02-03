from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .validate import validate_case_frames


def _utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _default_out_path(case_id: str) -> Path:
    # MVP preference: match other dev-time storage in repo root .localdata
    # while still allowing override via out_path in tests/callers.
    return Path.cwd() / ".localdata" / "cases" / case_id / "case_frames.json"


def generate_case_frames(
    con: sqlite3.Connection,
    case_id: str,
    *,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """
    Generate CaseFrames JSON (MVP) from uploaded documents + derived chunks.
    Writes pretty JSON to out_path (or default) and returns the dict.
    """
    docs = con.execute(
        """
        SELECT
          id, case_id, original_name, mime, size_bytes, sha256_hex, storage_relpath, created_at_utc
        FROM case_documents
        WHERE case_id = ?
        ORDER BY created_at_utc ASC, id ASC;
        """,
        (case_id,),
    ).fetchall()

    documents: list[dict[str, Any]] = []
    total_chunks = 0
    total_words = 0

    for doc_id, _case_id, original_name, mime, _size_bytes, sha256_hex, _storage_relpath, created_at_utc in docs:
        chunks = con.execute(
            """
            SELECT chunk_index, start_offset, end_offset, word_count, text
            FROM document_chunks
            WHERE document_id = ?
            ORDER BY chunk_index ASC;
            """,
            (doc_id,),
        ).fetchall()

        chunk_count = len(chunks)
        doc_words = sum(int(r[3]) for r in chunks)
        total_chunks += chunk_count
        total_words += doc_words

        sample_quotes: list[dict[str, Any]] = []
        for r in chunks[:3]:
            chunk_index, start_offset, end_offset, _wc, text = r
            preview = (text or "")[:200]
            sample_quotes.append(
                {
                    "chunk_index": int(chunk_index),
                    "start_offset": int(start_offset),
                    "end_offset": int(end_offset),
                    "text_preview": preview,
                }
            )

        documents.append(
            {
                "document_id": str(doc_id),
                "original_name": str(original_name),
                "mime": str(mime),
                "sha256": str(sha256_hex),
                "created_at": str(created_at_utc),
                "chunk_count": int(chunk_count),
                "total_words": int(doc_words),
                "sample_quotes": sample_quotes,
            }
        )

    out: dict[str, Any] = {
        "schema_version": "1.0",
        "case_id": case_id,
        "generated_at_utc": _utc_now_iso_z(),
        "documents_count": len(documents),
        "total_chunks": int(total_chunks),
        "total_words": int(total_words),
        "documents": documents,
    }

    validate_case_frames(out)

    out_path = out_path or _default_out_path(case_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out

