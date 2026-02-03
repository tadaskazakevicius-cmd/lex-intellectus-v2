from __future__ import annotations

import hashlib
import os
import sqlite3
import sys
from pathlib import Path

from .chunking import normalize_text
from .pipeline import process_document
from .text_extract import extract_text


def _repo_root() -> Path:
    # .../apps/server/src/lex_server/documents/test_pipeline.py -> repo root
    return Path(__file__).resolve().parents[5]


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    root = _repo_root()
    os.chdir(root)

    # Isolated test DB (do NOT touch app.db)
    test_db = root / ".localdata" / "test_pipeline.db"
    test_db.parent.mkdir(parents=True, exist_ok=True)
    if test_db.exists():
        test_db.unlink()

    # Load migrate() from apps/server/db/migrate.py via namespace package import.
    server_dir = root / "apps" / "server"
    if str(server_dir) not in sys.path:
        sys.path.insert(0, str(server_dir))
    from db.migrate import migrate  # type: ignore

    v = migrate(
        db_path=test_db,
        schema_sql_path=Path("apps/server/db/schema.sql"),
        migrations_dir=Path("apps/server/db/migrations"),
    )
    if int(v) < 5:
        raise SystemExit(f"DB migration failed; user_version={v}")

    con = sqlite3.connect(test_db)
    try:
        con.execute("PRAGMA foreign_keys = ON;")

        case_id = "testcase_pipeline"
        golden = Path("apps/server/src/lex_server/documents/golden/sample.txt")
        content = golden.read_bytes()
        sha = _sha256_hex(content)
        size = len(content)

        # Copy real file into the storage location the pipeline expects.
        storage_rel = Path("user_docs") / case_id / f"{sha}__sample.txt"
        abs_storage = root / ".localdata" / storage_rel
        abs_storage.parent.mkdir(parents=True, exist_ok=True)
        abs_storage.write_bytes(content)

        with con:
            cur = con.execute(
                """
                INSERT INTO case_documents(case_id, original_name, mime, size_bytes, sha256_hex, storage_relpath)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (case_id, "sample.txt", "text/plain", int(size), sha, storage_rel.as_posix()),
            )
            document_id = int(cur.lastrowid)

        # Process into chunks.
        n_chunks = process_document(con, document_id, base_data_dir=root / ".localdata")
        if n_chunks <= 0:
            raise SystemExit("Expected >0 chunks inserted")

        rows = con.execute(
            """
            SELECT chunk_index, start_offset, end_offset, word_count, text
            FROM document_chunks
            WHERE document_id = ?
            ORDER BY chunk_index ASC;
            """,
            (document_id,),
        ).fetchall()
        if len(rows) <= 0:
            raise SystemExit("No rows found in document_chunks")

        # Recompute normalized text like the pipeline does.
        raw_text = extract_text(abs_storage, mime="text/plain")
        norm_text = normalize_text(raw_text)

        idxs = [int(r[0]) for r in rows]
        if idxs != list(range(len(rows))):
            raise SystemExit(f"chunk_index not 0..N-1: {idxs}")

        prev_end = None
        for chunk_index, start, end, wc, txt in rows:
            start = int(start)
            end = int(end)
            wc = int(wc)
            if not (start < end):
                raise SystemExit(f"Bad offsets for chunk {chunk_index}: {start}..{end}")
            if wc <= 0:
                raise SystemExit(f"Bad word_count for chunk {chunk_index}: {wc}")
            if txt != norm_text[start:end]:
                raise SystemExit(
                    f"Text mismatch for chunk {chunk_index}: "
                    f"expected slice {norm_text[start:end]!r} got {txt!r}"
                )
            if prev_end is not None and prev_end > start:
                raise SystemExit("Offsets are not monotonic / overlap detected")
            prev_end = end

    finally:
        con.close()

    print("D3 pipeline test OK")


if __name__ == "__main__":
    main()

