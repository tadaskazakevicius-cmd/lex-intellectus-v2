from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from pathlib import Path


def _repo_root() -> Path:
    # .../apps/server/src/lex_server/caseframe/test_caseframe.py -> repo root
    return Path(__file__).resolve().parents[5]


def _count_words(s: str) -> int:
    return len(re.findall(r"\S+", s))


def main() -> None:
    root = _repo_root()
    os.chdir(root)

    # Ensure imports for db.migrate (apps/server) and lex_server (apps/server/src)
    server_dir = root / "apps" / "server"
    src_dir = server_dir / "src"
    if str(server_dir) not in sys.path:
        sys.path.insert(0, str(server_dir))
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from db.migrate import migrate  # type: ignore
    from lex_server.caseframe.generator import generate_case_frames
    from lex_server.caseframe.validate import validate_case_frames

    test_db = root / ".localdata" / "test_caseframe.db"
    test_db.parent.mkdir(parents=True, exist_ok=True)
    if test_db.exists():
        test_db.unlink()

    v = migrate(
        db_path=test_db,
        schema_sql_path=Path("apps/server/db/schema.sql"),
        migrations_dir=Path("apps/server/db/migrations"),
    )
    if int(v) < 5:
        raise SystemExit(f"Migration failed; user_version={v}")

    con = sqlite3.connect(test_db)
    try:
        con.execute("PRAGMA foreign_keys = ON;")

        case_id = "case_test"
        with con:
            cur = con.execute(
                """
                INSERT INTO case_documents(case_id, original_name, mime, size_bytes, sha256_hex, storage_relpath)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (case_id, "dummy.txt", "text/plain", 10, "00" * 32, "user_docs/case_test/dummy.txt"),
            )
            doc_id = int(cur.lastrowid)

        # Build a fake normalized text and slice it into 4 chunks with consistent offsets.
        norm_text = (
            "PVM deklaracija FR0600.\n\n"
            "Tai testinis dokumentas.\n"
            "Cituojama eilute: \"PVM\".\n\n"
            "Pabaiga."
        )

        # Make 4 contiguous chunks.
        parts = [
            (0, 24),   # "PVM deklaracija FR0600."
            (24, 50),
            (50, 95),
            (95, len(norm_text)),
        ]
        with con:
            for i, (s, e) in enumerate(parts):
                txt = norm_text[s:e]
                con.execute(
                    """
                    INSERT INTO document_chunks(
                      id, document_id, chunk_index, start_offset, end_offset, word_count, text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (f"{doc_id}:{i}", doc_id, i, s, e, _count_words(txt), txt),
                )

        out_path = root / ".localdata" / "cases" / case_id / "case_frames.json"
        frames = generate_case_frames(con, case_id, out_path=out_path)
        validate_case_frames(frames)

        assert frames["case_id"] == case_id
        assert frames["documents_count"] == 1
        assert frames["total_chunks"] == 4
        assert frames["total_words"] == sum(_count_words(norm_text[s:e]) for s, e in parts)

        doc = frames["documents"][0]
        assert doc["chunk_count"] == 4
        assert doc["total_words"] == frames["total_words"]
        assert len(doc["sample_quotes"]) <= 3
        for q in doc["sample_quotes"]:
            assert q["start_offset"] < q["end_offset"]
            assert isinstance(q["text_preview"], str)
            assert len(q["text_preview"]) <= 200

        # Ensure file written and is valid JSON
        on_disk = json.loads(out_path.read_text(encoding="utf-8"))
        validate_case_frames(on_disk)

    finally:
        con.close()

    print("D4 caseframe OK")


if __name__ == "__main__":
    main()

