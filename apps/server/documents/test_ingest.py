from __future__ import annotations

import hashlib
import sqlite3
import sys
from pathlib import Path


def main() -> None:
    root = Path.cwd()
    server_dir = root / "apps" / "server"
    src_dir = server_dir / "src"

    # Ensure imports work for both `lex_server.*` (from src) and `documents.*` (from apps/server).
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    if str(server_dir) not in sys.path:
        sys.path.insert(0, str(server_dir))

    # Migrate .localdata/app.db to version 4
    from db.migrate import migrate  # type: ignore

    db_path = root / ".localdata" / "app.db"
    v = migrate(
        db_path=db_path,
        schema_sql_path=Path("apps/server/db/schema.sql"),
        migrations_dir=Path("apps/server/db/migrations"),
    )
    if int(v) < 4:
        raise SystemExit(f"DB migration failed; user_version={v}")

    from fastapi.testclient import TestClient
    from lex_server.main import app

    client = TestClient(app)

    content = b"Hello ingest!\n"
    sha = hashlib.sha256(content).hexdigest()
    size = len(content)

    resp = client.post(
        "/api/cases/testcase1/documents",
        files={"file": ("hello.txt", content, "text/plain")},
    )
    if resp.status_code != 200:
        raise SystemExit(f"Upload failed: status={resp.status_code}, body={resp.text}")

    con = sqlite3.connect(db_path)
    try:
        con.execute("PRAGMA foreign_keys = ON;")
        row = con.execute(
            """
            SELECT sha256_hex, size_bytes
            FROM case_documents
            WHERE case_id = ? AND sha256_hex = ?;
            """,
            ("testcase1", sha),
        ).fetchone()
        if not row:
            raise SystemExit("No row found in case_documents")
        if row[0] != sha or int(row[1]) != size:
            raise SystemExit(f"Row mismatch: got={row}, expected_sha={sha}, expected_size={size}")
    finally:
        con.close()

    print("D1 ingest OK")


if __name__ == "__main__":
    main()

