PRAGMA foreign_keys = ON;

-- D1: User-uploaded documents per case.
CREATE TABLE IF NOT EXISTS case_documents (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  case_id         TEXT NOT NULL,
  original_name   TEXT NOT NULL,
  mime            TEXT NOT NULL,
  size_bytes      INTEGER NOT NULL,
  sha256_hex      TEXT NOT NULL,
  storage_relpath TEXT NOT NULL,
  created_at_utc  TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(case_id, sha256_hex)
);

CREATE INDEX IF NOT EXISTS idx_case_documents_case_id ON case_documents(case_id);

INSERT OR IGNORE INTO schema_migrations(version) VALUES (4);
PRAGMA user_version = 4;

