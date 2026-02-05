-- 0010_create_case_documents.sql
-- Minimal table needed for G1 uploads + status tracking.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS case_documents (
  id TEXT PRIMARY KEY,
  case_id TEXT NOT NULL,
  filename TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  storage_path TEXT NOT NULL,
  mime_type TEXT,
  size_bytes INTEGER,
  uploaded_at_utc TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'queued',
  error TEXT,
  deduped INTEGER NOT NULL DEFAULT 0,
  FOREIGN KEY(case_id) REFERENCES cases(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_case_documents_case_id
  ON case_documents(case_id);

CREATE UNIQUE INDEX IF NOT EXISTS ux_case_documents_case_sha256
  ON case_documents(case_id, sha256);

PRAGMA user_version = 10;
