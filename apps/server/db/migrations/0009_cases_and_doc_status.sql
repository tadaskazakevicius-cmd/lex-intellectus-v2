PRAGMA foreign_keys = ON;

-- G1: Cases metadata + document processing status.

CREATE TABLE IF NOT EXISTS cases (
  id             TEXT PRIMARY KEY, -- uuid4
  title          TEXT NOT NULL,
  description    TEXT,
  category       TEXT,
  created_at_utc TEXT NOT NULL
);

-- Extend existing case_documents with processing status fields (MVP).
ALTER TABLE case_documents ADD COLUMN status TEXT NOT NULL DEFAULT 'queued';
ALTER TABLE case_documents ADD COLUMN error TEXT;
ALTER TABLE case_documents ADD COLUMN updated_at_utc TEXT;

CREATE INDEX IF NOT EXISTS idx_cases_created_at ON cases(created_at_utc);

INSERT OR IGNORE INTO schema_migrations(version) VALUES (9);
PRAGMA user_version = 9;

