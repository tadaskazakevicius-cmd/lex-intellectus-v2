PRAGMA foreign_keys = ON;

-- F5: audit log for LLM generations (determinism + replayability).
CREATE TABLE IF NOT EXISTS audit_log (
  id               INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at       TEXT NOT NULL, -- ISO8601 UTC with Z
  event            TEXT NOT NULL, -- "llm_generate_defense"
  model            TEXT NOT NULL,
  pack_version     TEXT NOT NULL,
  retrieval_run_id TEXT,          -- nullable
  params_json      TEXT NOT NULL, -- stable JSON dumps
  output_json      TEXT NOT NULL, -- stable JSON dumps of response
  output_sha256    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON audit_log(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_log_event ON audit_log(event);
CREATE INDEX IF NOT EXISTS idx_audit_log_retrieval_run_id ON audit_log(retrieval_run_id);

INSERT OR IGNORE INTO schema_migrations(version) VALUES (8);
PRAGMA user_version = 8;

