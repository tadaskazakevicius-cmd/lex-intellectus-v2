PRAGMA foreign_keys = ON;

-- Embeddings metadata (vectors are stored on disk in the HNSW index, not in SQLite).
CREATE TABLE IF NOT EXISTS embeddings (
  chunk_id    TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
  dim         INTEGER NOT NULL,
  model       TEXT,
  updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

INSERT OR IGNORE INTO schema_migrations(version) VALUES (3);
PRAGMA user_version = 3;
