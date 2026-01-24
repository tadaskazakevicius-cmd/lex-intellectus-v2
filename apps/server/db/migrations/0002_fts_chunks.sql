PRAGMA foreign_keys = ON;

-- Full-text search index for chunks.text (SQLite FTS5).
-- Uses rowid to keep stable linkage; also stores chunk_id for convenience.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  chunk_id UNINDEXED,
  text,
  tokenize = 'unicode61'
);

-- Backfill existing rows (idempotent via INSERT OR IGNORE on rowid).
INSERT OR IGNORE INTO chunks_fts(rowid, chunk_id, text)
SELECT rowid, id, text
FROM chunks;

-- Keep FTS table in sync with base table.
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, chunk_id, text)
  VALUES (new.rowid, new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, text)
  VALUES ('delete', old.rowid, old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF text, id ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, text)
  VALUES ('delete', old.rowid, old.id, old.text);
  INSERT INTO chunks_fts(rowid, chunk_id, text)
  VALUES (new.rowid, new.id, new.text);
END;

INSERT OR IGNORE INTO schema_migrations(version) VALUES (2);
PRAGMA user_version = 2;
