PRAGMA foreign_keys = ON;

-- E2: FTS5 index for user document chunks (document_chunks.text).
CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts USING fts5(
  chunk_id UNINDEXED,
  text,
  tokenize = 'unicode61'
);

-- Backfill existing rows (idempotent).
INSERT INTO document_chunks_fts(rowid, chunk_id, text)
SELECT rowid, id, text
FROM document_chunks
WHERE rowid NOT IN (SELECT rowid FROM document_chunks_fts);

-- Keep FTS in sync with base table.
CREATE TRIGGER IF NOT EXISTS document_chunks_ai AFTER INSERT ON document_chunks BEGIN
  INSERT INTO document_chunks_fts(rowid, chunk_id, text)
  VALUES (new.rowid, new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS document_chunks_ad AFTER DELETE ON document_chunks BEGIN
  INSERT INTO document_chunks_fts(document_chunks_fts, rowid, chunk_id, text)
  VALUES ('delete', old.rowid, old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS document_chunks_au AFTER UPDATE OF text, id ON document_chunks BEGIN
  INSERT INTO document_chunks_fts(document_chunks_fts, rowid, chunk_id, text)
  VALUES ('delete', old.rowid, old.id, old.text);
  INSERT INTO document_chunks_fts(rowid, chunk_id, text)
  VALUES (new.rowid, new.id, new.text);
END;

INSERT OR IGNORE INTO schema_migrations(version) VALUES (6);
PRAGMA user_version = 6;

