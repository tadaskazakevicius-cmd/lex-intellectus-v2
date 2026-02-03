PRAGMA foreign_keys = ON;

-- D3: stable document chunks with char offsets against normalized text.
CREATE TABLE IF NOT EXISTS document_chunks (
  id           TEXT PRIMARY KEY, -- f"{document_id}:{chunk_index}"
  document_id  TEXT NOT NULL REFERENCES case_documents(id) ON DELETE CASCADE,
  chunk_index  INTEGER NOT NULL,
  start_offset INTEGER NOT NULL,
  end_offset   INTEGER NOT NULL,
  word_count   INTEGER NOT NULL,
  text         TEXT NOT NULL,
  created_at   TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_doc_idx ON document_chunks(document_id, chunk_index);

INSERT OR IGNORE INTO schema_migrations(version) VALUES (5);
PRAGMA user_version = 5;

