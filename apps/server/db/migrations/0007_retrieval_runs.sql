PRAGMA foreign_keys = ON;

-- E5: Persisted retrieval runs (UI reload determinism).
CREATE TABLE IF NOT EXISTS retrieval_runs (
  id            TEXT PRIMARY KEY,  -- uuid4
  created_at    TEXT NOT NULL,      -- ISO8601 UTC
  query         TEXT NOT NULL,
  top_n         INTEGER NOT NULL,
  filters_json  TEXT,
  use_fts       INTEGER NOT NULL,   -- 0/1
  use_vector    INTEGER NOT NULL,   -- 0/1
  algo_version  TEXT NOT NULL,      -- e.g. hybrid_v1
  meta_json     TEXT
);

CREATE TABLE IF NOT EXISTS retrieval_run_hits (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id          TEXT NOT NULL REFERENCES retrieval_runs(id) ON DELETE CASCADE,
  rank            INTEGER NOT NULL,
  chunk_id        TEXT NOT NULL,
  practice_doc_id TEXT NOT NULL,
  score           REAL NOT NULL,
  fts_bm25         REAL,
  vector_distance  REAL,
  UNIQUE(run_id, rank),
  UNIQUE(run_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_retrieval_run_hits_run_id ON retrieval_run_hits(run_id);

CREATE TABLE IF NOT EXISTS retrieval_run_citations (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  hit_id     INTEGER NOT NULL REFERENCES retrieval_run_hits(id) ON DELETE CASCADE,
  idx        INTEGER NOT NULL,
  quote      TEXT NOT NULL,
  start      INTEGER NOT NULL,
  end        INTEGER NOT NULL,
  source_url TEXT,
  UNIQUE(hit_id, idx)
);

CREATE INDEX IF NOT EXISTS idx_retrieval_run_citations_hit_id ON retrieval_run_citations(hit_id);

INSERT OR IGNORE INTO schema_migrations(version) VALUES (7);
PRAGMA user_version = 7;

