from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from lex_server.retrieval.vector_index import VectorIndex
from lex_server.retrieval.vector_index_build import build_vector_index, load_idmap


class FakeEmbedder:
    def __init__(self, mapping: dict[str, np.ndarray]) -> None:
        self.mapping = {k: np.asarray(v, dtype=np.float32) for k, v in mapping.items()}
        dims = {v.shape for v in self.mapping.values()}
        assert len(dims) == 1
        self.dim = next(iter(dims))[0]

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vecs = [self.mapping[t] for t in texts]
        return np.vstack(vecs).astype(np.float32)


def _make_db(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys = ON;")
    con.executescript(
        """
        CREATE TABLE chunks (
          id INTEGER PRIMARY KEY,
          chunk_id TEXT NOT NULL,
          content TEXT NOT NULL,
          practice_doc_id TEXT NOT NULL
        );
        """
    )
    return con


def test_build_then_load_then_search_deterministic(tmp_path: Path) -> None:
    db_path = tmp_path / "t.db"
    con = _make_db(db_path)
    try:
        # 5 chunks with hard-coded embeddings (dim=4, unit vectors)
        texts = ["t0", "t1", "t2", "t3", "t4"]
        vecs = {
            "t0": np.array([1, 0, 0, 0], dtype=np.float32),
            "t1": np.array([0, 1, 0, 0], dtype=np.float32),
            "t2": np.array([0, 0, 1, 0], dtype=np.float32),
            "t3": np.array([0, 0, 0, 1], dtype=np.float32),
            "t4": np.array([1, 0, 0, 0], dtype=np.float32),  # same as t0
        }
        embedder = FakeEmbedder(vecs)

        with con:
            for i, t in enumerate(texts, start=1):
                con.execute(
                    "INSERT INTO chunks(id, chunk_id, content, practice_doc_id) VALUES (?, ?, ?, ?);",
                    (i, f"c{i}", t, "doc1" if i <= 3 else "doc2"),
                )

        out_index = tmp_path / "idx.bin"
        out_idmap = tmp_path / "idmap.json"
        build_vector_index(con, embedder, out_index, out_idmap, space="cosine", batch_size=2)

        idx = VectorIndex.load(out_index, dim=4, space="cosine")
        q = vecs["t2"]
        ids1, d1 = idx.search(q, top_k=3)
        ids2, d2 = idx.search(q, top_k=3)
        assert ids1.tolist() == ids2.tolist()
        assert d1.tolist() == d2.tolist()
        assert ids1[0] == 3  # chunk id=3 has content t2 -> closest
    finally:
        con.close()


def test_idmap_written_and_readable(tmp_path: Path) -> None:
    db_path = tmp_path / "t.db"
    con = _make_db(db_path)
    try:
        embedder = FakeEmbedder(
            {
                "a": np.array([1, 0, 0, 0], dtype=np.float32),
                "b": np.array([0, 1, 0, 0], dtype=np.float32),
                "c": np.array([0, 0, 1, 0], dtype=np.float32),
                "d": np.array([0, 0, 0, 1], dtype=np.float32),
                "e": np.array([1, 0, 0, 0], dtype=np.float32),
            }
        )
        with con:
            for i, t in enumerate(["a", "b", "c", "d", "e"], start=1):
                con.execute(
                    "INSERT INTO chunks(id, chunk_id, content, practice_doc_id) VALUES (?, ?, ?, ?);",
                    (i, f"chunk_{i}", t, "doc"),
                )

        out_index = tmp_path / "idx.bin"
        out_idmap = tmp_path / "idmap.json"
        build_vector_index(con, embedder, out_index, out_idmap, space="cosine", batch_size=5)

        assert out_idmap.exists()
        m = load_idmap(out_idmap)
        assert m[1] == "chunk_1"
        assert m[5] == "chunk_5"
        # json is object with string keys
        raw = json.loads(out_idmap.read_text(encoding="utf-8"))
        assert "1" in raw
    finally:
        con.close()

