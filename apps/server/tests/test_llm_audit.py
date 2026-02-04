from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from lex_server.llm.audit import sha256_text, stable_json_dumps
from lex_server.llm.llama_cpp_runtime import LlamaParams
from lex_server.llm.orchestrator import generate_defense_directions


class _FakeRuntime:
    def __init__(self, outputs: list[str], *, model_path: Path) -> None:
        self._outputs = list(outputs)
        self.model_path = model_path
        self._backend_selected = "cpu"

    @property
    def backend_selected(self) -> str:
        return self._backend_selected

    def generate(self, prompt: str, params=None) -> str:  # match LlamaCppRuntime shape
        if not self._outputs:
            raise AssertionError("Fake runtime ran out of outputs")
        return self._outputs.pop(0)


def _apply_migration_0008(conn: sqlite3.Connection) -> None:
    # Ensure schema_migrations exists for migration script compatibility.
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE IF NOT EXISTS schema_migrations (
          version INTEGER PRIMARY KEY,
          applied_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )
    mig = (Path(__file__).resolve().parents[1] / "db" / "migrations" / "0008_audit_llm_generation.sql").read_text(
        encoding="utf-8"
    )
    conn.executescript(mig)


def _fetch_audit_rows(dbp: Path) -> list[dict[str, Any]]:
    con = sqlite3.connect(dbp)
    try:
        cur = con.execute(
            """
            SELECT
              id, created_at, event, model, pack_version, retrieval_run_id,
              params_json, output_json, output_sha256
            FROM audit_log
            ORDER BY id
            """
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]
    finally:
        con.close()


def test_audit_log_written_and_hash_stable(tmp_path: Path, monkeypatch) -> None:
    dbp = tmp_path / "audit.db"
    con = sqlite3.connect(dbp)
    try:
        _apply_migration_0008(con)
    finally:
        con.close()

    import lex_server.llm.orchestrator as orch_mod

    monkeypatch.setattr(orch_mod, "_db_path", lambda: dbp)
    monkeypatch.setenv("LEX_PACK_VERSION", "testpack")
    monkeypatch.delenv("LEX_DB_PATH", raising=False)

    raw = json.dumps(
        {
            "argument_paths": [
                {
                    "title": "Kryptis A",
                    "claims": ["Teiginys 1"],
                    "supporting_citations": [{"quote": "Q1", "chunk_id": "c1"}],
                }
            ],
            "counterarguments": [],
            "risks": [],
            "missing_info": [],
            "insufficient_authority": False,
        },
        ensure_ascii=False,
    )

    rt = _FakeRuntime([raw, raw], model_path=Path("C:/models/test.gguf"))
    p = LlamaParams(seed=123, temperature=0.0, top_p=1.0, top_k=40, repeat_penalty=1.1, ctx=256, n_predict=32)

    resp1 = generate_defense_directions(
        runtime=rt,  # type: ignore[arg-type]
        query="q",
        citations=[{"quote": "Q1", "chunk_id": "c1"}],
        params=p,
        retrieval_run_id="run-123",
    )
    resp2 = generate_defense_directions(
        runtime=rt,  # type: ignore[arg-type]
        query="q",
        citations=[{"quote": "Q1", "chunk_id": "c1"}],
        params=p,
        retrieval_run_id="run-123",
    )

    assert resp1.model_dump() == resp2.model_dump()

    expected_output_json = stable_json_dumps(resp1.model_dump())
    expected_sha = sha256_text(expected_output_json)

    rows = _fetch_audit_rows(dbp)
    assert len(rows) == 2
    r0 = rows[0]
    assert r0["event"] == "llm_generate_defense"
    assert r0["pack_version"] == "testpack"
    assert r0["retrieval_run_id"] == "run-123"
    assert r0["model"]
    assert json.loads(r0["params_json"])["seed"] == 123
    assert r0["output_json"] == expected_output_json
    assert r0["output_sha256"] == expected_sha
    assert rows[1]["output_sha256"] == expected_sha

