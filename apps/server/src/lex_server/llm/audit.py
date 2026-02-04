from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any


logger = logging.getLogger(__name__)


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def try_audit_llm_generation_to_db(
    conn: sqlite3.Connection,
    *,
    model: str,
    pack_version: str,
    retrieval_run_id: str | None,
    params_json: str,
    output_json: str,
) -> int | None:
    """
    Best-effort audit write. Never raises.

    Returns inserted audit id on success, else None.
    """

    try:
        created_at = _utc_now_iso_z()
        output_sha256 = sha256_text(output_json)

        with conn:
            cur = conn.execute(
                """
                INSERT INTO audit_log(
                  created_at, event, model, pack_version, retrieval_run_id,
                  params_json, output_json, output_sha256
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    "llm_generate_defense",
                    model,
                    pack_version,
                    retrieval_run_id,
                    params_json,
                    output_json,
                    output_sha256,
                ),
            )
            return int(cur.lastrowid)
    except Exception as e:  # pragma: no cover
        logger.warning("audit write failed: %s", e)
        return None

