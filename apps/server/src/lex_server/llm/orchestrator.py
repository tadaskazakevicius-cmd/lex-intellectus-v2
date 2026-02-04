from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import asdict
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .audit import stable_json_dumps, try_audit_llm_generation_to_db
from .enforcement import enforce_no_citation_no_claim
from .llama_cpp_runtime import LlamaCppRuntime, LlamaParams
from .prompting import defense_prompt
from .schemas import DefenseDirectionsResponse


logger = logging.getLogger(__name__)


def _db_path() -> Path:
    env_db = (os.environ.get("LEX_DB_PATH") or "").strip()
    if env_db:
        return Path(env_db).expanduser().resolve()

    local = Path.cwd() / ".localdata" / "app.db"
    if local.exists():
        return local

    from ..paths import get_paths

    return get_paths().data_dir / "app.db"


def _extract_json_object(raw: str) -> Any:
    """
    Best-effort extraction of a JSON object from an LLM string output.
    Tries full parse first, then substring from first '{' to last '}'.
    """

    raw_s = (raw or "").strip()
    if not raw_s:
        raise json.JSONDecodeError("Empty string", raw_s, 0)

    try:
        return json.loads(raw_s)
    except json.JSONDecodeError:
        pass

    i = raw_s.find("{")
    j = raw_s.rfind("}")
    if i == -1 or j == -1 or j <= i:
        raise json.JSONDecodeError("No JSON object found", raw_s, 0)

    candidate = raw_s[i : j + 1]
    return json.loads(candidate)


def _repair_prompt(*, schema_json: str, raw: str, error_summary: str) -> str:
    return (
        "You MUST output ONLY a single valid JSON object and nothing else.\n"
        "No markdown. No code fences. No prose.\n"
        "\n"
        "The previous output did not match the required JSON schema.\n"
        "Fix the JSON so it matches the schema EXACTLY (extra keys forbidden).\n"
        "\n"
        "REQUIRED JSON SCHEMA:\n"
        f"{schema_json}\n"
        "\n"
        "ERROR SUMMARY:\n"
        f"{error_summary}\n"
        "\n"
        "PREVIOUS RAW OUTPUT (for reference):\n"
        "-----BEGIN RAW-----\n"
        f"{raw}\n"
        "-----END RAW-----\n"
        "\n"
        "Return the corrected JSON now.\n"
    )


def generate_defense_directions(
    runtime: LlamaCppRuntime,
    query: str,
    citations: list[dict[str, Any]],
    params: LlamaParams | None = None,
    retrieval_run_id: str | None = None,
) -> DefenseDirectionsResponse:
    """
    Orchestrate generation + JSON parsing + Pydantic schema validation.

    Behavior:
    - Prompt requires ONLY JSON.
    - Parse JSON (robust extraction).
    - Validate DefenseDirectionsResponse.
    - If invalid: do 1 repair attempt.
    - If still invalid: return schema-valid fallback with insufficient_authority=true.
    """

    def _audit_best_effort(resp_final: DefenseDirectionsResponse, *, p_effective: LlamaParams) -> None:
        pack_version = (os.environ.get("LEX_PACK_VERSION") or "").strip() or "dev"
        model_path = getattr(runtime, "model_path", None)
        model = (
            str(model_path)
            if model_path is not None
            else ((os.environ.get("LEX_MODEL_GGUF") or "").strip() or "unknown")
        )

        params_dict = asdict(p_effective)
        params_dict["threads_resolved"] = int(
            p_effective.threads if p_effective.threads is not None else (os.cpu_count() or 4)
        )
        params_dict["backend_selected"] = getattr(runtime, "backend_selected", None)

        params_json = stable_json_dumps(params_dict)
        output_json = stable_json_dumps(resp_final.model_dump())

        try:
            dbp = _db_path()
            if not dbp.exists():
                logger.warning("audit skipped: DB not found at %s", dbp)
                return

            con = sqlite3.connect(dbp)
            try:
                con.execute("PRAGMA foreign_keys = ON;")
                _ = try_audit_llm_generation_to_db(
                    con,
                    model=model,
                    pack_version=pack_version,
                    retrieval_run_id=retrieval_run_id,
                    params_json=params_json,
                    output_json=output_json,
                )
            finally:
                con.close()
        except Exception as e:  # pragma: no cover
            logger.warning("audit write failed: %s", e)

    schema_json = DefenseDirectionsResponse.schema_json()
    prompt = defense_prompt(query=query, citations=citations, schema_json=schema_json)

    # Helpful stop tokens (best-effort): discourage trailing commentary.
    p_use = params
    if p_use is not None:
        stops = list(p_use.stop or [])
        for tok in ("\n\n", "\n```", "\n---"):
            if tok not in stops:
                stops.append(tok)
        p_use = dc_replace(p_use, stop=stops)

    p_effective = p_use or getattr(runtime, "params", LlamaParams())

    raw1 = runtime.generate(prompt, params=p_use)
    try:
        parsed1 = _extract_json_object(raw1)
        resp1 = DefenseDirectionsResponse.model_validate(parsed1)
        final1 = enforce_no_citation_no_claim(resp1)
        _audit_best_effort(final1, p_effective=p_effective)
        return final1
    except (json.JSONDecodeError, ValidationError) as e1:
        error_summary = str(e1)

    repair = _repair_prompt(schema_json=schema_json, raw=raw1, error_summary=error_summary)
    raw2 = runtime.generate(repair, params=p_use)
    try:
        parsed2 = _extract_json_object(raw2)
        resp2 = DefenseDirectionsResponse.model_validate(parsed2)
        final2 = enforce_no_citation_no_claim(resp2)
        _audit_best_effort(final2, p_effective=p_effective)
        return final2
    except (json.JSONDecodeError, ValidationError) as e2:
        info = [
            "LLM output was not valid JSON per schema after repair attempt.",
            f"first_error={error_summary[:500]}",
            f"second_error={str(e2)[:500]}",
        ]
        final3 = enforce_no_citation_no_claim(DefenseDirectionsResponse.fallback(missing_info=info))
        _audit_best_effort(final3, p_effective=p_effective)
        return final3
