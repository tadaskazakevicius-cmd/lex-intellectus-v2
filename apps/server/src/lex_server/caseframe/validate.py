from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _schema_path() -> Path:
    return Path(__file__).resolve().parents[1] / "schemas" / "caseframe.schema.json"


def validate_case_frames(data: dict[str, Any]) -> None:
    """
    Validate generated CaseFrames JSON against the bundled JSON Schema.
    Raises ValueError on validation errors.
    """
    try:
        from jsonschema import Draft202012Validator  # type: ignore

        schema = json.loads(_schema_path().read_text(encoding="utf-8"))
        v = Draft202012Validator(schema)
        errs = sorted(v.iter_errors(data), key=lambda e: list(e.path))
        if errs:
            msg = "; ".join([e.message for e in errs[:5]])
            raise ValueError(msg)
        return
    except ModuleNotFoundError:
        # Minimal fallback validator (keeps runnable scripts working in bare envs).
        _fallback_validate(data)


def _fallback_validate(data: dict[str, Any]) -> None:
    req_top = ["schema_version", "case_id", "generated_at_utc", "documents_count", "total_chunks", "total_words", "documents"]
    for k in req_top:
        if k not in data:
            raise ValueError(f"Missing field: {k}")
    if data.get("schema_version") != "1.0":
        raise ValueError("schema_version must be '1.0'")
    if not isinstance(data["documents"], list):
        raise ValueError("documents must be a list")
    for d in data["documents"]:
        if not isinstance(d, dict):
            raise ValueError("documents[] must be objects")
        for k in ["document_id", "original_name", "mime", "sha256", "created_at", "chunk_count", "total_words", "sample_quotes"]:
            if k not in d:
                raise ValueError(f"Missing document field: {k}")
        if not isinstance(d["sample_quotes"], list):
            raise ValueError("sample_quotes must be a list")
        if len(d["sample_quotes"]) > 3:
            raise ValueError("sample_quotes maxItems=3")
        for q in d["sample_quotes"]:
            if not isinstance(q, dict):
                raise ValueError("sample_quotes[] must be objects")
            for k in ["chunk_index", "start_offset", "end_offset", "text_preview"]:
                if k not in q:
                    raise ValueError(f"Missing sample_quote field: {k}")

