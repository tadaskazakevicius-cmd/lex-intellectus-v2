from __future__ import annotations

import json
from typing import Any


def defense_prompt(query: str, citations: list[dict[str, Any]], schema_json: str) -> str:
    """
    Build a strict "ONLY JSON" prompt for DefenseDirectionsResponse.

    `citations` is expected to be a list of dicts with at least:
      - quote: str
      - chunk_id/practice_doc_id/source_url/start/end: optional
    """

    citations_compact = []
    for c in citations:
        if not isinstance(c, dict):
            continue
        # MVP: pass through only the schema-supported citation keys (ignore extras).
        citations_compact.append(
            {
                "quote": c.get("quote", ""),
                "chunk_id": c.get("chunk_id"),
                "practice_doc_id": c.get("practice_doc_id"),
                "source_url": c.get("source_url"),
                "start": c.get("start"),
                "end": c.get("end"),
            }
        )

    citations_json = json.dumps(citations_compact, ensure_ascii=False, indent=2)

    # Tiny valid example (minimal but schema-valid).
    example = {
        "argument_paths": [
            {
                "title": "Proceso pažeidimų linija",
                "claims": ["Procesiniai pažeidimai galėjo paveikti sprendimo teisėtumą."],
                "supporting_citations": [
                    {
                        "quote": "…",
                        "chunk_id": "chunk_123",
                        "practice_doc_id": None,
                        "source_url": None,
                        "start": None,
                        "end": None,
                    }
                ],
            }
        ],
        "counterarguments": ["Prokuroras teigs, kad pažeidimai nereikšmingi."],
        "risks": ["Nepakankamai duomenų apie įrodymų rinkimo aplinkybes."],
        "missing_info": ["Kokie konkretūs procesiniai veiksmai buvo atlikti ir kada."],
        "insufficient_authority": True,
    }
    example_json = json.dumps(example, ensure_ascii=False, indent=2)

    return (
        "You are a legal assistant. Your task: propose defense directions based on the query and the provided citations.\n"
        "\n"
        "CRITICAL OUTPUT RULES:\n"
        "- Output ONLY a single valid JSON object.\n"
        "- No markdown. No code fences. No prose. No commentary.\n"
        "- Do not include any text before or after the JSON.\n"
        "\n"
        "JSON CONTRACT (must match exactly; extra keys forbidden):\n"
        f"{schema_json}\n"
        "\n"
        "FIELD GUIDANCE:\n"
        "- argument_paths: array of {title, claims, supporting_citations}\n"
        "- supporting_citations: MUST be non-empty; use the provided citations; the 'quote' MUST be copied from them.\n"
        "- counterarguments/risks/missing_info: arrays of strings (can be empty).\n"
        "- If citations are insufficient or key facts are missing: set insufficient_authority=true and add items to missing_info.\n"
        "\n"
        "USER QUERY:\n"
        f"{query}\n"
        "\n"
        "AVAILABLE CITATIONS (use these only):\n"
        f"{citations_json}\n"
        "\n"
        "VALID EXAMPLE (shape only, keep yours grounded in citations):\n"
        f"{example_json}\n"
        "\n"
        "Now produce the JSON response.\n"
    )
