from __future__ import annotations

from typing import Any

import pytest

from lex_server.llm.orchestrator import generate_defense_directions
from lex_server.llm.schemas import DefenseDirectionsResponse


class _FakeRuntime:
    def __init__(self, outputs: list[str]) -> None:
        self._outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    def generate(self, prompt: str, params=None) -> str:  # match LlamaCppRuntime shape
        self.calls.append({"prompt": prompt, "params": params})
        if not self._outputs:
            raise AssertionError("Fake runtime ran out of outputs")
        return self._outputs.pop(0)


def test_schema_accepts_valid_json() -> None:
    payload = {
        "argument_paths": [
            {
                "title": "Procesiniai pažeidimai",
                "claims": ["Procesiniai pažeidimai galėjo turėti reikšmės."],
                "supporting_citations": [
                    {
                        "quote": "Citata iš dokumento.",
                        "chunk_id": "c1",
                        "practice_doc_id": None,
                        "source_url": None,
                        "start": 10,
                        "end": 20,
                    }
                ],
            }
        ],
        "counterarguments": ["Prokuroras teigs priešingai."],
        "risks": ["Gali trūkti bylos duomenų."],
        "missing_info": ["Trūksta faktinių aplinkybių."],
        "insufficient_authority": False,
    }

    model = DefenseDirectionsResponse.model_validate(payload)
    assert model.insufficient_authority is False
    assert model.argument_paths and model.argument_paths[0].supporting_citations


def test_orchestrator_parses_json_with_noise() -> None:
    valid = (
        '{ "argument_paths": [ { "title": "Kryptis A", "claims": ["Teiginys 1"],'
        ' "supporting_citations": [ { "quote": "Q1", "chunk_id": "c1" } ] } ],'
        ' "counterarguments": [], "risks": [], "missing_info": [], "insufficient_authority": false }'
    )
    rt = _FakeRuntime([f"SURE! {valid} thanks"])

    out = generate_defense_directions(
        runtime=rt,  # type: ignore[arg-type]
        query="gynybos kryptys",
        citations=[{"quote": "Q1", "chunk_id": "c1"}],
        params=None,
    )
    assert isinstance(out, DefenseDirectionsResponse)
    assert out.argument_paths[0].title == "Kryptis A"


def test_orchestrator_repair_on_invalid() -> None:
    # First output is JSON but invalid schema (supporting_citations is empty).
    invalid_schema = (
        '{ "argument_paths": [ { "title": "Kryptis A", "claims": ["Teiginys 1"],'
        ' "supporting_citations": [] } ],'
        ' "counterarguments": [], "risks": [], "missing_info": [], "insufficient_authority": false }'
    )
    valid = (
        '{ "argument_paths": [ { "title": "Kryptis A", "claims": ["Teiginys 1"],'
        ' "supporting_citations": [ { "quote": "Q1", "chunk_id": "c1" } ] } ],'
        ' "counterarguments": [], "risks": [], "missing_info": [], "insufficient_authority": false }'
    )
    rt = _FakeRuntime([invalid_schema, valid])

    out = generate_defense_directions(
        runtime=rt,  # type: ignore[arg-type]
        query="gynybos kryptys",
        citations=[{"quote": "Q1", "chunk_id": "c1"}],
        params=None,
    )

    assert out.argument_paths[0].supporting_citations[0].quote == "Q1"
    assert len(rt.calls) == 2
    assert "Fix the JSON" in rt.calls[1]["prompt"]


def test_orchestrator_fallback_after_two_failures() -> None:
    rt = _FakeRuntime(["not json at all", "still not json"])

    out = generate_defense_directions(
        runtime=rt,  # type: ignore[arg-type]
        query="gynybos kryptys",
        citations=[],
        params=None,
    )

    assert out.insufficient_authority is True
    assert out.missing_info
    assert out.argument_paths == []

