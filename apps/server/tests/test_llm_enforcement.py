from __future__ import annotations

from typing import Any

from lex_server.llm.enforcement import enforce_no_citation_no_claim
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


def test_removes_paths_without_citations() -> None:
    resp = DefenseDirectionsResponse.model_validate(
        {
            "argument_paths": [
                {
                    "title": "Path A",
                    "claims": ["A1"],
                    "supporting_citations": [{"quote": "x", "chunk_id": "c1"}],
                },
                {
                    "title": "Path B",
                    "claims": ["B1"],
                    "supporting_citations": [{"quote": "y", "chunk_id": "c2"}],
                },
            ],
            "counterarguments": [],
            "risks": [],
            "missing_info": [],
            "insufficient_authority": False,
        }
    )

    # Make Path A unsupported by dropping its citations (simulate hallucination).
    resp2 = resp.model_copy(deep=True)
    resp2.argument_paths[0].supporting_citations = []

    out = enforce_no_citation_no_claim(resp2)
    assert [p.title for p in out.argument_paths] == ["Path B"]
    assert sum(len(p.claims) for p in out.argument_paths) == 1
    assert out.insufficient_authority is False


def test_sets_insufficient_authority_when_all_removed() -> None:
    resp = DefenseDirectionsResponse.model_validate(
        {
            "argument_paths": [
                {"title": "Only path", "claims": ["C1"], "supporting_citations": [{"quote": "q", "chunk_id": "c"}]}
            ],
            "counterarguments": [],
            "risks": [],
            "missing_info": [],
            "insufficient_authority": False,
        }
    )
    resp2 = resp.model_copy(deep=True)
    resp2.argument_paths[0].supporting_citations = []

    out = enforce_no_citation_no_claim(resp2)
    assert out.argument_paths == []
    assert out.insufficient_authority is True
    assert out.missing_info


def test_orchestrator_applies_enforcement() -> None:
    # Valid schema JSON: MUST include at least 1 supporting citation, so we provide one
    # but keep it below enforcement threshold by calling with min_citations_per_path=1
    # and returning empty citations in response itself. To keep schema valid, we include
    # a citation object but set enforcement to remove claims when citations are empty.
    #
    # Instead, we make schema-valid response with 1 citation in JSON but pass
    # min_citations_per_path=2 via enforcement defaults? Defaults are 1, so we'd keep it.
    # For orchestrator-level test per spec, use supporting_citations=[] in JSON and rely on
    # orchestrator repair to return schema-valid? That would fail schema validation (min_length=1).
    #
    # Therefore, we model "hallucination" as citations present but not enough per threshold by
    # passing a stricter min_citations_per_path into enforcement via the function itself is not
    # wired in orchestrator. So we follow spec intent: claims removed when no citations provided,
    # by returning schema-valid response whose citations list is non-empty but quotes don't match
    # retrieval citations (still grounded enforcement is path-level only). We keep the test focused:
    # orchestrator applies enforcement and can set insufficient_authority when content removed.

    raw = """
    {
      "argument_paths": [
        {
          "title": "Hallucinated path",
          "claims": ["Unsupported claim"],
          "supporting_citations": []
        }
      ],
      "counterarguments": [],
      "risks": [],
      "missing_info": [],
      "insufficient_authority": false
    }
    """.strip()

    rt = _FakeRuntime([raw, raw])

    out = generate_defense_directions(
        runtime=rt,  # type: ignore[arg-type]
        query="gynybos kryptys",
        citations=[],
        params=None,
    )

    assert out.insufficient_authority is True
    assert out.argument_paths == []
    assert out.missing_info

