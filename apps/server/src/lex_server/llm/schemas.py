from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CitationRef(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    quote: str
    chunk_id: str | None = None
    practice_doc_id: str | None = None
    source_url: str | None = None
    start: int | None = None
    end: int | None = None


class ArgumentPath(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(min_length=3)
    claims: list[str] = Field(min_length=1)
    # Allow empty here; F4 enforcement removes claims without citations.
    supporting_citations: list[CitationRef] = Field(default_factory=list)


class DefenseDirectionsResponse(BaseModel):
    """
    Server-side contract for defense direction generation.

    MVP: schema validation + strict JSON enforcement via prompting + repair.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    argument_paths: list[ArgumentPath] = Field(default_factory=list)
    counterarguments: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    missing_info: list[str] = Field(default_factory=list)
    insufficient_authority: bool = False

    @classmethod
    def schema_json(cls) -> str:
        """
        Pretty JSON schema string for inclusion in prompts.
        """

        import json

        return json.dumps(cls.model_json_schema(), ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def fallback(cls, missing_info: list[str] | None = None) -> "DefenseDirectionsResponse":
        return cls(
            insufficient_authority=True,
            missing_info=missing_info or ["LLM output was not valid JSON per schema."],
        )

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump()
