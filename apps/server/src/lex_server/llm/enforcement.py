from __future__ import annotations

from .schemas import DefenseDirectionsResponse


_DEFAULT_INSUFFICIENT_MSG = (
    "Insufficient grounded content: removed claims without citations; "
    "provide more sources or refine query."
)


def enforce_no_citation_no_claim(
    resp: DefenseDirectionsResponse,
    *,
    min_paths: int = 1,
    min_total_claims: int = 1,
    min_citations_per_path: int = 1,
) -> DefenseDirectionsResponse:
    """
    MVP enforcement: "No citation -> no claim" at path level.

    Because the schema does not map citations to individual claims, we enforce the
    constraint at ArgumentPath level:
    - If a path has too few supporting citations, all claims in that path are removed.
    - Then empty paths (no claims) are removed.
    - If too little content remains, insufficient_authority is set and missing_info
      is augmented.

    Deterministic and non-mutating: returns a new model using deep copy.
    """

    out = resp.model_copy(deep=True)

    missing_info = list(out.missing_info or [])
    new_paths = []

    for p in out.argument_paths:
        citations_count = len(p.supporting_citations or [])
        if citations_count < int(min_citations_per_path):
            # Remove all claims in this path (no grounding).
            p.claims = []
            missing_info.append(
                f"Removed claims in path '{p.title}' because no supporting citations were provided."
            )
        if p.claims:
            new_paths.append(p)

    out.argument_paths = new_paths
    out.missing_info = missing_info

    paths_left = len(out.argument_paths)
    claims_left = sum(len(p.claims) for p in out.argument_paths)
    if paths_left < int(min_paths) or claims_left < int(min_total_claims):
        out.insufficient_authority = True
        if _DEFAULT_INSUFFICIENT_MSG not in out.missing_info:
            out.missing_info.append(_DEFAULT_INSUFFICIENT_MSG)

    return out
