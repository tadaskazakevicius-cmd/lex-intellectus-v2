from __future__ import annotations

from lex_server.retrieval.query_builder import build_query_plan


def test_build_query_plan_full_deterministic_order() -> None:
    case_frame = {
        "case_id": "case1",
        "facts": {
            "summary": "Pirkimo–pardavimo sutartis. Neįvykdymas ir žalos atlyginimas.",
            "keywords": ["sutartis", "žala", "CK 6.248", "FR0600"],
        },
        "issues": ["Ar yra civilinė atsakomybė?", "Kokie įrodymai reikalingi?"],
        "norms": ["CK 6.248 str.", "CK 6.256 str."],
    }
    plan = build_query_plan(case_frame, k=6)
    atoms = plan.atoms

    assert [a.kind for a in atoms] == ["phrase", "norm", "norm", "phrase", "phrase", "keywords"]
    assert atoms[0].text == '"Pirkimo–pardavimo sutartis. Neįvykdymas ir žalos atlyginimas."'
    assert atoms[0].weight == 1.4

    assert atoms[1].text == "CK 6.248 str."
    assert atoms[1].kind == "norm"
    assert atoms[1].weight == 1.3

    assert atoms[2].text == "CK 6.256 str."
    assert atoms[2].weight == 1.3

    assert atoms[3].text == '"Ar yra civilinė atsakomybė?"'
    assert atoms[3].weight == 1.2
    assert atoms[4].text == '"Kokie įrodymai reikalingi?"'
    assert atoms[4].weight == 1.2

    assert atoms[5].kind == "keywords"
    assert atoms[5].weight == 1.0
    assert atoms[5].text == "sutartis žala CK 6.248 FR0600"


def test_build_query_plan_norms_only_limit_k() -> None:
    case_frame = {
        "case_id": "c2",
        "norms": ["CK 6.248 str.", "CK 6.256 str.", "CK 1.5 str.", "ATPĮ 12 str."],
    }
    plan = build_query_plan(case_frame, k=3)
    assert [a.kind for a in plan.atoms] == ["norm", "norm", "norm"]
    assert [a.text for a in plan.atoms] == ["CK 6.248 str.", "CK 6.256 str.", "CK 1.5 str."]


def test_dedup_case_insensitive_and_truncate_phrase_to_160() -> None:
    long = " ".join(["Labaiilgaszodis"] * 40)  # > 160 chars with spaces
    case_frame = {
        "case_id": "c3",
        "facts": {"summary": long},
        "issues": [long.upper()],  # duplicate in different case
        "keywords": ["A", "B"],
    }
    plan = build_query_plan(case_frame, k=6)

    # Summary phrase present, issue duplicate removed.
    assert plan.atoms[0].kind == "phrase"
    assert len(plan.atoms) == 2  # phrase + keywords

    phrase = plan.atoms[0].text
    assert phrase.startswith('"') and phrase.endswith('"')
    inner = phrase[1:-1]
    assert len(inner) <= 160

