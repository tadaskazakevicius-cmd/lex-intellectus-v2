from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path


EXPECTED = (
    "PVM deklaracija FR0600.\n"
    "Tai testinis dokumentas.\n"
    'Cituojama eilute: "PVM".\n'
)


def normalize(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _golden_dir() -> Path:
    return Path(__file__).resolve().parent / "golden"


def _xml_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _write_docx_minimal(path: Path, lines: list[str]) -> None:
    """
    Create a deterministic minimal DOCX by writing the required OOXML parts.
    Avoids dependency on python-docx inside the test itself.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    content_types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""
    rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""
    paras_xml = "\n".join([f"<w:p><w:r><w:t>{_xml_escape(t)}</w:t></w:r></w:p>" for t in lines])
    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {paras_xml}
  </w:body>
</w:document>
"""

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", document_xml)


def _first_diff(a: str, b: str) -> str:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return f"diff@{i}: got={a[max(0, i-20):i+20]!r} expected={b[max(0, i-20):i+20]!r}"
    if len(a) != len(b):
        return f"len mismatch: got={len(a)}, expected={len(b)}"
    return "no diff"


def _ensure_imports_work() -> None:
    """
    Allow running both:
      - python -m apps.server.src.lex_server.documents.test_text_extract
      - python apps/server/src/lex_server/documents/test_text_extract.py

    When run as a script, Python doesn't know about the package root.
    We add apps/server/src to sys.path to make relative imports possible.
    """
    if __package__:
        return

    # __file__ = .../apps/server/src/lex_server/documents/test_text_extract.py
    # parents[2] = .../apps/server/src
    src_root = Path(__file__).resolve().parents[2]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


def main() -> None:
    _ensure_imports_work()

    from .text_extract import extract_text  # import after sys.path fix

    golden = _golden_dir()
    golden.mkdir(parents=True, exist_ok=True)

    sample_txt = golden / "sample.txt"
    expected_path = golden / "expected.txt"

    # Keep expected snapshot deterministic in repo
    if not expected_path.exists():
        expected_path.write_text(EXPECTED, encoding="utf-8", newline="\n")
    expected = expected_path.read_text(encoding="utf-8")
    expected_n = normalize(expected)

    # Write sample.txt deterministically too
    if not sample_txt.exists():
        sample_txt.write_text(expected, encoding="utf-8", newline="\n")

    lines = [ln for ln in expected.replace("\r\n", "\n").split("\n") if ln.strip()]

    # Generate DOCX deterministically (OK to generate)
    sample_docx = golden / "sample.docx"
    _write_docx_minimal(sample_docx, lines)

    # ✅ Variant 1: PDF must exist as a golden file (selectable-text).
    sample_pdf = golden / "sample.pdf"
    if not sample_pdf.exists():
        raise SystemExit(
            "Missing golden/sample.pdf (selectable-text). Create it once via:\n"
            "1) Open golden/sample.txt in Notepad/Word\n"
            "2) Print -> Microsoft Print to PDF\n"
            "3) Save as apps/server/src/lex_server/documents/golden/sample.pdf\n"
        )

    for p in (sample_txt, sample_docx, sample_pdf):
        got = extract_text(p)
        got_n = normalize(got)
        if got_n != expected_n:
            raise SystemExit(f"Mismatch for {p.name}: {_first_diff(got_n, expected_n)}")

    print("D2 text extract OK")


if __name__ == "__main__":
    main()
