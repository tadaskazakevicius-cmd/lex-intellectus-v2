from __future__ import annotations

import mimetypes
from pathlib import Path


PDF_MIME = "application/pdf"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
TXT_MIME = "text/plain"


def _infer_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return PDF_MIME
    if ext == ".docx":
        return DOCX_MIME
    if ext == ".txt":
        return TXT_MIME
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def _normalize_light(s: str) -> str:
    # convert CRLF -> LF
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # strip trailing spaces per line
    s = "\n".join([ln.rstrip() for ln in s.split("\n")])
    return s.strip()


def extract_text(path: Path, mime: str | None = None) -> str:
    """
    D2 extractor entry point (MVP, no OCR):
    - TXT: utf-8 read (errors="replace")
    - DOCX: python-docx, join non-empty paragraphs with "\\n"
    - PDF: selectable text only via pypdf; join pages with "\\n\\n"
    """
    mime = mime or _infer_mime(path)

    if mime == TXT_MIME:
        return _normalize_light(path.read_text(encoding="utf-8", errors="replace"))

    if mime == DOCX_MIME:
        from docx import Document

        doc = Document(str(path))
        paras = [(p.text or "") for p in doc.paragraphs]
        paras = [t for t in paras if t.strip() != ""]
        return _normalize_light("\n".join(paras))

    if mime == PDF_MIME:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages_text = [(page.extract_text() or "") for page in reader.pages]
        text = "\n\n".join(pages_text)
        return _normalize_light(text)

    raise ValueError(f"Unsupported mime for extraction: {mime}")

