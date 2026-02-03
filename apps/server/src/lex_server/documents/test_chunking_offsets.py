from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .chunking import chunk_text, normalize_text, word_spans


def main() -> None:
    golden_dir = Path(__file__).resolve().parent / "golden"
    expected_txt = golden_dir / "expected.txt"
    raw = expected_txt.read_text(encoding="utf-8")
    norm = normalize_text(raw)

    norm2, chunks = chunk_text(raw)
    assert norm2 == norm
    assert len(chunks) >= 1

    spans = word_spans(norm)
    assert len(spans) > 0

    # indices 0..N-1
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    # invariants
    for c in chunks:
        assert c.start_offset < c.end_offset
        assert c.text != ""
        assert c.text == norm[c.start_offset : c.end_offset]

    # monotonic offsets; no overlap by default
    for i in range(len(chunks) - 1):
        a = chunks[i]
        b = chunks[i + 1]
        assert a.end_offset <= b.start_offset

    # coverage: from first word start to last word end
    assert chunks[0].start_offset == spans[0][0]
    assert chunks[-1].end_offset == spans[-1][1]

    snap = []
    for c in chunks:
        sha = hashlib.sha256(c.text.encode("utf-8")).hexdigest()
        snap.append(
            {
                "i": c.chunk_index,
                "start": c.start_offset,
                "end": c.end_offset,
                "words": c.word_count,
                "sha256": sha,
            }
        )

    snap_path = golden_dir / "chunks_snapshot.json"
    if not snap_path.exists():
        snap_path.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        expected_snap = json.loads(snap_path.read_text(encoding="utf-8"))
        if expected_snap != snap:
            raise SystemExit(
                "Chunk snapshot mismatch.\n"
                f"Got: {json.dumps(snap, ensure_ascii=False, indent=2)}\n"
                f"Expected: {json.dumps(expected_snap, ensure_ascii=False, indent=2)}\n"
            )

    print("D3 chunking offsets OK")


if __name__ == "__main__":
    main()

