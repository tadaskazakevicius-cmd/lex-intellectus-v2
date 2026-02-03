from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def list_files(root: Path) -> list[Path]:
    """
    Recursively list files under root in stable order.
    Returns absolute Paths.
    """
    files = [p for p in root.rglob("*") if p.is_file()]
    files.sort(key=lambda p: p.relative_to(root).as_posix())
    return files


def file_entry(root: Path, file_path: Path) -> dict:
    """
    Build a manifest file entry:
      {"path": "<posix relative>", "size": <bytes>, "sha256": "<hex>"}
    """
    rel = file_path.relative_to(root).as_posix()
    st = file_path.stat()
    return {"path": rel, "size": int(st.st_size), "sha256": sha256_file(file_path)}

