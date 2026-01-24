from __future__ import annotations

from pathlib import Path


def indices_root() -> Path:
    """
    MVP: store indices under repo-local `.localdata/indices`.

    Note:
    - This is intentionally simple and OS-agnostic.
    - Callers can pass an explicit root Path if they want different storage.
    """
    return Path(".localdata") / "indices"


def pack_index_dir(root: Path, pack_id: str) -> Path:
    return root / pack_id

