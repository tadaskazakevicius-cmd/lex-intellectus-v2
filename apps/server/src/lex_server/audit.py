from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__


@dataclass(frozen=True)
class AuditEvent:
    event_type: str
    timestamp_utc: str
    app_version: str
    details: dict[str, Any] | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def log_event(audit_log_path: Path, *, event_type: str, details: dict[str, Any] | None = None) -> None:
    ev = AuditEvent(
        event_type=event_type,
        timestamp_utc=utc_now_iso(),
        app_version=__version__,
        details=details,
    )
    append_jsonl(audit_log_path, asdict(ev))

