from __future__ import annotations

import argparse
import base64
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow `python apps/server/packs/build_snapshot.py ...`
SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from packs.hashing import file_entry, list_files  # noqa: E402
from packs.signing import canonical_json_bytes, sign_manifest  # noqa: E402


def build_snapshot(
    snapshot_dir: Path,
    pack_id: str,
    channel: str,
    version: str,
    private_key_b64: str,
) -> None:
    payload_dir = snapshot_dir / "payload"
    if not payload_dir.exists():
        raise ValueError(f"Missing payload dir: {payload_dir}")

    priv_raw = base64.b64decode(private_key_b64, validate=True)
    if len(priv_raw) != 32:
        raise ValueError("private_key_b64 must decode to exactly 32 bytes")

    files = [file_entry(snapshot_dir, p) for p in list_files(payload_dir)]

    manifest = {
        "format": "snapshot",
        "pack_id": pack_id,
        "channel": channel,
        "version": version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }

    manifest_path = snapshot_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    sig = sign_manifest(priv_raw, manifest)
    sig_b64 = base64.b64encode(sig).decode("ascii")
    (snapshot_dir / "manifest.sig").write_text(sig_b64 + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Build a snapshot manifest + signature (C2).")
    p.add_argument("snapshot_dir", type=Path)
    p.add_argument("pack_id", type=str)
    p.add_argument("version", type=str)
    p.add_argument("private_key_b64", type=str)
    p.add_argument("--channel", type=str, default="stable")
    args = p.parse_args()

    build_snapshot(
        snapshot_dir=args.snapshot_dir,
        pack_id=args.pack_id,
        channel=args.channel,
        version=args.version,
        private_key_b64=args.private_key_b64,
    )


if __name__ == "__main__":
    main()

