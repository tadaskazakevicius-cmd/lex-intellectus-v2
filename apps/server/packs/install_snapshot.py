from __future__ import annotations

import base64
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow `python apps/server/packs/install_snapshot.py ...` (if a CLI is added later)
SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from packs.hashing import sha256_file  # noqa: E402
from packs.signing import verify_manifest  # noqa: E402


def _read_manifest(snapshot_dir: Path) -> dict:
    return json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))


def _read_sig(snapshot_dir: Path) -> bytes:
    sig_b64 = (snapshot_dir / "manifest.sig").read_text(encoding="utf-8").strip()
    return base64.b64decode(sig_b64, validate=True)


def verify_snapshot(snapshot_dir: Path, public_key_b64: str) -> None:
    manifest = _read_manifest(snapshot_dir)
    sig = _read_sig(snapshot_dir)

    pub_raw = base64.b64decode(public_key_b64, validate=True)
    if len(pub_raw) != 32:
        raise ValueError("public_key_b64 must decode to exactly 32 bytes")

    if not verify_manifest(pub_raw, manifest, sig):
        raise ValueError("Snapshot signature verification failed")

    files = manifest.get("files", [])
    if not isinstance(files, list):
        raise ValueError("Invalid manifest: files must be a list")

    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("Invalid manifest: file entry must be an object")
        rel = entry.get("path")
        size = entry.get("size")
        sha = entry.get("sha256")
        if not isinstance(rel, str) or not isinstance(size, int) or not isinstance(sha, str):
            raise ValueError("Invalid manifest: bad file entry fields")

        fp = snapshot_dir / rel
        if not fp.exists() or not fp.is_file():
            raise ValueError(f"Missing file: {rel}")

        st = fp.stat()
        if int(st.st_size) != int(size):
            raise ValueError(f"Size mismatch for {rel}: expected={size}, got={st.st_size}")

        got_sha = sha256_file(fp)
        if got_sha != sha:
            raise ValueError(f"SHA256 mismatch for {rel}: expected={sha}, got={got_sha}")


def _copy_tree(src: Path, dst: Path) -> None:
    for p in src.rglob("*"):
        rel = p.relative_to(src)
        out = dst / rel
        if p.is_dir():
            out.mkdir(parents=True, exist_ok=True)
        elif p.is_file():
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)


def install_snapshot(snapshot_dir: Path, data_dir: Path, public_key_b64: str) -> None:
    """
    Verify then install snapshot payload into an (optionally empty) data_dir.

    MVP atomic approach on Windows:
    - create packs/staging_<ts>/
    - copy payload/* into that staging dir
    - write packs/ACTIVE.txt with staging dir name using atomic replace
    """
    verify_snapshot(snapshot_dir, public_key_b64)

    payload_dir = snapshot_dir / "payload"
    if not payload_dir.exists():
        raise ValueError(f"Missing payload dir: {payload_dir}")

    packs_dir = data_dir / "packs"
    packs_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    staging_name = f"staging_{ts}"
    staging_dir = packs_dir / staging_name
    staging_dir.mkdir(parents=True, exist_ok=False)

    _copy_tree(payload_dir, staging_dir)

    active_path = packs_dir / "ACTIVE.txt"
    tmp = packs_dir / "ACTIVE.txt.tmp"
    tmp.write_text(staging_name + "\n", encoding="utf-8")
    tmp.replace(active_path)

