from __future__ import annotations

import base64
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow `python apps/server/packs/...` style usage.
SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from packs.hashing import sha256_file  # noqa: E402
from packs.signing import canonical_json_bytes, sign_manifest, verify_manifest  # noqa: E402


def canonical_manifest_sha256(manifest_obj: object) -> str:
    b = canonical_json_bytes(manifest_obj)
    return hashlib.sha256(b).hexdigest()


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(obj))


def _files_map(manifest_obj: object) -> dict[str, tuple[str, int]]:
    if not isinstance(manifest_obj, dict):
        raise ValueError("manifest must be an object")
    files = manifest_obj.get("files")
    if not isinstance(files, list):
        raise ValueError("manifest.files must be a list")
    out: dict[str, tuple[str, int]] = {}
    for e in files:
        if not isinstance(e, dict):
            raise ValueError("manifest.files entry must be an object")
        p = e.get("path")
        sha = e.get("sha256")
        size = e.get("size")
        if not isinstance(p, str) or not isinstance(sha, str) or not isinstance(size, int):
            raise ValueError("invalid file entry in manifest")
        out[p] = (sha, int(size))
    return out


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_delta(
    from_snapshot_dir: Path,
    to_snapshot_dir: Path,
    delta_dir: Path,
    private_key_b64: str,
    channel: str = "stable",
) -> None:
    from_manifest = load_json(from_snapshot_dir / "manifest.json")
    to_manifest = load_json(to_snapshot_dir / "manifest.json")

    from_sha = canonical_manifest_sha256(from_manifest)
    to_sha = canonical_manifest_sha256(to_manifest)

    fm = _files_map(from_manifest)
    tm = _files_map(to_manifest)

    add_or_replace: list[dict] = []
    delete: list[dict] = []

    for p, (sha, size) in tm.items():
        if p not in fm or fm[p][0] != sha:
            add_or_replace.append({"path": p, "size": size, "sha256": sha})

    for p in fm.keys():
        if p not in tm:
            delete.append({"path": p})

    # Copy changed/new files from TO snapshot into delta_dir preserving manifest paths.
    for e in add_or_replace:
        rel = str(e["path"])
        _copy_file(to_snapshot_dir / rel, delta_dir / rel)

    delta_manifest = {
        "format": "delta",
        "channel": channel,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "from": {
            "pack_id": str(from_manifest.get("pack_id", "")),
            "version": str(from_manifest.get("version", "")),
            "manifest_sha256": from_sha,
        },
        "to": {
            "pack_id": str(to_manifest.get("pack_id", "")),
            "version": str(to_manifest.get("version", "")),
            "manifest_sha256": to_sha,
        },
        "ops": {"add_or_replace": add_or_replace, "delete": delete},
    }

    write_json(delta_dir / "delta_manifest.json", delta_manifest)

    priv_raw = base64.b64decode(private_key_b64, validate=True)
    if len(priv_raw) != 32:
        raise ValueError("private_key_b64 must decode to exactly 32 bytes")
    sig = sign_manifest(priv_raw, delta_manifest)
    (delta_dir / "delta_manifest.sig").write_text(
        base64.b64encode(sig).decode("ascii") + "\n", encoding="utf-8"
    )


def verify_delta(delta_dir: Path, public_key_b64: str) -> object:
    manifest = load_json(delta_dir / "delta_manifest.json")
    sig_b64 = (delta_dir / "delta_manifest.sig").read_text(encoding="utf-8").strip()
    sig = base64.b64decode(sig_b64, validate=True)

    pub_raw = base64.b64decode(public_key_b64, validate=True)
    if len(pub_raw) != 32:
        raise ValueError("public_key_b64 must decode to exactly 32 bytes")

    if not verify_manifest(pub_raw, manifest, sig):
        raise ValueError("Delta signature verification failed")

    if not isinstance(manifest, dict):
        raise ValueError("delta_manifest must be an object")
    ops = manifest.get("ops")
    if not isinstance(ops, dict):
        raise ValueError("delta_manifest.ops must be an object")
    aor = ops.get("add_or_replace", [])
    if not isinstance(aor, list):
        raise ValueError("delta_manifest.ops.add_or_replace must be a list")

    for e in aor:
        if not isinstance(e, dict):
            raise ValueError("add_or_replace entry must be an object")
        rel = e.get("path")
        size = e.get("size")
        sha = e.get("sha256")
        if not isinstance(rel, str) or not isinstance(size, int) or not isinstance(sha, str):
            raise ValueError("invalid add_or_replace entry")
        fp = delta_dir / rel
        if not fp.exists() or not fp.is_file():
            raise ValueError(f"Missing delta payload file: {rel}")
        st = fp.stat()
        if int(st.st_size) != int(size):
            raise ValueError(f"Size mismatch for {rel}: expected={size}, got={st.st_size}")
        if sha256_file(fp) != sha:
            raise ValueError(f"SHA256 mismatch for {rel}")

    return manifest


def _read_active_pack_dir(data_dir: Path) -> Path:
    active_path = data_dir / "packs" / "ACTIVE.txt"
    if not active_path.exists():
        raise ValueError("no active pack; install snapshot first")
    name = active_path.read_text(encoding="utf-8").strip()
    if not name:
        raise ValueError("ACTIVE.txt is empty")
    p = data_dir / "packs" / name
    if not p.exists() or not p.is_dir():
        raise ValueError(f"Active pack dir missing: {p}")
    return p


def _copy_tree(src: Path, dst: Path) -> None:
    for p in src.rglob("*"):
        rel = p.relative_to(src)
        out = dst / rel
        if p.is_dir():
            out.mkdir(parents=True, exist_ok=True)
        elif p.is_file():
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)


def _pack_rel_from_manifest_path(p: str) -> Path:
    # Installed pack stores snapshot payload contents at pack root (no `payload/` dir).
    prefix = "payload/"
    if p.startswith(prefix):
        p = p[len(prefix) :]
    return Path(*p.split("/"))


def _list_pack_payload_files(pack_root: Path) -> set[str]:
    out: set[str] = set()
    for p in pack_root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(pack_root).as_posix()
        if rel in ("manifest.json", "manifest.sig"):
            continue
        out.add(rel)
    return out


def apply_delta(
    delta_dir: Path,
    data_dir: Path,
    public_key_b64: str,
    to_snapshot_dir: Path | None = None,
) -> None:
    delta_manifest = verify_delta(delta_dir, public_key_b64)
    if not isinstance(delta_manifest, dict):
        raise ValueError("delta_manifest must be an object")

    active_pack = _read_active_pack_dir(data_dir)

    # Validate "from" manifest hash against installed pack's manifest.json.
    from_expected = (
        delta_manifest.get("from", {}).get("manifest_sha256")
        if isinstance(delta_manifest.get("from"), dict)
        else None
    )
    if not isinstance(from_expected, str) or not from_expected:
        raise ValueError("delta_manifest.from.manifest_sha256 missing")
    active_manifest = load_json(active_pack / "manifest.json")
    active_sha = canonical_manifest_sha256(active_manifest)
    if active_sha != from_expected:
        raise ValueError("Active pack does not match delta 'from' manifest")

    packs_dir = data_dir / "packs"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    staging_name = f"staging_{ts}"
    staging_dir = packs_dir / staging_name
    staging_dir.mkdir(parents=True, exist_ok=False)

    # Copy whole active pack to staging (simple + reliable MVP).
    _copy_tree(active_pack, staging_dir)

    ops = delta_manifest.get("ops")
    if not isinstance(ops, dict):
        raise ValueError("delta_manifest.ops missing")
    deletes = ops.get("delete", [])
    aor = ops.get("add_or_replace", [])
    if not isinstance(deletes, list) or not isinstance(aor, list):
        raise ValueError("delta_manifest.ops invalid")

    # Apply deletes.
    for e in deletes:
        if not isinstance(e, dict):
            continue
        p = e.get("path")
        if not isinstance(p, str):
            continue
        tgt = staging_dir / _pack_rel_from_manifest_path(p)
        if tgt.exists():
            try:
                tgt.unlink()
            except IsADirectoryError:
                shutil.rmtree(tgt, ignore_errors=True)

    # Apply add_or_replace.
    for e in aor:
        if not isinstance(e, dict):
            raise ValueError("add_or_replace entry must be object")
        p = e.get("path")
        if not isinstance(p, str):
            raise ValueError("add_or_replace.path missing")
        src = delta_dir / p
        dst = staging_dir / _pack_rel_from_manifest_path(p)
        _copy_file(src, dst)

    # Strict final verification against TO snapshot, if provided.
    if to_snapshot_dir is not None:
        from packs.install_snapshot import verify_snapshot  # local import to avoid cycles

        verify_snapshot(to_snapshot_dir, public_key_b64)
        to_manifest = load_json(to_snapshot_dir / "manifest.json")
        to_sha = canonical_manifest_sha256(to_manifest)
        to_expected = (
            delta_manifest.get("to", {}).get("manifest_sha256")
            if isinstance(delta_manifest.get("to"), dict)
            else None
        )
        if not isinstance(to_expected, str) or not to_expected:
            raise ValueError("delta_manifest.to.manifest_sha256 missing")
        if to_sha != to_expected:
            raise ValueError("Provided TO snapshot does not match delta 'to' manifest")

        tm = _files_map(to_manifest)
        expected_paths = {_pack_rel_from_manifest_path(p).as_posix() for p in tm.keys()}
        got_paths = _list_pack_payload_files(staging_dir)
        if got_paths != expected_paths:
            extra = sorted(got_paths - expected_paths)
            missing = sorted(expected_paths - got_paths)
            raise ValueError(f"Payload file set mismatch. extra={extra}, missing={missing}")

        # Verify sizes + sha256 for every TO file.
        for p, (sha, size) in tm.items():
            fp = staging_dir / _pack_rel_from_manifest_path(p)
            st = fp.stat()
            if int(st.st_size) != int(size):
                raise ValueError(f"Size mismatch for {p}")
            if sha256_file(fp) != sha:
                raise ValueError(f"SHA256 mismatch for {p}")

        # Update installed pack manifest to TO (keeps future deltas consistent).
        shutil.copy2(to_snapshot_dir / "manifest.json", staging_dir / "manifest.json")
        shutil.copy2(to_snapshot_dir / "manifest.sig", staging_dir / "manifest.sig")

    # Atomic switch ACTIVE.txt (keep previous for debugging).
    active_path = packs_dir / "ACTIVE.txt"
    prev_path = packs_dir / "ACTIVE.prev"
    if active_path.exists():
        prev_path.write_text(active_path.read_text(encoding="utf-8"), encoding="utf-8")
    tmp = packs_dir / "ACTIVE.txt.tmp"
    tmp.write_text(staging_name + "\n", encoding="utf-8")
    tmp.replace(active_path)

