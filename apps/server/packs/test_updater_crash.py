from __future__ import annotations

import base64
import hashlib
import json
import shutil
import sqlite3
import sys
from pathlib import Path


def _repo_root() -> Path:
    # .../apps/server/packs/test_updater_crash.py -> repo root
    return Path(__file__).resolve().parents[3]


def _write_sqlite_db(path: Path, marker: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    try:
        con.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, v TEXT);")
        con.execute("DELETE FROM t;")
        con.execute("INSERT INTO t(v) VALUES (?);", (marker,))
        con.commit()
    finally:
        con.close()


def _manifest_sha(snapshot_dir: Path) -> str:
    from packs.signing import canonical_json_bytes

    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


def _assert_pack_matches_snapshot(pack_dir: Path, snapshot_dir: Path) -> None:
    from packs.hashing import sha256_file

    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    files = manifest["files"]
    expected_paths = []
    for e in files:
        p = e["path"]
        assert isinstance(p, str)
        assert p.startswith("payload/")
        expected_paths.append(p[len("payload/") :])
    expected_paths.sort()

    got_paths = []
    for p in pack_dir.rglob("*"):
        if p.is_file():
            rel = p.relative_to(pack_dir).as_posix()
            if rel in ("manifest.json", "manifest.sig"):
                continue
            got_paths.append(rel)
    got_paths.sort()

    if got_paths != expected_paths:
        raise AssertionError(f"payload set mismatch\nexpected={expected_paths}\n got={got_paths}")

    for e in files:
        rel = e["path"][len("payload/") :]
        exp_sha = e["sha256"]
        exp_size = e["size"]
        fp = pack_dir / Path(*rel.split("/"))
        if not fp.exists():
            raise AssertionError(f"missing file: {rel}")
        if fp.stat().st_size != exp_size:
            raise AssertionError(f"size mismatch for {rel}")
        if sha256_file(fp) != exp_sha:
            raise AssertionError(f"sha mismatch for {rel}")


def main() -> None:
    root = _repo_root()
    server_dir = root / "apps" / "server"
    if str(server_dir) not in sys.path:
        sys.path.insert(0, str(server_dir))

    from packs.build_snapshot import build_snapshot  # noqa: E402
    from packs.delta import build_delta  # noqa: E402
    from packs.install_snapshot import install_snapshot  # noqa: E402
    from packs.signing import generate_ed25519_keypair  # noqa: E402
    from packs.updater import OfflineUpdater, FAILED_RETRYABLE, IDLE  # noqa: E402

    base = root / ".localtemp" / "updater_test"
    remote_dir = base / "remote_repo"
    data_dir = base / "updater_data"

    if base.exists():
        shutil.rmtree(base)
    (remote_dir / "stable" / "snapshots").mkdir(parents=True, exist_ok=True)
    (remote_dir / "stable" / "deltas").mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    pack_id = "pack_updater_test"
    v1 = "2026.01.24.1"
    v2 = "2026.01.25.0"

    # Build remote snapshots v1 and v2.
    s1 = remote_dir / "stable" / "snapshots" / v1
    s2 = remote_dir / "stable" / "snapshots" / v2
    (s1 / "payload" / "docs").mkdir(parents=True, exist_ok=True)
    (s2 / "payload" / "docs").mkdir(parents=True, exist_ok=True)

    # v1 payload
    _write_sqlite_db(s1 / "payload" / "app.db", marker="v1")
    (s1 / "payload" / "docs" / "a.txt").write_text("v1\n", encoding="utf-8")
    # v2 payload differs
    _write_sqlite_db(s2 / "payload" / "app.db", marker="v2")
    (s2 / "payload" / "docs" / "a.txt").write_text("v2\n", encoding="utf-8")
    (s2 / "payload" / "docs" / "b.txt").write_text("new\n", encoding="utf-8")

    priv, pub = generate_ed25519_keypair()
    priv_b64 = base64.b64encode(priv).decode("ascii")
    pub_b64 = base64.b64encode(pub).decode("ascii")

    build_snapshot(s1, pack_id=pack_id, channel="stable", version=v1, private_key_b64=priv_b64)
    build_snapshot(s2, pack_id=pack_id, channel="stable", version=v2, private_key_b64=priv_b64)

    # Build remote delta v1 -> v2.
    d12 = remote_dir / "stable" / "deltas" / f"{v1}__{v2}"
    d12.mkdir(parents=True, exist_ok=True)
    build_delta(from_snapshot_dir=s1, to_snapshot_dir=s2, delta_dir=d12, private_key_b64=priv_b64, channel="stable")

    # Create latest.json
    latest = {
        "pack_id": pack_id,
        "channel": "stable",
        "latest_version": v2,
        "snapshot_path": f"snapshots/{v2}",
        "to_manifest_sha256": _manifest_sha(s2),
        "delta": {
            "from_manifest_sha256": _manifest_sha(s1),
            "from_version": v1,
            "path": f"deltas/{v1}__{v2}",
        },
    }
    (remote_dir / "stable" / "latest.json").write_text(json.dumps(latest, indent=2), encoding="utf-8")

    # Install v1 into empty data_dir.
    install_snapshot(snapshot_dir=s1, data_dir=data_dir, public_key_b64=pub_b64)
    active_before = (data_dir / "packs" / "ACTIVE.txt").read_text(encoding="utf-8").strip()

    # A) Crash simulation during apply
    updater_crash = OfflineUpdater(
        data_dir=data_dir,
        public_key_b64=pub_b64,
        remote_dir=remote_dir,
        fault_injection={"crash_mid_copy": True},
    )
    try:
        updater_crash.run_once(channel="stable")
        raise AssertionError("run_once should have crashed")
    except Exception:
        pass

    active_after = (data_dir / "packs" / "ACTIVE.txt").read_text(encoding="utf-8").strip()
    if active_after != active_before:
        raise AssertionError("ACTIVE changed despite crash")

    state = json.loads((data_dir / "packs" / "state.json").read_text(encoding="utf-8"))
    if state.get("state") != FAILED_RETRYABLE or not state.get("error"):
        raise AssertionError("state.json not marked FAILED_RETRYABLE with error")

    # B) Recovery + normal run
    updater_ok = OfflineUpdater(data_dir=data_dir, public_key_b64=pub_b64, remote_dir=remote_dir)
    updater_ok.recover_on_startup()
    state2 = json.loads((data_dir / "packs" / "state.json").read_text(encoding="utf-8"))
    if state2.get("state") != IDLE:
        raise AssertionError("recover_on_startup did not reset state to IDLE")
    active_after_recover = (data_dir / "packs" / "ACTIVE.txt").read_text(encoding="utf-8").strip()
    if active_after_recover != active_before:
        raise AssertionError("ACTIVE changed during recovery")

    updater_ok.run_once(channel="stable")
    active_final = (data_dir / "packs" / "ACTIVE.txt").read_text(encoding="utf-8").strip()
    if active_final == active_before:
        raise AssertionError("ACTIVE did not change after successful update")

    active_dir = data_dir / "packs" / active_final
    _assert_pack_matches_snapshot(active_dir, s2)

    print("C4 updater crash safety OK")


if __name__ == "__main__":
    main()

