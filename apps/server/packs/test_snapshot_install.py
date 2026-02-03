from __future__ import annotations

import base64
import shutil
import sqlite3
import sys
from pathlib import Path


def _repo_root() -> Path:
    # .../apps/server/packs/test_snapshot_install.py -> repo root
    return Path(__file__).resolve().parents[3]


def _write_sqlite_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    try:
        con.execute("PRAGMA foreign_keys = ON;")
        con.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, v TEXT);")
        con.execute("INSERT INTO t(v) VALUES ('hello');")
        con.commit()
    finally:
        con.close()


def main() -> None:
    root = _repo_root()

    # Ensure `packs.*` imports work when running as a script.
    server_dir = root / "apps" / "server"
    if str(server_dir) not in sys.path:
        sys.path.insert(0, str(server_dir))

    from packs.build_snapshot import build_snapshot  # noqa: E402
    from packs.install_snapshot import install_snapshot, verify_snapshot  # noqa: E402
    from packs.signing import generate_ed25519_keypair  # noqa: E402

    snapshot_dir = root / ".localtemp" / "snapshot_test"
    data_dir = root / ".localtemp" / "data_test"
    pack_id = "pack_snapshot_test"

    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    if data_dir.exists():
        shutil.rmtree(data_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    payload = snapshot_dir / "payload"
    (payload / "indices" / pack_id).mkdir(parents=True, exist_ok=True)
    (payload / "docs").mkdir(parents=True, exist_ok=True)

    # Minimal payload files.
    _write_sqlite_db(payload / "app.db")
    (payload / "indices" / pack_id / "meta.json").write_text('{"ok":true}\n', encoding="utf-8")
    (payload / "indices" / pack_id / "idmap.json").write_text('{"label_to_chunk_id":{},"chunk_id_to_label":{}}\n', encoding="utf-8")
    (payload / "indices" / pack_id / "hnsw.bin").write_bytes(b"\x00HNSW\x01\x02\x03")
    (payload / "docs" / "readme.txt").write_text("docs ok\n", encoding="utf-8")

    priv, pub = generate_ed25519_keypair()
    priv_b64 = base64.b64encode(priv).decode("ascii")
    pub_b64 = base64.b64encode(pub).decode("ascii")

    build_snapshot(
        snapshot_dir=snapshot_dir,
        pack_id=pack_id,
        channel="stable",
        version="test-1",
        private_key_b64=priv_b64,
    )

    # Verify should pass.
    verify_snapshot(snapshot_dir, pub_b64)

    # Mutate manifest.json (keep old signature) -> verify must fail.
    manifest_path = snapshot_dir / "manifest.json"
    manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    manifest["channel"] = "stableX"
    manifest_path.write_text(__import__("json").dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    try:
        verify_snapshot(snapshot_dir, pub_b64)
        raise AssertionError("verify_snapshot should have failed after manifest mutation")
    except ValueError:
        pass

    # Restore correct manifest+sig.
    build_snapshot(
        snapshot_dir=snapshot_dir,
        pack_id=pack_id,
        channel="stable",
        version="test-1",
        private_key_b64=priv_b64,
    )
    verify_snapshot(snapshot_dir, pub_b64)

    # Install into empty data dir.
    install_snapshot(snapshot_dir=snapshot_dir, data_dir=data_dir, public_key_b64=pub_b64)

    active = (data_dir / "packs" / "ACTIVE.txt").read_text(encoding="utf-8").strip()
    active_dir = data_dir / "packs" / active
    assert active_dir.exists() and active_dir.is_dir()
    assert (active_dir / "app.db").exists()
    assert (active_dir / "indices" / pack_id / "hnsw.bin").exists()
    assert (active_dir / "docs" / "readme.txt").exists()

    print("C2 snapshot install OK")


if __name__ == "__main__":
    main()

