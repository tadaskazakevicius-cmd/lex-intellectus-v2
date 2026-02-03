from __future__ import annotations

import base64
import shutil
import sqlite3
import sys
from pathlib import Path


def _repo_root() -> Path:
    # .../apps/server/packs/test_delta_apply.py -> repo root
    return Path(__file__).resolve().parents[3]


def _write_sqlite_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    try:
        con.execute("PRAGMA foreign_keys = ON;")
        con.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, v TEXT);")
        con.execute("INSERT INTO t(v) VALUES ('x');")
        con.commit()
    finally:
        con.close()


def _list_files(root: Path) -> list[str]:
    out: list[str] = []
    for p in root.rglob("*"):
        if p.is_file():
            out.append(p.relative_to(root).as_posix())
    out.sort()
    return out


def main() -> None:
    root = _repo_root()
    server_dir = root / "apps" / "server"
    if str(server_dir) not in sys.path:
        sys.path.insert(0, str(server_dir))

    from packs.build_snapshot import build_snapshot  # noqa: E402
    from packs.delta import apply_delta, build_delta  # noqa: E402
    from packs.hashing import sha256_file  # noqa: E402
    from packs.install_snapshot import install_snapshot  # noqa: E402
    from packs.signing import generate_ed25519_keypair  # noqa: E402

    base = root / ".localtemp" / "delta_test"
    from_snapshot_dir = base / "from_snapshot"
    to_snapshot_dir = base / "to_snapshot"
    delta_dir = base / "delta"
    data_dir = base / "data_dir"

    if base.exists():
        shutil.rmtree(base)
    from_snapshot_dir.mkdir(parents=True, exist_ok=True)
    to_snapshot_dir.mkdir(parents=True, exist_ok=True)
    delta_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    pack_id = "pack_delta_test"

    # FROM snapshot payload
    fp = from_snapshot_dir / "payload"
    (fp / "indices" / pack_id).mkdir(parents=True, exist_ok=True)
    (fp / "docs").mkdir(parents=True, exist_ok=True)
    _write_sqlite_db(fp / "app.db")
    (fp / "docs" / "a.txt").write_text("hello v1\n", encoding="utf-8")
    (fp / "docs" / "old.txt").write_text("to be deleted\n", encoding="utf-8")
    (fp / "indices" / pack_id / "meta.json").write_text('{"v":1}\n', encoding="utf-8")
    (fp / "indices" / pack_id / "hnsw.bin").write_bytes(b"FROMHNSW")

    # TO snapshot payload changes:
    tp = to_snapshot_dir / "payload"
    (tp / "indices" / pack_id).mkdir(parents=True, exist_ok=True)
    (tp / "docs").mkdir(parents=True, exist_ok=True)
    _write_sqlite_db(tp / "app.db")
    (tp / "docs" / "a.txt").write_text("hello v2 (modified)\n", encoding="utf-8")
    (tp / "docs" / "b.txt").write_text("new file\n", encoding="utf-8")
    # old.txt deleted (do not create)
    (tp / "indices" / pack_id / "meta.json").write_text('{"v":2}\n', encoding="utf-8")
    (tp / "indices" / pack_id / "hnsw.bin").write_bytes(b"FROMHNSW")  # unchanged

    priv, pub = generate_ed25519_keypair()
    priv_b64 = base64.b64encode(priv).decode("ascii")
    pub_b64 = base64.b64encode(pub).decode("ascii")

    build_snapshot(from_snapshot_dir, pack_id=pack_id, channel="stable", version="from-1", private_key_b64=priv_b64)
    build_snapshot(to_snapshot_dir, pack_id=pack_id, channel="stable", version="to-2", private_key_b64=priv_b64)

    # Install FROM snapshot into empty data dir.
    install_snapshot(from_snapshot_dir, data_dir=data_dir, public_key_b64=pub_b64)

    # Build + apply delta FROM -> TO.
    build_delta(
        from_snapshot_dir=from_snapshot_dir,
        to_snapshot_dir=to_snapshot_dir,
        delta_dir=delta_dir,
        private_key_b64=priv_b64,
        channel="stable",
    )
    apply_delta(delta_dir=delta_dir, data_dir=data_dir, public_key_b64=pub_b64, to_snapshot_dir=to_snapshot_dir)

    # Assert active pack matches TO snapshot payload exactly.
    active = (data_dir / "packs" / "ACTIVE.txt").read_text(encoding="utf-8").strip()
    active_dir = data_dir / "packs" / active

    # Compare payload file lists (installed pack stores payload contents at root).
    expected_files = _list_files(to_snapshot_dir / "payload")
    got_files = _list_files(active_dir)
    # remove manifest files from installed pack view
    got_files = [p for p in got_files if p not in ("manifest.json", "manifest.sig")]

    if got_files != expected_files:
        raise AssertionError(f"File list mismatch.\nexpected={expected_files}\n got={got_files}")

    # Compare sha256 for every file.
    for rel in expected_files:
        exp = sha256_file(to_snapshot_dir / "payload" / rel)
        got = sha256_file(active_dir / rel)
        if exp != got:
            raise AssertionError(f"SHA mismatch for {rel}: expected={exp}, got={got}")

    print("C3 delta apply OK")


if __name__ == "__main__":
    main()

