from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

# Allow `python apps/server/packs/...` usage.
SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from packs.hashing import sha256_file  # noqa: E402
from packs.install_snapshot import verify_snapshot  # noqa: E402
from packs.signing import canonical_json_bytes  # noqa: E402
from packs.delta import verify_delta, load_json, canonical_manifest_sha256  # noqa: E402


IDLE = "IDLE"
CHECKING = "CHECKING"
DOWNLOADING = "DOWNLOADING"
STAGING = "STAGING"
VERIFYING = "VERIFYING"
APPLYING = "APPLYING"
CLEANUP = "CLEANUP"
ROLLBACK = "ROLLBACK"
FAILED_RETRYABLE = "FAILED_RETRYABLE"
FAILED_HARD = "FAILED_HARD"


@dataclass(frozen=True)
class UpdatePlan:
    plan_type: Literal["snapshot", "delta"]
    channel: str
    pack_id: str
    from_version: str | None
    to_version: str
    artifact_kind: Literal["snapshot_dir", "delta_dir", "snapshot_zip", "delta_zip"]
    artifact_ref: str
    from_manifest_sha256: str | None
    to_manifest_sha256: str


class OfflineUpdater:
    def __init__(
        self,
        data_dir: Path,
        public_key_b64: str,
        remote_dir: Path,
        *,
        fault_injection: dict | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.remote_dir = remote_dir
        self.fault_injection = fault_injection or {}

        pub_raw = base64.b64decode(public_key_b64, validate=True)
        if len(pub_raw) != 32:
            raise ValueError("public_key_b64 must decode to exactly 32 bytes")
        self.public_key_b64 = public_key_b64

        self.packs_dir = self.data_dir / "packs"
        self.cache_dir = self.packs_dir / "cache"
        self.lock_path = self.packs_dir / "lock"
        self.state_path = self.packs_dir / "state.json"

        self.packs_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------
    # Atomic write helpers
    # ------------------------
    def _atomic_write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    def _atomic_write_json(self, path: Path, obj: object) -> None:
        b = canonical_json_bytes(obj)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(b)
        os.replace(tmp, path)

    # ------------------------
    # Locking (best-effort)
    # ------------------------
    def _acquire_lock(self) -> None:
        self.packs_dir.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError("Updater lock already held")
        else:
            os.close(fd)

    def _release_lock(self) -> None:
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass

    # ------------------------
    # State
    # ------------------------
    def _load_state(self) -> dict:
        if not self.state_path.exists():
            return {"state": IDLE}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {"state": FAILED_RETRYABLE, "error": {"kind": "state_parse", "message": "invalid state.json"}}

    def _save_state(self, state: dict) -> None:
        self._atomic_write_json(self.state_path, state)

    # ------------------------
    # Active pack helpers
    # ------------------------
    def _read_active_name(self) -> str:
        p = self.packs_dir / "ACTIVE.txt"
        if not p.exists():
            raise ValueError("no active pack; install snapshot first")
        name = p.read_text(encoding="utf-8").strip()
        if not name:
            raise ValueError("ACTIVE.txt is empty")
        return name

    def _get_active_pack_dir(self) -> Path:
        name = self._read_active_name()
        d = self.packs_dir / name
        if not d.exists() or not d.is_dir():
            raise ValueError("ACTIVE points to missing pack dir")
        return d

    def _set_active_name_atomic(self, new_name: str) -> None:
        active_path = self.packs_dir / "ACTIVE.txt"
        prev_path = self.packs_dir / "ACTIVE.prev"
        if active_path.exists():
            self._atomic_write_text(prev_path, active_path.read_text(encoding="utf-8"))
        self._atomic_write_text(active_path, new_name + "\n")

    def _active_manifest_sha(self) -> str:
        active_dir = self._get_active_pack_dir()
        manifest = load_json(active_dir / "manifest.json")
        return canonical_manifest_sha256(manifest)

    # ------------------------
    # Cleanup helpers
    # ------------------------
    def _cleanup_dir(self, p: Path) -> None:
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)

    # ------------------------
    # Public API
    # ------------------------
    def check_updates(self, channel: str) -> UpdatePlan | None:
        active_sha = self._active_manifest_sha()
        latest_path = self.remote_dir / channel / "latest.json"
        latest = json.loads(latest_path.read_text(encoding="utf-8"))

        pack_id = str(latest["pack_id"])
        to_version = str(latest["latest_version"])
        snapshot_ref = str(latest["snapshot_path"])
        to_sha = str(latest["to_manifest_sha256"])

        if active_sha == to_sha:
            return None

        delta = latest.get("delta")
        if isinstance(delta, dict) and str(delta.get("from_manifest_sha256", "")) == active_sha:
            return UpdatePlan(
                plan_type="delta",
                channel=channel,
                pack_id=pack_id,
                from_version=str(delta.get("from_version")) if delta.get("from_version") is not None else None,
                to_version=to_version,
                artifact_kind="delta_dir",
                artifact_ref=str(delta["path"]),
                from_manifest_sha256=str(delta["from_manifest_sha256"]),
                to_manifest_sha256=to_sha,
            )

        return UpdatePlan(
            plan_type="snapshot",
            channel=channel,
            pack_id=pack_id,
            from_version=None,
            to_version=to_version,
            artifact_kind="snapshot_dir",
            artifact_ref=snapshot_ref,
            from_manifest_sha256=None,
            to_manifest_sha256=to_sha,
        )

    def recover_on_startup(self) -> None:
        state = self._load_state()
        if state.get("state") == IDLE:
            return

        # Ensure ACTIVE is sane and prefer active_before if recorded.
        active_before = state.get("active_before")
        active_path = self.packs_dir / "ACTIVE.txt"
        if isinstance(active_before, str) and active_before:
            if active_path.exists():
                current = active_path.read_text(encoding="utf-8").strip()
                if current and current != active_before:
                    self._set_active_name_atomic(active_before)

        # Cleanup recorded staging dir (if any).
        staging_name = state.get("staging_dir")
        if isinstance(staging_name, str) and staging_name:
            self._cleanup_dir(self.packs_dir / staging_name)

        # Cleanup cache path (if any).
        cache_rel = state.get("cache_path")
        if isinstance(cache_rel, str) and cache_rel:
            self._cleanup_dir(self.packs_dir / cache_rel)

        state["state"] = IDLE
        self._save_state(state)

    def run_once(self, channel: str, trigger: str = "manual") -> None:
        self._acquire_lock()
        try:
            self.recover_on_startup()
            state = {"state": CHECKING, "channel": channel, "trigger": trigger, "started_at_utc": _utc_now()}
            self._save_state(state)

            plan = self.check_updates(channel)
            if plan is None:
                self._save_state({"state": IDLE})
                return

            state.update(
                {
                    "plan_type": plan.plan_type,
                    "channel": plan.channel,
                    "from_manifest_sha256": plan.from_manifest_sha256,
                    "to_manifest_sha256": plan.to_manifest_sha256,
                    "active_before": self._read_active_name(),
                    "staging_dir": None,
                    "cache_path": None,
                    "error": None,
                }
            )
            self._save_state(state)

            # Download
            state["state"] = DOWNLOADING
            self._save_state(state)
            cache_path = self._download(plan)
            state["cache_path"] = str(cache_path.relative_to(self.packs_dir).as_posix())
            self._save_state(state)

            # Staging (unpack)
            # MVP for directory artifacts: treat cache directory as the staged artifact.
            state["state"] = STAGING
            self._save_state(state)
            staged_artifact = self._stage(cache_path)

            # Verify artifact
            state["state"] = VERIFYING
            self._save_state(state)
            self._verify(plan, staged_artifact)

            # Apply
            state["state"] = APPLYING
            self._save_state(state)
            self._apply(plan, staged_artifact, state)

            # Cleanup
            state["state"] = CLEANUP
            self._save_state(state)
            self._cleanup_dir(cache_path)
            state = {"state": IDLE}
            self._save_state(state)
        except ValueError as e:
            st = self._load_state()
            st["state"] = FAILED_HARD
            st["error"] = {"kind": "hard", "message": str(e)}
            self._save_state(st)
            raise
        except Exception as e:
            st = self._load_state()
            st["state"] = FAILED_RETRYABLE
            st["error"] = {"kind": "retryable", "message": str(e)}
            self._save_state(st)
            raise
        finally:
            self._release_lock()

    # ------------------------
    # Steps
    # ------------------------
    def _download(self, plan: UpdatePlan) -> Path:
        src = self.remote_dir / plan.channel / plan.artifact_ref
        if not src.exists():
            raise ValueError(f"Remote artifact missing: {src}")

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        dst = self.cache_dir / f"cache_{plan.plan_type}_{ts}"
        if dst.exists():
            shutil.rmtree(dst)

        _copy_tree(src, dst)
        return dst

    def _stage(self, cache_path: Path) -> Path:
        return cache_path

    def _verify(self, plan: UpdatePlan, staged_artifact: Path) -> None:
        if plan.plan_type == "snapshot":
            verify_snapshot(staged_artifact, self.public_key_b64)
            # Ensure manifest sha matches expectation from latest.json (hard fail).
            manifest = load_json(staged_artifact / "manifest.json")
            got = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
            if got != plan.to_manifest_sha256:
                raise ValueError("Downloaded snapshot does not match expected manifest sha")
            return

        if plan.plan_type == "delta":
            dm = verify_delta(staged_artifact, self.public_key_b64)
            if not isinstance(dm, dict):
                raise ValueError("delta_manifest must be object")
            to_sha = dm.get("to", {}).get("manifest_sha256") if isinstance(dm.get("to"), dict) else None
            if str(to_sha) != plan.to_manifest_sha256:
                raise ValueError("Downloaded delta does not match expected to manifest sha")
            return

        raise ValueError("Unknown plan_type")

    def _apply(self, plan: UpdatePlan, staged_artifact: Path, state: dict) -> None:
        if plan.plan_type == "snapshot":
            self._apply_snapshot(staged_artifact, state)
            return

        # delta
        if self.fault_injection.get("crash_mid_copy"):
            self._apply_delta_with_injection(staged_artifact, plan, state)
        else:
            # Use existing apply_delta for normal operation.
            from packs.delta import apply_delta  # local import

            to_snapshot_dir = self.remote_dir / plan.channel / f"snapshots/{plan.to_version}"
            apply_delta(
                delta_dir=staged_artifact,
                data_dir=self.data_dir,
                public_key_b64=self.public_key_b64,
                to_snapshot_dir=to_snapshot_dir,
            )

    def _apply_snapshot(self, snapshot_dir: Path, state: dict) -> None:
        # Implement snapshot install here so we can inject a crash mid-copy if requested.
        payload_dir = snapshot_dir / "payload"
        if not payload_dir.exists():
            raise ValueError("snapshot missing payload/")

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        staging_name = f"staging_{ts}"
        staging_dir = self.packs_dir / staging_name
        staging_dir.mkdir(parents=True, exist_ok=False)

        state["staging_dir"] = staging_name
        self._save_state(state)

        # Copy payload files to pack root.
        files = [p for p in payload_dir.rglob("*") if p.is_file()]
        files.sort(key=lambda p: p.relative_to(payload_dir).as_posix())
        crash = bool(self.fault_injection.get("crash_mid_copy"))
        crash_after = max(1, len(files) // 2) if crash else None
        copied = 0
        for p in files:
            rel = p.relative_to(payload_dir)
            dst = staging_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            copied += 1
            if crash_after is not None and copied >= crash_after:
                raise RuntimeError("Injected crash mid-copy (snapshot)")

        # Copy manifest into pack root for future delta checks.
        shutil.copy2(snapshot_dir / "manifest.json", staging_dir / "manifest.json")
        shutil.copy2(snapshot_dir / "manifest.sig", staging_dir / "manifest.sig")

        self._set_active_name_atomic(staging_name)

    def _apply_delta_with_injection(self, delta_dir: Path, plan: UpdatePlan, state: dict) -> None:
        # Like packs.delta.apply_delta, but with crash injection during the initial copy of active pack.
        dm = verify_delta(delta_dir, self.public_key_b64)
        if not isinstance(dm, dict):
            raise ValueError("delta_manifest must be object")

        active_before_name = self._read_active_name()
        active_pack = self._get_active_pack_dir()

        from_expected = dm.get("from", {}).get("manifest_sha256") if isinstance(dm.get("from"), dict) else None
        if not isinstance(from_expected, str) or not from_expected:
            raise ValueError("delta missing from.manifest_sha256")
        active_manifest = load_json(active_pack / "manifest.json")
        active_sha = hashlib.sha256(canonical_json_bytes(active_manifest)).hexdigest()
        if active_sha != from_expected:
            raise ValueError("Active pack does not match delta 'from' manifest")

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        staging_name = f"staging_{ts}"
        staging_dir = self.packs_dir / staging_name
        staging_dir.mkdir(parents=True, exist_ok=False)

        state["staging_dir"] = staging_name
        self._save_state(state)

        # Copy whole active pack with injection after ~50% files.
        src_files = [p for p in active_pack.rglob("*") if p.is_file()]
        src_files.sort(key=lambda p: p.relative_to(active_pack).as_posix())
        crash_after = max(1, len(src_files) // 2)
        copied = 0
        for p in src_files:
            rel = p.relative_to(active_pack)
            out = staging_dir / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)
            copied += 1
            if copied >= crash_after:
                raise RuntimeError("Injected crash mid-copy (delta)")

        # If we got here (no crash), we'd apply ops and switch ACTIVE, but injection always crashes above.
        # Ensure ACTIVE not changed:
        if self._read_active_name() != active_before_name:
            self._set_active_name_atomic(active_before_name)


def _copy_tree(src: Path, dst: Path) -> None:
    for p in src.rglob("*"):
        rel = p.relative_to(src)
        out = dst / rel
        if p.is_dir():
            out.mkdir(parents=True, exist_ok=True)
        elif p.is_file():
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

