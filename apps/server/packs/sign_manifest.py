from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from .signing import canonical_json_bytes, sign_bytes


def main() -> None:
    p = argparse.ArgumentParser(description="Canonicalize and sign a manifest (ed25519).")
    p.add_argument("manifest_json", type=Path)
    p.add_argument("private_key_b64", type=str, help="Base64-encoded 32-byte Ed25519 private key (raw).")
    p.add_argument("--out", type=Path, default=None, help="Signature output path (base64).")
    args = p.parse_args()

    manifest_path: Path = args.manifest_json
    priv_b64: str = args.private_key_b64
    out_path: Path | None = args.out

    manifest_obj = json.loads(manifest_path.read_text(encoding="utf-8"))

    priv_raw = base64.b64decode(priv_b64, validate=True)
    if len(priv_raw) != 32:
        raise SystemExit("private_key_b64 must decode to exactly 32 bytes")

    data = canonical_json_bytes(manifest_obj)
    sig = sign_bytes(priv_raw, data)
    sig_b64 = base64.b64encode(sig).decode("ascii")

    print(sig_b64)

    if out_path is None:
        out_path = manifest_path.with_suffix(manifest_path.suffix + ".sig")
    out_path.write_text(sig_b64 + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

