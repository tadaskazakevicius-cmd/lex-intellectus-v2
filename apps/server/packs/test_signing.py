from __future__ import annotations

from packs.signing import (
    canonical_json_bytes,
    generate_ed25519_keypair,
    sign_manifest,
    verify_bytes,
    verify_manifest,
)


def test_canonical_bytes_stable() -> None:
    manifest = {
        "name": "Test",
        "version": 1,
        "meta": {"b": 2, "a": 1},
        "files": [{"path": "a.txt", "size": 10}, {"size": 5, "path": "b.txt"}],
        "unicode": "ąčęėįšųūž",
    }
    b1 = canonical_json_bytes(manifest)
    b2 = canonical_json_bytes(manifest)
    assert b1 == b2


def test_sign_verify_roundtrip_and_tamper_fails() -> None:
    priv, pub = generate_ed25519_keypair()
    manifest = {"pack_id": "p1", "files": [{"path": "x.txt", "sha256": "00" * 32}], "n": 1}

    sig = sign_manifest(priv, manifest)
    assert verify_manifest(pub, manifest, sig) is True

    # Flip one byte in canonical bytes -> signature must not verify for modified data.
    canon = canonical_json_bytes(manifest)
    tampered = bytearray(canon)
    tampered[0] ^= 0x01
    assert verify_bytes(pub, bytes(tampered), sig) is False

