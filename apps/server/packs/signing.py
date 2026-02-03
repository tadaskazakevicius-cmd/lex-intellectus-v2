from __future__ import annotations

import json

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)


def canonical_json_bytes(obj: object) -> bytes:
    """
    Canonical JSON rules (C1):
    - UTF-8 encoding
    - No whitespace (separators=(",", ":"))
    - ensure_ascii=False
    - sort_keys=True (recursive)
    - allow_nan=False (reject NaN/Infinity)

    The canonical form is the exact bytes output of json.dumps(...) with the above settings.
    """
    s = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return s.encode("utf-8")


def generate_ed25519_keypair() -> tuple[bytes, bytes]:
    """
    Returns (private_key_raw_32, public_key_raw_32).

    Both are raw 32-byte values suitable for:
    - Ed25519PrivateKey.from_private_bytes(...)
    - Ed25519PublicKey.from_public_bytes(...)
    """
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()

    priv_raw = priv.private_bytes(
        encoding=Encoding.Raw,
        format=PrivateFormat.Raw,
        encryption_algorithm=NoEncryption(),
    )
    pub_raw = pub.public_bytes(
        encoding=Encoding.Raw,
        format=PublicFormat.Raw,
    )
    if len(priv_raw) != 32 or len(pub_raw) != 32:
        raise RuntimeError("Unexpected Ed25519 raw key length")
    return priv_raw, pub_raw


def sign_bytes(private_key_raw_32: bytes, data: bytes) -> bytes:
    if len(private_key_raw_32) != 32:
        raise ValueError("private_key_raw_32 must be 32 bytes")
    key = Ed25519PrivateKey.from_private_bytes(private_key_raw_32)
    return key.sign(data)


def verify_bytes(public_key_raw_32: bytes, data: bytes, signature: bytes) -> bool:
    if len(public_key_raw_32) != 32:
        raise ValueError("public_key_raw_32 must be 32 bytes")
    key = Ed25519PublicKey.from_public_bytes(public_key_raw_32)
    try:
        key.verify(signature, data)
        return True
    except InvalidSignature:
        return False


def sign_manifest(private_key_raw_32: bytes, manifest_obj: object) -> bytes:
    return sign_bytes(private_key_raw_32, canonical_json_bytes(manifest_obj))


def verify_manifest(public_key_raw_32: bytes, manifest_obj: object, signature: bytes) -> bool:
    return verify_bytes(public_key_raw_32, canonical_json_bytes(manifest_obj), signature)

