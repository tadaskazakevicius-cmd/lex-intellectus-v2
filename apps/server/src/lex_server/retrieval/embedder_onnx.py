from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import onnxruntime as ort
except Exception as e:  # pragma: no cover
    raise RuntimeError("onnxruntime is required for ONNX embeddings") from e

try:
    from tokenizers import Tokenizer
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "HuggingFace `tokenizers` is required for ONNX embeddings. "
        "Install with: pip install tokenizers"
    ) from e


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom


def _mean_pool(last_hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """
    Mean pooling over tokens using attention mask.
    last_hidden: (B, T, H)
    attention_mask: (B, T)
    """
    mask = attention_mask.astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)  # (B,T,1)
    summed = (last_hidden * mask).sum(axis=1)  # (B,H)
    denom = mask.sum(axis=1) + 1e-12  # (B,1)
    return (summed / denom).astype(np.float32)


@dataclass
class OnnxEmbedder:
    model_path: Path
    tokenizer_path: Path | None = None
    providers: list[str] | None = None
    max_length: int = 256
    normalize: bool = True

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path)

        if self.tokenizer_path is None:
            # default: tokenizer.json next to model
            self.tokenizer_path = self.model_path.parent / "tokenizer.json"

        self.tokenizer_path = Path(self.tokenizer_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"tokenizer.json not found: {self.tokenizer_path}")

        self.providers = self.providers or ["CPUExecutionProvider"]

        # load tokenizer
        self._tok = Tokenizer.from_file(str(self.tokenizer_path))
        # padding/truncation for batch
        self._tok.enable_truncation(max_length=int(self.max_length))
        self._tok.enable_padding(length=int(self.max_length))

        # load ONNX session
        self._sess = ort.InferenceSession(str(self.model_path), providers=self.providers)

        # figure out expected inputs
        self._input_names = [i.name for i in self._sess.get_inputs()]
        # Typical: input_ids, attention_mask, token_type_ids
        # Some models might not have token_type_ids.
        # We'll only feed what exists.

        self._output_names = [o.name for o in self._sess.get_outputs()]

    def _encode_batch(self, texts: list[str]) -> dict[str, np.ndarray]:
        enc = self._tok.encode_batch(texts)

        input_ids = np.asarray([e.ids for e in enc], dtype=np.int64)
        attention_mask = np.asarray([e.attention_mask for e in enc], dtype=np.int64)

        feeds: dict[str, np.ndarray] = {}

        if "input_ids" in self._input_names:
            feeds["input_ids"] = input_ids
        if "attention_mask" in self._input_names:
            feeds["attention_mask"] = attention_mask
        if "token_type_ids" in self._input_names:
            # Many ST models expect this even if it's always zeros
            feeds["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)

        # Some exports use different names; if so, fail loudly with clear message.
        missing = [n for n in ("input_ids", "attention_mask") if n not in self._input_names]
        if missing:
            raise RuntimeError(
                f"ONNX model inputs not supported by this wrapper. "
                f"Model expects inputs={self._input_names}. "
                f"Wrapper expects at least input_ids + attention_mask."
            )

        return feeds

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        feeds = self._encode_batch(texts)

        # run model
        outs = self._sess.run(None, feeds)

        # Try to find sentence embedding directly
        # Common names: 'sentence_embedding', 'embeddings'
        out_by_name = {name: outs[i] for i, name in enumerate(self._output_names)}
        vec: Any | None = None
        for key in ("sentence_embedding", "embeddings", "sentence_embeddings"):
            if key in out_by_name:
                vec = out_by_name[key]
                break

        if vec is None:
            # fallback: take first output and pool if it's token-level
            first = outs[0]
            arr = np.asarray(first)
            if arr.ndim == 3:
                # (B,T,H) -> mean pool
                vec = _mean_pool(arr.astype(np.float32), feeds["attention_mask"])
            elif arr.ndim == 2:
                vec = arr.astype(np.float32)
            else:
                raise RuntimeError(f"Unexpected ONNX output shape: {arr.shape} outputs={self._output_names}")

        vec = np.asarray(vec, dtype=np.float32)
        if vec.ndim != 2:
            raise RuntimeError(f"Expected embeddings shape (B,H), got {vec.shape}")

        if self.normalize:
            vec = _l2_normalize_rows(vec)

        return vec
