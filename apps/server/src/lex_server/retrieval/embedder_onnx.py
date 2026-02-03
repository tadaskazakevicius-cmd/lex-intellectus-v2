from __future__ import annotations

"""
E3: ONNX Runtime (CPU) embedder wrapper.

This module is structured so unit tests don't require a real ONNX model file:
- `onnxruntime` is imported lazily in __init__.
- Retrieval code can accept a "pluggable" embedder with `embed_text(s)` methods.
"""

from pathlib import Path
from typing import Any

import numpy as np


class OnnxEmbedder:
    def __init__(self, model_path: Path, providers: list[str] | None = None) -> None:
        self.model_path = Path(model_path)
        self.providers = providers or ["CPUExecutionProvider"]

        try:
            import onnxruntime as ort  # type: ignore
        except ModuleNotFoundError as e:  # pragma: no cover
            raise RuntimeError(
                "onnxruntime is required to use OnnxEmbedder. Install server deps."
            ) from e

        # NOTE:
        # This is a minimal wrapper. The actual tokenizer/input construction depends on the chosen model.
        # For now we require the model to accept a single string input named 'text' OR already-embedded vectors.
        self._ort = ort
        self._sess = ort.InferenceSession(str(self.model_path), providers=self.providers)
        self._input_names = [i.name for i in self._sess.get_inputs()]
        self._output_names = [o.name for o in self._sess.get_outputs()]

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        if x.ndim == 1:
            denom = np.linalg.norm(x) + 1e-12
            return x / denom
        denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / denom

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Returns embeddings as float32 array of shape (n, d).

        Model input expectations vary; this wrapper supports a minimal contract:
        - If the model has a single input called 'text', it will be fed a numpy object array of strings.
        Otherwise, raise with guidance (caller must adapt tokenizer/model contract).
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        if self._input_names == ["text"]:
            inp = {"text": np.asarray(texts, dtype=object)}
            out = self._sess.run(self._output_names, inp)[0]
            emb = np.asarray(out, dtype=np.float32)
            if emb.ndim != 2:
                raise RuntimeError(f"Unexpected embedding output shape: {emb.shape}")
            return self._l2_normalize(emb)

        raise RuntimeError(
            "OnnxEmbedder model inputs are not supported by this MVP wrapper. "
            f"Expected single input ['text'], got {self._input_names}. "
            "Adapt embedder to your tokenizer/model input signature."
        )

    def embed_text(self, text: str) -> np.ndarray:
        emb = self.embed_texts([text])
        if emb.shape[0] != 1:
            raise RuntimeError("Unexpected embedding batch size")
        return emb[0]

