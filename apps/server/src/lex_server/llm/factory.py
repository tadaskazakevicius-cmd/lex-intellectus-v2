from __future__ import annotations

from pathlib import Path

from ..paths import get_paths
from .llama_cpp_runtime import LlamaCppRuntime, LlamaParams, find_gguf_model, find_llama_bin


def get_llm_runtime(params: LlamaParams | None = None) -> LlamaCppRuntime:
    """
    Factory for offline llama.cpp runtime using A2 paths + env overrides.
    """
    paths = get_paths()
    llama_bin = find_llama_bin(paths.app_dir, paths.data_dir)
    model_path = find_gguf_model(paths.model_dir)
    return LlamaCppRuntime(llama_bin=llama_bin, model_path=model_path, params=params or LlamaParams())

