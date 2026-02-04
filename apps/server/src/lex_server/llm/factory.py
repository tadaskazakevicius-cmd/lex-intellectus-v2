from __future__ import annotations

import os
from dataclasses import replace as dc_replace
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

    p = params or LlamaParams()
    backend_env = (os.environ.get("LEX_LLAMA_BACKEND") or "").strip().lower()
    ngl_env = os.environ.get("LEX_LLAMA_N_GPU_LAYERS")

    if backend_env:
        if backend_env == "auto":
            p = dc_replace(p, backend=None)
        else:
            p = dc_replace(p, backend=backend_env)
    if ngl_env and p.n_gpu_layers is None:
        p = dc_replace(p, n_gpu_layers=int(ngl_env))

    return LlamaCppRuntime(llama_bin=llama_bin, model_path=model_path, params=p)

