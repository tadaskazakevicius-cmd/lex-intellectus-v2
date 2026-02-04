from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from lex_server.llm.llama_cpp_runtime import LlamaCppRuntime, LlamaParams, detect_backend


def _make_fake_llama_with_help(tmp_path: Path, help_text: str, *, fail_on_ngl: bool = False) -> Path:
    fake_py = tmp_path / "fake_llama.py"
    fail_flag = "True" if fail_on_ngl else "False"
    fake_py.write_text(
        f"""
import sys

args = sys.argv[1:]
if "--help" in args or "--version" in args:
    sys.stdout.write({help_text!r} + "\\n")
    sys.exit(0)

if {fail_flag} and "--n-gpu-layers" in args:
    sys.stderr.write("unknown option --n-gpu-layers\\n")
    sys.exit(1)

def _get_arg(name, default=""):
    if name in args:
        i = args.index(name)
        if i + 1 < len(args):
            return args[i+1]
    return default

prompt = _get_arg("-p", "")
sys.stdout.write("OK: " + prompt + "\\n")
sys.exit(0)
""".lstrip(),
        encoding="utf-8",
    )

    if os.name == "nt":
        cmd = tmp_path / "llama-cli.cmd"
        cmd.write_text(f'@echo off\r\n"{sys.executable}" "{fake_py}" %*\r\n', encoding="utf-8")
        return cmd

    sh = tmp_path / "llama-cli"
    sh.write_text(f"#!/bin/sh\n\"{sys.executable}\" \"{fake_py}\" \"$@\"\n", encoding="utf-8")
    sh.chmod(0o755)
    return sh


def test_detect_backend_env_override(tmp_path: Path, monkeypatch) -> None:
    llama = _make_fake_llama_with_help(tmp_path, "cuBLAS CUDA")
    monkeypatch.setenv("LEX_LLAMA_BACKEND", "cpu")
    assert detect_backend(llama) == "cpu"


def test_detect_backend_from_help_cuda(tmp_path: Path, monkeypatch) -> None:
    llama = _make_fake_llama_with_help(tmp_path, "This build uses cuBLAS")
    monkeypatch.delenv("LEX_LLAMA_BACKEND", raising=False)
    assert detect_backend(llama) == "cuda"


def test_detect_backend_from_help_metal(tmp_path: Path, monkeypatch) -> None:
    llama = _make_fake_llama_with_help(tmp_path, "metal backend available")
    monkeypatch.delenv("LEX_LLAMA_BACKEND", raising=False)
    assert detect_backend(llama) == "metal"


def test_gpu_flag_retry_fallback(tmp_path: Path, monkeypatch) -> None:
    llama = _make_fake_llama_with_help(tmp_path, "cuBLAS CUDA", fail_on_ngl=True)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"dummy")

    monkeypatch.delenv("LEX_LLAMA_BACKEND", raising=False)
    rt = LlamaCppRuntime(
        llama_bin=llama,
        model_path=model,
        params=LlamaParams(backend="cuda", n_gpu_layers=9999, n_predict=4, ctx=256, timeout_sec=10),
    )
    out = rt.generate("hello")
    assert "OK:" in out
    assert rt.backend_selected == "cpu"

