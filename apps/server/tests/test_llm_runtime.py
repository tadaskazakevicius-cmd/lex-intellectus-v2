from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from lex_server.llm.factory import get_llm_runtime
from lex_server.llm.llama_cpp_runtime import LlamaCppRuntime, LlamaParams


def _make_fake_llama(tmp_path: Path) -> Path:
    """
    Create a fake llama-cli wrapper executable that prints a deterministic response.
    """
    fake_py = tmp_path / "fake_llama.py"
    fake_py.write_text(
        """
import sys

def _get_arg(name, default=None):
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return sys.argv[i+1]
    return default

prompt = _get_arg("-p", _get_arg("--prompt", ""))
sys.stdout.write(f"You said: {prompt}\\n")
sys.exit(0)
""".lstrip(),
        encoding="utf-8",
    )

    if os.name == "nt":
        cmd = tmp_path / "llama-cli.cmd"
        cmd.write_text(
            f'@echo off\r\n"{sys.executable}" "{fake_py}" %*\r\n',
            encoding="utf-8",
        )
        return cmd

    sh = tmp_path / "llama-cli"
    sh.write_text(
        f"#!/bin/sh\n\"{sys.executable}\" \"{fake_py}\" \"$@\"\n",
        encoding="utf-8",
    )
    sh.chmod(0o755)
    return sh


def test_llm_runtime_fake_exec(tmp_path: Path) -> None:
    llama_bin = _make_fake_llama(tmp_path)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"dummy")

    rt = LlamaCppRuntime(llama_bin=llama_bin, model_path=model, params=LlamaParams(n_predict=8, ctx=256))
    out = rt.generate("hello")
    assert "You said:" in out
    assert "hello" in out


def test_llm_runtime_real_optional() -> None:
    if not os.environ.get("LEX_LLAMA_BIN") or not os.environ.get("LEX_MODEL_GGUF"):
        pytest.skip("real llama.cpp smoke test requires LEX_LLAMA_BIN and LEX_MODEL_GGUF")

    rt = get_llm_runtime(LlamaParams(n_predict=16, ctx=512, timeout_sec=120))
    out = rt.generate("hello")
    assert isinstance(out, str)
    assert out.strip() != ""

