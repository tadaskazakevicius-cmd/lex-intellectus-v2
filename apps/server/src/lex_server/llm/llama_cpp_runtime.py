from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class LlamaParams:
    temperature: float = 0.1
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1
    seed: int = 42
    ctx: int = 4096
    n_predict: int = 256
    threads: int | None = None
    batch: int | None = None
    stop: list[str] = field(default_factory=list)
    timeout_sec: int = 120


def _default_threads() -> int:
    return int(os.cpu_count() or 4)


def _is_windows() -> bool:
    return os.name == "nt"


def _run_cmd(args: list[str], *, timeout_sec: int) -> subprocess.CompletedProcess[str]:
    # Windows: .cmd/.bat need cmd.exe to execute.
    if _is_windows() and args and args[0].lower().endswith((".cmd", ".bat")):
        args = ["cmd.exe", "/c", args[0], *args[1:]]
    return subprocess.run(
        args,
        input=None,
        text=True,
        capture_output=True,
        timeout=int(timeout_sec),
        check=False,
    )


class LlamaCppRuntime:
    def __init__(self, llama_bin: Path, model_path: Path, params: LlamaParams | None = None) -> None:
        self.llama_bin = Path(llama_bin)
        self.model_path = Path(model_path)
        self.params = params or LlamaParams()

    def _build_args(self, prompt: str, params: LlamaParams) -> list[str]:
        t = params.threads if params.threads is not None else _default_threads()
        args: list[str] = [
            str(self.llama_bin),
            "-m",
            str(self.model_path),
            "-p",
            prompt,
            "-n",
            str(int(params.n_predict)),
            "-c",
            str(int(params.ctx)),
            "-t",
            str(int(t)),
            "--temp",
            str(float(params.temperature)),
            "--top-p",
            str(float(params.top_p)),
            "--top-k",
            str(int(params.top_k)),
            "--repeat-penalty",
            str(float(params.repeat_penalty)),
            "--seed",
            str(int(params.seed)),
        ]

        if params.batch is not None:
            args += ["--batch-size", str(int(params.batch))]

        for s in params.stop:
            if s:
                args += ["--stop", s]

        # Best-effort: many llama.cpp CLIs support these; fake CLI in tests ignores unknown flags.
        args += ["--no-display-prompt", "--silent"]
        return args

    def generate(self, prompt: str, params: LlamaParams | None = None) -> str:
        p = params or self.params
        if not self.llama_bin.exists():
            raise RuntimeError(f"llama.cpp binary not found: {self.llama_bin}")
        if not self.model_path.exists():
            raise RuntimeError(f"GGUF model not found: {self.model_path}")

        args = self._build_args(prompt, p)
        try:
            cp = _run_cmd(args, timeout_sec=int(p.timeout_sec))
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"llama.cpp timeout after {p.timeout_sec}s") from e

        if cp.returncode != 0:
            err = (cp.stderr or "").strip()
            out = (cp.stdout or "").strip()
            msg = err or out or f"returncode={cp.returncode}"
            raise RuntimeError(f"llama.cpp failed: {msg[:4000]}")

        return (cp.stdout or "").strip()

    def healthcheck(self) -> dict:
        # Best-effort version probe.
        args = [str(self.llama_bin), "--version"]
        try:
            cp = _run_cmd(args, timeout_sec=10)
            return {
                "ok": cp.returncode == 0,
                "stdout": (cp.stdout or "").strip(),
                "stderr": (cp.stderr or "").strip(),
            }
        except Exception as e:  # pragma: no cover
            return {"ok": False, "error": str(e)}


def find_llama_bin(app_dir: Path, data_dir: Path) -> Path:
    """
    Resolve llama.cpp CLI executable path.

    Priority:
    1) env LEX_LLAMA_BIN (full path)
    2) data_dir/bin/llama-cli(.exe)
    3) app_dir/bin/llama-cli(.exe)
    4) data_dir/bin/main(.exe)
    5) app_dir/bin/main(.exe)
    """
    env = os.environ.get("LEX_LLAMA_BIN")
    if env:
        p = Path(env).expanduser().resolve()
        return p

    candidates = ["llama-cli", "main"]
    exts = [".exe"] if _is_windows() else [""]

    for base in candidates:
        for ext in exts:
            for root in (data_dir / "bin", app_dir / "bin"):
                p = (root / f"{base}{ext}").resolve()
                if p.exists():
                    return p

    raise RuntimeError(
        "llama.cpp binary not found. Set LEX_LLAMA_BIN to the full path of llama-cli/main."
    )


def find_gguf_model(model_dir: Path) -> Path:
    """
    Resolve GGUF model path.

    Priority:
    1) env LEX_MODEL_GGUF
    2) if exactly one *.gguf in model_dir
    """
    env = os.environ.get("LEX_MODEL_GGUF")
    if env:
        return Path(env).expanduser().resolve()

    ggufs = sorted(model_dir.glob("*.gguf"))
    if len(ggufs) == 1:
        return ggufs[0].resolve()

    raise RuntimeError(
        f"GGUF model not found. Set LEX_MODEL_GGUF or place exactly one .gguf in {model_dir}."
    )

