from __future__ import annotations

import os
import logging
import platform
import subprocess
from dataclasses import dataclass, field, replace as dc_replace
from pathlib import Path


logger = logging.getLogger(__name__)


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
    backend: str | None = None  # None = auto
    n_gpu_layers: int | None = None


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


def detect_backend(llama_bin: Path) -> str:
    """
    Best-effort backend detection for llama.cpp CLI.

    Priority:
    1) env LEX_LLAMA_BACKEND (cpu|cuda|metal|vulkan|auto)
    2) auto-detect from `llama_bin --help` output keywords
    Fallback: cpu
    """
    env = (os.environ.get("LEX_LLAMA_BACKEND") or "").strip().lower()
    if env and env != "auto":
        return env

    try:
        cp = _run_cmd([str(llama_bin), "--help"], timeout_sec=5)
        txt = ((cp.stdout or "") + "\n" + (cp.stderr or "")).lower()
    except Exception:
        return "cpu"

    # Prefer Metal if advertised.
    if "metal" in txt:
        return "metal"
    if "vulkan" in txt:
        return "vulkan"
    if ("cublas" in txt) or ("cuda" in txt):
        return "cuda"
    return "cpu"


_GPU_FLAG_ERR_HINTS = (
    "unknown option",
    "unrecognized",
    "invalid option",
    "unknown argument",
    "not recognized",
)


class LlamaCppRuntime:
    def __init__(self, llama_bin: Path, model_path: Path, params: LlamaParams | None = None) -> None:
        self.llama_bin = Path(llama_bin)
        self.model_path = Path(model_path)
        self.params = params or LlamaParams()
        self._backend_selected: str | None = None

    @property
    def backend_selected(self) -> str | None:
        return self._backend_selected

    def _resolve_backend(self, params: LlamaParams) -> tuple[str, int]:
        backend = (params.backend or "").strip().lower() if params.backend else None
        if backend is None:
            backend = self._backend_selected or detect_backend(self.llama_bin)
        if backend in ("", "auto"):
            backend = "cpu"

        n_gpu_layers = 0
        if backend != "cpu":
            env_ngl = os.environ.get("LEX_LLAMA_N_GPU_LAYERS")
            if params.n_gpu_layers is not None:
                n_gpu_layers = int(params.n_gpu_layers)
            elif env_ngl:
                n_gpu_layers = int(env_ngl)
            else:
                n_gpu_layers = 9999

        self._backend_selected = backend
        logger.info("LLM backend selected: %s", backend)
        return backend, int(n_gpu_layers)

    def _build_args(self, prompt: str, params: LlamaParams, *, with_gpu: bool) -> list[str]:
        t = params.threads if params.threads is not None else _default_threads()
        backend, n_gpu_layers = self._resolve_backend(params)
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

        if with_gpu and backend != "cpu" and n_gpu_layers > 0:
            args += ["--n-gpu-layers", str(int(n_gpu_layers))]

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

        def _try(with_gpu: bool, p_use: LlamaParams) -> subprocess.CompletedProcess[str]:
            args = self._build_args(prompt, p_use, with_gpu=with_gpu)
            return _run_cmd(args, timeout_sec=int(p.timeout_sec))

        try:
            cp = _try(with_gpu=True, p_use=p)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"llama.cpp timeout after {p.timeout_sec}s") from e

        if cp.returncode != 0:
            err = (cp.stderr or "").strip()
            err_l = err.lower()
            # Fail-safe retry: unknown GPU flag => rerun once on CPU.
            if any(h in err_l for h in _GPU_FLAG_ERR_HINTS) and "--n-gpu-layers" in " ".join(map(str, cp.args)):
                logger.warning("GPU flag unsupported, falling back to CPU: %s", err[:200])
                self._backend_selected = "cpu"
                p_cpu = dc_replace(p, backend="cpu", n_gpu_layers=0)
                cp2 = _try(with_gpu=False, p_use=p_cpu)
                if cp2.returncode == 0:
                    return (cp2.stdout or "").strip()
                err2 = (cp2.stderr or "").strip()
                out2 = (cp2.stdout or "").strip()
                msg2 = err2 or out2 or f"returncode={cp2.returncode}"
                raise RuntimeError(f"llama.cpp failed: {msg2[:4000]}")

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

