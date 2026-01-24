from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from platformdirs import PlatformDirs

APP_NAME = "Lex Intellectus"


@dataclass(frozen=True)
class AppPaths:
    """OS-agnostic paths used by the app.

    Environment overrides (optional):
    - LEX_APP_DIR
    - LEX_DATA_DIR
    - LEX_MODEL_DIR
    - LEX_TEMP_DIR
    """

    app_dir: Path
    data_dir: Path
    model_dir: Path
    temp_dir: Path


def _default_app_dir() -> Path:
    # "binary/runtime root": best-effort approximation that works for both
    # packaged apps and `python -m uvicorn ...` dev runs.
    env = os.environ.get("LEX_APP_DIR")
    if env:
        return Path(env).expanduser().resolve()

    # If frozen (PyInstaller, etc.), sys.executable points to the packaged binary.
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent

    # Dev: treat current working directory as runtime root.
    return Path.cwd().resolve()


def get_paths() -> AppPaths:
    app_dir = _default_app_dir()

    data_env = os.environ.get("LEX_DATA_DIR")
    model_env = os.environ.get("LEX_MODEL_DIR")
    temp_env = os.environ.get("LEX_TEMP_DIR")

    dirs = PlatformDirs(appname=APP_NAME, appauthor=False)
    data_dir = (
        Path(data_env).expanduser().resolve() if data_env else Path(dirs.user_data_dir).resolve()
    )
    model_dir = (
        Path(model_env).expanduser().resolve()
        if model_env
        else (data_dir / "models").resolve()
    )
    temp_dir = (
        Path(temp_env).expanduser().resolve()
        if temp_env
        else (Path(tempfile.gettempdir()).resolve() / APP_NAME)
    )

    return AppPaths(app_dir=app_dir, data_dir=data_dir, model_dir=model_dir, temp_dir=temp_dir)


def ensure_dirs(paths: AppPaths) -> None:
    paths.app_dir.mkdir(parents=True, exist_ok=True)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.model_dir.mkdir(parents=True, exist_ok=True)
    paths.temp_dir.mkdir(parents=True, exist_ok=True)

