from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from platformdirs import PlatformDirs

# ---------------------------------------------------------------------
# App identifiers
# ---------------------------------------------------------------------

# Slug naudojamas katalogams, temp, platformdirs (be tarpų!)
APP_SLUG = "lex-intellectus"

# Display name (UI, audit, logs)
APP_DISPLAY_NAME = "Lex Intellectus"


# ---------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class AppPaths:
    """
    OS-agnostic paths used by the app.

    Environment overrides (optional):
    - LEX_APP_DIR
    - LEX_DATA_DIR
    - LEX_MODEL_DIR
    - LEX_TEMP_DIR
    """

    app_dir: Path     # runtime root (read-only)
    data_dir: Path    # mutable app data (db, packs, audit)
    model_dir: Path   # local LLM / embedding models
    temp_dir: Path    # temp working dir


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _default_app_dir() -> Path:
    """
    Resolve application runtime directory.

    Priority:
    1. LEX_APP_DIR env override
    2. Frozen app (PyInstaller, etc.) → directory of executable
    3. Dev mode → current working directory
    """
    env = os.environ.get("LEX_APP_DIR")
    if env:
        return Path(env).expanduser().resolve()

    # Frozen binary (PyInstaller / similar)
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent

    # Dev mode: assume cwd is repo / runtime root
    return Path.cwd().resolve()


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def get_paths() -> AppPaths:
    """
    Compute all filesystem paths used by the app.
    """
    app_dir = _default_app_dir()

    data_env = os.environ.get("LEX_DATA_DIR")
    model_env = os.environ.get("LEX_MODEL_DIR")
    temp_env = os.environ.get("LEX_TEMP_DIR")

    # Platform-specific user data directory
    dirs = PlatformDirs(appname=APP_SLUG, appauthor=False)

    data_dir = (
        Path(data_env).expanduser().resolve()
        if data_env
        else Path(dirs.user_data_dir).resolve()
    )

    model_dir = (
        Path(model_env).expanduser().resolve()
        if model_env
        else (data_dir / "models").resolve()
    )

    temp_dir = (
        Path(temp_env).expanduser().resolve()
        if temp_env
        else (Path(tempfile.gettempdir()).resolve() / APP_SLUG)
    )

    return AppPaths(
        app_dir=app_dir,
        data_dir=data_dir,
        model_dir=model_dir,
        temp_dir=temp_dir,
    )


def ensure_dirs(paths: AppPaths) -> None:
    """
    Ensure mutable directories exist.

    IMPORTANT:
    - app_dir is treated as read-only and is NOT created here
      (installer or dev environment is responsible for it).
    """
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.model_dir.mkdir(parents=True, exist_ok=True)
    paths.temp_dir.mkdir(parents=True, exist_ok=True)
