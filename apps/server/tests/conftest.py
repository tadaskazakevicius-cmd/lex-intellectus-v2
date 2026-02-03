from __future__ import annotations

import sys
from pathlib import Path


# Ensure `lex_server` (under ./src) is importable when running `pytest` from apps/server.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

