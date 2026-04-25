"""Make `evaluation` and `src` importable when running `pytest evaluation/tests`."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PKG_SRC = _REPO_ROOT / "evaluation" / "src"

for path in (_REPO_ROOT, _PKG_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
