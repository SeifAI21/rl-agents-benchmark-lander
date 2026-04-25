"""Top-level launcher so `python -m evaluation ...` works from repo root.

The actual implementation lives in `evaluation/src/evaluation/__main__.py`.
"""

from evaluation.src.evaluation.__main__ import main


if __name__ == "__main__":
    raise SystemExit(main())
