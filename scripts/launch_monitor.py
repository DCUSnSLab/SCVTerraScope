"""Convenience launcher equivalent to `python -m scvterrascope.gui`.

Useful when running from the repo without installing the package — the
project root just needs to be on `PYTHONPATH` (set automatically when run
via `python scripts/launch_monitor.py` since `__main__` is in `src/`).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src/scvterrascope` importable when run uninstalled.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from scvterrascope.gui.__main__ import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
