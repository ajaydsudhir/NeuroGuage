"""Live entry point aligned to demo dashboard defaults.

This script intentionally reuses ``scripts/run_demo.py`` forwarding logic so
both ``run_live`` and ``run_demo`` launch with the same dashboard/runtime
profile unless the caller overrides flags explicitly.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuroguage.cli import main_run_live
from run_demo import main


if __name__ == "__main__":
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        raise SystemExit(main_run_live())
    raise SystemExit(main())
