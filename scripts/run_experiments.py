#!/usr/bin/env python3
"""Shared parallel entry point; experiment implementations live in the package."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flow2pc.experiments.parallel import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
