"""Compatibility entry for archived drivers in the public release.

Changed 2026-10-25: remove the private research workspace write restriction.
The public runner writes to the output directory supplied by its caller.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


def install(original: Path):
    """Retain the archived call signature without a machine-specific path ban."""
    sys.dont_write_bytecode = True


if __name__ == "__main__":
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")
