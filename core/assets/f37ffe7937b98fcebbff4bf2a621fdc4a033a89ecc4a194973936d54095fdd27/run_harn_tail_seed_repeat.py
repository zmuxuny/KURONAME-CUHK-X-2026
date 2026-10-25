#!/usr/bin/env python3
"""Repeat the four-epoch subject-scoped video fit with one fixed new seed."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from cuhkx.harn_tail_seed_repeat import SeedRepeatTailClassifier
from scripts import run_harn_tail_nested as original

BASE_VALIDATE = original.validate_config
SOURCES = [*original.SOURCES, "scripts/run_harn_tail_seed_repeat.py", "src/cuhkx/harn_tail_seed_repeat.py"]


def validate_config(config):
    BASE_VALIDATE(config)
    if config.get("repeat_protocol") != {
        "original_visual_seed": 20262905,
        "repeated_visual_seed": 20262906,
        "skeleton_seed_change": 0,
        "repeats": 1,
        "ensemble_visual_weights": [0.5, 0.5],
        "ensemble_skeleton_weights": [0.0, 0.5, 0.75, 1.0],
        "ensemble_weight_selection": "inner subjects only",
    }:
        raise ValueError("configuration differs from the single fixed seed and equal ensemble")


def main():
    original.validate_config = validate_config
    original.SOURCES = SOURCES
    original.tail.TailClassifier = SeedRepeatTailClassifier
    return original.main()


if __name__ == "__main__":
    raise SystemExit(main())
