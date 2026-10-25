#!/usr/bin/env python3
"""Replay Testing extractors with the verified GPU crop and complete metadata."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import reproduce_raw_test_features as engine

SCALARS = ("backbone", "weights", "frames", "windows", "window_seconds")
EXTRA_SOURCES = [
    "scripts/reproduce_raw_test_clean_cuda.py", "scripts/extract_video_midfeatures.py",
    "src/cuhkx/video_midfeatures.py", "scripts/infer_adapted_experts.py",
    "configs/runtime_cuda_linux_py311_v1.json", "requirements-cuda-raw-linux-py311.lock",
]


def complete_arrays(arrays, config):
    if set(SCALARS) & set(arrays):
        raise ValueError("archive builder unexpectedly already supplied scalar metadata")
    return {**arrays, **{key: np.asarray(config[key]) for key in SCALARS}}


def main():
    entrypoint = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    original_write = engine.write
    engine.SOURCES = [*engine.SOURCES, *EXTRA_SOURCES]

    def write(path, value):
        if Path(path).name == "frozen_contract.json":
            value["detector"]["device"] = "0"
            value["execution_adapter"] = {
                "command": entrypoint, "python_prefix": sys.prefix,
                "python_base_prefix": sys.base_prefix,
                "generic_archive_scalars": list(SCALARS),
                "reason": "Use the previously verified GPU ROI recipe and include the original extractor's five serialization scalars at creation.",
            }
            for name in ("checkpoint_inference_harn_tail_v1.json",
                         "checkpoint_inference_emotion_v6_harn_tail_v1.json"):
                config_path = ROOT / "configs" / name
                config = json.loads(config_path.read_text())
                for file in [config_path, *(ROOT / p for p in config["checkpoints"].values())]:
                    value["fixed_inputs"][str(file)] = engine.sha(file)
        elif Path(path).parent.name == "reports":
            value["actual_adapter_command"] = entrypoint
            value["python_prefix"] = sys.prefix
        original_write(path, value)

    engine.write = write
    if "emotion_visual" in sys.argv:
        extractor = importlib.import_module("extract_generic_hau_windows")
        original_arrays = extractor.archive_arrays

        def archive_arrays(inputs, config):
            return complete_arrays(original_arrays(inputs, config), config)

        extractor.archive_arrays = archive_arrays
    return engine.main()


if __name__ == "__main__":
    raise SystemExit(main())
