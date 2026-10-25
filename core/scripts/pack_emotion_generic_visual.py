#!/usr/bin/env python3
"""Freeze keyed generic-video mean/std descriptors as an emotion modality."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuhkx.emotion_visual_nested import aggregate_visual
from cuhkx.validation import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--extraction-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists() or args.run_dir.exists():
        raise FileExistsError("visual aggregation outputs must be new")
    source_paths = [Path(__file__), ROOT / "src/cuhkx/emotion_visual_nested.py",
                    ROOT / "src/cuhkx/validation.py"]
    input_paths = [args.windows, args.extraction_run / "report.json",
                   args.extraction_run / "frozen_config.json", args.extraction_run / "input_manifest.json"]
    inputs = {str(path.resolve()): sha256_file(path) for path in input_paths}
    upstream = json.loads((args.extraction_run / "report.json").read_text())
    if sha256_file(args.windows) != upstream["sha256"]:
        raise ValueError("window archive differs from its completed extraction report")
    frozen = {"input_hashes": inputs, "source_hashes": {
        str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths}}
    args.run_dir.mkdir(parents=True)
    (args.run_dir / "frozen_inputs.json").write_text(json.dumps(frozen, indent=2) + "\n")
    for path in source_paths:
        target = args.run_dir / "source_snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    with np.load(args.windows, allow_pickle=False) as archive:
        arrays, report = aggregate_visual({name: archive[name].copy() for name in archive.files})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    for path, digest in inputs.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError("frozen aggregation input changed")
    report.update({"status": "generic_visual_emotion_descriptors_aggregated", "inputs": inputs,
                   "sources": frozen["source_hashes"], "output": str(args.output.resolve()),
                   "sha256": sha256_file(args.output), "bytes": args.output.stat().st_size,
                   "numpy_version": np.__version__})
    for path in (args.output.with_suffix(".json"), args.run_dir / "report.json"):
        path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "shape", "accepted_units", "rejected_units", "sha256", "bytes")}, indent=2))


if __name__ == "__main__":
    main()
