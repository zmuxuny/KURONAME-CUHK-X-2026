#!/usr/bin/env python3
"""Raw finalist adapter with hard links compatible with resolved video paths.

The original v1 preparation and failed symlink run remain intact. Hard links
retain Testing-only discovery without copying media or escaping the view root
when frozen visual extractors call Path.resolve().
"""

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import run_finalist_pipeline as original


def make_raw_view(run, records, modalities):
    for record in records:
        target = run / "raw_view" / record["kind"] / record["relative"]
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(record["path"], target)
    sensor_root = run / "raw_view/sensors"
    sensor_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(modalities).to_csv(sensor_root / "manifest_nonvisual.csv", index=False)


def verify_raw_view(run, records):
    for record in records:
        target = run / "raw_view" / record["kind"] / record["relative"]
        if target.is_symlink() or not target.samefile(record["path"]):
            raise ValueError(f"Testing media view no longer targets the frozen raw input: {target}")
    original.verify(records)


def stage_command(run, name, attempt):
    return [sys.executable, str(Path(__file__).resolve()), "_stage", "--output-dir", str(run),
            "--stage", name, "--stage-dir", str(attempt)]


def configure():
    original.KIND = "cuhkx_raw_testing_equal_seed_finalist_v2"
    original.make_raw_view = make_raw_view
    original.verify_raw_view = verify_raw_view
    original.stage_command = stage_command
    original.EXTRACTOR_SOURCES = [*original.EXTRACTOR_SOURCES,
                                  "scripts/run_finalist_pipeline_v2.py", "finalist_v2.sh"]


def main():
    configure()
    if len(sys.argv) > 1 and sys.argv[1] != "_stage" and "--config" not in sys.argv:
        sys.argv.extend(["--config", str(ROOT / "configs/finalist_pipeline_v2.json")])
    return original.main()


if __name__ == "__main__":
    raise SystemExit(main())
