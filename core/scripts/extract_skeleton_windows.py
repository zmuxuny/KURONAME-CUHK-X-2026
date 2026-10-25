#!/usr/bin/env python3
"""Cache compact sliding-window Skeleton features for temporal localization."""

from __future__ import annotations

import argparse
import concurrent.futures
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuhkx.skeleton import load_sequence
from cuhkx.skeleton_windows import extract_window_features, window_bounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-frames", type=int, default=32)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def extract_one(
    task: tuple[str, str, int, int],
) -> tuple[str, list[tuple[int, int, float, np.ndarray]], tuple[str, ...] | None, str]:
    root, unit, size, stride = task
    try:
        poses, confidence = load_sequence(Path(root) / unit / "Skeleton" / "predictions")
        rows = []
        for start, stop in window_bounds(len(poses), size=size, stride=stride):
            values, names = extract_window_features(poses[start:stop], confidence[start:stop])
            center = (start + stop - 1) / (2 * max(len(poses) - 1, 1))
            rows.append((start, stop, center, values))
        return unit, rows, names, ""
    except Exception as error:  # noqa: BLE001 - persisted and surfaced to the caller
        return unit, [], None, f"{type(error).__name__}: {error}"


def main() -> int:
    args = parse_args()
    manifest = pd.read_csv(
        args.sensor_root / "manifest_nonvisual.csv", dtype=str, keep_default_na=False
    )
    units = manifest.loc[
        manifest["modalities"].str.split(";").map(lambda values: "Skeleton" in values), "unit"
    ].tolist()
    tasks = [
        (str(args.sensor_root.resolve()), unit, args.window_frames, args.stride) for unit in units
    ]
    records = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, record in enumerate(executor.map(extract_one, tasks, chunksize=4), start=1):
            records.append(record)
            if index % 100 == 0 or index == len(tasks):
                failures = sum(bool(item[3]) for item in records)
                windows = sum(len(item[1]) for item in records)
                print(
                    f"processed={index}/{len(tasks)} windows={windows} failures={failures}",
                    flush=True,
                )

    good = [record for record in records if record[1]]
    bad = [(unit, error) for unit, _, _, error in records if error]
    if not good:
        raise RuntimeError("no Skeleton windows were extracted")
    feature_names = good[0][2]
    if any(record[2] != feature_names for record in good):
        raise RuntimeError("feature names differ between units")
    flat = [(unit, *window) for unit, rows, _, _ in good for window in rows]
    matrix = np.stack([row[4] for row in flat]).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        X=matrix,
        units=np.asarray([row[0] for row in flat]),
        starts=np.asarray([row[1] for row in flat]),
        stops=np.asarray([row[2] for row in flat]),
        centers=np.asarray([row[3] for row in flat], dtype=np.float32),
        feature_names=np.asarray(feature_names),
        failed_units=np.asarray([item[0] for item in bad]),
        failure_messages=np.asarray([item[1] for item in bad]),
    )
    print(f"wrote={args.output.resolve()} shape={matrix.shape} failures={len(bad)}", flush=True)
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
