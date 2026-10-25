#!/usr/bin/env python3
"""Cache deterministic skeleton features for every available train/test unit."""

from __future__ import annotations

import argparse
import concurrent.futures
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuhkx.skeleton import extract_unit_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def extract_one(task: tuple[str, str]) -> tuple[str, np.ndarray | None, tuple[str, ...] | None, str]:
    root, unit = task
    try:
        values, names = extract_unit_features(Path(root) / unit)
        return unit, values, names, ""
    except Exception as error:  # noqa: BLE001 - persisted and surfaced to the caller
        return unit, None, None, f"{type(error).__name__}: {error}"


def main() -> int:
    args = parse_args()
    manifest = pd.read_csv(
        args.sensor_root / "manifest_nonvisual.csv", dtype=str, keep_default_na=False
    )
    units = manifest.loc[
        manifest["modalities"].str.split(";").map(lambda values: "Skeleton" in values), "unit"
    ].tolist()
    tasks = [(str(args.sensor_root.resolve()), unit) for unit in units]
    records: list[tuple[str, np.ndarray | None, tuple[str, ...] | None, str]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, record in enumerate(executor.map(extract_one, tasks, chunksize=8), start=1):
            records.append(record)
            if index % 100 == 0 or index == len(tasks):
                failures = sum(bool(item[3]) for item in records)
                print(f"processed={index}/{len(tasks)} failures={failures}", flush=True)

    good = [record for record in records if record[1] is not None]
    bad = [(unit, error) for unit, _, _, error in records if error]
    if not good:
        raise RuntimeError("no skeleton features were extracted")
    feature_names = good[0][2]
    if any(record[2] != feature_names for record in good):
        raise RuntimeError("feature names differ between units")
    matrix = np.stack([record[1] for record in good]).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        X=matrix,
        units=np.asarray([record[0] for record in good]),
        feature_names=np.asarray(feature_names),
        failed_units=np.asarray([item[0] for item in bad]),
        failure_messages=np.asarray([item[1] for item in bad]),
    )
    print(
        f"wrote={args.output.resolve()} shape={matrix.shape} failures={len(bad)}",
        flush=True,
    )
    for unit, error in bad[:20]:
        print(f"FAILED {unit}: {error}", file=sys.stderr)
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
