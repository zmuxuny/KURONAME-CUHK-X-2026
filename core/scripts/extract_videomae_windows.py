#!/usr/bin/env python3
"""Extract label-free, explicitly keyed VideoMAE windows from a fixed raw plan."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np

from cuhkx.validation import sha256_file
from cuhkx.videomae_legacy_weights import load_runtime
from cuhkx.videomae_windows import encode_window, read_windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    hashes = {str(args.config): sha256_file(args.config)}
    for entry in [config["records_file"], *config["implementation"]]:
        path = ROOT / entry["path"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError("VideoMAE extraction declaration changed")
        hashes[str(path)] = entry["sha256"]
    records = json.loads((ROOT / config["records_file"]["path"]).read_text())["records"]
    keys = [(record["unit"], record["modality"]) for record in records]
    if len(keys) != len(set(keys)) or len(keys) != config["expected_records"]:
        raise ValueError("VideoMAE record keys or count differ")
    count, windows = len(records), config["windows"]
    arrays = {name: np.lib.format.open_memmap(output / (name + ".npy"), mode="w+", dtype=np.float32,
                                             shape=(count, windows, width))
              for name, width in (("features", 768), ("logits", 174))}
    available = np.asarray([record["available"] for record in records], dtype=bool)
    completed = np.zeros(count, dtype=bool)
    frames = np.full((count, windows, 16), -1, dtype=np.int32)
    from cuhkx.video import temporal_window_indices

    for i, record in enumerate(records):
        if available[i]:
            meta = record["metadata"]
            frames[i] = temporal_window_indices(meta["frame_count"], meta["fps"], windows, 16,
                                               config["window_seconds"])
    np.savez_compressed(output / "metadata.npz", units=np.asarray([key[0] for key in keys]),
                        modalities=np.asarray([key[1] for key in keys]), available=available,
                        frame_indices=frames,
                        alignment_ok=np.asarray([record["alignment_ok"] for record in records]))
    state = {"status": "running", "pid": os.getpid(), "completed_records": 0, "records": count,
             "config_sha256": hashes[str(args.config)], "started_utc": datetime.now(UTC).isoformat()}

    def checkpoint():
        for array in arrays.values():
            array.flush()
        np.save(output / "completed.npy", completed)
        state["completed_records"] = int(completed.sum())
        state["updated_utc"] = datetime.now(UTC).isoformat()
        (output / "state.json").write_text(json.dumps(state, indent=2) + "\n")

    checkpoint()
    started = time.perf_counter()
    runtime = load_runtime(config)
    (output / "strict_load_audit.json").write_text(json.dumps(runtime[1].cuhkx_load_audit, indent=2) + "\n")
    try:
        for row, record in enumerate(records):
            if available[row]:
                values = read_windows(record, config)
                for window, value in enumerate(values):
                    feature, logits = encode_window(runtime, value, check_classifier=row == 0 and window == 0)
                    arrays["features"][row, window] = feature
                    arrays["logits"][row, window] = logits
            else:
                arrays["features"][row] = 0
                arrays["logits"][row] = 0
            completed[row] = True
            if (row + 1) % 25 == 0 or row + 1 == count:
                state["elapsed_seconds"] = time.perf_counter() - started
                checkpoint()
                print(json.dumps(state), flush=True)
        for path, digest in hashes.items():
            assert sha256_file(Path(path)) == digest
        # Re-read raw first/last available windows and compare against the stored arrays.
        controls = []
        for row in np.flatnonzero(available)[[0, -1]]:
            replay = encode_window(runtime, read_windows(records[row], config)[0])[0]
            np.testing.assert_array_equal(arrays["features"][row, 0], replay)
            controls.append(int(row))
        manifest = {"kind": config["kind"], "config_sha256": hashes[str(args.config)],
                    "input_hashes": hashes, "records": count, "windows": windows,
                    "available_records": int(available.sum()), "raw_replay_rows_exact": controls,
                    "feature_shape": list(arrays["features"].shape), "labels_used": False,
                    "new_neural_fits": 0, "peak_allocated_bytes": runtime[0].cuda.max_memory_allocated(),
                    "peak_reserved_bytes": runtime[0].cuda.max_memory_reserved(),
                    "files": [{"path": name, "sha256": sha256_file(output / name),
                               "bytes": (output / name).stat().st_size}
                              for name in ["features.npy", "logits.npy", "metadata.npz", "completed.npy"]]}
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        state["status"] = "complete"
        checkpoint()
    except Exception as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        checkpoint()
        raise


if __name__ == "__main__":
    main()
