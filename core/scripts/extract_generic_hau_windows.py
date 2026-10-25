#!/usr/bin/env python3
"""Extract generic Kinetics HAU windows with explicit keys and time alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuhkx.generic_video_windows import (
    sample_preprocessed,
    temporal_alignment,
    video_metadata,
    window_timing,
)
from cuhkx.metadata import feature_unit
from cuhkx.video import discover_videos

SOURCE_FILES = ["scripts/extract_generic_hau_windows.py", "src/cuhkx/generic_video_windows.py",
                "src/cuhkx/video.py", "src/cuhkx/metadata.py", "scripts/extract_video_embeddings.py"]


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    def default(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value, indent=2, default=default) + "\n")


def check_config(config):
    fixed = {"backbone": "r2plus1d_18", "windows": 8, "frames": 16, "window_seconds": 2.5,
             "workers": 1, "torch_threads": 1, "opencv_threads": 1, "batch_size": 8,
             "modalities": ["Depth", "IR"], "device": "cuda"}
    if not config.get("frozen_before_extraction") or any(config[key] != value for key, value in fixed.items()):
        raise ValueError("the extraction must preserve the predeclared generic window configuration")


def setup_runtime(config):
    import cv2
    import torch
    from torch import nn
    from torchvision.models.video import r2plus1d_18

    cv2.setNumThreads(config["opencv_threads"])
    torch.set_num_threads(config["torch_threads"])
    torch.set_num_interop_threads(config["torch_threads"])
    if not torch.cuda.is_available():
        raise RuntimeError("the predeclared generic GPU extraction requires CUDA")
    weights = resolve(config["weights_path"])
    if sha(weights) != config["weights_sha256"]:
        raise ValueError("cached generic Kinetics weights failed SHA256 verification")
    model = r2plus1d_18(weights=None)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    model.fc = nn.Identity()
    model.eval().to(config["device"])
    return model


def embeddings(model, tensor, config):
    import torch
    from torch.nn import functional

    if tensor.shape != (config["batch_size"], 3, config["frames"], 112, 112):
        raise ValueError("each GPU batch must contain exactly eight preprocessed windows")
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        values = model(tensor.to(config["device"], non_blocking=True))
        values = functional.normalize(values.float(), dim=1)
    result = values.cpu().numpy()
    if result.shape != (8, 512) or not np.isfinite(result).all():
        raise ValueError("generic backbone emitted invalid embeddings")
    return result


def collect_inputs(config):
    media = resolve(config["media_root"])
    questions = {}
    for split in ("train", "test"):
        qa = pd.read_csv(resolve(config[split + "_csv"]), usecols=["qa_id", "source", "path"],
                         dtype=str, keep_default_na=False)
        questions[split] = sorted({feature_unit(value, training=split == "train")
                                   for value in qa.loc[qa.source.eq("HAU"), "path"]})
    discovered = discover_videos(media, tuple(config["modalities"]), ("Training/HAU", "Testing/"))
    locations = {(unit, modality): path for unit, modality, path in discovered}
    training = {unit for unit, modality, _ in discovered if unit.startswith("Training/HAU") and modality == "IR"}
    if len(training) != config["expected_training_units"]:
        raise ValueError("training HAU media count differs from the frozen contract")
    for split, expected in (("train", "expected_training_qa_units"), ("test", "expected_test_qa_units")):
        if len(questions[split]) != config[expected]:
            raise ValueError("official HAU QA scope differs from the frozen contract")
    if not set(questions["train"]) <= training:
        raise ValueError("official training HAU QA lacks IR media")
    units = sorted(training | set(questions["test"]))
    crops, cache_inputs = {}, []
    for name in config["person_window_caches"]:
        path = resolve(name)
        payload = json.loads(path.read_text())
        cache_inputs.append({"path": str(path.resolve()), "sha256": sha(path), "stats": payload.get("stats")})
        for unit, crop in payload["windows"].items():
            if unit not in units:
                continue
            if len(crop) != 4 or not (0 <= crop[0] < crop[2] <= 1 and 0 <= crop[1] < crop[3] <= 1):
                raise ValueError("person ROI cache contains an invalid normalized crop")
            if unit in crops and crops[unit] != crop:
                raise ValueError("person ROI caches disagree on a target unit")
            crops[unit] = crop
    if set(crops) != set(units):
        raise ValueError("person ROI caches do not cover every requested HAU unit")
    records, alignment = [], {}
    for unit in units:
        metadata = {}
        for modality in config["modalities"]:
            path = locations.get((unit, modality))
            meta = None if path is None else video_metadata(path)
            metadata[modality] = meta
            records.append({"unit": unit, "modality": modality, "available": path is not None,
                            "path": None if path is None else str(path.resolve()), "roi": crops[unit],
                            "sha256": None if path is None else sha(path),
                            "size_bytes": 0 if path is None else path.stat().st_size,
                            "metadata": meta, "timing": None if meta is None else window_timing(meta, config)})
        alignment[unit] = temporal_alignment(metadata["Depth"], metadata["IR"], config)
    return {"units": units, "records": records, "alignment": alignment,
            "qa_required_units": questions, "roi_cache_inputs": cache_inputs}


def archive_arrays(inputs, config):
    count, windows = len(inputs["records"]), config["windows"]
    total = count * windows
    arrays = {"X": np.zeros((total, 512), dtype=np.float32),
              "units": np.repeat([row["unit"] for row in inputs["records"]], windows),
              "modalities": np.repeat([row["modality"] for row in inputs["records"]], windows),
              "window_indices": np.tile(np.arange(windows, dtype=np.int16), count),
              "frame_indices": np.full((total, config["frames"]), -1, dtype=np.int32),
              "available": np.repeat([row["available"] for row in inputs["records"]], windows),
              "alignment_ok": np.repeat([inputs["alignment"][row["unit"]]["alignment_ok"] for row in inputs["records"]], windows),
              "frame_count": np.zeros(total, dtype=np.int32), "fps": np.zeros(total, dtype=np.float64),
              "duration_seconds": np.zeros(total, dtype=np.float64)}
    time_fields = ("starts", "stops", "centers", "starts_seconds", "stops_seconds", "centers_seconds")
    arrays.update({field: np.full(total, -1.0, dtype=np.float64) for field in time_fields})
    for index, row in enumerate(inputs["records"]):
        if not row["available"]:
            continue
        selected = slice(index * windows, (index + 1) * windows)
        for field in ("frame_count", "fps", "duration_seconds"):
            arrays[field][selected] = row["metadata"][field]
        for field in ("frame_indices", *time_fields):
            arrays[field][selected] = row["timing"][field]
    keys = set(zip(arrays["units"], arrays["modalities"], arrays["window_indices"], strict=True))
    if len(keys) != total:
        raise ValueError("feature rows do not have unique unit/modality/window keys")
    return arrays


def extract(args):
    config = json.loads(args.config.read_text())
    check_config(config)
    output = resolve(config["output_npz"])
    if args.output_dir.exists() or output.exists():
        raise FileExistsError("feature run and output NPZ must both be new")
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    source_hashes = {name: sha(ROOT / name) for name in SOURCE_FILES}
    fixed_inputs = {str(resolve(config[key]).resolve()): sha(resolve(config[key]))
                    for key in ("train_csv", "test_csv", "weights_path")}
    fixed_inputs[str(args.config.resolve())] = sha(args.config)
    fixed_inputs.update({str(resolve(path).resolve()): sha(resolve(path)) for path in config["person_window_caches"]})
    write_json(args.output_dir / "frozen_config.json", {"config": config, "frozen_utc": datetime.now(UTC).isoformat(),
               "source_hashes": source_hashes, "input_hashes": fixed_inputs, "before_any_extraction": True})
    for name in SOURCE_FILES:
        target = args.output_dir / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    model = setup_runtime(config)
    inputs = collect_inputs(config)
    write_json(args.output_dir / "input_manifest.json", inputs)
    arrays = archive_arrays(inputs, config)
    available = [(index, row) for index, row in enumerate(inputs["records"]) if row["available"]]
    import torch
    from torch.utils.data import DataLoader, Dataset

    class Videos(Dataset):
        def __len__(self):
            return len(available)

        def __getitem__(self, index):
            grid_index, row = available[index]
            return sample_preprocessed(row["path"], row["roi"], config), grid_index

    loader = DataLoader(Videos(), batch_size=1, shuffle=False, num_workers=1,
                        pin_memory=True, persistent_workers=True, prefetch_factor=1)
    extraction_started = time.perf_counter()
    for index, (clips, indices) in enumerate(loader, start=1):
        grid_index = int(indices.item())
        arrays["X"][grid_index * 8:(grid_index + 1) * 8] = embeddings(model, clips[0], config)
        if index % 100 == 0 or index == len(available):
            print(f"generic_videos={index}/{len(available)} windows={index * 8}", flush=True)
    extraction_seconds = time.perf_counter() - extraction_started
    valid = arrays["available"]
    if not np.allclose(np.linalg.norm(arrays["X"][valid], axis=1), 1, atol=1e-6):
        raise ValueError("available generic embeddings are not L2 normalized")
    if np.any(arrays["X"][~valid]):
        raise ValueError("missing feature rows must remain masked zeros")
    for row in inputs["records"]:
        if row["available"] and sha(row["path"]) != row["sha256"]:
            raise ValueError("raw video changed during extraction")
    for path, digest in fixed_inputs.items():
        if sha(path) != digest:
            raise ValueError("fixed extraction input changed")
    for path, digest in source_hashes.items():
        if sha(ROOT / path) != digest or sha(args.output_dir / "source_snapshot" / path) != digest:
            raise ValueError("extraction source changed")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays, backbone=np.asarray(config["backbone"]), weights=np.asarray(config["weights"]),
                        frames=np.asarray(16), windows=np.asarray(8), window_seconds=np.asarray(2.5))
    missing = [{"unit": row["unit"], "modality": row["modality"]} for row in inputs["records"] if not row["available"]]
    misaligned = {unit: audit for unit, audit in inputs["alignment"].items() if not audit["alignment_ok"]}
    report = {"status": "generic_windows_extracted", "created_utc": datetime.now(UTC).isoformat(),
              "output": str(output.resolve()), "sha256": sha(output), "shape": list(arrays["X"].shape),
              "available_rows": int(valid.sum()), "aligned_rows": int(arrays["alignment_ok"].sum()),
              "units": len(inputs["units"]), "modality_videos": len(available), "missing_modalities": missing,
              "misaligned_or_missing_units": misaligned,
              "qa_required_units": inputs["qa_required_units"],
              "qa_aligned_units": {split: sum(inputs["alignment"][unit]["alignment_ok"] for unit in units)
                                   for split, units in inputs["qa_required_units"].items()},
              "extraction_seconds": extraction_seconds, "total_seconds": time.perf_counter() - started,
              "source_hashes": source_hashes, "input_hashes": fixed_inputs,
              "input_manifest_sha256": sha(args.output_dir / "input_manifest.json"),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "environment": {"torch": torch.__version__, "torchvision": __import__("torchvision").__version__,
                              "opencv": __import__("cv2").__version__, "numpy": np.__version__},
              "limitations": ["Alignment audits relative video timing, not hardware clock synchronization or spatial calibration.",
                              "Features have no labels or predictions. Any supervised HARn-to-HAU transfer still requires subject exclusion."]}
    write_json(args.output_dir / "report.json", report)
    print(json.dumps({key: report[key] for key in ("status", "shape", "available_rows", "aligned_rows", "qa_aligned_units", "extraction_seconds")}, indent=2))


def replay(args):
    if args.output_dir.exists():
        raise FileExistsError("replay output must be new")
    frozen = json.loads((args.run_dir / "frozen_config.json").read_text())
    config = frozen["config"]
    check_config(config)
    report = json.loads((args.run_dir / "report.json").read_text())
    manifest_path = args.run_dir / "input_manifest.json"
    if sha(manifest_path) != report["input_manifest_sha256"] or sha(report["output"]) != report["sha256"]:
        raise ValueError("feature artifact or input manifest failed SHA256 verification")
    for name, digest in frozen["source_hashes"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("replay source differs from the frozen extraction code")
    inputs = json.loads(manifest_path.read_text())
    selected_units = [inputs["qa_required_units"][split][0] for split in ("train", "test")]
    model = setup_runtime(config)
    checked, replay_values = [], []
    with np.load(report["output"], allow_pickle=False) as archive:
        for grid_index, row in enumerate(inputs["records"]):
            if row["unit"] not in selected_units or not row["available"]:
                continue
            if sha(row["path"]) != row["sha256"]:
                raise ValueError("replay raw video differs from extraction")
            metadata = video_metadata(Path(row["path"]))
            if metadata != row["metadata"]:
                raise ValueError("replay video metadata differs")
            timing = window_timing(metadata, config)
            indices = np.arange(grid_index * 8, (grid_index + 1) * 8)
            if not np.all(archive["units"][indices] == row["unit"]) or not np.all(archive["modalities"][indices] == row["modality"]):
                raise ValueError("replay unit/modality key alignment failed")
            if not np.array_equal(archive["window_indices"][indices], np.arange(8)):
                raise ValueError("replay window IDs are misaligned")
            for field, value in timing.items():
                if not np.array_equal(archive[field][indices], value):
                    raise ValueError("replayed physical sampling indices/times differ")
            actual = embeddings(model, sample_preprocessed(row["path"], row["roi"], config), config)
            expected = archive["X"][indices]
            exact = np.array_equal(actual, expected)
            close = np.allclose(actual, expected, atol=config["replay_atol"], rtol=config["replay_rtol"])
            checked.append({"unit": row["unit"], "modality": row["modality"], "windows": 8,
                            "feature_values": int(actual.size), "maximum_absolute_difference": float(np.abs(actual - expected).max()),
                            "byte_values_exact": exact, "within_frozen_tolerance": bool(close),
                            "raw_video_sha256": row["sha256"], "timing_exact": True})
            replay_values.append(actual)
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(args.output_dir / "replayed_features.npz", X=np.concatenate(replay_values),
                        units=np.repeat([row["unit"] for row in checked], 8),
                        modalities=np.repeat([row["modality"] for row in checked], 8),
                        window_indices=np.tile(np.arange(8), len(checked)))
    complete = len(checked) == 4 and all(row["within_frozen_tolerance"] for row in checked)
    write_json(args.output_dir / "report.json", {"status": "raw_window_replay_verified" if complete else "replay_failed",
               "selected_units": selected_units, "selection_rule": "lexicographically first official training-QA unit and first test-QA unit",
               "checks": checked, "feature_source_sha256": report["sha256"], "source_hashes": frozen["source_hashes"],
               "replay_sha256": sha(args.output_dir / "replayed_features.npz")})
    if not complete:
        raise ValueError("two-clip raw replay failed its predeclared tolerance or coverage")
    print(json.dumps({"status": "raw_window_replay_verified", "videos": len(checked),
                      "feature_values": sum(row["feature_values"] for row in checked),
                      "all_exact": all(row["byte_values_exact"] for row in checked)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    extraction = sub.add_parser("extract")
    extraction.add_argument("--config", type=Path, required=True)
    repeating = sub.add_parser("replay")
    repeating.add_argument("--run-dir", type=Path, required=True)
    for command in (extraction, repeating):
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    extract(args) if args.command == "extract" else replay(args)


if __name__ == "__main__":
    main()
