#!/usr/bin/env python3
"""Freeze/extract generic R34 whole-clip and HAU-window middle activations.

prepare reads only QA source/path metadata or explicit unit declarations. extract
uses the frozen raw files, pinned image-derived ROIs and the original public
IG65M/Kinetics weights. No CUHK-X supervised checkpoint is an input.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from cuhkx.generic_video_windows import video_metadata
from cuhkx.metadata import feature_unit
from cuhkx.r34_video import (
    KIND,
    MID_SHAPE,
    MODALITIES,
    SPEC,
    WEIGHTS_BYTES,
    WEIGHTS_SHA,
    alignment,
    frozen_core,
    metadata_arrays,
    pretrained_model,
    sample_preprocessed,
    sha256_file,
    timing,
)

SOURCES = ["scripts/extract_r34_midfeatures.py", "src/cuhkx/r34_video.py",
           "src/cuhkx/generic_video_windows.py", "src/cuhkx/video.py", "src/cuhkx/metadata.py"]


def write_json(path, value):
    def default(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=default) + "\n")
    temporary.replace(path)


def file_entry(path, expected=None):
    path = Path(path).resolve()
    digest = sha256_file(path)
    if expected is not None and digest != expected:
        raise ValueError(f"pinned SHA256 differs: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def verify(records):
    for record in records:
        path = Path(record["path"])
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"frozen R34 input changed: {path}")


def resolve(path):
    return (ROOT / path).resolve()


def validate_config(config):
    expected = {"schema_version": 1, "kind": KIND, "weights_sha256": WEIGHTS_SHA,
                "weights_bytes": WEIGHTS_BYTES, "modalities": list(MODALITIES), "frames": 32,
                "hau_windows": 8, "hau_window_seconds": 2.5, "alignment_tolerance_frames": 1.0,
                "batch_size": 2, "workers": 2, "torch_threads": 2, "opencv_threads": 1,
                "device": "cuda", "autocast": "float16", "cudnn_benchmark": False,
                "cudnn_deterministic": True, "matmul_precision": "high", "storage": "float16",
                "invalid_metadata_policy": "mask_with_recorded_error",
                "missing_modality_policy": "keep_zero_row_no_Depth_substitution"}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("R34 extraction parameters differ from the predeclared recipe")


def select_units(*, media_root, sources, train_qa=None, test_qa=None,
                 unit_list=None, discover_training=(), maximum_units=None):
    """Return opaque unit keys and task sources; never parse an action label."""
    selected = {}

    def add(unit, source):
        unit = str(unit).replace("\\", "/").strip("/")
        parts = unit.split("/")
        if (source not in ("HARn", "HAU") or len(parts) < 3 or ".." in parts
                or parts[0] not in ("Training", "Testing")
                or (parts[0] == "Training" and parts[1] != source)):
            raise ValueError("unit keys require an explicit consistent Training/Testing and HARn/HAU source")
        if source not in sources:
            return
        if unit in selected and selected[unit] != source:
            raise ValueError("the same unit was assigned conflicting sources")
        selected[unit] = source

    for path, training in ((train_qa, True), (test_qa, False)):
        if path is None:
            continue
        frame = pd.read_csv(path, usecols=["source", "path"], dtype=str, keep_default_na=False)
        for source, raw in frame[["source", "path"]].itertuples(index=False, name=None):
            add(feature_unit(raw, training=training), source)
    if unit_list is not None:
        declarations = json.loads(Path(unit_list).read_text())
        if not isinstance(declarations, list):
            raise ValueError("unit-list must be a JSON list of unit/source objects")
        for item in declarations:
            if not isinstance(item, dict) or set(item) != {"unit", "source"}:
                raise ValueError("unit-list entries must contain only unit and source")
            add(item["unit"], item["source"])
    for source in discover_training:
        if source not in ("HARn", "HAU"):
            raise ValueError("only Training/HARn and Training/HAU can be discovered")
        base = media_root / "Training" / source
        for modality in MODALITIES:
            for path in sorted(base.rglob(f"{modality}/{modality}.mp4")):
                # Unit identity comes from the declared layout, not a resolved link target.
                add(str(path.parents[1].relative_to(media_root)), source)
    result = [{"unit": unit, "source": selected[unit]} for unit in sorted(selected)]
    if maximum_units is not None:
        if maximum_units <= 0:
            raise ValueError("maximum-units must be positive")
        result = result[:maximum_units]
    if not result:
        raise ValueError("no raw units were selected")
    return result


def read_rois(paths, digests, units):
    if len(paths) != len(digests) or not paths:
        raise ValueError("each declared ROI cache requires its matching --roi-sha256")
    wanted = set(units)
    windows, references = {}, []
    for path, digest in zip(paths, digests, strict=True):
        references.append(file_entry(path, digest))
        payload = json.loads(Path(path).read_text())
        for unit, crop in payload["windows"].items():
            if unit not in wanted:
                continue
            if (len(crop) != 4 or not all(np.isfinite(crop))
                    or not (0 <= crop[0] < crop[2] <= 1 and 0 <= crop[1] < crop[3] <= 1)):
                raise ValueError("invalid normalized person ROI")
            if unit in windows and windows[unit] != crop:
                raise ValueError("pinned ROI caches disagree for a requested unit")
            windows[unit] = crop
    return windows, references


def collect_inputs(media_root, units, crops):
    records, alignments, cursor = [], {}, 0
    for declaration in units:
        unit, source = declaration["unit"], declaration["source"]
        metadata = {}
        for modality in MODALITIES:
            path = media_root / unit / modality / f"{modality}.mp4"
            exists = path.is_file()
            fingerprint = file_entry(path) if exists else None
            status, error, meta = "missing", None, None
            if exists:
                try:
                    meta = video_metadata(path)
                    status = "available"
                except ValueError as failure:
                    status, error = "invalid_metadata", str(failure)
            metadata[modality] = meta
            if meta is not None and unit not in crops:
                raise ValueError(f"available video lacks a pinned person ROI: {unit}")
            number = 1 if source == "HARn" else 8
            records.append({"unit": unit, "source": source, "modality": modality,
                            "available": meta is not None, "raw_status": status, "error": error,
                            "path": str(path.resolve()) if exists else None,
                            "raw": fingerprint, "metadata": meta, "roi": crops.get(unit),
                            "timing": timing(meta, source) if meta is not None else None,
                            "row_start": cursor, "row_count": number})
            cursor += number
        alignments[unit] = alignment(metadata["Depth_Color"], metadata["IR"], source)
    return {"units": units, "records": records, "alignment": alignments,
            "row_count": cursor, "numeric_action_path_features": False}


def prepare(args):
    import cv2

    if args.output_dir.exists():
        raise FileExistsError("R34 extraction plan needs a new output directory")
    config = json.loads(args.config.read_text())
    validate_config(config)
    media_root = args.media_root.resolve()
    units = select_units(media_root=media_root, sources=args.sources, train_qa=args.train_qa,
                         test_qa=args.test_qa, unit_list=args.unit_list,
                         discover_training=args.discover_training or (), maximum_units=args.maximum_units)
    crops, roi_inputs = read_rois(args.roi_cache, args.roi_sha256, [item["unit"] for item in units])
    inputs = collect_inputs(media_root, units, crops)
    weights = file_entry(resolve(config["weights"]), WEIGHTS_SHA)
    if weights["bytes"] != WEIGHTS_BYTES:
        raise ValueError("public R34 weight byte count differs")
    dependencies = [weights, *roi_inputs, file_entry(args.config)]
    dependencies += [file_entry(path) for path in (args.train_qa, args.test_qa, args.unit_list) if path is not None]
    dependencies += [file_entry(resolve(path)) for path in config["provenance_files"]]
    architecture = config["architecture_reference"]
    dependencies.append(file_entry(resolve(architecture["path"]), architecture["sha256"]))
    dependencies += [file_entry(path) for path in args.provenance_file]
    sources = [file_entry(ROOT / name) for name in SOURCES]
    metadata = metadata_arrays(inputs)
    out = args.output_dir.resolve()
    out.mkdir(parents=True)
    write_json(out / "input_manifest.json", inputs)
    np.savez_compressed(out / "metadata.npz", **metadata)
    frozen = {"kind": KIND, "created_utc": datetime.now(UTC).isoformat(), "spec": SPEC, "config": config,
              "dependencies": dependencies, "source_manifest": sources, "roi_inputs": roi_inputs,
              "qa_columns_read": ["source", "path"], "labels_used": False, "source_selection": args.sources,
              "discover_training": args.discover_training, "maximum_units": args.maximum_units,
              "scope": "bounded_raw_smoke" if args.maximum_units else "declared_unit_scope",
              "input_manifest_sha256": sha256_file(out / "input_manifest.json"),
              "metadata_sha256": sha256_file(out / "metadata.npz"),
              "shape": [inputs["row_count"], *MID_SHAPE], "dtype": "float16",
              "estimated_feature_bytes": inputs["row_count"] * int(np.prod(MID_SHAPE)) * 2,
              "versions": {**{name: importlib.metadata.version(name) for name in
                              ("numpy", "pandas", "torch", "torchvision")},
                           "opencv_runtime": cv2.__version__}}
    for record in sources:
        path = Path(record["path"])
        destination = out / "source_snapshot" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        if sha256_file(destination) != record["sha256"]:
            raise ValueError("R34 extraction source changed while freezing")
    write_json(out / "frozen_config.json", frozen)
    (out / "frozen_config.sha256").write_text(sha256_file(out / "frozen_config.json") + "\n")
    write_json(out / "state.json", {"status": "prepared", "rows": inputs["row_count"], "completed_rows": 0})
    print(json.dumps({"status": "prepared", "units": len(units), "rows": inputs["row_count"],
                      "feature_bytes": frozen["estimated_feature_bytes"],
                      "available_views": sum(record["available"] for record in inputs["records"]),
                      "output_dir": str(out)}), flush=True)


def load_plan(run):
    path = run / "frozen_config.json"
    if sha256_file(path) != (run / "frozen_config.sha256").read_text().strip():
        raise ValueError("R34 frozen extraction plan changed")
    frozen = json.loads(path.read_text())
    if frozen["kind"] != KIND or frozen["spec"] != SPEC:
        raise ValueError("R34 frozen feature specification differs")
    validate_config(frozen["config"])
    verify([*frozen["dependencies"], *frozen["source_manifest"]])
    for name, key in (("input_manifest.json", "input_manifest_sha256"), ("metadata.npz", "metadata_sha256")):
        if sha256_file(run / name) != frozen[key]:
            raise ValueError("R34 frozen raw window/metadata manifest changed")
    inputs = json.loads((run / "input_manifest.json").read_text())
    verify([record["raw"] for record in inputs["records"] if record["raw"] is not None])
    return frozen, inputs


class RawViews:
    def __init__(self, records, pending):
        self.records = records
        self.pending = pending

    def __len__(self):
        return len(self.pending)

    def __getitem__(self, index):
        record_index = self.pending[index]
        return record_index, sample_preprocessed(self.records[record_index])


def loader_worker(_index):
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def extract(args):
    import cv2
    import torch
    from torch.utils.data import DataLoader

    run = args.run_dir.resolve()
    if (run / "manifest.json").exists():
        raise FileExistsError("R34 extraction is already complete; load its manifest instead of extracting again")
    frozen, inputs = load_plan(run)
    config = frozen["config"]
    if not torch.cuda.is_available():
        raise RuntimeError("the declared R34 raw extraction requires CUDA")
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:config["workers"]])
    cv2.setNumThreads(config["opencv_threads"])
    torch.set_num_threads(config["torch_threads"])
    torch.backends.cudnn.benchmark = config["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = config["cudnn_deterministic"]
    torch.set_float32_matmul_precision(config["matmul_precision"])
    shape = tuple(frozen["shape"])
    progress_path, feature_path = run / "progress.npz", run / "features.npy"
    if args.resume:
        with np.load(progress_path, allow_pickle=False) as progress:
            written = progress["written"].copy()
            if str(progress["frozen_config_sha256"]) != sha256_file(run / "frozen_config.json"):
                raise ValueError("R34 partial extraction belongs to a different frozen plan")
        cache = np.load(feature_path, mmap_mode="r+", allow_pickle=False)
        if written.shape != (shape[0],) or written.dtype != np.bool_:
            raise ValueError("invalid R34 resume progress mask")
    else:
        if feature_path.exists() or progress_path.exists():
            raise FileExistsError("partial R34 outputs exist; use extract --resume")
        cache = np.lib.format.open_memmap(feature_path, mode="w+", dtype=np.float16, shape=shape)
        written = np.zeros(shape[0], dtype=bool)
        for record in inputs["records"]:
            if not record["available"]:
                selected = slice(record["row_start"], record["row_start"] + record["row_count"])
                cache[selected] = 0
                written[selected] = True
    if cache.shape != shape or cache.dtype != np.float16:
        raise ValueError("partial R34 feature shape or storage changed")
    state = {"status": "running", "started_utc": datetime.now(UTC).isoformat(), "rows": shape[0],
             "device": "cuda", "resume": args.resume, "cpu_affinity": sorted(os.sched_getaffinity(0))}

    def save_progress():
        cache.flush()
        temporary = progress_path.with_name("progress.tmp.npz")
        np.savez_compressed(temporary, written=written,
                            frozen_config_sha256=np.asarray(sha256_file(run / "frozen_config.json")))
        temporary.replace(progress_path)
        state["completed_rows"] = int(written.sum())
        write_json(run / "state.json", state)

    save_progress()
    pending = [index for index, record in enumerate(inputs["records"]) if record["available"]
               and not written[record["row_start"]:record["row_start"] + record["row_count"]].all()]
    model = pretrained_model(resolve(config["weights"]))
    core = frozen_core(model).to(config["device"])
    loader = DataLoader(RawViews(inputs["records"], pending), batch_size=None, shuffle=False,
                        num_workers=config["workers"], pin_memory=True, persistent_workers=True,
                        worker_init_fn=loader_worker, prefetch_factor=1)
    tensors, row_indices, steps = [], [], 0
    started = time.monotonic()

    def forward_pending():
        nonlocal steps
        if not tensors:
            return
        batch = torch.stack(tensors).to(config["device"], non_blocking=True)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            values = core(batch).half().cpu().numpy()
        if values.shape != (len(row_indices), *MID_SHAPE) or not np.isfinite(values).all():
            raise ValueError("R34 backbone returned invalid intermediate features")
        if written[row_indices].any():
            raise ValueError("duplicate R34 feature row write")
        cache[row_indices] = values
        written[row_indices] = True
        tensors.clear()
        row_indices.clear()
        steps += 1

    try:
        for position, (record_index, tensor) in enumerate(loader, start=1):
            record = inputs["records"][int(record_index)]
            if tuple(tensor.shape) != (record["row_count"], 3, 32, 112, 112):
                raise ValueError("decoded video does not match its declared window layout")
            for window in range(record["row_count"]):
                row = record["row_start"] + window
                if written[row]:
                    continue
                tensors.append(tensor[window])
                row_indices.append(row)
                if len(tensors) == config["batch_size"]:
                    forward_pending()
            if position % 32 == 0 or position == len(pending):
                forward_pending()
                save_progress()
                print(f"r34_rows={int(written.sum())}/{len(written)} views={position}/{len(pending)} seconds={time.monotonic()-started:.1f}", flush=True)
        forward_pending()
        save_progress()
        if not written.all():
            raise ValueError("R34 extraction did not write every declared row")
        load_plan(run)
        manifest = {"kind": KIND, "schema_version": 1, "spec": SPEC,
                    "units": [record["unit"] for record in inputs["units"]],
                    "unit_sources": {record["unit"]: record["source"] for record in inputs["units"]},
                    "features": {"path": "features.npy", "sha256": sha256_file(feature_path)},
                    "metadata": {"path": "metadata.npz", "sha256": sha256_file(run / "metadata.npz")},
                    "frozen_config_sha256": sha256_file(run / "frozen_config.json"),
                    "input_manifest_sha256": sha256_file(run / "input_manifest.json"),
                    "shape": list(shape), "labels_used": False, "new_neural_fits": 0}
        write_json(run / "manifest.json", manifest)
        state.update(status="complete", completed_utc=datetime.now(UTC).isoformat(),
                     elapsed_seconds=time.monotonic() - started, forward_batches_this_execution=steps,
                     manifest_sha256=sha256_file(run / "manifest.json"),
                     sha256=sha256_file(run / "manifest.json"), feature_bytes=cache.nbytes)
        write_json(run / "state.json", state)
        write_json(run / "report.json", state)
        print(json.dumps(state, indent=2), flush=True)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_progress()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    planning = sub.add_parser("prepare")
    planning.add_argument("--config", type=Path, default=ROOT / "configs/r34_midfeatures_v1.json")
    planning.add_argument("--media-root", type=Path, required=True)
    planning.add_argument("--output-dir", type=Path, required=True)
    planning.add_argument("--sources", choices=("HARn", "HAU"), nargs="+", default=["HARn", "HAU"])
    planning.add_argument("--train-qa", type=Path)
    planning.add_argument("--test-qa", type=Path)
    planning.add_argument("--unit-list", type=Path, help="JSON list with exactly unit/source fields")
    planning.add_argument("--discover-training", choices=("HARn", "HAU"), nargs="+")
    planning.add_argument("--maximum-units", type=int, help="bounded raw smoke only; recorded in the manifest")
    planning.add_argument("--roi-cache", type=Path, action="append", default=[])
    planning.add_argument("--roi-sha256", action="append", default=[])
    planning.add_argument("--provenance-file", type=Path, action="append", default=[])
    extraction = sub.add_parser("extract")
    extraction.add_argument("--run-dir", type=Path, required=True)
    extraction.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        extract(args)


if __name__ == "__main__":
    main()
