#!/usr/bin/env python3
"""Extract time-resolved action logits from multi-action HAU videos.

This applies the public K-KUNO single-action expert to fixed-duration windows.
The expert was trained on the separate 40-class HAR corpus, whose class order
matches the official HAU action vocabulary. No HAU labels are used here.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from extract_kuno_transfer_logits import (
    KINETICS_MEAN,
    KINETICS_STD,
    build_r2plus1d34,
    dequantize_state,
    detect_windows,
    discover_units,
    sha256_file,
)

from cuhkx.video import sample_video_windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefixes", nargs="+", default=["Training/HAU", "Testing/"])
    parser.add_argument("--unit-regex", help="optional regular expression applied to full unit IDs")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--segments", type=int, default=8)
    parser.add_argument("--window-seconds", type=float, default=2.5)
    parser.add_argument("--detection-frames", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2, help="temporal windows per GPU batch")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--detector-device", default="0")
    parser.add_argument("--windows-json", type=Path)
    parser.add_argument("--disable-person-crop", action="store_true")
    return parser.parse_args()


def load_or_detect_windows(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    detector_path: Path,
) -> tuple[dict[str, tuple[float, float, float, float]], dict[str, int]]:
    expected_units = [str(record["unit"]) for record in records]
    if args.windows_json:
        payload = json.loads(args.windows_json.read_text(encoding="utf-8"))
        missing = sorted(set(expected_units) - set(payload["windows"]))
        if missing:
            raise ValueError(f"person-window cache is missing {len(missing)} requested units")
        windows = {unit: tuple(payload["windows"][unit]) for unit in expected_units}
        detected = sum(window != (0.0, 0.0, 1.0, 1.0) for window in windows.values())
        return windows, {
            "clips": len(records),
            "detected": detected,
            "fallback": len(records) - detected,
        }
    if args.disable_person_crop:
        return (
            {unit: (0.0, 0.0, 1.0, 1.0) for unit in expected_units},
            {"clips": len(records), "detected": 0, "fallback": len(records)},
        )
    windows, stats = detect_windows(
        records, detector_path, args.detection_frames, args.detector_device
    )
    cache_path = args.output.with_suffix(".windows.json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {"units": expected_units, "windows": windows, "stats": stats},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return windows, stats


def prepare_clip(record: dict[str, Any], window, args: argparse.Namespace):
    import cv2
    import torch

    infrared = sample_video_windows(
        record["ir"], args.segments, args.frames, args.window_seconds
    )
    if record["depth_color"] is None:
        depth_color = np.zeros_like(infrared)
    else:
        depth_color = sample_video_windows(
            record["depth_color"], args.segments, args.frames, args.window_seconds
        )
    x0, y0, x1, y1 = window
    output = np.zeros((args.segments, args.frames, 4, 128, 128), dtype=np.uint8)
    for segment in range(args.segments):
        for frame in range(args.frames):
            for source, begin, channels in ((depth_color, 0, 3), (infrared, 3, 1)):
                height, width = source[segment, frame].shape[:2]
                crop = source[
                    segment,
                    frame,
                    round(y0 * height) : round(y1 * height),
                    round(x0 * width) : round(x1 * width),
                ]
                if crop.size == 0:
                    crop = source[segment, frame]
                image = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_LINEAR)
                if channels == 1:
                    output[segment, frame, begin] = image.mean(axis=2).astype(np.uint8)
                else:
                    output[segment, frame, begin : begin + channels] = image.transpose(2, 0, 1)
    tensor = torch.from_numpy(output).float().div_(255.0)
    mean = torch.tensor((*KINETICS_MEAN, sum(KINETICS_MEAN) / 3.0)).view(1, 1, 4, 1, 1)
    std = torch.tensor((*KINETICS_STD, sum(KINETICS_STD) / 3.0)).view(1, 1, 4, 1, 1)
    return tensor.sub_(mean).div_(std)


def main() -> int:
    args = parse_args()
    try:
        import cv2
        import torch
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise RuntimeError("install the project visual dependencies first") from error

    if args.frames != 16:
        raise ValueError("the frozen K-KUNO expert was trained with exactly 16 frames")
    if args.segments < 2:
        raise ValueError("temporal inference requires at least two segments")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    model_path = args.model_dir / "ensemble_packed.pt"
    architecture_path = args.model_dir / "ig65m_models.py"
    detector_path = args.model_dir / "yolo11n.pt"
    for path in (model_path, architecture_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    records = discover_units(args.media_root, tuple(args.prefixes))
    if args.unit_regex:
        pattern = re.compile(args.unit_regex)
        records = [record for record in records if pattern.search(str(record["unit"]))]
    if not records:
        raise ValueError("no matching IR videos found")
    units = [str(record["unit"]) for record in records]
    windows, detection_stats = load_or_detect_windows(args, records, detector_path)

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != "kuno-yolo-r2p1d-packed-ensemble/v1":
        raise ValueError("unexpected public checkpoint schema")
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    models = []
    for fold, packed_state in zip(
        checkpoint["folds"], checkpoint["models_packed"], strict=True
    ):
        model = build_r2plus1d34(architecture_path).to(device)
        state = dequantize_state(packed_state)
        model.load_state_dict(state)
        del state
        model.eval()
        models.append(model)
        print(f"loaded_fold={fold}", flush=True)

    logits = np.zeros(
        (len(models), len(records), args.segments, 40), dtype=np.float32
    )
    started = time.perf_counter()
    with torch.inference_mode():
        for record_index, record in enumerate(records, start=1):
            clip = prepare_clip(record, windows[str(record["unit"])], args)
            for fold_index, model in enumerate(models):
                for begin in range(0, args.segments, args.batch_size):
                    batch = clip[begin : begin + args.batch_size].to(device, non_blocking=True)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=device.type == "cuda",
                    ):
                        output = model(batch) + model(torch.flip(batch, dims=(-1,)))
                    logits[fold_index, record_index - 1, begin : begin + len(batch)] = (
                        output.float().cpu().numpy()
                    )
            if record_index % 10 == 0 or record_index == len(records):
                elapsed = time.perf_counter() - started
                print(
                    f"clips={record_index}/{len(records)} elapsed={elapsed:.1f}s",
                    flush=True,
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        logits=logits,
        units=np.asarray(units),
        folds=np.asarray(checkpoint["folds"]),
        weights=np.asarray(checkpoint["weights"], dtype=np.float32),
        classes=np.arange(40),
        segments=np.asarray(args.segments),
        frames=np.asarray(args.frames),
        window_seconds=np.asarray(args.window_seconds),
    )
    report = {
        "output": str(args.output.resolve()),
        "sha256": sha256_file(args.output),
        "shape": list(logits.shape),
        "units": len(records),
        "missing_depth_color": sum(record["depth_color"] is None for record in records),
        "detection": detection_stats,
        "source_model_sha256": sha256_file(model_path),
        "source_architecture_sha256": sha256_file(architecture_path),
        "source_detector_sha256": sha256_file(detector_path) if detector_path.is_file() else None,
        "source_parent_validation_accuracy": checkpoint["parent_val_acc"],
        "frames": args.frames,
        "segments": args.segments,
        "window_seconds": args.window_seconds,
        "horizontal_flip_tta": True,
        "runtime_seconds": time.perf_counter() - started,
        "torch": torch.__version__,
        "opencv": cv2.__version__,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
