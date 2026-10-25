#!/usr/bin/env python3
"""Run the public K-KUNO Small-Track video expert on Large-Track MP4 clips.

The public model and reference inference are Apache-2.0; IG65M code is MIT and
the optional YOLO11n detector is AGPL-3.0. Third-party assets are loaded from
an external directory and are never copied into this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuhkx.video import media_unit, sample_video_frames

KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD = (0.22803, 0.22145, 0.216989)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefixes", nargs="+", default=["Training/HARn", "Testing/"])
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--detection-frames", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--detector-device", default="cpu")
    parser.add_argument("--disable-person-crop", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_units(media_root: Path, prefixes: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ir_path in media_root.rglob("IR/IR.mp4"):
        unit = media_unit(ir_path, media_root)
        if prefixes and not unit.startswith(prefixes):
            continue
        depth_color_path = ir_path.parents[1] / "Depth_Color" / "Depth_Color.mp4"
        rows.append(
            {
                "unit": unit,
                "ir": ir_path,
                "depth_color": depth_color_path if depth_color_path.is_file() else None,
            }
        )
    rows.sort(key=lambda row: str(row["unit"]))
    if len({str(row["unit"]) for row in rows}) != len(rows):
        raise ValueError("duplicate video units discovered")
    return rows


def square_window(
    boxes: list[tuple[float, float, float, float]],
    width: int,
    height: int,
    *,
    margin: float = 1.4,
    minimum_side_fraction: float = 0.35,
) -> tuple[float, float, float, float]:
    if not boxes:
        return 0.0, 0.0, 1.0, 1.0
    values = np.asarray(boxes, dtype=np.float64)
    x0, y0 = values[:, :2].min(axis=0)
    x1, y1 = values[:, 2:].max(axis=0)
    center_x, center_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = max(x1 - x0, y1 - y0) * margin
    side = max(side, minimum_side_fraction * max(width, height))
    return (
        max(center_x - side / 2.0, 0.0) / width,
        max(center_y - side / 2.0, 0.0) / height,
        min(center_x + side / 2.0, float(width)) / width,
        min(center_y + side / 2.0, float(height)) / height,
    )


def detect_windows(
    records: list[dict[str, Any]],
    weights: Path,
    probes: int,
    device: str,
) -> tuple[dict[str, tuple[float, float, float, float]], dict[str, int]]:
    try:
        from ultralytics import YOLO
    except ImportError as error:  # pragma: no cover - optional external dependency
        raise RuntimeError("install ultralytics to reproduce K-KUNO person crops") from error

    model = YOLO(str(weights))
    windows: dict[str, tuple[float, float, float, float]] = {}
    detected = 0
    for index, record in enumerate(records, start=1):
        frames = sample_video_frames(record["ir"], probes)
        # Ultralytics 8.4 treats a rank-4 ndarray as one malformed image;
        # provide an explicit image list to preserve the eight-frame batch.
        results = model.predict(
            list(frames), classes=[0], conf=0.25, verbose=False, device=device
        )
        boxes = []
        height, width = frames[0].shape[:2]
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            best = int(result.boxes.conf.argmax().item())
            boxes.append(tuple(float(value) for value in result.boxes.xyxy[best].cpu().tolist()))
        windows[str(record["unit"])] = square_window(boxes, width, height)
        detected += bool(boxes)
        if index % 100 == 0 or index == len(records):
            print(f"person_windows={index}/{len(records)} detected={detected}", flush=True)
    return windows, {"clips": len(records), "detected": detected, "fallback": len(records) - detected}


def unpack_signed(packed, shape: tuple[int, ...], bits: int):
    import torch

    count = math.prod(shape)
    starts = torch.arange(count, dtype=torch.int64) * bits
    codes = torch.zeros(count, dtype=torch.int16)
    source = packed.to(torch.int16)
    for bit in range(bits):
        positions = starts + bit
        byte_indices = positions >> 3
        shifts = positions & 7
        values = torch.bitwise_and(source[byte_indices] >> shifts, 1)
        codes |= values << bit
    sign, modulus = 1 << (bits - 1), 1 << bits
    signed = torch.where(codes >= sign, codes - modulus, codes)
    return signed.to(torch.int8).reshape(shape)


def dequantize_state(state: dict[str, object]) -> dict[str, object]:
    output = {}
    for key, value in state.items():
        if isinstance(value, dict):
            shape = tuple(int(item) for item in value["shape"])
            quantized = unpack_signed(value["packed"], shape, int(value["bits"]))
            output[key] = quantized.float() * value["scale"].float()
        else:
            output[key] = value.float() if value.is_floating_point() else value
    return output


def load_architecture(source: Path):
    spec = importlib.util.spec_from_file_location("cuhkx_external_ig65m", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load architecture source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_r2plus1d34(architecture_path: Path):
    """Build the exact four-channel architecture used by the public expert."""

    import torch
    from torch import nn

    architecture = load_architecture(architecture_path)

    def adapt_input_conv(conv, channels: int):
        replacement = type(conv)(
            channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            bias=conv.bias is not None,
        )
        with torch.no_grad():
            replacement.weight[:, :3] = conv.weight
            replacement.weight[:, 3:] = conv.weight.mean(dim=1, keepdim=True).expand(
                -1, channels - 3, -1, -1, -1
            )
            if conv.bias is not None:
                replacement.bias.copy_(conv.bias)
        return replacement

    class R2Plus1D34(nn.Module):
        def __init__(self):
            super().__init__()
            network = architecture.r2plus1d_34_32_kinetics(num_classes=400, pretrained=False)
            network.stem[0] = adapt_input_conv(network.stem[0], 4)
            features = network.fc.in_features
            network.fc = nn.Identity()
            self.encoder = network
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(features, 40))

        def forward(self, inputs):
            return self.head(self.encoder(inputs.permute(0, 2, 1, 3, 4)))

    return R2Plus1D34()


def main() -> int:
    args = parse_args()
    try:
        import cv2
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise RuntimeError("install the project visual dependencies first") from error

    if args.frames != 16:
        raise ValueError("the frozen K-KUNO expert was trained with exactly 16 frames")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    model_path = args.model_dir / "ensemble_packed.pt"
    architecture_path = args.model_dir / "ig65m_models.py"
    detector_path = args.model_dir / "yolo11n.pt"
    for path in (model_path, architecture_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    records = discover_units(args.media_root, tuple(args.prefixes))
    if not records:
        raise ValueError("no matching IR videos found")

    window_path = args.output.with_suffix(".windows.json")
    expected_units = [str(record["unit"]) for record in records]
    detection_stats: dict[str, int]
    if window_path.is_file():
        window_payload = json.loads(window_path.read_text(encoding="utf-8"))
        if window_payload.get("units") != expected_units:
            raise ValueError("cached person windows do not match discovered video units")
        windows = {key: tuple(value) for key, value in window_payload["windows"].items()}
        detection_stats = window_payload["stats"]
    elif args.disable_person_crop:
        windows = {unit: (0.0, 0.0, 1.0, 1.0) for unit in expected_units}
        detection_stats = {"clips": len(records), "detected": 0, "fallback": len(records)}
    else:
        if not detector_path.is_file():
            raise FileNotFoundError(detector_path)
        windows, detection_stats = detect_windows(
            records, detector_path, args.detection_frames, args.detector_device
        )
    if not window_path.is_file():
        window_path.parent.mkdir(parents=True, exist_ok=True)
        window_path.write_text(
            json.dumps(
                {"units": expected_units, "windows": windows, "stats": detection_stats},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    class ClipDataset(Dataset):
        def __len__(self) -> int:
            return len(records)

        def __getitem__(self, index: int):
            record = records[index]
            ir = sample_video_frames(record["ir"], args.frames)
            if record["depth_color"] is None:
                depth_color = np.zeros_like(ir)
            else:
                depth_color = sample_video_frames(record["depth_color"], args.frames)
            x0, y0, x1, y1 = windows[str(record["unit"])]
            output = np.zeros((args.frames, 4, 128, 128), dtype=np.float32)
            for frame_index in range(args.frames):
                for source, begin, channels in ((depth_color, 0, 3), (ir, 3, 1)):
                    height, width = source[frame_index].shape[:2]
                    crop = source[frame_index][
                        round(y0 * height) : round(y1 * height),
                        round(x0 * width) : round(x1 * width),
                    ]
                    if crop.size == 0:
                        crop = source[frame_index]
                    image = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_LINEAR)
                    if channels == 1:
                        output[frame_index, begin] = image.mean(axis=2) / 255.0
                    else:
                        output[frame_index, begin : begin + channels] = (
                            image.transpose(2, 0, 1) / 255.0
                        )
            mean = torch.tensor((*KINETICS_MEAN, sum(KINETICS_MEAN) / 3.0))[None, :, None, None]
            std = torch.tensor((*KINETICS_STD, sum(KINETICS_STD) / 3.0))[None, :, None, None]
            return (torch.from_numpy(output) - mean) / std, index

    device = torch.device(args.device)
    loader = DataLoader(
        ClipDataset(),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != "kuno-yolo-r2p1d-packed-ensemble/v1":
        raise ValueError("unexpected public checkpoint schema")
    fold_logits = np.zeros((len(checkpoint["models_packed"]), len(records), 40), dtype=np.float32)
    started = time.perf_counter()
    with torch.inference_mode():
        for fold_index, (fold, packed_state) in enumerate(
            zip(checkpoint["folds"], checkpoint["models_packed"], strict=True)
        ):
            model = build_r2plus1d34(architecture_path).to(device)
            model.load_state_dict(dequantize_state(packed_state))
            model.eval()
            for batch_index, (clips, indices) in enumerate(loader, start=1):
                clips = clips.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(clips) + model(torch.flip(clips, dims=(-1,)))
                fold_logits[fold_index, indices.numpy()] = logits.float().cpu().numpy()
                if batch_index % 25 == 0 or batch_index == len(loader):
                    print(
                        f"fold={fold} processed={min(batch_index * args.batch_size, len(records))}/"
                        f"{len(records)}",
                        flush=True,
                    )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        logits=fold_logits,
        units=np.asarray(expected_units),
        folds=np.asarray(checkpoint["folds"]),
        weights=np.asarray(checkpoint["weights"], dtype=np.float32),
        classes=np.arange(40),
    )
    report = {
        "output": str(args.output.resolve()),
        "sha256": sha256_file(args.output),
        "shape": list(fold_logits.shape),
        "units": len(records),
        "missing_depth_color": sum(record["depth_color"] is None for record in records),
        "detection": detection_stats,
        "source_model_sha256": sha256_file(model_path),
        "source_architecture_sha256": sha256_file(architecture_path),
        "source_detector_sha256": sha256_file(detector_path) if detector_path.is_file() else None,
        "source_parent_validation_accuracy": checkpoint["parent_val_acc"],
        "frames": args.frames,
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
