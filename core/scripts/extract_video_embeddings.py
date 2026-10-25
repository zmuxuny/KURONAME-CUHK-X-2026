#!/usr/bin/env python3
"""Extract frozen Kinetics video embeddings from privacy-preserving modalities."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuhkx.video import crop_video_frames, discover_videos, sample_video_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modalities", nargs="+", default=["Depth", "IR"])
    parser.add_argument("--prefixes", nargs="+", default=["Training/HARn", "Testing/"])
    parser.add_argument(
        "--backbone", choices=("r3d_18", "mc3_18", "r2plus1d_18"), default="r3d_18"
    )
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--windows-json", type=Path)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    try:
        import cv2
        import torch
        from torch import nn
        from torch.nn import functional
        from torch.utils.data import DataLoader, Dataset
        from torchvision.models.video import (
            MC3_18_Weights,
            R2Plus1D_18_Weights,
            R3D_18_Weights,
            mc3_18,
            r2plus1d_18,
            r3d_18,
        )
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise RuntimeError("install the project visual dependencies first") from error

    records = discover_videos(
        args.media_root,
        tuple(args.modalities),
        tuple(args.prefixes),
    )
    if not records:
        raise ValueError("no matching videos found")
    windows = {}
    if args.windows_json:
        payload = json.loads(args.windows_json.read_text(encoding="utf-8"))
        windows = {key: tuple(value) for key, value in payload["windows"].items()}

    class VideoDataset(Dataset):
        def __len__(self) -> int:
            return len(records)

        def __getitem__(self, index: int):
            unit, _, path = records[index]
            frames = sample_video_frames(path, args.frames)
            if unit in windows:
                frames = crop_video_frames(frames, windows[unit])
            resized = []
            for frame in frames:
                image = cv2.resize(frame, (171, 128), interpolation=cv2.INTER_AREA)
                resized.append(image[8:120, 29:141])
            array = np.stack(resized).astype(np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(3, 0, 1, 2)
            mean = torch.tensor([0.43216, 0.394666, 0.37645])[:, None, None, None]
            std = torch.tensor([0.22803, 0.22145, 0.216989])[:, None, None, None]
            return (tensor - mean) / std, index

    factories = {
        "r3d_18": (r3d_18, R3D_18_Weights.DEFAULT),
        "mc3_18": (mc3_18, MC3_18_Weights.DEFAULT),
        "r2plus1d_18": (r2plus1d_18, R2Plus1D_18_Weights.DEFAULT),
    }
    factory, weights = factories[args.backbone]
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    model = factory(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(device)
    loader = DataLoader(
        VideoDataset(),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    output = np.zeros((len(records), 512), dtype=np.float32)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (clips, indices) in enumerate(loader, start=1):
            clips = clips.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                embeddings = model(clips)
                embeddings = functional.normalize(embeddings.float(), dim=1)
            output[indices.numpy()] = embeddings.cpu().numpy()
            if batch_index % 25 == 0 or batch_index == len(loader):
                print(
                    f"processed={min(batch_index * args.batch_size, len(records))}/{len(records)}",
                    flush=True,
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        X=output,
        units=np.asarray([unit for unit, _, _ in records]),
        modalities=np.asarray([modality for _, modality, _ in records]),
        backbone=np.asarray(args.backbone),
        weights=np.asarray(str(weights)),
        frames=np.asarray(args.frames),
    )
    report = {
        "output": str(args.output.resolve()),
        "sha256": file_sha256(args.output),
        "shape": list(output.shape),
        "backbone": args.backbone,
        "weights": str(weights),
        "frames": args.frames,
        "modalities": args.modalities,
        "prefixes": args.prefixes,
        "device": str(device),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "opencv": cv2.__version__,
        "runtime_seconds": time.perf_counter() - started,
        "windows_json": str(args.windows_json.resolve()) if args.windows_json else None,
        "window_coverage": sum(unit in windows for unit, _, _ in records),
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
