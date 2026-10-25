#!/usr/bin/env python3
"""Cache generic pretrained R2Plus1D features before layer4, without fitting."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuhkx.harn_nested import sha256_file, write_json
from cuhkx.video import discover_videos, sample_video_frames
from cuhkx.video_midfeatures import MID_SHAPE, SPEC, generic_model, preprocess

SOURCES = ["scripts/extract_video_midfeatures.py", "src/cuhkx/video_midfeatures.py",
           "src/cuhkx/video.py", "scripts/extract_video_embeddings.py"]


def main():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--windows-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--prefixes", nargs="+", default=["Training/HARn", "Testing/"])
    parser.add_argument("--maximum-units", type=int, help="bounded extraction control-flow/raw replay only")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not 1 <= args.workers <= 3 or args.batch_size != 8:
        raise ValueError("fixed CUDA extraction requires batch8 and 1..3 loader workers")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    discovered = discover_videos(args.media_root, ("Depth", "IR"), tuple(args.prefixes))
    lookup = {(unit, modality): path for unit, modality, path in discovered}
    if len(lookup) != len(discovered):
        raise ValueError("duplicate raw unit/modality")
    units = sorted({unit for unit, modality in lookup if modality == "Depth"}
                   & {unit for unit, modality in lookup if modality == "IR"})
    if args.maximum_units:
        units = units[:args.maximum_units]
    if not units:
        raise ValueError("no complete paired Depth/IR videos")
    records = [(unit, modality, lookup[(unit, modality)]) for unit in units for modality in SPEC["modalities"]]
    windows = json.loads(args.windows_json.read_text())["windows"]
    if any(unit not in windows for unit in units):
        raise ValueError("every selected video needs the declared generic person crop")
    sources = [{"path": name, "sha256": sha256_file(ROOT / name)} for name in SOURCES]
    frozen = {"spec": SPEC, "source_manifest": sources,
              "weights": {"path": str(args.weights.resolve()), "sha256": sha256_file(args.weights)},
              "crop": {"path": str(args.windows_json.resolve()), "sha256": sha256_file(args.windows_json)},
              "raw_inputs": [{"unit": unit, "modality": modality, "path": str(path.resolve()),
                              "bytes": path.stat().st_size, "sha256": sha256_file(path)} for unit, modality, path in records],
              "training_or_test_labels_read": False, "maximum_units": args.maximum_units,
              "execution": {"device": "cuda:0", "batch_size": 8, "workers": args.workers,
                            "autocast": "float16", "cudnn_benchmark": False, "cudnn_deterministic": True,
                            "torch": torch.__version__, "cuda": torch.version.cuda}}
    write_json(out / "frozen_inputs.json", frozen)
    for entry in sources:
        destination = out / "source_snapshot" / entry["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / entry["path"], destination)
        if sha256_file(destination) != entry["sha256"]:
            raise ValueError("source changed while freezing the raw extraction")
    model = generic_model(args.weights).eval().cuda()
    core = nn.Sequential(model.stem, model.layer1, model.layer2, model.layer3).eval()
    for parameter in core.parameters():
        parameter.requires_grad_(False)
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cache = np.lib.format.open_memmap(out / "features.npy", mode="w+", dtype=np.float16,
                                      shape=(len(units), 2, *MID_SHAPE))
    written = np.zeros((len(units), 2), dtype=bool)

    class Videos(Dataset):
        def __len__(self):
            return len(records)

        def __getitem__(self, index):
            unit, _, path = records[index]
            return preprocess(sample_video_frames(path, 16), tuple(windows[unit])), index

    loader = DataLoader(Videos(), batch_size=8, shuffle=False, num_workers=args.workers,
                        pin_memory=True, persistent_workers=True)
    started = time.monotonic()
    with torch.inference_mode():
        for step, (batch, indices) in enumerate(loader):
            with torch.autocast(device_type="cuda"):
                values = core(batch.cuda(non_blocking=True)).half().cpu().numpy()
            if values.shape != (len(indices), *MID_SHAPE) or not np.isfinite(values).all():
                raise ValueError("frozen core returned invalid middle activations")
            for index, value in zip(indices.tolist(), values, strict=True):
                row, modality = divmod(index, 2)
                if written[row, modality]:
                    raise ValueError("duplicate middle feature write")
                cache[row, modality] = value
                written[row, modality] = True
            if (step + 1) % 50 == 0 or (step + 1) * 8 >= len(records):
                print(f"midfeatures={int(written.sum())}/{len(records)} seconds={time.monotonic()-started:.1f}", flush=True)
    if not written.all():
        raise ValueError("raw extraction missed unit/modality rows")
    cache.flush()
    np.savez_compressed(out / "units.npz", units=np.asarray(units))
    manifest = {"schema_version": 1, "spec": SPEC,
                **{name: {"path": file, "sha256": sha256_file(out / file)} for name, file in
                   [("features", "features.npy"), ("units", "units.npz")]},
                "frozen_inputs_sha256": sha256_file(out / "frozen_inputs.json")}
    write_json(out / "manifest.json", manifest)
    for entry in sources:
        if sha256_file(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError("extractor source changed during execution")
    write_json(out / "report.json", {"sha256": sha256_file(out / "manifest.json"),
                                     "units": len(units), "array_shape": list(cache.shape),
                                     "array_bytes": cache.nbytes, "core_fitting": False,
                                     "seconds": time.monotonic() - started, "torch": torch.__version__,
                                     "cuda": torch.version.cuda, "stored_dtype": "float16"})


if __name__ == "__main__":
    main()
