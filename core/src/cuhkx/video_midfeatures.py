"""Frozen generic video features before the final residual stage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cuhkx.harn_nested import FeatureBank, sha256_file

MID_SHAPE = (256, 4, 14, 14)
WEIGHTS_SHA = "91a641e6c2ab531d1aca5f4321b4d802ec5c3babc15df855cdb6e39c6a1107c8"
SPEC = {"kind": "r2plus1d_18_layer3", "weights_sha256": WEIGHTS_SHA,
        "frames": 16, "modalities": ["Depth", "IR"], "shape": list(MID_SHAPE),
        "storage": "float16", "core_fitting": "Kinetics400 pretrained only; never fitted on CUHK-X"}


def preprocess(frames, crop=None):
    import cv2

    from cuhkx.video import crop_video_frames

    if crop is not None:
        frames = crop_video_frames(frames, crop)
    images = [cv2.resize(frame, (171, 128), interpolation=cv2.INTER_AREA)[8:120, 29:141] for frame in frames]
    tensor = np.stack(images).astype(np.float32).transpose(3, 0, 1, 2) / 255.0
    mean = np.asarray([0.43216, 0.394666, 0.37645], dtype=np.float32)[:, None, None, None]
    std = np.asarray([0.22803, 0.22145, 0.216989], dtype=np.float32)[:, None, None, None]
    return (tensor - mean) / std


def generic_model(weights_path):
    import torch
    from torchvision.models.video import r2plus1d_18

    if sha256_file(Path(weights_path)) != WEIGHTS_SHA:
        raise ValueError("local generic Kinetics weights differ from the declared public checkpoint")
    network = r2plus1d_18(weights=None)
    network.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    return network


class MidFeatureBank(FeatureBank):
    @classmethod
    def load(cls, skeleton_path, visual_path):
        with np.load(skeleton_path, allow_pickle=False) as source:
            skeleton_units = source["units"].astype(str)
            skeleton = source["X"].astype(np.float32)
            names = source["feature_names"].astype(str)
        if (skeleton.shape != (len(skeleton_units), len(names)) or not np.isfinite(skeleton).all()
                or len(set(skeleton_units)) != len(skeleton_units)):
            raise ValueError("invalid Skeleton feature axes or values")
        payload = json.loads(Path(visual_path).read_text())
        if payload["spec"] != SPEC:
            raise ValueError("unexpected frozen middle feature specification")
        references = {}
        for name in ("features", "units"):
            entry = payload[name]
            path = (Path(visual_path).parent / entry["path"]).resolve()
            if sha256_file(path) != entry["sha256"]:
                raise ValueError(f"middle feature {name} SHA differs")
            references[name] = path
        with np.load(references["units"], allow_pickle=False) as source:
            units = source["units"].astype(str)
        visual = np.load(references["features"], mmap_mode="r", allow_pickle=False)
        if (visual.shape != (len(units), 2, *MID_SHAPE) or visual.dtype != np.float16
                or len(set(units)) != len(units)):
            raise ValueError("middle feature axes or storage differ")
        for start in range(0, len(units), 64):
            if not np.isfinite(visual[start:start + 64]).all():
                raise ValueError("nonfinite middle features")
        bank = cls(skeleton_units, skeleton, names, units, visual.reshape(len(units), -1), SPEC)
        bank.mid_manifest = payload
        bank.mid_manifest_path = str(Path(visual_path).resolve())
        return bank
