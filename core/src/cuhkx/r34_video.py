"""Generic IG65M/Kinetics R(2+1)D-34 features with explicit video/window keys.

The architecture matches moabitcoin/ig65m-pytorch (MIT): torchvision's
VideoResNet plus the published Caffe2 convolution and BatchNorm corrections.
Only the original 400-class public checkpoint can be loaded by this module.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from cuhkx.generic_video_windows import preprocess_windows as generic_preprocess
from cuhkx.generic_video_windows import temporal_alignment, window_timing
from cuhkx.video import sample_video_frames, sample_video_windows, uniform_frame_indices

KIND = "cuhkx_generic_r34_layer4_1_v1"
WEIGHTS_SHA = "ade133f18548b59da3a36d6a70d44a4a7cc4b90460654c338cb8a023e9a3e0e1"
WEIGHTS_BYTES = 255061555
MID_SHAPE = (512, 4, 7, 7)
MODALITIES = ("Depth_Color", "IR")
WINDOW_CONFIG = {"windows": 8, "frames": 32, "window_seconds": 2.5,
                 "alignment_tolerance_frames": 1.0}
SPEC = {
    "kind": KIND, "backbone": "r2plus1d_34_32_kinetics", "weights_sha256": WEIGHTS_SHA,
    "frames": 32, "modalities": list(MODALITIES), "shape": list(MID_SHAPE),
    "cut_after": "layer4[1]", "remaining_pretrained_block": "layer4[2]",
    "storage": "float16", "row_keys": ["units", "modalities", "window_indices"],
    "regimes": {"HARn": {"name": "whole", "windows": 1},
                "HAU": {"name": "temporal", "windows": 8, "window_seconds": 2.5}},
    "preprocessing": "RGB Depth_Color; RGB-to-gray IR replicated3; normalized person ROI; INTER_AREA171x128; center112; Kinetics mean/std",
    "missing_policy": "Keep zero rows with available=false; never substitute Depth for missing Depth_Color",
    "core_fitting": "Generic IG65M then Kinetics400 pretraining only; no CUHK-X labels or model weights",
}
TIME_FIELDS = ("starts", "stops", "centers", "starts_seconds", "stops_seconds", "centers_seconds")


def sha256_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def build_r34():
    """Reconstruct the published 3-channel/400-class Caffe2-compatible model."""
    from torch import nn
    from torchvision.models.video.resnet import BasicBlock, Conv2Plus1D, R2Plus1dStem, VideoResNet

    model = VideoResNet(block=BasicBlock, conv_makers=[Conv2Plus1D] * 4,
                       layers=[3, 4, 6, 3], stem=R2Plus1dStem, num_classes=400)
    for layer, channels, intermediate in ((model.layer2, 128, 288),
                                          (model.layer3, 256, 576), (model.layer4, 512, 1152)):
        layer[0].conv2[0] = Conv2Plus1D(channels, channels, intermediate)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm3d):
            module.eps, module.momentum = 1e-3, 0.9
    return model


def pretrained_model(weights_path):
    import torch

    path = Path(weights_path)
    if path.stat().st_size != WEIGHTS_BYTES or sha256_file(path) != WEIGHTS_SHA:
        raise ValueError("R34 requires the exact declared public IG65M/Kinetics checkpoint")
    model = build_r34()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.eval()


def frozen_core(model):
    from torch import nn

    core = nn.Sequential(model.stem, model.layer1, model.layer2, model.layer3,
                         model.layer4[0], model.layer4[1]).eval()
    for parameter in core.parameters():
        parameter.requires_grad_(False)
    return core


def timing(metadata, source):
    if source == "HAU":
        return window_timing(metadata, WINDOW_CONFIG)
    if source != "HARn":
        raise ValueError("unknown video source")
    indices = uniform_frame_indices(metadata["frame_count"], 32)[None, :].astype(np.int32)
    starts = indices[:, 0] / metadata["fps"]
    stops = (indices[:, -1] + 1) / metadata["fps"]
    centers = (starts + stops) / 2
    duration = metadata["duration_seconds"]
    return {"frame_indices": indices, "starts_seconds": starts, "stops_seconds": stops,
            "centers_seconds": centers, "starts": starts / duration,
            "stops": stops / duration, "centers": centers / duration}


def alignment(left, right, source):
    if source == "HAU":
        return temporal_alignment(left, right, WINDOW_CONFIG)
    if source != "HARn":
        raise ValueError("unknown video source")
    if left is None or right is None:
        return {"alignment_ok": False, "reason": ["missing_modality"], "tolerance_seconds": None}
    tolerance = max(1 / left["fps"], 1 / right["fps"])
    a, b = timing(left, source), timing(right, source)
    gaps = {"duration_seconds": abs(left["duration_seconds"] - right["duration_seconds"])}
    gaps.update({key: float(np.max(np.abs(a[key] - b[key]))) for key in TIME_FIELDS if key.endswith("seconds")})
    reasons = [key + "_mismatch" for key, value in gaps.items() if value > tolerance + 1e-9]
    return {"alignment_ok": not reasons, "reason": reasons, "tolerance_seconds": tolerance,
            "absolute_gaps": gaps}


def preprocess(windows, roi, modality):
    if windows.ndim != 5 or windows.shape[1] != 32 or windows.shape[-1] != 3:
        raise ValueError("R34 input must be [windows,32,height,width,3] RGB frames")
    if modality not in MODALITIES:
        raise ValueError("only Depth_Color and replicated grayscale IR are supported")
    if modality == "IR":
        import cv2
        gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in windows.reshape(-1, *windows.shape[2:])])
        windows = np.repeat(gray[..., None], 3, axis=-1).reshape(windows.shape)
    return generic_preprocess(windows, roi)


def sample_preprocessed(record):
    if not record["available"]:
        raise ValueError("a missing-modality record has no frames to sample")
    path = Path(record["path"])
    if record["source"] == "HARn":
        frames = sample_video_frames(path, 32)[None, :]
    else:
        frames = sample_video_windows(path, 8, 32, 2.5)
    return preprocess(frames, record["roi"], record["modality"])


def metadata_arrays(inputs):
    records, by_unit = inputs["records"], inputs["alignment"]
    count = sum(1 if record["source"] == "HARn" else 8 for record in records)
    arrays = {"units": [], "sources": [], "modalities": [], "regimes": [], "window_indices": [],
              "available": [], "alignment_ok": [], "raw_status": []}
    arrays.update(frame_indices=np.full((count, 32), -1, dtype=np.int32),
                  frame_count=np.zeros(count, dtype=np.int32), fps=np.zeros(count, dtype=np.float64),
                  duration_seconds=np.zeros(count, dtype=np.float64), roi=np.full((count, 4), -1.0, dtype=np.float64))
    arrays.update({key: np.full(count, -1.0, dtype=np.float64) for key in TIME_FIELDS})
    cursor = 0
    for record in records:
        number = 1 if record["source"] == "HARn" else 8
        selected = slice(cursor, cursor + number)
        for key, value in (("units", record["unit"]), ("sources", record["source"]),
                           ("modalities", record["modality"]), ("raw_status", record["raw_status"]),
                           ("regimes", "whole" if number == 1 else "temporal"),
                           ("available", record["available"]), ("alignment_ok", by_unit[record["unit"]]["alignment_ok"])):
            arrays[key].extend([value] * number)
        arrays["window_indices"].extend(range(number))
        if record["roi"] is not None:
            arrays["roi"][selected] = record["roi"]
        if record["available"]:
            for key in ("frame_count", "fps", "duration_seconds"):
                arrays[key][selected] = record["metadata"][key]
            for key in ("frame_indices", *TIME_FIELDS):
                arrays[key][selected] = record["timing"][key]
        cursor += number
    arrays = {key: np.asarray(value) for key, value in arrays.items()}
    arrays["window_indices"] = arrays["window_indices"].astype(np.int16)
    validate_metadata(arrays)
    return arrays


def validate_metadata(metadata):
    required = {"units", "sources", "modalities", "regimes", "window_indices", "available",
                "alignment_ok", "frame_indices", "frame_count", "fps", "duration_seconds", "roi", "raw_status", *TIME_FIELDS}
    if set(metadata) != required:
        raise ValueError("R34 window metadata schema differs")
    count = len(metadata["units"])
    if not count or any(len(value) != count for value in metadata.values()):
        raise ValueError("R34 metadata row axes differ")
    if metadata["frame_indices"].shape != (count, 32) or metadata["roi"].shape != (count, 4):
        raise ValueError("R34 frame/ROI axes differ")
    keys = list(zip(metadata["units"].astype(str), metadata["modalities"].astype(str),
                    metadata["window_indices"].tolist(), strict=True))
    if len(set(keys)) != count:
        raise ValueError("duplicate R34 unit/modality/window keys")
    expected = set()
    for unit in sorted(set(metadata["units"].astype(str))):
        selected = metadata["units"] == unit
        sources = set(metadata["sources"][selected].astype(str))
        if len(sources) != 1 or not sources <= {"HARn", "HAU"}:
            raise ValueError("each unit requires one explicit HARn/HAU source")
        source = sources.pop()
        number = 1 if source == "HARn" else 8
        expected.update((unit, modality, window) for modality in MODALITIES for window in range(number))
        if not np.all(metadata["regimes"][selected] == ("whole" if number == 1 else "temporal")):
            raise ValueError("source/window regime mismatch")
    if expected != set(keys):
        raise ValueError("R34 metadata must retain the complete declared modality/window grid")
    if metadata["available"].dtype != np.bool_ or metadata["alignment_ok"].dtype != np.bool_:
        raise ValueError("R34 availability/alignment masks must be Boolean")
    missing = ~metadata["available"]
    if (np.any(metadata["frame_indices"][missing] != -1)
            or any(np.any(metadata[key][missing] != -1) for key in TIME_FIELDS)
            or np.any(metadata["alignment_ok"][missing])):
        raise ValueError("missing modality rows must retain explicit invalid timing/alignment")


class R34FeatureBank:
    """A hash-checked row-keyed memmap; labels never enter the feature interface."""

    @classmethod
    def load(cls, manifest_path):
        path = Path(manifest_path).resolve()
        manifest = json.loads(path.read_text())
        if manifest.get("kind") != KIND or manifest.get("spec") != SPEC:
            raise ValueError("not the declared generic R34 middle-feature bank")
        files = {}
        for name in ("features", "metadata"):
            record = manifest[name]
            actual = (path.parent / record["path"]).resolve()
            if sha256_file(actual) != record["sha256"]:
                raise ValueError(f"R34 {name} SHA256 differs")
            files[name] = actual
        with np.load(files["metadata"], allow_pickle=False) as source:
            metadata = {key: source[key] for key in source.files}
        validate_metadata(metadata)
        features = np.load(files["features"], mmap_mode="r", allow_pickle=False)
        if features.dtype != np.float16 or features.shape != (len(metadata["units"]), *MID_SHAPE):
            raise ValueError("R34 feature shape or storage differs")
        if sorted(set(metadata["units"].astype(str))) != manifest["units"]:
            raise ValueError("R34 manifest/metadata unit coverage differs")
        bank = cls()
        bank.features, bank.metadata, bank.manifest = features, metadata, manifest
        bank.manifest_path = str(path)
        bank.row_index = {key: index for index, key in enumerate(zip(metadata["units"].astype(str),
            metadata["modalities"].astype(str), metadata["window_indices"].tolist(), strict=True))}
        return bank

    def rows_for_units(self, units, *, available_only=False):
        units = set(map(str, units))
        if not units <= set(self.metadata["units"].astype(str)):
            raise ValueError("requested units are absent from the R34 bank")
        selected = np.isin(self.metadata["units"], sorted(units))
        if available_only:
            selected &= self.metadata["available"]
        return np.flatnonzero(selected)
