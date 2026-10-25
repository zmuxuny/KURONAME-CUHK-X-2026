"""Deterministic generic video windows and explicit physical-time alignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cuhkx.video import crop_video_frames, sample_video_windows, temporal_window_indices

KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
KINETICS_STD = [0.22803, 0.22145, 0.216989]


def video_metadata(path: Path):
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open video metadata: {path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if frame_count <= 0 or not np.isfinite(fps) or fps <= 0 or min(width, height) <= 0:
        raise ValueError(f"invalid video metadata: {path}")
    return {"frame_count": frame_count, "fps": fps, "width": width, "height": height,
            "duration_seconds": frame_count / fps}


def window_timing(metadata, config):
    indices = temporal_window_indices(metadata["frame_count"], metadata["fps"],
                                      config["windows"], config["frames"], config["window_seconds"])
    starts = indices[:, 0] / metadata["fps"]
    stops = (indices[:, -1] + 1) / metadata["fps"]
    centers = (starts + stops) / 2
    duration = metadata["duration_seconds"]
    return {"frame_indices": indices.astype(np.int32), "starts_seconds": starts,
            "stops_seconds": stops, "centers_seconds": centers,
            "starts": starts / duration, "stops": stops / duration, "centers": centers / duration}


def temporal_alignment(depth, infrared, config):
    if depth is None or infrared is None:
        return {"alignment_ok": False, "reason": ["missing_modality"], "tolerance_seconds": None}
    tolerance = config["alignment_tolerance_frames"] * max(1 / depth["fps"], 1 / infrared["fps"])
    left, right = window_timing(depth, config), window_timing(infrared, config)
    gaps = {"duration_seconds": abs(depth["duration_seconds"] - infrared["duration_seconds"])}
    gaps.update({field: float(np.abs(left[field] - right[field]).max())
                 for field in ("starts_seconds", "stops_seconds", "centers_seconds")})
    reasons = [field + "_mismatch" for field, value in gaps.items() if value > tolerance + 1e-9]
    return {"alignment_ok": not reasons, "reason": reasons, "tolerance_seconds": tolerance,
            "absolute_gaps": gaps}


def preprocess_windows(windows, roi):
    """Exactly reuse the established 16-frame generic HARn preprocessing."""
    import cv2
    import torch

    if windows.ndim != 5 or windows.shape[-1] != 3:
        raise ValueError("windows must have shape [W,T,H,W,3]")
    tensors = []
    for raw in windows:
        frames = crop_video_frames(raw, tuple(roi))
        resized = []
        for frame in frames:
            image = cv2.resize(frame, (171, 128), interpolation=cv2.INTER_AREA)
            resized.append(image[8:120, 29:141])
        array = np.stack(resized).astype(np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(3, 0, 1, 2)
        mean = torch.tensor(KINETICS_MEAN)[:, None, None, None]
        std = torch.tensor(KINETICS_STD)[:, None, None, None]
        tensors.append((tensor - mean) / std)
    return torch.stack(tensors)


def sample_preprocessed(path, roi, config):
    windows = sample_video_windows(Path(path), config["windows"], config["frames"], config["window_seconds"])
    return preprocess_windows(windows, roi)
