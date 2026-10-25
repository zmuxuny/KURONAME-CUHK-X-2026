"""Compact window features for temporal Skeleton action localization."""

from __future__ import annotations

import numpy as np

from cuhkx.skeleton import COORDINATES, EDGES, JOINTS


def window_bounds(length: int, *, size: int = 32, stride: int = 12) -> list[tuple[int, int]]:
    if length <= 0:
        raise ValueError("sequence length must be positive")
    if size <= 0 or stride <= 0:
        raise ValueError("window size and stride must be positive")
    if length <= size:
        return [(0, length)]
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] != length - size:
        starts.append(length - size)
    return [(start, start + size) for start in starts]


def _resample(sequence: np.ndarray, length: int) -> np.ndarray:
    if len(sequence) == length:
        return sequence
    if len(sequence) == 1:
        return np.repeat(sequence, length, axis=0)
    old_time = np.linspace(0.0, 1.0, len(sequence))
    new_time = np.linspace(0.0, 1.0, length)
    flat = sequence.reshape(len(sequence), -1)
    result = np.column_stack(
        [np.interp(new_time, old_time, flat[:, column]) for column in range(flat.shape[1])]
    )
    return result.reshape((length, *sequence.shape[1:]))


def extract_window_features(
    poses: np.ndarray,
    confidences: np.ndarray,
    *,
    resample_frames: int = 16,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if poses.ndim != 3 or poses.shape[1:] != (JOINTS, COORDINATES):
        raise ValueError(f"unexpected pose shape {poses.shape}")
    if confidences.shape != poses.shape[:2]:
        raise ValueError(f"unexpected confidence shape {confidences.shape}")
    pelvis = poses[:, :1, :]
    centered = poses - pelvis
    edge_lengths = np.stack(
        [np.linalg.norm(centered[:, left] - centered[:, right], axis=1) for left, right in EDGES],
        axis=1,
    )
    positive = edge_lengths[edge_lengths > 1e-6]
    scale = float(np.median(positive)) if len(positive) else 1.0
    normalized = centered / max(scale, 1e-6)
    sampled = _resample(normalized, resample_frames)

    values: list[float] = [
        float(len(poses)),
        scale,
        float(confidences.mean()),
        float(confidences.min()),
    ]
    names = ["window_frames", "body_scale", "score_mean", "score_min"]
    values.extend(sampled.reshape(-1).tolist())
    names.extend(f"pose_{index:04d}" for index in range(sampled.size))

    flat = sampled.reshape(resample_frames, -1)
    for statistic, vector in (
        ("mean", flat.mean(axis=0)),
        ("std", flat.std(axis=0)),
        ("range", np.ptp(flat, axis=0)),
    ):
        values.extend(vector.tolist())
        names.extend(f"pose_{statistic}_{index:03d}" for index in range(len(vector)))

    difference = np.diff(sampled, axis=0).reshape(resample_frames - 1, -1)
    for statistic, vector in (
        ("mean_abs", np.mean(np.abs(difference), axis=0)),
        ("std", difference.std(axis=0)),
        ("max_abs", np.max(np.abs(difference), axis=0)),
    ):
        values.extend(vector.tolist())
        names.extend(f"velocity_{statistic}_{index:03d}" for index in range(len(vector)))

    normalized_bones = edge_lengths / max(scale, 1e-6)
    for statistic, vector in (
        ("mean", normalized_bones.mean(axis=0)),
        ("std", normalized_bones.std(axis=0)),
    ):
        values.extend(vector.tolist())
        names.extend(f"bone_{statistic}_{index:02d}" for index in range(len(vector)))

    root = (pelvis[:, 0] - pelvis[0, 0]) / max(scale, 1e-6)
    sampled_root = _resample(root, resample_frames)
    values.extend(sampled_root.reshape(-1).tolist())
    names.extend(f"root_delta_{index:03d}" for index in range(sampled_root.size))

    result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("skeleton window feature vector contains a non-finite value")
    return result, tuple(names)
