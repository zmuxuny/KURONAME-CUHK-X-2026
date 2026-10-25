"""Feature extraction for CUHK-X 17-joint 3D skeleton sequences."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

JOINTS = 17
COORDINATES = 3
EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (8, 11),
    (11, 12),
    (12, 13),
    (8, 14),
    (14, 15),
    (15, 16),
)
FRAME_PATTERN = re.compile(r"_(\d+)\.json$")


def _frame_order(path: Path) -> tuple[int, str]:
    match = FRAME_PATTERN.search(path.name)
    return (int(match.group(1)) if match else 0, path.name)


def load_sequence(predictions_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the highest-confidence detected person from each valid frame."""

    poses: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    for path in sorted(predictions_dir.glob("*.json"), key=_frame_order):
        people = json.loads(path.read_text(encoding="utf-8"))
        if not people:
            continue
        person = max(
            people,
            key=lambda item: float(np.asarray(item.get("keypoint_scores", []), dtype=float).mean())
            if item.get("keypoint_scores")
            else -1.0,
        )
        pose = np.asarray(person.get("keypoints"), dtype=np.float64)
        score = np.asarray(person.get("keypoint_scores"), dtype=np.float64)
        if pose.shape != (JOINTS, COORDINATES) or score.shape != (JOINTS,):
            continue
        if not np.isfinite(pose).all():
            continue
        poses.append(pose)
        confidences.append(np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0))
    if not poses:
        raise ValueError(f"no valid skeleton frames in {predictions_dir}")
    return np.stack(poses), np.stack(confidences)


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


def _summary_features(array: np.ndarray, prefix: str) -> tuple[list[float], list[str]]:
    flat = array.reshape(len(array), -1)
    statistics = {
        "mean": np.mean(flat, axis=0),
        "std": np.std(flat, axis=0),
        "min": np.min(flat, axis=0),
        "max": np.max(flat, axis=0),
        "q10": np.quantile(flat, 0.10, axis=0),
        "q25": np.quantile(flat, 0.25, axis=0),
        "q50": np.quantile(flat, 0.50, axis=0),
        "q75": np.quantile(flat, 0.75, axis=0),
        "q90": np.quantile(flat, 0.90, axis=0),
    }
    values: list[float] = []
    names: list[str] = []
    for statistic, vector in statistics.items():
        values.extend(vector.tolist())
        names.extend(f"{prefix}_{statistic}_{index:03d}" for index in range(len(vector)))
    return values, names


def extract_features(
    poses: np.ndarray,
    confidences: np.ndarray,
    *,
    resample_frames: int = 32,
    temporal_bins: int = 8,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Extract subject-robust kinematic, temporal, and spectral features."""

    if poses.ndim != 3 or poses.shape[1:] != (JOINTS, COORDINATES):
        raise ValueError(f"unexpected pose shape {poses.shape}")
    pelvis = poses[:, :1, :]
    centered = poses - pelvis
    edge_lengths = np.stack(
        [np.linalg.norm(centered[:, left] - centered[:, right], axis=1) for left, right in EDGES],
        axis=1,
    )
    positive_lengths = edge_lengths[edge_lengths > 1e-6]
    scale = float(np.median(positive_lengths)) if len(positive_lengths) else 1.0
    normalized = centered / max(scale, 1e-6)

    values: list[float] = [
        float(len(poses)),
        float(len(poses) / 10.0),
        scale,
        float(np.mean(confidences)),
        float(np.min(confidences)),
    ]
    names: list[str] = ["frame_count", "duration_seconds", "body_scale", "score_mean", "score_min"]

    for array, prefix in ((normalized, "pose"), (edge_lengths / max(scale, 1e-6), "bone")):
        part_values, part_names = _summary_features(array, prefix)
        values.extend(part_values)
        names.extend(part_names)

    for order, difference in (
        (1, np.diff(normalized, n=1, axis=0)),
        (2, np.diff(normalized, n=2, axis=0)),
    ):
        if len(difference) == 0:
            difference = np.zeros((1, JOINTS, COORDINATES), dtype=np.float64)
        flat = difference.reshape(len(difference), -1)
        stats = {
            "mean_abs": np.mean(np.abs(flat), axis=0),
            "std": np.std(flat, axis=0),
            "q90_abs": np.quantile(np.abs(flat), 0.90, axis=0),
            "max_abs": np.max(np.abs(flat), axis=0),
        }
        for statistic, vector in stats.items():
            values.extend(vector.tolist())
            names.extend(f"diff{order}_{statistic}_{index:03d}" for index in range(len(vector)))

    sampled = _resample(normalized, resample_frames)
    values.extend(sampled.reshape(-1).tolist())
    names.extend(f"resampled_pose_{index:04d}" for index in range(sampled.size))

    for bin_index, indices in enumerate(np.array_split(np.arange(len(normalized)), temporal_bins)):
        block = normalized[indices] if len(indices) else normalized[-1:]
        for statistic, vector in (("mean", block.mean(axis=0)), ("std", block.std(axis=0))):
            vector = vector.reshape(-1)
            values.extend(vector.tolist())
            names.extend(
                f"bin{bin_index:02d}_{statistic}_{index:03d}" for index in range(len(vector))
            )

    spectral_input = sampled.reshape(resample_frames, -1)
    spectrum = np.abs(np.fft.rfft(spectral_input - spectral_input.mean(axis=0), axis=0))[1:6]
    values.extend(spectrum.reshape(-1).tolist())
    names.extend(f"fft_1to5_{index:04d}" for index in range(spectrum.size))

    # Preserve camera-relative root/depth motion separately from centered pose.
    root_values, root_names = _summary_features(pelvis, "root")
    values.extend(root_values)
    names.extend(root_names)

    result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("feature vector contains a non-finite value")
    return result, tuple(names)


def extract_unit_features(unit_dir: Path) -> tuple[np.ndarray, tuple[str, ...]]:
    poses, confidences = load_sequence(unit_dir / "Skeleton" / "predictions")
    return extract_features(poses, confidences)

