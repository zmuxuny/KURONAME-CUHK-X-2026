"""Compact, irreversible statistics for CUHK-X mmWave radar point clouds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC_COLUMNS = ("frame", "x", "y", "z", "v", "snr", "noise")
POINT_COLUMNS = (
    "x",
    "y",
    "z",
    "v",
    "abs_v",
    "range",
    "horizontal_range",
    "snr",
    "noise",
    "contrast",
)
FRAME_COLUMNS = (
    "count",
    "x_mean",
    "y_mean",
    "z_mean",
    "x_std",
    "y_std",
    "z_std",
    "v_mean",
    "v_std",
    "abs_v_mean",
    "abs_v_q90",
    "range_mean",
    "range_std",
    "snr_mean",
    "snr_std",
    "noise_mean",
    "noise_std",
    "contrast_mean",
)
SUMMARY_STATISTICS = ("mean", "std", "min", "max", "q10", "q25", "q50", "q75", "q90")


def load_points(path: Path) -> tuple[pd.DataFrame, float]:
    """Load finite radar detections and return them with observed wall-clock duration."""

    frame = pd.read_csv(path, dtype={"timestamp": str})
    required = {"timestamp", *NUMERIC_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"radar file omits columns: {missing}")
    if frame.empty:
        raise ValueError(f"radar file contains no detections: {path}")
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    finite = np.isfinite(frame[list(NUMERIC_COLUMNS)].to_numpy(dtype=np.float64)).all(axis=1)
    frame = frame.loc[finite].copy()
    if frame.empty:
        raise ValueError(f"radar file contains no finite detections: {path}")

    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce", format="mixed")
    valid_timestamps = timestamps.dropna()
    duration = 0.0
    if len(valid_timestamps) >= 2:
        duration = float(
            (valid_timestamps.max() - valid_timestamps.min()).total_seconds()
        )
    return frame, duration


def _summary(array: np.ndarray, prefix: str) -> tuple[list[float], list[str]]:
    if array.ndim == 1:
        array = array[:, np.newaxis]
    statistics = (
        np.mean(array, axis=0),
        np.std(array, axis=0),
        np.min(array, axis=0),
        np.max(array, axis=0),
        np.quantile(array, 0.10, axis=0),
        np.quantile(array, 0.25, axis=0),
        np.quantile(array, 0.50, axis=0),
        np.quantile(array, 0.75, axis=0),
        np.quantile(array, 0.90, axis=0),
    )
    values: list[float] = []
    names: list[str] = []
    for statistic_name, statistic in zip(SUMMARY_STATISTICS, statistics, strict=True):
        values.extend(statistic.tolist())
        names.extend(
            f"{prefix}_{statistic_name}_{index:02d}" for index in range(array.shape[1])
        )
    return values, names


def _frame_descriptors(points: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    grouped = points.groupby("frame", sort=True)
    means = grouped[["x", "y", "z", "v", "abs_v", "range", "snr", "noise", "contrast"]].mean()
    deviations = grouped[["x", "y", "z", "v", "range", "snr", "noise"]].std(ddof=0)
    descriptor = np.column_stack(
        [
            grouped.size().to_numpy(dtype=np.float64),
            means[["x", "y", "z"]].to_numpy(),
            deviations[["x", "y", "z"]].to_numpy(),
            means["v"].to_numpy(),
            deviations["v"].to_numpy(),
            means["abs_v"].to_numpy(),
            grouped["abs_v"].quantile(0.90).to_numpy(),
            means["range"].to_numpy(),
            deviations["range"].to_numpy(),
            means["snr"].to_numpy(),
            deviations["snr"].to_numpy(),
            means["noise"].to_numpy(),
            deviations["noise"].to_numpy(),
            means["contrast"].to_numpy(),
        ]
    )
    return grouped.size().index.to_numpy(dtype=np.float64), descriptor


def _resample(sequence: np.ndarray, length: int) -> np.ndarray:
    if len(sequence) == 1:
        return np.repeat(sequence, length, axis=0)
    old_time = np.linspace(0.0, 1.0, len(sequence))
    new_time = np.linspace(0.0, 1.0, length)
    return np.column_stack(
        [np.interp(new_time, old_time, sequence[:, column]) for column in range(sequence.shape[1])]
    )


def extract_features(
    points: pd.DataFrame,
    *,
    duration_seconds: float = 0.0,
    temporal_bins: int = 8,
    spectral_frames: int = 32,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Summarize geometry, Doppler, density, and their time variation."""

    missing = sorted(set(NUMERIC_COLUMNS) - set(points.columns))
    if missing:
        raise ValueError(f"radar points omit columns: {missing}")
    points = points.copy()
    points["abs_v"] = points["v"].abs()
    points["range"] = np.sqrt(points["x"] ** 2 + points["y"] ** 2 + points["z"] ** 2)
    points["horizontal_range"] = np.sqrt(points["x"] ** 2 + points["y"] ** 2)
    points["contrast"] = points["snr"] - points["noise"]
    point_values = points[list(POINT_COLUMNS)].to_numpy(dtype=np.float64)
    if len(point_values) == 0 or not np.isfinite(point_values).all():
        raise ValueError("radar detections must be nonempty and finite")

    frame_numbers, descriptors = _frame_descriptors(points)
    frame_span = float(frame_numbers.max() - frame_numbers.min() + 1.0)
    values: list[float] = [
        float(len(points)),
        float(len(descriptors)),
        frame_span,
        float(len(descriptors) / max(frame_span, 1.0)),
        float(max(duration_seconds, 0.0)),
        float(len(descriptors) / max(duration_seconds, 1e-6))
        if duration_seconds > 0
        else 0.0,
    ]
    names: list[str] = [
        "meta_point_count",
        "meta_detected_frames",
        "meta_frame_span",
        "meta_frame_density",
        "meta_duration_seconds",
        "meta_detected_fps",
    ]

    summary_values, summary_names = _summary(point_values, "point")
    values.extend(summary_values)
    names.extend(summary_names)
    summary_values, summary_names = _summary(descriptors, "frame")
    values.extend(summary_values)
    names.extend(summary_names)

    differences = np.diff(descriptors, axis=0)
    if len(differences) == 0:
        differences = np.zeros((1, descriptors.shape[1]), dtype=np.float64)
    difference_statistics = (
        ("mean_abs", np.mean(np.abs(differences), axis=0)),
        ("std", np.std(differences, axis=0)),
        ("q90_abs", np.quantile(np.abs(differences), 0.90, axis=0)),
        ("max_abs", np.max(np.abs(differences), axis=0)),
    )
    for statistic, vector in difference_statistics:
        values.extend(vector.tolist())
        names.extend(f"diff_{statistic}_{index:02d}" for index in range(len(vector)))

    for bin_index, indices in enumerate(np.array_split(np.arange(len(descriptors)), temporal_bins)):
        block = descriptors[indices] if len(indices) else descriptors[-1:]
        for statistic, vector in (("mean", block.mean(axis=0)), ("std", block.std(axis=0))):
            values.extend(vector.tolist())
            names.extend(
                f"bin{bin_index:02d}_{statistic}_{index:02d}" for index in range(len(vector))
            )

    sampled = _resample(descriptors, spectral_frames)
    spectrum = np.abs(np.fft.rfft(sampled - sampled.mean(axis=0), axis=0))[1:6]
    values.extend(spectrum.reshape(-1).tolist())
    names.extend(f"fft_1to5_{index:03d}" for index in range(spectrum.size))

    result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("radar feature vector contains a non-finite value")
    return result, tuple(names)


def extract_unit_features(unit_dir: Path) -> tuple[np.ndarray, tuple[str, ...]]:
    points, duration = load_points(unit_dir / "Radar" / "Radar.csv")
    return extract_features(points, duration_seconds=duration)
