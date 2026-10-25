"""Deterministic spectral and temporal features for CUHK-X wearable IMU streams."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEVICES = ("WTC", "WTRA", "WTLA", "WTRL", "WTLL")
CHANNELS = (
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "angle_x",
    "angle_y",
    "angle_z",
    "mag_x",
    "mag_y",
    "mag_z",
    "quat_0",
    "quat_1",
    "quat_2",
    "quat_3",
)

TimedStream = tuple[np.ndarray, np.ndarray]


def _device_code(value: object) -> str:
    text = str(value).strip().upper()
    for device in DEVICES:
        if text.startswith(device):
            return device
    return text.split("(", maxsplit=1)[0]


def load_unit_timed_streams(imu_dir: Path) -> dict[str, TimedStream]:
    """Read both body CSVs and return timestamps plus values for each device."""

    chunks: list[pd.DataFrame] = []
    for path in sorted(imu_dir.glob("*.csv")):
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        if len(frame.columns) < 18:
            raise ValueError(f"unexpected IMU schema with {len(frame.columns)} columns: {path}")
        if frame.empty:
            continue
        chunk = pd.DataFrame(
            {
                "time": pd.to_datetime(frame.iloc[:, 0], errors="coerce", format="mixed"),
                "device": frame.iloc[:, 1].map(_device_code),
            }
        )
        for index, channel in enumerate(CHANNELS, start=2):
            chunk[channel] = pd.to_numeric(frame.iloc[:, index], errors="coerce")
        chunks.append(chunk)
    if not chunks:
        raise ValueError(f"no non-empty IMU rows in {imu_dir}")

    combined = pd.concat(chunks, ignore_index=True)
    streams: dict[str, TimedStream] = {}
    for device in DEVICES:
        rows = combined[combined["device"] == device].sort_values("time", kind="stable")
        if rows.empty:
            continue
        values = rows.loc[:, CHANNELS].to_numpy(dtype=np.float64)
        for column in range(values.shape[1]):
            finite = np.isfinite(values[:, column])
            replacement = float(np.median(values[finite, column])) if finite.any() else 0.0
            values[~finite, column] = replacement
        timestamp_ns = rows["time"].astype("int64").to_numpy()
        valid_time = timestamp_ns != np.iinfo(np.int64).min
        if valid_time.sum() >= 2:
            timestamp = timestamp_ns.astype(np.float64) / 1e9
            valid_indices = np.flatnonzero(valid_time)
            fallback = np.interp(
                np.arange(len(timestamp)), valid_indices, timestamp[valid_indices]
            )
            timestamp[~valid_time] = fallback[~valid_time]
            timestamp -= timestamp[0]
        else:
            timestamp = np.arange(len(values), dtype=np.float64) * 0.1
        streams[device] = (timestamp, values)
    if not streams:
        raise ValueError(f"no recognized devices in {imu_dir}")
    return streams


def load_unit_streams(imu_dir: Path) -> dict[str, np.ndarray]:
    """Read numeric streams while preserving the v1 array-only interface."""

    return {device: values for device, (_, values) in load_unit_timed_streams(imu_dir).items()}


def _resample(sequence: np.ndarray, length: int) -> np.ndarray:
    if len(sequence) == 1:
        return np.repeat(sequence, length, axis=0)
    old_time = np.linspace(0.0, 1.0, len(sequence))
    new_time = np.linspace(0.0, 1.0, length)
    return np.column_stack(
        [np.interp(new_time, old_time, sequence[:, column]) for column in range(sequence.shape[1])]
    )


def _autocorrelation(sequence: np.ndarray, lag: int) -> np.ndarray:
    if len(sequence) <= lag:
        return np.zeros(sequence.shape[1], dtype=np.float64)
    left = sequence[:-lag] - sequence[:-lag].mean(axis=0)
    right = sequence[lag:] - sequence[lag:].mean(axis=0)
    denominator = np.sqrt(np.sum(left * left, axis=0) * np.sum(right * right, axis=0))
    return np.divide(
        np.sum(left * right, axis=0),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1e-12,
    )


def _stream_features(
    stream: np.ndarray, device: str, *, resample_length: int = 64, temporal_bins: int = 8
) -> tuple[list[float], list[str]]:
    sample = _resample(stream, resample_length)
    difference = np.diff(sample, axis=0)
    values: list[float] = [float(len(stream))]
    names: list[str] = [f"{device}_rows"]

    statistics = {
        "mean": sample.mean(axis=0),
        "std": sample.std(axis=0),
        "min": sample.min(axis=0),
        "max": sample.max(axis=0),
        "q10": np.quantile(sample, 0.10, axis=0),
        "q25": np.quantile(sample, 0.25, axis=0),
        "q50": np.quantile(sample, 0.50, axis=0),
        "q75": np.quantile(sample, 0.75, axis=0),
        "q90": np.quantile(sample, 0.90, axis=0),
        "diff_mean_abs": np.mean(np.abs(difference), axis=0),
        "diff_std": np.std(difference, axis=0),
        "diff_q90_abs": np.quantile(np.abs(difference), 0.90, axis=0),
        "diff_max_abs": np.max(np.abs(difference), axis=0),
    }
    for statistic, vector in statistics.items():
        values.extend(vector.tolist())
        names.extend(f"{device}_{statistic}_{channel}" for channel in CHANNELS)

    centered = sample - sample.mean(axis=0)
    power = np.abs(np.fft.rfft(centered, axis=0)) ** 2
    nonzero = power[1:]
    total = nonzero.sum(axis=0)
    normalized = np.divide(
        nonzero, total, out=np.zeros_like(nonzero), where=total[np.newaxis, :] > 1e-12
    )
    bands = ((0, 2), (2, 4), (4, 8), (8, 16), (16, len(nonzero)))
    for band_index, (start, stop) in enumerate(bands):
        vector = normalized[start:min(stop, len(normalized))].sum(axis=0)
        values.extend(vector.tolist())
        names.extend(f"{device}_band{band_index}_{channel}" for channel in CHANNELS)
    entropy = -(normalized * np.log(normalized + 1e-12)).sum(axis=0)
    dominant = np.argmax(nonzero, axis=0) + 1
    values.extend(entropy.tolist())
    names.extend(f"{device}_spectral_entropy_{channel}" for channel in CHANNELS)
    values.extend(dominant.astype(float).tolist())
    names.extend(f"{device}_dominant_bin_{channel}" for channel in CHANNELS)
    for lag in (1, 2, 5, 10):
        vector = _autocorrelation(sample, lag)
        values.extend(vector.tolist())
        names.extend(f"{device}_autocorr{lag}_{channel}" for channel in CHANNELS)

    for bin_index, indices in enumerate(np.array_split(np.arange(resample_length), temporal_bins)):
        block = sample[indices]
        for statistic, vector in (("mean", block.mean(axis=0)), ("std", block.std(axis=0))):
            values.extend(vector.tolist())
            names.extend(
                f"{device}_bin{bin_index:02d}_{statistic}_{channel}" for channel in CHANNELS
            )

    for start, stop, label in ((0, 3, "acc"), (3, 6, "gyro"), (9, 12, "mag"), (12, 16, "quat")):
        magnitude = np.linalg.norm(sample[:, start:stop], axis=1)
        magnitude_diff = np.diff(magnitude)
        aggregate = (
            magnitude.mean(),
            magnitude.std(),
            np.quantile(magnitude, 0.10),
            np.quantile(magnitude, 0.50),
            np.quantile(magnitude, 0.90),
            np.mean(np.abs(magnitude_diff)),
            np.quantile(np.abs(magnitude_diff), 0.90),
        )
        values.extend(aggregate)
        names.extend(
            f"{device}_{label}_{statistic}"
            for statistic in ("mean", "std", "q10", "q50", "q90", "diff_mean_abs", "diff_q90_abs")
        )
    return values, names


def extract_features(streams: dict[str, np.ndarray]) -> tuple[np.ndarray, tuple[str, ...]]:
    values: list[float] = []
    names: list[str] = []
    # A dummy stream defines stable zero-filled features for missing devices.
    dummy = np.zeros((2, len(CHANNELS)), dtype=np.float64)
    for device in DEVICES:
        present = device in streams
        values.append(float(present))
        names.append(f"{device}_present")
        stream_values, stream_names = _stream_features(streams.get(device, dummy), device)
        values.extend(stream_values)
        names.extend(stream_names)
    result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("IMU feature vector contains a non-finite value")
    return result, tuple(names)


def _timed_stream_features(
    timestamp: np.ndarray, stream: np.ndarray, device: str
) -> tuple[list[float], list[str]]:
    """Extract pace-aware derivatives and spectra in physical time units."""

    timestamp = np.asarray(timestamp, dtype=np.float64)
    stream = np.asarray(stream, dtype=np.float64)
    if stream.ndim != 2 or stream.shape[1] != len(CHANNELS) or len(timestamp) != len(stream):
        raise ValueError(f"invalid timed stream shape: time={timestamp.shape} values={stream.shape}")
    positive_dt = np.diff(timestamp)
    positive_dt = positive_dt[np.isfinite(positive_dt) & (positive_dt > 0.002)]
    median_dt = float(np.median(positive_dt)) if len(positive_dt) else 0.1
    duration = float(max(timestamp[-1] - timestamp[0], median_dt)) if len(timestamp) else median_dt
    sample_rate = 1.0 / max(median_dt, 1e-3)
    values: list[float] = [
        duration,
        sample_rate,
        median_dt,
        float(np.quantile(positive_dt, 0.90)) if len(positive_dt) else median_dt,
        float(len(stream) / max(duration, 1e-3)),
    ]
    names = [
        f"{device}_duration_seconds",
        f"{device}_sample_rate_hz",
        f"{device}_dt_median",
        f"{device}_dt_q90",
        f"{device}_rows_per_second",
    ]

    delta_time = np.diff(timestamp)
    valid_delta = np.isfinite(delta_time) & (delta_time > 0.002)
    if valid_delta.any():
        derivative = np.diff(stream, axis=0)[valid_delta] / delta_time[valid_delta, np.newaxis]
    else:
        derivative = np.zeros((1, stream.shape[1]), dtype=np.float64)
    derivative_stats = {
        "physical_diff_mean_abs": np.mean(np.abs(derivative), axis=0),
        "physical_diff_std": np.std(derivative, axis=0),
        "physical_diff_q90_abs": np.quantile(np.abs(derivative), 0.90, axis=0),
    }
    for statistic, vector in derivative_stats.items():
        values.extend(vector.tolist())
        names.extend(f"{device}_{statistic}_{channel}" for channel in CHANNELS)

    # Interpolate to a regular physical-time grid.  This retains true pace,
    # unlike the normalized-length representation used by v1.
    regular_length = int(np.clip(round(duration / max(median_dt, 1e-3)) + 1, 16, 2048))
    regular_time = np.linspace(timestamp[0], timestamp[0] + duration, regular_length)
    regular = np.column_stack(
        [np.interp(regular_time, timestamp, stream[:, column]) for column in range(stream.shape[1])]
    )
    centered = regular - regular.mean(axis=0)
    power = np.abs(np.fft.rfft(centered, axis=0)) ** 2
    frequencies = np.fft.rfftfreq(regular_length, d=duration / max(regular_length - 1, 1))
    power = power[1:]
    frequencies = frequencies[1:]
    total = power.sum(axis=0)
    normalized = np.divide(
        power, total, out=np.zeros_like(power), where=total[np.newaxis, :] > 1e-12
    )
    bands = ((0.0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, np.inf))
    for band_index, (start, stop) in enumerate(bands):
        selected = (frequencies >= start) & (frequencies < stop)
        vector = normalized[selected].sum(axis=0) if selected.any() else np.zeros(stream.shape[1])
        values.extend(vector.tolist())
        names.extend(f"{device}_physical_band{band_index}_{channel}" for channel in CHANNELS)
    if len(power):
        dominant = frequencies[np.argmax(power, axis=0)]
        centroid = (normalized * frequencies[:, np.newaxis]).sum(axis=0)
        entropy = -(normalized * np.log(normalized + 1e-12)).sum(axis=0)
    else:
        dominant = centroid = entropy = np.zeros(stream.shape[1])
    for statistic, vector in (
        ("physical_dominant_hz", dominant),
        ("physical_spectral_centroid_hz", centroid),
        ("physical_spectral_entropy", entropy),
    ):
        values.extend(vector.tolist())
        names.extend(f"{device}_{statistic}_{channel}" for channel in CHANNELS)

    for start, stop, label in ((0, 3, "acc"), (3, 6, "gyro"), (9, 12, "mag"), (12, 16, "quat")):
        magnitude = np.linalg.norm(stream[:, start:stop], axis=1)
        if valid_delta.any():
            speed = np.diff(magnitude)[valid_delta] / delta_time[valid_delta]
        else:
            speed = np.zeros(1)
        aggregate = (
            np.mean(np.abs(speed)),
            np.std(speed),
            np.quantile(np.abs(speed), 0.90),
            np.max(np.abs(speed)),
        )
        values.extend(aggregate)
        names.extend(
            f"{device}_{label}_{statistic}"
            for statistic in (
                "physical_diff_mean_abs",
                "physical_diff_std",
                "physical_diff_q90_abs",
                "physical_diff_max_abs",
            )
        )
    return values, names


def extract_timed_features(
    streams: dict[str, TimedStream]
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Append pace-aware v2 features to the reproducible v1 representation."""

    base_values, base_names = extract_features(
        {device: stream for device, (_, stream) in streams.items()}
    )
    values = base_values.tolist()
    names = list(base_names)
    dummy_time = np.asarray([0.0, 0.1])
    dummy_stream = np.zeros((2, len(CHANNELS)), dtype=np.float64)
    for device in DEVICES:
        timestamp, stream = streams.get(device, (dummy_time, dummy_stream))
        stream_values, stream_names = _timed_stream_features(timestamp, stream, device)
        values.extend(stream_values)
        names.extend(stream_names)
    result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("timed IMU feature vector contains a non-finite value")
    return result, tuple(names)


def extract_unit_features(unit_dir: Path) -> tuple[np.ndarray, tuple[str, ...]]:
    return extract_features(load_unit_streams(unit_dir / "IMU"))


def extract_unit_timed_features(unit_dir: Path) -> tuple[np.ndarray, tuple[str, ...]]:
    return extract_timed_features(load_unit_timed_streams(unit_dir / "IMU"))
