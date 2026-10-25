"""Deterministic video discovery and uniform frame sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def uniform_frame_indices(frame_count: int, sample_count: int) -> np.ndarray:
    if frame_count <= 0 or sample_count <= 0:
        raise ValueError("frame_count and sample_count must be positive")
    if frame_count == 1:
        return np.zeros(sample_count, dtype=int)
    return np.rint(np.linspace(0, frame_count - 1, sample_count)).astype(int)


def temporal_window_indices(
    frame_count: int,
    fps: float,
    window_count: int,
    sample_count: int,
    window_seconds: float,
) -> np.ndarray:
    """Return centered fixed-duration sampling windows in chronological order."""

    if frame_count <= 0 or fps <= 0 or window_count <= 0 or sample_count <= 0:
        raise ValueError("frame metadata and sample counts must be positive")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    span = min(frame_count, max(1, round(fps * window_seconds)))
    centers = (np.arange(window_count, dtype=np.float64) + 0.5) / window_count
    centers *= frame_count - 1
    starts = np.rint(centers - (span - 1) / 2.0).astype(int)
    starts = np.clip(starts, 0, frame_count - span)
    offsets = uniform_frame_indices(span, sample_count)
    return starts[:, None] + offsets[None, :]


def media_unit(path: Path, media_root: Path) -> str:
    """Return the unit before the final ``Modality/file.mp4`` pair."""

    relative = path.resolve().relative_to(media_root.resolve())
    if len(relative.parts) < 3:
        raise ValueError(f"video is not nested below a unit: {path}")
    return "/".join(relative.parts[:-2])


def crop_video_frames(
    frames: np.ndarray, window: tuple[float, float, float, float]
) -> np.ndarray:
    """Apply one normalized fixed window to a ``[T,H,W,C]`` frame tensor."""

    if frames.ndim != 4:
        raise ValueError("frames must have shape [T,H,W,C]")
    x0, y0, x1, y1 = window
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError(f"invalid normalized crop window: {window}")
    height, width = frames.shape[1:3]
    left, right = round(x0 * width), round(x1 * width)
    top, bottom = round(y0 * height), round(y1 * height)
    if left >= right or top >= bottom:
        raise ValueError(f"crop window is empty after rounding: {window}")
    return frames[:, top:bottom, left:right]


def discover_videos(
    media_root: Path,
    modalities: tuple[str, ...],
    prefixes: tuple[str, ...],
) -> list[tuple[str, str, Path]]:
    records: list[tuple[str, str, Path]] = []
    for modality in modalities:
        for path in media_root.rglob(f"{modality}/{modality}.mp4"):
            unit = media_unit(path, media_root)
            if prefixes and not unit.startswith(prefixes):
                continue
            records.append((unit, modality, path))
    records.sort(key=lambda item: (item[0], item[1], str(item[2])))
    if len({(unit, modality) for unit, modality, _ in records}) != len(records):
        raise ValueError("duplicate unit/modality videos discovered")
    return records


def sample_video_frames(path: Path, sample_count: int) -> np.ndarray:
    """Read uniformly spaced RGB frames with OpenCV, failing closed."""

    try:
        import cv2
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise RuntimeError("opencv-python-headless is required for video sampling") from error

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        raise ValueError(f"video reports no frames: {path}")
    indices = uniform_frame_indices(frame_count, sample_count)
    return _decode_video_indices(capture, path, indices)


def sample_video_windows(
    path: Path,
    window_count: int,
    sample_count: int,
    window_seconds: float,
) -> np.ndarray:
    """Read several fixed-duration windows while decoding the MP4 only once."""

    try:
        import cv2
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise RuntimeError("opencv-python-headless is required for video sampling") from error

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if frame_count <= 0 or fps <= 0:
        capture.release()
        raise ValueError(f"video reports invalid metadata: {path}")
    indices = temporal_window_indices(
        frame_count, fps, window_count, sample_count, window_seconds
    )
    return _decode_video_indices(capture, path, indices)


def _decode_video_indices(capture, path: Path, indices: np.ndarray) -> np.ndarray:
    """Decode an arbitrary index tensor with one forward pass through a video."""

    try:
        import cv2
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise RuntimeError("opencv-python-headless is required for video sampling") from error

    flat_indices = np.asarray(indices, dtype=int).reshape(-1)
    if len(flat_indices) == 0 or flat_indices.min() < 0:
        capture.release()
        raise ValueError("frame indices must be non-empty and non-negative")
    positions: dict[int, list[int]] = {}
    for output_index, frame_index in enumerate(flat_indices):
        positions.setdefault(int(frame_index), []).append(output_index)
    frames: list[np.ndarray | None] = [None] * len(flat_indices)
    for frame_index in range(int(flat_indices.max()) + 1):
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            raise ValueError(f"failed to decode frame {frame_index} from {path}")
        if frame_index in positions:
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for output_index in positions[frame_index]:
                frames[output_index] = converted
    capture.release()
    if any(frame is None for frame in frames):  # pragma: no cover - defensive codec guard
        raise ValueError(f"not all requested frames were decoded from {path}")
    stacked = np.stack(frames)
    return stacked.reshape((*np.asarray(indices).shape, *stacked.shape[1:]))
