"""Path and text normalization shared across CUHK-X pipelines."""

from __future__ import annotations

import re
from pathlib import Path

MODALITIES = {"depth", "depth_color", "ir", "thermal", "imu", "skeleton", "radar"}


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def option_atoms(value: object) -> tuple[str, ...]:
    return tuple(
        normalize_text(piece) for piece in str(value).split(",") if str(piece).strip()
    )


def clip_key(path: object) -> str:
    parts = [piece for piece in str(path).replace("\\", "/").split("/") if piece]
    if len(parts) >= 2:
        modality = parts[-2].lower()
        stem = Path(parts[-1]).stem.lower()
        if modality in MODALITIES and stem in MODALITIES:
            parts = parts[:-2]
    return "/".join(parts)


def subject_id(path: object) -> str | None:
    match = re.search(r"(?:^|/)user(\d+)(?:/|$)", str(path).replace("\\", "/"))
    return f"user{int(match.group(1))}" if match else None


def feature_unit(path: object, *, training: bool) -> str:
    """Map a QA media path to the corresponding extracted sensor unit.

    Training QA paths already stop at the recording directory, while test
    paths include a final ``<Modality>/<Modality>.mp4`` pair.  ``clip_key``
    normalizes that difference before the on-disk split prefix is added.
    """

    key = clip_key(path)
    if training:
        if not key.startswith(("HAU/", "HARn/")):
            raise ValueError(f"unexpected training path: {path!r}")
        return f"Training/{key}"
    return f"Testing/{key}"
