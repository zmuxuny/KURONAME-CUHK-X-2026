"""Assemble strict keyed VideoMAE Train or Test windows without changing frozen banks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cuhkx.action_window_head import CHANNELS, FEATURE_DIMENSION, VIEWS, WINDOWS
from cuhkx.validation import sha256_file


def read_window_bank(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest["kind"] not in {"cuhkx_videomae_ssv2_hau_train_windows_v1", "cuhkx_videomae_ssv2_hau_test_windows_v1"}
            or manifest["labels_used"] or manifest["windows"] != WINDOWS):
        raise ValueError("not the declared label-free VideoMAE window bank")
    for entry in manifest["files"]:
        path = directory / entry["path"]
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError("VideoMAE window bank changed")
    completed = np.load(directory / "completed.npy", allow_pickle=False)
    features = np.load(directory / "features.npy", mmap_mode="r", allow_pickle=False)
    with np.load(directory / "metadata.npz", allow_pickle=False) as archive:
        metadata = {key: archive[key] for key in archive.files}
    units, modalities, visible, alignment = [metadata[key] for key in ["units", "modalities", "available", "alignment_ok"]]
    if (features.shape != (len(units), WINDOWS, CHANNELS) or not completed.all()
            or completed.shape != (len(units),) or visible.dtype != bool or alignment.dtype != bool
            or metadata["frame_indices"].shape != (len(units), WINDOWS, 16)
            or not np.isfinite(features).all()):
        raise ValueError("incomplete or malformed VideoMAE descriptors")
    lookup = {(str(unit), str(modality)): i for i, (unit, modality) in enumerate(zip(units, modalities, strict=True))}
    if len(lookup) != len(units) or set(modalities) != {"Depth_Color", "IR"}:
        raise ValueError("duplicated or unknown VideoMAE modality keys")
    ordered = sorted(set(units.astype(str)))
    values = np.zeros((len(ordered), WINDOWS, VIEWS, CHANNELS), dtype=np.float32)
    mask = np.zeros((len(ordered), WINDOWS, VIEWS), dtype=bool)
    for i, unit in enumerate(ordered):
        rows = [lookup[(unit, modality)] for modality in ("Depth_Color", "IR")]
        for view, row in enumerate(rows):
            values[i, :, view] = features[row]
            mask[i, :, view] = visible[row]
        if visible[rows].all() and not alignment[rows].all():
            mask[i, :, 0] = False
    matrix = np.concatenate((values.reshape(len(ordered), -1), mask.reshape(len(ordered), -1)), axis=1)
    assert matrix.shape == (len(ordered), FEATURE_DIMENSION)
    names = np.asarray([f"videomae_w{w}_{view}_{c:03d}" for w in range(WINDOWS)
                        for view in ("depthcolor", "ir") for c in range(CHANNELS)]
                       + [f"available_w{w}_{view}" for w in range(WINDOWS) for view in ("depthcolor", "ir")])
    return {"X": matrix, "units": np.asarray(ordered), "feature_names": names,
            "index": {unit: i for i, unit in enumerate(ordered) if mask[i].any()},
            "source_manifest_sha256": sha256_file(directory / "manifest.json")}
