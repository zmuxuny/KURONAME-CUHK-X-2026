"""Subject-matched R34 temporal inference with explicit single-view fallback."""

from pathlib import Path

import numpy as np

from cuhkx.r34_tail import MODALITIES, PairedRows, configure_torch, reduce_views, tail_network
from cuhkx.r34_video import R34FeatureBank
from cuhkx.sequence_skeleton_nested import validate_target
from cuhkx.sequence_tail_transfer import temporal_evidence

__all__ = ["WindowBank", "select_views", "temporal_evidence", "window_probabilities"]


def select_views(metadata, lookup, unit, *, allow_duplicate_windows=False):
    """Use aligned paired views, otherwise prefer the complete IR timeline."""
    rows = np.asarray([[lookup[(unit, modality, window)] for modality in MODALITIES]
                       for window in range(8)], dtype=np.int64)
    available = metadata["available"][rows]
    complete = available.all(axis=0)
    use_pair = bool(complete.all() and metadata["alignment_ok"][rows].all())
    if use_pair:
        mask = np.ones((8, 2), dtype=bool)
        source = "aligned_Depth_Color_IR"
    elif complete[1]:
        mask = np.tile([False, True], (8, 1))
        source = "IR_only"
    elif complete[0]:
        mask = np.tile([True, False], (8, 1))
        source = "Depth_Color_only"
    else:
        return None, None, None, "no_complete_temporal_view"
    selected = rows[:, mask[0]]
    fps, counts, frames = (metadata[key][selected] for key in ("fps", "frame_count", "frame_indices"))
    if (not np.isfinite(fps).all() or (fps <= 0).any() or (counts <= 0).any()
            or (frames < 0).any() or (frames >= counts[..., None]).any()
            or (np.diff(frames, axis=-1) < 0).any()):
        raise ValueError("invalid R34 available window frame metadata")
    duration = counts / fps
    starts, stops = frames[..., 0] / fps, (frames[..., -1] + 1) / fps
    centers = (starts + stops) / 2
    for key, expected in (("duration_seconds", duration), ("starts_seconds", starts),
                          ("stops_seconds", stops), ("centers_seconds", centers),
                          ("starts", starts / duration), ("stops", stops / duration),
                          ("centers", centers / duration)):
        if not np.array_equal(metadata[key][selected], expected):
            raise ValueError("R34 timing contradicts sampled physical frame indices")
        # Physical fields establish pairing; normalized fields use each view's duration.
        if (use_pair and key.endswith("seconds")
                and np.abs(expected[:, 0] - expected[:, 1]).max() > max(1 / fps.ravel()) + 1e-6):
            raise ValueError("R34 paired alignment mask contradicts physical timestamps")
    times = np.stack([metadata[key][selected].mean(axis=1) for key in ("starts", "stops", "centers")], axis=1)
    if not allow_duplicate_windows and len(set(zip(times[:, 0], times[:, 1], strict=True))) != 8:
        return None, None, None, "duplicate_physical_windows"
    return rows, mask, times, source


class WindowBank:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.raw = R34FeatureBank.load(path)
        self.manifest = self.raw.manifest
        metadata = self.raw.metadata
        units = sorted(set(metadata["units"][metadata["sources"] == "HAU"].astype(str)))
        self.units = np.asarray(units, dtype=str)
        self.timing, self.invalid, self.indices, self.available, self.view_sources = {}, {}, {}, {}, {}
        for unit in units:
            rows, mask, times, source = select_views(metadata, self.raw.row_index, unit)
            if rows is None:
                self.invalid[unit] = source
            else:
                self.indices[unit], self.available[unit], self.timing[unit] = rows, mask, times
                self.view_sources[unit] = source


def window_probabilities(model, bank, target, training_subjects):
    validate_target(target, training_subjects)
    units = list(dict.fromkeys(unit for unit in target.unit if unit in bank.timing))
    shape = (len(units), 8, len(model.classes_))
    if not units:
        return {"units": np.asarray([], dtype=str), "classes": model.classes_,
                "probabilities": np.empty(shape, dtype=np.float32)}
    matrix = PairedRows(bank.raw.features, np.concatenate([bank.indices[unit] for unit in units]),
                        np.concatenate([bank.available[unit] for unit in units]))
    torch = configure_torch()
    network = tail_network()
    network.load_state_dict({key: torch.from_numpy(value) for key, value in model.state.items()})
    network.eval().cuda()
    pieces = []
    with torch.inference_mode():
        for start in range(0, len(matrix), model.settings["batch_units"]):
            batch = matrix[start:start + model.settings["batch_units"]]
            owners = np.nonzero(batch.available)[0]
            tensor = torch.from_numpy(batch.values[batch.available].copy()).to("cuda", dtype=torch.float32)
            with torch.autocast(device_type="cuda"):
                logits = network(tensor)
            probabilities = logits.float()[:, model.classes_].softmax(dim=1).cpu().numpy()
            pieces.append(reduce_views(probabilities, owners, len(batch)))
    del network
    torch.cuda.empty_cache()
    return {"units": np.asarray(units, dtype=str), "classes": model.classes_,
            "probabilities": np.concatenate(pieces).reshape(shape).astype(np.float32)}
