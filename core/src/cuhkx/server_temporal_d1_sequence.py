"""Matched D1 coarse/fine temporal curves for frozen action checkpoints.

C0 and F1 use the same server-runtime raw bank.  Curves are produced from
per-bin logits before MIL pooling, with the stored positive-weight correction.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.special import expit

from cuhkx.pairwise_temporal import PAIR_INDICES, pair_curve_features
from cuhkx.sequence_skeleton_nested import temporal_before_probability
from cuhkx.server_temporal_d1_head import MODALITIES, unpack_d1


def read_d1_centers(directory: Path, split: str, mode: str) -> dict[tuple[str, str], np.ndarray]:
    directory = Path(directory)
    with np.load(directory / "metadata.npz", allow_pickle=False) as archive:
        units = archive["units"].astype(str)
        modalities = archive["modalities"].astype(str)
        splits = archive["split"].astype(str)
        available = archive["available"].astype(bool)
        fine = archive["fine_normalized"].astype(np.float64)
    keep = np.flatnonzero(splits == split)
    result: dict[tuple[str, str], np.ndarray] = {}
    for row in keep:
        if not available[row]:
            continue
        axis = fine[row]
        if axis.shape != (16, 8) or not np.isfinite(axis).all():
            raise ValueError("D1 fine normalized time axis is invalid")
        if (axis < 0).any() or (axis > 1).any() or (np.diff(axis, axis=1) < 0).any():
            raise ValueError("D1 fine centers must be bounded and monotone within each source window")
        if mode == "coarse":
            times = axis.mean(axis=1)
            if (np.diff(times) < 0).any():
                raise ValueError("D1 coarse physical centers are nonmonotone")
        elif mode == "fine":
            # Source windows overlap in physical time, so flattening [window,bin]
            # is intentionally not globally monotone. d1_window_curves groups equal
            # physical centers and sorts them before interpolation.
            times = axis.reshape(-1)
        else:
            raise ValueError("D1 mode must be coarse/fine")
        result[(units[row], modalities[row])] = times
    return result


def d1_window_curves(head, matrix, units, centers, *, target_points: int = 128):
    """Project one frozen D1 action head to per-action physical-time curves."""
    import torch

    features, mask = unpack_d1(matrix, head.bins)
    normalized = head._standardized(features, mask)
    with torch.inference_mode():
        hidden = torch.nn.functional.gelu(torch.nn.functional.linear(
            torch.from_numpy(normalized),
            torch.from_numpy(head.state["0.weight"]),
            torch.from_numpy(head.state["0.bias"]),
        ))
        logits = torch.nn.functional.linear(
            hidden,
            torch.from_numpy(head.state["3.weight"]),
            torch.from_numpy(head.state["3.bias"]),
        ).numpy()
    logits = logits.astype(np.float64) - np.log(head.positive_weight)
    probability = expit(logits)
    probability[..., ~head.supported] = 0.5
    target = np.linspace(0.0, 1.0, target_points)
    result = np.zeros((len(units), target_points, 40), dtype=np.float64)
    for index, unit in enumerate(map(str, units)):
        views = []
        for view, modality in enumerate(MODALITIES):
            present = mask[index, :, view]
            if not present.any():
                continue
            times = np.asarray(centers[(unit, modality)], dtype=np.float64)[present]
            values = probability[index, present, view]
            unique = np.unique(times)
            values = np.stack([values[times == time].mean(axis=0) for time in unique])
            views.append(np.stack([
                np.interp(target, unique, values[:, action]) for action in range(40)
            ], axis=1))
        if not views:
            raise ValueError("D1 sequence curve requires one available view")
        result[index] = np.mean(views, axis=0)
    if not np.isfinite(result).all():
        raise ValueError("D1 sequence curve is nonfinite")
    return result


def d1_pair_inputs(frame, curves):
    """Use 128-point timing for moments/mass while pair shapes stay resampled to 16."""
    forward, reverse, mass = [], [], []
    curves = np.asarray(curves, dtype=np.float64)
    if curves.ndim != 3 or curves.shape[1:] != (128, 40):
        raise ValueError("D1 curves must have shape [rows,128,40]")
    physical = np.linspace(0.0, 1.0, 128)
    for row, curve in zip(frame.itertuples(), curves, strict=True):
        local = []
        for left, right in PAIR_INDICES:
            a, b = row.classes[left], row.classes[right]
            forward.append(pair_curve_features([curve], a, b, resample_length=16))
            reverse.append(pair_curve_features([curve], b, a, resample_length=16))
            local.append(temporal_before_probability(
                physical, curve[:, a], curve[:, b], mode="mass_before"
            ))
        mass.append(local)
    return {
        "forward": np.asarray(forward),
        "reverse": np.asarray(reverse),
        "mass": np.asarray(mass),
    }
