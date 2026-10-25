"""Partial-label HAU supervision over eight visual windows per recording."""

from __future__ import annotations

import numpy as np

from cuhkx.metadata import subject_id

MODALITIES = ("Depth_Color", "IR")


def window_rows(bank, units):
    indices = np.full((len(units), 8, 2), -1, dtype=np.int64)
    available = np.zeros_like(indices, dtype=bool)
    for i, unit in enumerate(units):
        for window in range(8):
            for view, modality in enumerate(MODALITIES):
                row = bank.row_index.get((unit, modality, window), -1)
                indices[i, window, view] = row
                available[i, window, view] = row >= 0 and bool(bank.metadata["available"][row])
            if available[i, window].all() and not bank.metadata["alignment_ok"][indices[i, window]].all():
                available[i, window, 0] = False
        if not available[i].any():
            raise ValueError("HAU MIL recording has no available window")
    return indices, available


def check_fit_scope(units, train_subjects, validation_subjects):
    from cuhkx.emotion_nested import OFFICIAL_SUBJECTS

    train, validation = set(train_subjects), set(validation_subjects)
    if not train or train & validation or not train | validation <= OFFICIAL_SUBJECTS:
        raise ValueError("invalid HAU MIL subject split")
    if len(set(units)) != len(units) or any(not unit.startswith("Training/HAU/") for unit in units):
        raise ValueError("HAU MIL fitting requires unique official Training HAU units")
    if {subject_id(unit) for unit in units} != train:
        raise ValueError("actual HAU MIL fitting people differ from the declared scope")


def positive_weights(targets):
    values = np.asarray(targets)
    if values.ndim != 2 or values.shape[1] != 40 or not np.isin(values, [-1, 0, 1]).all():
        raise ValueError("HAU MIL targets must be40 known/unknown action cells")
    if not (values != -1).any(axis=1).all():
        raise ValueError("HAU MIL training clip has no known labels")
    positive, negative = (values == 1).sum(axis=0), (values == 0).sum(axis=0)
    weights = np.clip((negative + 1) / (positive + 1), .25, 4).astype(np.float32)
    return weights, (positive > 0) & (negative > 0)


def pool_logits(logits, available, temperature):
    """Average aligned views, then normalized smooth maximum over visible windows."""
    import torch

    if logits.ndim != 4 or logits.shape[1:] != (8, 2, 40) or available.shape != logits.shape[:3]:
        raise ValueError("HAU MIL needs batch/window/view/action axes")
    if temperature <= 0 or not available.any(dim=(1, 2)).all():
        raise ValueError("HAU MIL pooling needs positive temperature and visible windows")
    views = available.sum(dim=2)
    pooled_views = (logits * available[..., None]).sum(dim=2) / views.clamp_min(1)[..., None]
    window_mask = views > 0
    scaled = (pooled_views / temperature).masked_fill(~window_mask[..., None], -torch.inf)
    return temperature * (torch.logsumexp(scaled, dim=1) - window_mask.sum(dim=1).log()[:, None])


def masked_loss(logits, available, targets, weights, temperature):
    from torch.nn import functional

    pooled = pool_logits(logits, available, temperature)
    known = targets != -1
    if targets.shape != pooled.shape or not known.any(dim=1).all():
        raise ValueError("HAU MIL loss needs aligned nonempty partial labels")
    per_cell = functional.binary_cross_entropy_with_logits(pooled, targets.clamp_min(0),
                                                          pos_weight=weights, reduction="none")
    return ((per_cell * known).sum(dim=1) / known.sum(dim=1)).mean()


def network(weights=None):
    import torch

    from cuhkx.r34_tail import tail_network

    result = tail_network(weights)
    result[3] = torch.nn.Linear(512, 40)
    return result
