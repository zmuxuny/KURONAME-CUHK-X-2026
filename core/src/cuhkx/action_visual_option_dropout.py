"""Transparent per-option visual summaries from a subject-excluded action head."""
from __future__ import annotations

import numpy as np

from cuhkx.actions import HAU_TEXT_CLASS
from cuhkx.metadata import option_atoms

FEATURES = ('known_logodds_min', 'known_logodds_mean', 'known_logodds_max', 'known_logodds_sum',
            'known_atom_count', 'known_atom_fraction', 'IR_window_fraction', 'DepthColor_window_fraction',
            'IR_present', 'DepthColor_present')


def option_visual_features(option, probabilities, known_classes, availability):
    atoms = option_atoms(option)
    classes = [HAU_TEXT_CLASS[atom] for atom in atoms]
    supported = [index for index in classes if index in set(known_classes)]
    p = np.clip(np.asarray(probabilities)[supported], 1e-5, 1 - 1e-5)
    logits = np.log(p / (1 - p))
    summary = [logits.min(), logits.mean(), logits.max(), logits.sum()] if len(logits) else [0.] * 4
    availability = np.asarray(availability, dtype=float)
    assert availability.shape == (2,) and np.all((availability >= 0) & (availability <= 1))
    result = np.asarray([*summary, len(supported), len(supported) / len(classes),
                         *availability, *(availability > 0)], dtype=np.float32)
    assert result.shape == (10,) and np.isfinite(result).all()
    return result
