"""Use Skeleton evidence within its class support without erasing unseen actions."""

from __future__ import annotations

import numpy as np

from cuhkx.harn_nested import decode_questions
from cuhkx.harn_seed_ensemble import validate_evidence


def supported_evidence(evidence, weight):
    """Preserve visual mass outside the shared class support exactly."""
    if not 0 <= weight <= 1:
        raise ValueError("weight must be between zero and one")
    validate_evidence(evidence)
    result = {key: value.copy() for key, value in evidence.items()}
    classes = np.intersect1d(evidence["skeleton_classes"], evidence["visual_classes"])
    if not len(classes) or weight == 0:
        return result
    rows = np.flatnonzero(evidence["skeleton_present"] & evidence["visual_present"])
    visual = evidence["visual_probabilities"][np.ix_(rows, classes)]
    skeleton = evidence["skeleton_probabilities"][np.ix_(rows, classes)]
    skeleton_mass = skeleton.sum(axis=1, keepdims=True)
    usable = skeleton_mass[:, 0] > 0
    conditional = skeleton[usable] / skeleton_mass[usable]
    updated = (1 - weight) * visual[usable] + weight * visual[usable].sum(axis=1, keepdims=True) * conditional
    result["visual_probabilities"][np.ix_(rows[usable], classes)] = updated
    return result


def decode_supported(qa, evidence, dictionary, mode, weight, *, training_paths):
    """Only replace the old unsupported-Skeleton route; retain all other routes."""
    original = decode_questions(qa, evidence, dictionary, weight, training_paths=training_paths)
    if mode == "baseline":
        return original
    if mode != "supported":
        raise ValueError("unknown fusion mode")
    eligible = original.skeleton_present & original.visual_present & ~original.skeleton_fully_supported
    if not eligible.any():
        return original
    fused = decode_questions(
        qa, supported_evidence(evidence, weight), dictionary, weight,
        training_paths=training_paths,
    )
    fused.loc[eligible & ~fused.unique_pair_applied, "evidence_source"] = "support_preserving_class_fusion"
    original.loc[eligible, original.columns] = fused.loc[eligible, original.columns]
    return original
