"""Shared constraints for within-scenario HAU emotion decoding."""

from __future__ import annotations

import itertools

import numpy as np


def triplet_assignment_probabilities(
    probabilities: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """Condition three-way row probabilities on one use of each state per triplet.

    The result is the exact marginal under the six possible assignments, using
    the independent classifier probabilities as likelihoods.
    """

    probabilities = np.asarray(probabilities, dtype=np.float64)
    groups = np.asarray(groups)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError("probabilities must have shape (rows, 3)")
    if len(groups) != len(probabilities):
        raise ValueError("groups and probabilities have different row counts")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("probabilities must be finite and nonnegative")

    result = np.zeros_like(probabilities)
    permutations = np.asarray(list(itertools.permutations(range(3))), dtype=int)
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        if len(rows) != 3:
            raise ValueError(f"group {group!r} has {len(rows)} rows instead of three")
        local = np.clip(probabilities[rows], 1e-12, None)
        log_weights = np.asarray(
            [np.log(local[np.arange(3), permutation]).sum() for permutation in permutations]
        )
        weights = np.exp(log_weights - log_weights.max())
        weights /= weights.sum()
        for permutation, weight in zip(permutations, weights, strict=True):
            result[rows, permutation] += weight
    return result.astype(np.float32)
