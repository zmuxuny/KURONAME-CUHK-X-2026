"""Option scoring from time-resolved 40-class action probabilities."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence

import numpy as np


def softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = np.asarray(values, dtype=np.float64) / temperature
    scaled -= scaled.max(axis=-1, keepdims=True)
    exponentials = np.exp(scaled)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def temporal_presence(probabilities: np.ndarray, top_windows: int = 2) -> np.ndarray:
    """Pool each action over its strongest temporal windows."""

    if probabilities.ndim != 2 or not 1 <= top_windows <= probabilities.shape[0]:
        raise ValueError("expected [segments, classes] probabilities and a valid pool size")
    selected = np.partition(probabilities, -top_windows, axis=0)[-top_windows:]
    return selected.mean(axis=0)


def normalized_score_margin(scores: Mapping[str, float]) -> tuple[str, float, float]:
    """Decode arbitrary option scores after converting them to a simplex."""

    if not scores:
        raise ValueError("cannot decode empty scores")
    labels = sorted(scores)
    values = np.asarray([scores[label] for label in labels], dtype=np.float64)
    probabilities = softmax(values)
    ranked = sorted(zip(labels, probabilities, strict=True), key=lambda item: (-item[1], item[0]))
    margin = float(ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else float(ranked[0][1])
    return ranked[0][0], float(ranked[0][1]), margin


def monotonic_path_score(log_emissions: np.ndarray, order: Sequence[int]) -> float:
    """Best strictly increasing segment assignment for one class order."""

    segments, _ = log_emissions.shape
    if not order or segments < len(order):
        raise ValueError("the timeline must have at least one segment per ordered class")
    dynamic = log_emissions[:, int(order[0])].copy()
    for class_id in order[1:]:
        previous = dynamic.copy()
        prefix = np.maximum.accumulate(previous)
        dynamic.fill(-np.inf)
        dynamic[1:] = prefix[:-1] + log_emissions[1:, int(class_id)]
    return float(dynamic.max())


def temporal_order_scores(
    probabilities: np.ndarray,
    letter_classes: Mapping[str, int],
    *,
    normalize_time: bool = True,
) -> dict[str, float]:
    """Score every letter permutation with monotonic temporal alignment."""

    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [segments, classes]")
    letters = sorted(letter_classes)
    if len(letters) > probabilities.shape[0]:
        raise ValueError("not enough temporal segments for all options")
    emissions = np.asarray(probabilities, dtype=np.float64).clip(1e-12)
    if normalize_time:
        emissions = emissions / emissions.sum(axis=0, keepdims=True).clip(1e-12)
    log_emissions = np.log(emissions)
    scores = {}
    for permutation in itertools.permutations(letters):
        order = [int(letter_classes[letter]) for letter in permutation]
        scores["".join(permutation)] = monotonic_path_score(log_emissions, order)
    return scores
