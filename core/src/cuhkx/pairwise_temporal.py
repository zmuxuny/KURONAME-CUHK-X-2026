"""Pairwise temporal features and constrained four-action order decoding."""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np

PAIR_INDICES = tuple(itertools.combinations(range(4), 2))
PERMUTATIONS = tuple(itertools.permutations(range(4)))


def _resample(values: np.ndarray, length: int = 16) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("a temporal curve must be a nonempty vector")
    if length < 2:
        raise ValueError("resample length must be at least two")
    return np.interp(
        np.linspace(0.0, 1.0, length),
        np.linspace(0.0, 1.0, len(values)),
        values,
    )


def pair_curve_features(
    probability_sources: Sequence[np.ndarray],
    left_class: int,
    right_class: int,
    *,
    class_count: int = 40,
    resample_length: int = 16,
) -> np.ndarray:
    """Describe the relative timing of two actions across temporal scales.

    Each source has shape ``[segments, classes]``.  The final class one-hot
    terms let the subject-disjoint learner account for systematic detector and
    routine-order biases without ever using a validation answer.
    """

    if not 0 <= left_class < class_count or not 0 <= right_class < class_count:
        raise ValueError("action class is outside the configured vocabulary")
    if left_class == right_class:
        raise ValueError("a temporal pair must contain two different actions")

    features: list[float] = []
    for source in probability_sources:
        source = np.asarray(source, dtype=np.float64)
        if source.ndim != 2 or source.shape[1] != class_count:
            raise ValueError("probability source must have shape [segments, classes]")
        if not np.isfinite(source).all() or (source < 0).any():
            raise ValueError("probability source must be finite and nonnegative")
        left_raw = source[:, left_class]
        right_raw = source[:, right_class]
        left = left_raw / max(float(left_raw.sum()), 1e-12)
        right = right_raw / max(float(right_raw.sum()), 1e-12)
        time = np.linspace(0.0, 1.0, len(left))
        left_resampled = _resample(left, resample_length)
        right_resampled = _resample(right, resample_length)
        left_raw_resampled = _resample(left_raw, resample_length)
        right_raw_resampled = _resample(right_raw, resample_length)
        left_cdf = np.cumsum(left)
        right_cdf = np.cumsum(right)
        left_median = time[min(int(np.searchsorted(left_cdf, 0.5)), len(time) - 1)]
        right_median = time[min(int(np.searchsorted(right_cdf, 0.5)), len(time) - 1)]
        left_before = sum(
            left[index] * right[index + 1 :].sum() for index in range(len(left))
        )
        right_before = sum(
            right[index] * left[index + 1 :].sum() for index in range(len(left))
        )
        features.extend(left_resampled)
        features.extend(right_resampled)
        features.extend(left_resampled - right_resampled)
        features.extend(left_raw_resampled)
        features.extend(right_raw_resampled)
        left_expectation = float(left @ time)
        right_expectation = float(right @ time)
        features.extend(
            (
                left_expectation,
                right_expectation,
                right_expectation - left_expectation,
                float(time[int(np.argmax(left))]),
                float(time[int(np.argmax(right))]),
                float(left_median),
                float(right_median),
                float(left_before),
                float(right_before),
                float(left_before - right_before),
                float(left_raw.max()),
                float(right_raw.max()),
                float(left_raw.mean()),
                float(right_raw.mean()),
            )
        )

    identity = np.zeros(class_count * 2, dtype=np.float32)
    identity[left_class] = 1.0
    identity[class_count + right_class] = 1.0
    result = np.concatenate((np.asarray(features, dtype=np.float32), identity))
    if not np.isfinite(result).all():
        raise ValueError("pairwise temporal feature vector is non-finite")
    return result


def before_counts(
    ordered_classes: Sequence[Sequence[int]],
) -> Counter[tuple[int, int]]:
    """Count ordered class pairs in a collection of ground-truth sequences."""

    result: Counter[tuple[int, int]] = Counter()
    for order in ordered_classes:
        for left_index, left in enumerate(order):
            for right in order[left_index + 1 :]:
                result[(int(left), int(right))] += 1
    return result


def decode_pairwise_order(
    pair_probabilities: Sequence[float],
    option_classes: Sequence[int],
    counts: Mapping[tuple[int, int], int],
    *,
    prior_weight: float = 1.0,
    prior_smoothing: float = 1.5,
) -> tuple[str, float]:
    """Find the best legal permutation under pair evidence and order priors."""

    probabilities = np.asarray(pair_probabilities, dtype=np.float64)
    classes = tuple(int(value) for value in option_classes)
    if probabilities.shape != (len(PAIR_INDICES),):
        raise ValueError("expected one probability for each of the six option pairs")
    if len(classes) != 4 or len(set(classes)) != 4:
        raise ValueError("sequence decoding requires four distinct action classes")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (
        probabilities > 1
    ).any():
        raise ValueError("pair probabilities must be finite and in [0, 1]")
    if prior_weight < 0 or prior_smoothing <= 0:
        raise ValueError("prior weight must be nonnegative and smoothing positive")

    pair_lookup = dict(zip(PAIR_INDICES, probabilities, strict=True))
    ranked: list[tuple[float, str]] = []
    for permutation in PERMUTATIONS:
        score = 0.0
        for left_position, left_option in enumerate(permutation):
            for right_option in permutation[left_position + 1 :]:
                low, high = sorted((left_option, right_option))
                probability = float(pair_lookup[(low, high)])
                if left_option != low:
                    probability = 1.0 - probability
                score += np.log(np.clip(probability, 1e-5, 1.0))

                left_class = classes[left_option]
                right_class = classes[right_option]
                left_first = int(counts.get((left_class, right_class), 0))
                right_first = int(counts.get((right_class, left_class), 0))
                prior = (left_first + prior_smoothing) / (
                    left_first + right_first + 2.0 * prior_smoothing
                )
                score += prior_weight * np.log(prior)
        answer = "".join("ABCD"[index] for index in permutation)
        ranked.append((float(score), answer))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked[0][1], float(ranked[0][0] - ranked[1][0])
