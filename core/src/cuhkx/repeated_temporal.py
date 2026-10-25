"""Temporal evidence sharing across repeated HAU scenario recordings."""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np

from cuhkx.pairwise_temporal import PAIR_INDICES, decode_pairwise_order


def canonical_pair_layout(option_classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return canonical action-class keys and option-to-key orientations."""

    classes = np.asarray(option_classes, dtype=int)
    if classes.ndim != 2 or classes.shape[1] != 4:
        raise ValueError("option classes must have shape [questions, 4]")
    if any(len(set(row)) != 4 for row in classes):
        raise ValueError("every sequence question must contain four distinct classes")
    keys = np.empty((len(classes), len(PAIR_INDICES), 2), dtype=int)
    orientations = np.empty((len(classes), len(PAIR_INDICES)), dtype=np.int8)
    for question, row in enumerate(classes):
        for pair, (left, right) in enumerate(PAIR_INDICES):
            left_class, right_class = int(row[left]), int(row[right])
            keys[question, pair] = sorted((left_class, right_class))
            orientations[question, pair] = 1 if left_class < right_class else -1
    return keys, orientations


def share_repeated_pair_probabilities(
    probabilities: np.ndarray,
    option_classes: np.ndarray,
    groups: Sequence[Hashable],
    *,
    family_indices: Sequence[int],
    sibling_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool like-for-like action-pair evidence across scenario repetitions.

    Probabilities are first oriented to the canonical lower-class action.  For
    a pair also observed in sibling recordings, family-mean own log odds are
    combined with each sibling occurrence's log odds.  Pairs seen only once
    retain their plain family-mean probability.
    """

    values = np.asarray(probabilities, dtype=np.float64)
    classes = np.asarray(option_classes, dtype=int)
    group_values = np.asarray(groups, dtype=object)
    families = np.asarray(tuple(family_indices), dtype=int)
    if values.ndim != 3 or values.shape[1:] != (len(classes), len(PAIR_INDICES)):
        raise ValueError("probabilities must have shape [families, questions, 6]")
    if len(group_values) != len(classes):
        raise ValueError("group count must equal question count")
    if not len(families) or (families < 0).any() or (families >= len(values)).any():
        raise ValueError("family indices are empty or out of range")
    if sibling_weight < 0:
        raise ValueError("sibling weight must be nonnegative")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("pair probabilities must be finite and in [0, 1]")

    keys, orientations = canonical_pair_layout(classes)
    selected = values[families]
    canonical = np.where(orientations[None, :, :] > 0, selected, 1.0 - selected)
    occurrences: dict[tuple[Hashable, int, int], list[tuple[int, int]]] = {}
    for question, group in enumerate(group_values):
        for pair, key in enumerate(keys[question]):
            occurrences.setdefault((group, int(key[0]), int(key[1])), []).append(
                (question, pair)
            )

    epsilon = 1e-5

    def logit(items: np.ndarray) -> np.ndarray:
        clipped = np.clip(items, epsilon, 1.0 - epsilon)
        return np.log(clipped / (1.0 - clipped))

    pooled = np.empty((len(classes), len(PAIR_INDICES)), dtype=np.float64)
    shared_counts = np.zeros(len(classes), dtype=int)
    for question, group in enumerate(group_values):
        for pair, key in enumerate(keys[question]):
            own = canonical[:, question, pair]
            siblings = [
                (other_question, other_pair)
                for other_question, other_pair in occurrences[
                    (group, int(key[0]), int(key[1]))
                ]
                if other_question != question
            ]
            if siblings:
                sibling_values = np.asarray(
                    [canonical[:, other_question, other_pair] for other_question, other_pair in siblings]
                ).reshape(-1)
                combined_logit = float(logit(own).mean()) + sibling_weight * float(
                    logit(sibling_values).sum()
                ) / len(families)
                canonical_probability = 1.0 / (1.0 + np.exp(-combined_logit))
                shared_counts[question] += 1
            else:
                canonical_probability = float(own.mean())
            pooled[question, pair] = (
                canonical_probability
                if orientations[question, pair] > 0
                else 1.0 - canonical_probability
            )
    return pooled.astype(np.float32), shared_counts


def decode_repeated_pair_probabilities(
    probabilities: np.ndarray,
    option_classes: np.ndarray,
    counts_by_question: Sequence[dict[tuple[int, int], int]],
    *,
    prior_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode each pooled six-pair vector into one legal four-action order."""

    if len(probabilities) != len(option_classes) or len(counts_by_question) != len(
        option_classes
    ):
        raise ValueError("probabilities, classes, and priors must align by question")
    predictions: list[str] = []
    margins: list[float] = []
    for pair_values, classes, counts in zip(
        probabilities, option_classes, counts_by_question, strict=True
    ):
        prediction, margin = decode_pairwise_order(
            pair_values,
            classes,
            counts,
            prior_weight=prior_weight,
        )
        predictions.append(prediction)
        margins.append(margin)
    return np.asarray(predictions), np.asarray(margins)


def adjacent_swap_pair(
    parent: str, candidate: str, option_classes: Sequence[int]
) -> tuple[int, int] | None:
    """Return the canonical action pair if candidate is one adjacent swap away."""

    parent = str(parent)
    candidate = str(candidate)
    if sorted(parent) != list("ABCD") or sorted(candidate) != list("ABCD"):
        return None
    changed = [index for index, values in enumerate(zip(parent, candidate)) if values[0] != values[1]]
    if len(changed) != 2 or changed[1] != changed[0] + 1:
        return None
    left, right = changed
    if parent[left] != candidate[right] or parent[right] != candidate[left]:
        return None
    classes = tuple(int(value) for value in option_classes)
    pair = (
        classes["ABCD".index(parent[left])],
        classes["ABCD".index(parent[right])],
    )
    return tuple(sorted(pair))


def pair_relation(order: str, option_classes: Sequence[int], pair: tuple[int, int]) -> bool:
    """Return whether the lower canonical class occurs first in an answer order."""

    classes = tuple(int(value) for value in option_classes)
    positions = {classes["ABCD".index(letter)]: index for index, letter in enumerate(order)}
    if pair[0] not in positions or pair[1] not in positions:
        raise ValueError("requested action pair is absent from the sequence question")
    return positions[pair[0]] < positions[pair[1]]
