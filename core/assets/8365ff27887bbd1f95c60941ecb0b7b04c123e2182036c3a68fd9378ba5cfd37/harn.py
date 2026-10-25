"""Utilities for option-constrained HARn action classification."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from cuhkx.metadata import normalize_text
from cuhkx.submission import LETTERS, available_letters

ACTION_PATTERN = re.compile(r"(?:^|/)HARn/(\d+)_[^/]+(?:/|$)")


def action_class(path: object) -> int:
    """Return the organizer-provided numeric action class from a HARn path."""

    match = ACTION_PATTERN.search(str(path).replace("\\", "/"))
    if not match:
        raise ValueError(f"cannot parse HARn action class from {path!r}")
    return int(match.group(1))


def action_text_classes(training_qa: pd.DataFrame) -> dict[str, tuple[int, ...]]:
    """Build the option-text to action-class dictionary from labeled HARn QA."""

    rows = training_qa.loc[
        training_qa["source"].eq("HARn") & training_qa["category"].eq("single")
    ]
    mapping: defaultdict[str, set[int]] = defaultdict(set)
    for _, row in rows.iterrows():
        label = str(row["answer"]).strip().upper()
        if label not in LETTERS or not str(row.get(label, "")).strip():
            raise ValueError(f"invalid HARn answer for qa_id={row.get('qa_id')}")
        mapping[normalize_text(row[label])].add(action_class(row["path"]))
    return {text: tuple(sorted(classes)) for text, classes in mapping.items()}


def object_text_classes(training_qa: pd.DataFrame) -> dict[str, tuple[int, ...]]:
    """Build the correct-object-text to HARn action-class dictionary."""

    rows = training_qa.loc[
        training_qa["source"].eq("HARn")
        & training_qa["category"].eq("object_interaction")
    ]
    mapping: defaultdict[str, set[int]] = defaultdict(set)
    for _, row in rows.iterrows():
        label = str(row["answer"]).strip().upper()
        if label not in LETTERS or not str(row.get(label, "")).strip():
            raise ValueError(f"invalid HARn object answer for qa_id={row.get('qa_id')}")
        mapping[normalize_text(row[label])].add(action_class(row["path"]))
    return {text: tuple(sorted(classes)) for text, classes in mapping.items()}


def paired_answer_candidates(
    rows: pd.DataFrame,
    action_classes: Mapping[str, Sequence[int]],
    object_classes: Mapping[str, Sequence[int]],
) -> tuple[tuple[str, str], ...]:
    """Return compatible answers for a paired HARn action/object question.

    A pair is compatible when the two selected option texts have at least one
    organizer action class in common.  Empty output means the rows are not an
    exact action/object pair or that the learned dictionaries do not cover it.
    """

    action_rows = rows.loc[rows["category"].eq("single")]
    object_rows = rows.loc[rows["category"].eq("object_interaction")]
    if len(action_rows) != 1 or len(object_rows) != 1:
        return ()
    action_row = action_rows.iloc[0]
    object_row = object_rows.iloc[0]
    candidates = []
    for action_letter in available_letters(action_row):
        action_ids = set(action_classes.get(normalize_text(action_row[action_letter]), ()))
        if not action_ids:
            continue
        for object_letter in available_letters(object_row):
            object_ids = set(object_classes.get(normalize_text(object_row[object_letter]), ()))
            if action_ids & object_ids:
                candidates.append((action_letter, object_letter))
    return tuple(candidates)


def constrained_scores(
    row: pd.Series,
    probabilities: np.ndarray,
    model_classes: Sequence[int],
    text_classes: Mapping[str, Sequence[int]],
) -> tuple[dict[str, float], bool]:
    """Sum class probabilities for each valid option.

    The boolean is true only when every option maps to at least one class the
    fitted model has seen.  Callers can abstain when new organizer classes are
    outside the released nonvisual training subset.
    """

    class_to_column = {int(value): index for index, value in enumerate(model_classes)}
    scores: dict[str, float] = {}
    fully_supported = True
    for letter in available_letters(row):
        candidates = text_classes.get(normalize_text(row[letter]), ())
        columns = [class_to_column[int(value)] for value in candidates if int(value) in class_to_column]
        if not candidates or len(columns) != len(candidates):
            fully_supported = False
        scores[letter] = float(probabilities[columns].sum()) if columns else 0.0
    return scores, fully_supported


def decode_scores(scores: Mapping[str, float]) -> tuple[str, float, float]:
    """Return best letter, top score, and top-two margin with stable ties."""

    if not scores:
        raise ValueError("cannot decode empty option scores")
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    top_score = float(ranked[0][1])
    margin = top_score - float(ranked[1][1]) if len(ranked) > 1 else top_score
    return ranked[0][0], top_score, margin
