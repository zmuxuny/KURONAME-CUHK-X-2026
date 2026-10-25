"""Deterministic consistency constraints between questions for one HAU clip."""

from __future__ import annotations

import re
from collections.abc import Mapping

import pandas as pd

from cuhkx.metadata import clip_key, option_atoms
from cuhkx.submission import available_letters

TEST_CLIP_PATTERN = re.compile(r"^(?P<prefix>.+/)?LM_test_(?P<number>\d+)$")
TRAIN_REPEAT_PATTERN = re.compile(r"^(?P<group>HAU/user\d+/.+)-(?P<repeat>[123])$")


def sequence_actions(frame: pd.DataFrame) -> dict[str, frozenset[str]]:
    """Return actions certified present by each HAU sequence question.

    Every valid option in a sequence question participates in its answer; the
    answer only specifies their order.  The option texts are therefore direct
    positive-presence evidence, independent of a media model.
    """

    result: dict[str, frozenset[str]] = {}
    rows = frame.loc[frame["source"].eq("HAU") & frame["category"].eq("sequence")]
    for _, row in rows.iterrows():
        key = clip_key(row["path"])
        actions = frozenset(
            atom
            for letter in available_letters(row)
            for atom in option_atoms(row[letter])
        )
        if key in result and result[key] != actions:
            raise ValueError(f"conflicting sequence questions for clip {key!r}")
        result[key] = actions
    return result


def repeated_recording_groups(frame: pd.DataFrame) -> dict[str, str]:
    """Map HAU clips to their three-recording scenario group.

    Training paths expose the final repeat index directly (``...-1/2/3``).  The
    anonymized test release preserves the same acquisition order as contiguous
    ``LM_test_XXXX`` triplets.  We require the complete HAU test interval to be
    contiguous before using that ordering, so a changed data layout fails
    closed instead of silently creating incorrect groups.
    """

    clips = tuple(
        dict.fromkeys(
            frame.loc[frame["source"].eq("HAU"), "path"].map(clip_key).astype(str)
        )
    )
    if not clips:
        return {}

    training_matches = [TRAIN_REPEAT_PATTERN.fullmatch(value) for value in clips]
    if all(match is not None for match in training_matches):
        return {
            clip: str(match.group("group"))
            for clip, match in zip(clips, training_matches, strict=True)
            if match is not None
        }

    test_matches = [TEST_CLIP_PATTERN.fullmatch(value) for value in clips]
    if not all(match is not None for match in test_matches):
        raise ValueError("HAU paths are neither repeat-indexed training paths nor test IDs")
    numbers = [int(match.group("number")) for match in test_matches if match is not None]
    ordered = sorted(numbers)
    if ordered != list(range(ordered[0], ordered[-1] + 1)):
        raise ValueError("HAU test clip IDs must form one contiguous interval")
    if len(ordered) % 3:
        raise ValueError("HAU test clip count must be divisible into recording triplets")
    start = ordered[0]
    return {
        clip: f"test_repeat_group_{(number - start) // 3:04d}"
        for clip, number in zip(clips, numbers, strict=True)
    }


def repeated_sequence_actions(frame: pd.DataFrame) -> dict[str, frozenset[str]]:
    """Return sequence-certified actions from the other recordings in a scenario."""

    group_by_clip = repeated_recording_groups(frame)
    direct = sequence_actions(frame)
    clips_by_group: dict[str, list[str]] = {}
    for clip, group in group_by_clip.items():
        clips_by_group.setdefault(group, []).append(clip)
    return {
        clip: frozenset(
            action
            for sibling in clips_by_group[group]
            if sibling != clip
            for action in direct.get(sibling, frozenset())
        )
        for clip, group in group_by_clip.items()
    }


def certified_letters(row: pd.Series, actions: frozenset[str]) -> tuple[str, ...]:
    """Find options whose actions are certified by a linked sequence question."""

    category = str(row["category"])
    if category not in {"single", "multi", "combination"}:
        return ()
    matches = []
    for letter in available_letters(row):
        atoms = option_atoms(row[letter])
        if atoms and set(atoms) <= actions:
            matches.append(letter)
    return tuple(matches)


def apply_sequence_constraints(
    frame: pd.DataFrame,
    parent_predictions: Mapping[str, str],
) -> tuple[pd.Series, pd.DataFrame]:
    """Overlay sequence-derived constraints without invalidating a correct parent.

    For a multi-answer question, all certified letters are unioned into the
    parent.  A correct parent must already contain them, so this operation can
    only change a parent answer that is certainly wrong.  For single and
    combination questions, a unique certified option is used; this rule is
    separately auditable on the labeled training set.
    """

    actions_by_clip = sequence_actions(frame)
    output: dict[str, str] = {}
    records: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        qa_id = str(row["qa_id"])
        if qa_id not in parent_predictions:
            raise ValueError(f"parent prediction missing qa_id {qa_id!r}")
        parent = str(parent_predictions[qa_id]).strip().upper()
        prediction = parent
        reason = ""
        key = clip_key(row["path"])
        matches = certified_letters(row, actions_by_clip.get(key, frozenset()))
        category = str(row["category"])
        if category == "multi" and matches:
            prediction = "".join(sorted(set(parent) | set(matches)))
            if prediction != parent:
                reason = "multi_add_certified_positive"
        elif category in {"single", "combination"} and len(matches) == 1:
            prediction = matches[0]
            if prediction != parent:
                reason = "unique_sequence_match"
        output[qa_id] = prediction
        if matches or reason:
            records.append(
                {
                    "qa_id": qa_id,
                    "clip": key,
                    "category": category,
                    "certified_letters": "".join(matches),
                    "parent": parent,
                    "prediction": prediction,
                    "changed": prediction != parent,
                    "reason": reason,
                }
            )
    prediction_series = frame["qa_id"].astype(str).map(output)
    return prediction_series, pd.DataFrame(records)


def audit_sequence_constraints(frame: pd.DataFrame) -> pd.DataFrame:
    """Measure rule coverage and correctness on a labeled frame."""

    if "answer" not in frame:
        raise ValueError("constraint audit requires an answer column")
    actions_by_clip = sequence_actions(frame)
    records: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        key = clip_key(row["path"])
        matches = certified_letters(row, actions_by_clip.get(key, frozenset()))
        if not matches:
            continue
        answer = str(row["answer"]).strip().upper()
        category = str(row["category"])
        if category == "multi":
            valid = set(matches) <= set(answer)
            rule = "certified_letters_are_positive"
        elif len(matches) == 1:
            valid = matches[0] == answer
            rule = "unique_sequence_match"
        else:
            continue
        records.append(
            {
                "qa_id": str(row["qa_id"]),
                "category": category,
                "rule": rule,
                "certified_letters": "".join(matches),
                "answer": answer,
                "valid": valid,
            }
        )
    return pd.DataFrame(records)


def apply_repeated_sequence_constraints(
    frame: pd.DataFrame,
    parent_predictions: Mapping[str, str],
) -> tuple[pd.Series, pd.DataFrame]:
    """Overlay action evidence certified by sibling recordings in one scenario.

    The operation mirrors :func:`apply_sequence_constraints`: multi answers are
    augmented only, while single and combination answers change only when one
    sibling-certified option is unique.  The cross-recording invariant is
    audited independently on labeled training subjects.
    """

    actions_by_clip = repeated_sequence_actions(frame)
    output: dict[str, str] = {}
    records: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        qa_id = str(row["qa_id"])
        if qa_id not in parent_predictions:
            raise ValueError(f"parent prediction missing qa_id {qa_id!r}")
        parent = str(parent_predictions[qa_id]).strip().upper()
        prediction = parent
        reason = ""
        key = clip_key(row["path"])
        matches = certified_letters(row, actions_by_clip.get(key, frozenset()))
        category = str(row["category"])
        if category == "multi" and matches:
            prediction = "".join(sorted(set(parent) | set(matches)))
            if prediction != parent:
                reason = "multi_add_repeated_sequence_positive"
        elif category in {"single", "combination"} and len(matches) == 1:
            prediction = matches[0]
            if prediction != parent:
                reason = "unique_repeated_sequence_match"
        output[qa_id] = prediction
        if matches or reason:
            records.append(
                {
                    "qa_id": qa_id,
                    "clip": key,
                    "category": category,
                    "certified_letters": "".join(matches),
                    "parent": parent,
                    "prediction": prediction,
                    "changed": prediction != parent,
                    "reason": reason,
                }
            )
    return frame["qa_id"].astype(str).map(output), pd.DataFrame(records)


def audit_repeated_sequence_constraints(frame: pd.DataFrame) -> pd.DataFrame:
    """Measure sibling-recording constraint coverage and correctness."""

    if "answer" not in frame:
        raise ValueError("constraint audit requires an answer column")
    actions_by_clip = repeated_sequence_actions(frame)
    records: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        key = clip_key(row["path"])
        matches = certified_letters(row, actions_by_clip.get(key, frozenset()))
        if not matches:
            continue
        answer = str(row["answer"]).strip().upper()
        category = str(row["category"])
        if category == "multi":
            valid = set(matches) <= set(answer)
            rule = "repeated_sequence_letters_are_positive"
        elif len(matches) == 1:
            valid = matches[0] == answer
            rule = "unique_repeated_sequence_match"
        else:
            continue
        records.append(
            {
                "qa_id": str(row["qa_id"]),
                "category": category,
                "rule": rule,
                "certified_letters": "".join(matches),
                "answer": answer,
                "valid": valid,
            }
        )
    return pd.DataFrame(records)
