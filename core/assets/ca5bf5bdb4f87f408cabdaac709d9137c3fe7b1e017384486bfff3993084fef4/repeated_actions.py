"""High-precision action evidence sharing across repeated HAU recordings."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Set as AbstractSet

import pandas as pd

from cuhkx.clip_constraints import repeated_recording_groups
from cuhkx.metadata import clip_key, option_atoms
from cuhkx.submission import available_letters

SOURCE_CATEGORIES = frozenset({"single", "combination"})
TARGET_CATEGORIES = frozenset({"single", "multi", "combination"})


def _validate_prediction(row: pd.Series, prediction: str) -> str:
    value = str(prediction).strip().upper()
    available = set(available_letters(row))
    if not value or len(set(value)) != len(value) or not set(value) <= available:
        raise ValueError(
            f"invalid source prediction {value!r} for qa_id {row['qa_id']!r}"
        )
    return value


def repeated_action_decisions(
    frame: pd.DataFrame,
    source_predictions: Mapping[str, str],
    eligible_source_ids: AbstractSet[str],
) -> pd.DataFrame:
    """Find target options supported by high-confidence sibling recordings.

    Sequence options are direct positive evidence.  Learned evidence is limited
    to eligible single and combination predictions; the caller is responsible
    for defining eligibility with an independently audited gate.  Evidence from
    the target recording itself is excluded.
    """

    group_by_clip = repeated_recording_groups(frame)
    evidence: dict[str, dict[str, list[tuple[str, str, str]]]] = {}
    hau = frame.loc[frame["source"].eq("HAU")]
    for _, row in hau.iterrows():
        category = str(row["category"])
        qa_id = str(row["qa_id"])
        clip = clip_key(row["path"])
        group = group_by_clip[clip]
        if category == "sequence":
            letters = available_letters(row)
            source_kind = "sequence"
        elif category in SOURCE_CATEGORIES and qa_id in eligible_source_ids:
            if qa_id not in source_predictions:
                raise ValueError(f"eligible source lacks prediction for {qa_id!r}")
            letters = _validate_prediction(row, source_predictions[qa_id])
            source_kind = "model"
        else:
            continue
        for letter in letters:
            for action in option_atoms(row[letter]):
                evidence.setdefault(group, {}).setdefault(action, []).append(
                    (clip, qa_id, source_kind)
                )

    records: list[dict[str, object]] = []
    for _, row in hau.loc[hau["category"].isin(TARGET_CATEGORIES)].iterrows():
        qa_id = str(row["qa_id"])
        clip = clip_key(row["path"])
        group = group_by_clip[clip]
        sibling = {
            action: [item for item in items if item[0] != clip]
            for action, items in evidence.get(group, {}).items()
        }
        sibling = {action: items for action, items in sibling.items() if items}
        matches = []
        matched_source_ids: set[str] = set()
        matched_source_kinds: set[str] = set()
        for letter in available_letters(row):
            atoms = option_atoms(row[letter])
            if atoms and set(atoms) <= set(sibling):
                matches.append(letter)
                for atom in atoms:
                    matched_source_ids.update(item[1] for item in sibling[atom])
                    matched_source_kinds.update(item[2] for item in sibling[atom])
        if matches:
            records.append(
                {
                    "qa_id": qa_id,
                    "clip": clip,
                    "repeat_group": group,
                    "category": str(row["category"]),
                    "certified_letters": "".join(matches),
                    "source_count": len(matched_source_ids),
                    "source_kinds": "+".join(sorted(matched_source_kinds)),
                    "source_qa_ids": "|".join(sorted(matched_source_ids)),
                }
            )
    return pd.DataFrame(records)


def apply_repeated_action_decisions(
    parent_predictions: Mapping[str, str], decisions: pd.DataFrame
) -> tuple[dict[str, str], pd.DataFrame]:
    """Apply sibling positives using risk-monotone target-category rules."""

    output = {str(qa_id): str(value).strip().upper() for qa_id, value in parent_predictions.items()}
    applied = decisions.copy()
    parents = []
    predictions = []
    reasons = []
    for row in decisions.itertuples(index=False):
        if row.qa_id not in output:
            raise ValueError(f"parent prediction missing qa_id {row.qa_id!r}")
        parent = output[row.qa_id]
        certified = str(row.certified_letters)
        prediction = parent
        reason = ""
        if row.category == "multi":
            prediction = "".join(sorted(set(parent) | set(certified)))
            if prediction != parent:
                reason = "multi_add_sibling_positive"
        elif row.category in {"single", "combination"} and len(certified) == 1:
            prediction = certified
            if prediction != parent:
                reason = "unique_sibling_positive_match"
        output[row.qa_id] = prediction
        parents.append(parent)
        predictions.append(prediction)
        reasons.append(reason)
    applied["parent"] = parents
    applied["prediction"] = predictions
    applied["changed"] = applied["parent"].ne(applied["prediction"])
    applied["reason"] = reasons
    return output, applied
