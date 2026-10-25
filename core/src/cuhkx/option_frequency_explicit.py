"""Option-frequency features using explicit scene membership, including singletons.

Derived from the frozen option_frequency builder; no identifier-based grouping.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from cuhkx.metadata import clip_key, option_atoms, subject_id
from cuhkx.submission import available_letters

ACTION_CATEGORIES = ("single", "multi", "combination")
EVIDENCE_CATEGORIES = ("single", "multi", "combination", "sequence")
LETTERS = "ABCD"


def build_option_frequency_features(
    frame: pd.DataFrame, *, groups: dict[str, str],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build one feature row for each option of each HAU action question.

    Counts use only official option text.  Sibling and same-recording evidence
    are separated, and the target question itself is excluded from same-recording
    counts.  Combination options aggregate the weakest, mean, strongest, and
    total evidence over their component actions.
    """

    hau = frame.loc[frame["source"].eq("HAU")].copy()
    group_by_clip = groups
    hau["clip_id"] = hau["path"].map(clip_key)
    hau["repeat_group"] = hau["clip_id"].map(group_by_clip)
    if hau["repeat_group"].isna().any():
        raise ValueError("failed to assign every HAU row to a repeated-recording group")

    occurrences: dict[
        tuple[str, str], list[tuple[str, str, str]]
    ] = defaultdict(list)
    for row in hau.itertuples():
        if row.category == "emotion":
            continue
        row_series = pd.Series(row._asdict())
        for letter in available_letters(row_series):
            for action in set(option_atoms(getattr(row, letter))):
                occurrences[(row.repeat_group, action)].append(
                    (row.clip_id, row.qa_id, row.category)
                )

    features: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    targets = hau.loc[hau["category"].isin(ACTION_CATEGORIES)]
    for row in targets.itertuples():
        row_series = pd.Series(row._asdict())
        if available_letters(row_series) != LETTERS:
            raise ValueError(f"HAU action row {row.qa_id!r} does not expose ABCD")
        raw_options = []
        for letter in LETTERS:
            atoms = option_atoms(getattr(row, letter))
            atom_features = []
            for action in atoms:
                items = occurrences[(row.repeat_group, action)]
                sibling = [item for item in items if item[0] != row.clip_id]
                same = [
                    item
                    for item in items
                    if item[0] == row.clip_id and item[1] != row.qa_id
                ]
                values = [
                    len(sibling),
                    len({item[1] for item in sibling}),
                    len({item[0] for item in sibling}),
                    len(items),
                    len(same),
                ]
                for category in EVIDENCE_CATEGORIES:
                    values.extend(
                        (
                            sum(item[2] == category for item in sibling),
                            len(
                                {
                                    item[1]
                                    for item in sibling
                                    if item[2] == category
                                }
                            ),
                            sum(item[2] == category for item in same),
                        )
                    )
                atom_features.append(values)
            values = np.asarray(atom_features, dtype=np.float32)
            raw_options.append(
                np.concatenate(
                    (
                        values.min(axis=0),
                        values.mean(axis=0),
                        values.max(axis=0),
                        values.sum(axis=0),
                        np.asarray([len(atoms)], dtype=np.float32),
                    )
                )
            )
        raw = np.stack(raw_options)
        for index, letter in enumerate(LETTERS):
            ranks = (raw < raw[index]).sum(axis=0).astype(np.float32)
            centered = raw[index] - raw.mean(axis=0)
            category = np.asarray(
                [row.category == value for value in ACTION_CATEGORIES],
                dtype=np.float32,
            )
            features.append(
                np.concatenate((raw[index], ranks, centered, category))
            )
            record: dict[str, object] = {
                "qa_id": row.qa_id,
                "category": row.category,
                "letter": letter,
                "subject": subject_id(row.path),
            }
            if "answer" in hau:
                record["label"] = int(letter in row.answer)
                record["answer"] = row.answer
            records.append(record)
    matrix = np.stack(features).astype(np.float32)
    metadata = pd.DataFrame(records)
    if not np.isfinite(matrix).all():
        raise ValueError("option-frequency features contain a non-finite value")
    return matrix, metadata

