"""Prepare action questions with declared scene groups or independent recordings."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cuhkx import action_generic_nested as base
from cuhkx.actions import HAU_TEXT_CLASS
from cuhkx.joint_actions import JOINT_CATEGORIES, action_universe, enumerate_joint_states
from cuhkx.metadata import clip_key, feature_unit, option_atoms, subject_id
from cuhkx.option_frequency_explicit import build_option_frequency_features


def prepare(qa, *, training, groups=None):
    """Groups map normalized clip IDs to caller-declared scenes; None means no sharing.

    Video identifiers have no implicit grouping meaning here. Training/inference
    uses the existing descriptor-key convention, independently of scene labels.
    """
    metadata = qa.loc[qa.source.eq("HAU"), base.METADATA].copy()
    if metadata.empty or metadata.qa_id.duplicated().any():
        raise ValueError("action questions need nonempty unique IDs")
    metadata["clip"] = metadata.path.map(clip_key)
    clips_present = set(metadata["clip"])
    if groups is None:
        groups = {clip: clip for clip in clips_present}
    if set(groups) != clips_present or any(not isinstance(value, str) or not value for value in groups.values()):
        raise ValueError("explicit scene mapping must cover exactly the provided clips")
    matrix, option_metadata = build_option_frequency_features(metadata.loc[:, base.METADATA], groups=groups)
    options = {(row.qa_id, row.letter): i for i, row in enumerate(option_metadata.itertuples())}
    metadata["subject"] = metadata.path.map(subject_id)
    metadata["unit"] = metadata.path.map(lambda value: feature_unit(value, training=training))
    frame = metadata.loc[metadata.category.isin(JOINT_CATEGORIES)].reset_index(drop=True)
    if training and frame.subject.isna().any():
        raise ValueError("training questions need their actual subject identifiers")
    clips = []
    for clip_id, group in metadata.groupby("clip", sort=False):
        targets = group.loc[group.category.isin(JOINT_CATEGORIES)]
        if targets.empty:
            continue
        if targets.category.duplicated().any():
            raise ValueError("one video has duplicate action categories")
        states = sorted(enumerate_joint_states(group), key=lambda state: state.answers)
        if not states:
            raise ValueError("action options admit no legal joint answer")
        option_keys = [(row.qa_id, letter) for row in targets.itertuples() for letter in "ABCD"]
        category_by_id = targets.set_index("qa_id").category.to_dict()
        action_masks, option_masks, answers = [], [], []
        for state in states:
            mask = np.zeros(40, dtype=np.float64)
            for atom in state.positives:
                mask[HAU_TEXT_CLASS[atom]] = 1
            answer_map = state.answer_map()
            action_masks.append(mask)
            option_masks.append([int(letter in answer_map[category_by_id[qa_id]]) for qa_id, letter in option_keys])
            answers.append([answer_map[row.category] for row in targets.itertuples()])
        clips.append({"clip": clip_id, "unit": group.iloc[0].unit, "subject": group.iloc[0].subject,
                      "repeat_group": groups[clip_id], "qa_ids": targets.qa_id.tolist(),
                      "action_masks": np.asarray(action_masks), "option_masks": np.asarray(option_masks),
                      "answers": answers, "option_rows": np.asarray([options[key] for key in option_keys]),
                      "atom_count": max(1, len(action_universe(group))),
                      "sequence_positives": sorted({atom for row in group.loc[group.category.eq("sequence")].itertuples()
                                                    for letter in "ABCD" for atom in option_atoms(getattr(row, letter))})})
    for scene in set(groups.values()):
        members = [clip for clip in clips if clip["repeat_group"] == scene]
        if len(members) > 3 or len({"unknown" if pd.isna(clip["subject"]) else clip["subject"] for clip in members}) > 1:
            raise ValueError("declared scene crosses subjects or exceeds the validated three-recording layout")
    return base.Prepared(frame, matrix, option_metadata, clips)
