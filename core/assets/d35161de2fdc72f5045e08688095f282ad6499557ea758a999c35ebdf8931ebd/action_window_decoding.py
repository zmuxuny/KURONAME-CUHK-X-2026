"""Reuse legal joint decoding with visual or training-prior window probabilities."""

from __future__ import annotations

import copy

import numpy as np
from scipy.special import expit

from cuhkx import action_generic_nested as base
from cuhkx.action_compiled_likelihood import compile_state, score_compiled
from cuhkx.action_mil_training import ProbabilityIdentity
from cuhkx.actions import HAU_CLASS_TEXT
from cuhkx.joint_actions import enumerate_joint_states
from cuhkx.metadata import clip_key, normalize_text


def compile_questions(qa, prepared):
    metadata = qa.loc[qa.source.eq("HAU"), base.METADATA].copy()
    metadata["clip"] = metadata.path.map(clip_key)
    groups = dict(tuple(metadata.groupby("clip", sort=False)))
    result = {}
    for clip in prepared.clips:
        group = groups[clip["clip"]]
        states = sorted(enumerate_joint_states(group), key=lambda state: state.answers)
        categories = prepared.frame.set_index("qa_id").loc[clip["qa_ids"], "category"].tolist()
        assert [[state.answer_map()[category] for category in categories] for state in states] == clip["answers"]
        result[clip["clip"]] = [compile_state(group, state) for state in states]
    return result


def evidence_from_probabilities(bundle, prepared, compiled, units, probabilities, qa_ids):
    names = np.asarray([f"probability_{i}" for i in range(40)])
    visual = {"X": probabilities, "index": {unit: i for i, unit in enumerate(units)}, "feature_names": names}
    identity = {"presence": ProbabilityIdentity(), "frequency": bundle["frequency"],
                "training_subjects": bundle["training_subjects"],
                "known_action_classes": bundle["known_action_classes"], "feature_names": names}
    minimal = base.evidence(identity, prepared, visual, qa_ids)
    marginal = copy.deepcopy(minimal)
    clipped = np.clip(probabilities.astype(np.float64), 1e-5, 1 - 1e-5)
    clipped[:, sorted(set(range(40)) - set(bundle["known_action_classes"]))] = .5
    odds = np.log(clipped / (1 - clipped))
    groups = {}
    for record in minimal["records"]:
        clip = record["clip"]
        groups.setdefault(clip["repeat_group"], []).append(visual["index"][clip["unit"]])
    for original, changed in zip(minimal["records"], marginal["records"], strict=True):
        clip = original["clip"]
        pooled = odds[groups[clip["repeat_group"]]].mean(axis=0)
        np.testing.assert_allclose(clip["action_masks"] @ pooled, original["action_scores"], rtol=0, atol=1e-12)
        scores = {normalize_text(HAU_CLASS_TEXT[i]): value for i, value in enumerate(expit(pooled))}
        changed["action_scores"] = np.asarray([score_compiled(event, scores) for event in compiled[clip["clip"]]])
        assert np.isfinite(changed["action_scores"]).all()
    return {"minimal_set": minimal, "marginal": marginal}
