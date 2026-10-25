from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cuhkx import action_generic_nested as model
from cuhkx.actions import HAU_TEXT_CLASS
from cuhkx.joint_actions import action_universe, decode_joint_state_with_option_evidence
from cuhkx.metadata import feature_unit
from scripts.run_action_generic_nested import validate_config


def fixture_qa(subjects=(1, 2), *, testing=False):
    records = []
    for subject in subjects:
        for repeat in (1, 2, 3):
            path = f"HAU/user{subject}/1-1-{repeat}" if not testing else f"large_model_track_test/LM_test_{repeat:04d}"
            options = {
                "single": ("Walking", "Running", "Sitting down", "Standing up"),
                "multi": ("Walking", "Drinking", "Mopping", "Squats"),
                "combination": ("Walking, Drinking", "Running, Mopping", "Sitting down, Squats", "Standing up, Drinking"),
            }
            label = (subject - 1) % 4
            for category, values in options.items():
                answer = "ABCD"[label] if category != "multi" else ["AB", "C", "D", "B"][label]
                records.append({"qa_id": f"q{subject}_{repeat}_{category}", "source": "HAU", "path": path,
                                "category": category, **dict(zip("ABCD", values, strict=True)), "answer": answer})
    return pd.DataFrame(records)


def fixture_visual(qa):
    units = sorted({feature_unit(path, training="user" in path) for path in qa.path})
    return {"X": np.random.default_rng(42).normal(size=(len(units), 8)), "units": np.asarray(units),
            "feature_names": np.asarray([f"visual_{i}" for i in range(8)]),
            "index": {unit: i for i, unit in enumerate(units)}}


def test_all_feature_and_state_preparation_ignores_poisoned_targets():
    qa = fixture_qa()
    prepared = model.prepare(qa, training=True)
    poisoned = qa.assign(answer="NEVER_READ", prediction="NEVER_READ")
    other = model.prepare(poisoned, training=True)
    np.testing.assert_array_equal(prepared.option_matrix, other.option_matrix)
    pd.testing.assert_frame_equal(prepared.frame, other.frame)
    for left, right in zip(prepared.clips, other.clips, strict=True):
        np.testing.assert_array_equal(left["action_masks"], right["action_masks"])
        assert left["answers"] == right["answers"]


def test_fitting_does_not_read_validation_labels_or_rows():
    qa = fixture_qa()
    prepared, visual = model.prepare(qa, training=True), fixture_visual(qa)
    arrays, scope = model.fitting_arrays(qa, prepared, visual, ["user1"], ["user2"])
    poisoned = qa.copy()
    poisoned.loc[poisoned.path.str.contains("user2"), "answer"] = "INVALID_VALIDATION_LABEL"
    repeated, actual_scope = model.fitting_arrays(poisoned, prepared, visual, ["user1"], ["user2"])
    assert actual_scope == scope
    for left, right in zip(arrays, repeated, strict=True):
        np.testing.assert_array_equal(left, right)
    assert arrays[1].shape == (3, 40)
    assert set(np.flatnonzero(arrays[1][0] == 1)) == {HAU_TEXT_CLASS["walking"], HAU_TEXT_CLASS["drinking"]}
    assert arrays[1][0, HAU_TEXT_CLASS["running"]] == 0
    assert arrays[1][0, HAU_TEXT_CLASS["checking the time"]] == -1
    assert all("user1" in unit for unit in scope["action_training_units"])
    with pytest.raises(ValueError, match="overlapping"):
        model.fitting_arrays(qa, prepared, visual, ["user1"], ["user1"])


def test_nonexhaustive_combination_does_not_negate_other_known_actions():
    qa = fixture_qa()
    qa.loc[qa.path.str.contains("user1") & qa.category.eq("multi"), "answer"] = "ABC"
    prepared, visual = model.prepare(qa, training=True), fixture_visual(qa)
    arrays, _ = model.fitting_arrays(qa, prepared, visual, ["user1"], ["user2"])
    # Walking and drinking occur in the correct combination; mopping is an
    # additional known positive from multi. False combination B includes both
    # running and mopping, and cannot be used to negate mopping.
    assert arrays[1][0, HAU_TEXT_CLASS["mopping"]] == 1
    assert arrays[1][0, HAU_TEXT_CLASS["running"]] == 0


def test_probe_fits_each_action_only_on_known_labels():
    config = json.loads(Path("configs/action_generic_nested_v1.json").read_text())["presence"]
    matrix = np.asarray([[1.0, 0.0], [0.5, 0.0], [-1.0, 0.0], [4.0, 0.0]])
    labels = np.full((4, 40), -1, dtype=np.int8)
    labels[:3, 0] = [1, 1, 0]
    probe = model.PartialActionClassifier(config, 41).fit(matrix, labels)
    assert probe.models[0] is not None
    assert all(estimator is None for estimator in probe.models[1:])
    np.testing.assert_array_equal(probe.fitting_rows[0], [0, 1, 2])
    np.testing.assert_array_equal(probe.predict(matrix)[:, 1:], 0.5)


class Presence:
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict(self, x):
        return np.tile(self.probabilities, (len(x), 1))


class Frequency:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, x):
        p = np.linspace(0.1, 0.9, len(x))
        return np.column_stack([1 - p, p])


def test_compiled_decoding_matches_reference_and_missing_sibling_pool():
    qa = fixture_qa((1,))
    prepared, visual = model.prepare(qa, training=True), fixture_visual(qa)
    # All action classes are known here, to compare the exact Bernoulli state
    # scores with the independent existing decoder, including its normalization.
    probabilities = np.random.default_rng(5).uniform(0.05, 0.95, size=40)
    bundle = {"presence": Presence(probabilities), "frequency": Frequency(), "training_subjects": ["user2"],
              "known_action_classes": list(range(40)), "feature_names": visual["feature_names"]}
    del visual["index"][prepared.clips[0]["unit"]]
    values = model.evidence(bundle, prepared, visual, prepared.frame.qa_id)
    decoder = {"name": "test", "visual_weight": 1.0, "option_weight": 0.05}
    actual = model.predict(prepared, values, decoder).set_index("qa_id")
    assert actual.evidence_available.all()
    option_probabilities = np.linspace(0.1, 0.9, len(prepared.frame) * 4)
    offset = 0
    for clip in prepared.clips:
        group = qa.loc[qa.qa_id.isin(clip["qa_ids"])]
        reference = decode_joint_state_with_option_evidence(
            group, {atom: probabilities[HAU_TEXT_CLASS[atom]] for atom in action_universe(group)},
            {qa_id: dict(zip("ABCD", option_probabilities[offset + i * 4:offset + (i + 1) * 4], strict=True))
             for i, qa_id in enumerate(clip["qa_ids"])}, option_weight=0.05)
        for row in group.itertuples():
            assert actual.loc[row.qa_id, "prediction"] == reference.state.answer_map()[row.category]
            assert actual.loc[row.qa_id, "margin"] == pytest.approx(reference.margin)
        offset += len(clip["qa_ids"]) * 4
    with pytest.raises(ValueError, match="upstream action model"):
        model.evidence({**bundle, "training_subjects": ["user1"]}, prepared, visual, prepared.frame.qa_id)
    with pytest.raises(ValueError, match="splits questions"):
        model.evidence(bundle, prepared, visual, [prepared.frame.qa_id.iloc[0]])


def test_unknown_classes_and_wholly_missing_group_are_neutral():
    qa = fixture_qa((1,))
    prepared, visual = model.prepare(qa, training=True), fixture_visual(qa)
    bundle = {"presence": Presence(np.zeros(40)), "frequency": Frequency(), "training_subjects": [],
              "known_action_classes": [], "feature_names": visual["feature_names"]}
    visual["index"] = {}
    values = model.evidence(bundle, prepared, visual, prepared.frame.qa_id)
    for record in values["records"]:
        assert not record["visual_available"]
        np.testing.assert_array_equal(record["action_scores"], 0)


def test_frozen_candidate_grid_rejects_changed_weights(tmp_path):
    config = json.loads(Path("configs/action_generic_nested_v1.json").read_text())
    validate_config(config, tmp_path / "synthetic.npz", synthetic=True)
    changed = copy.deepcopy(config)
    changed["decoders"][1]["option_weight"] = 0.1
    with pytest.raises(ValueError, match="predeclared"):
        validate_config(changed, tmp_path / "synthetic.npz", synthetic=True)
