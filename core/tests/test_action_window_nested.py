"""Actual scope fitting must ignore held-out labels and replay both evidence branches."""

import numpy as np
import pandas as pd
import torch
from test_action_generic_nested import fixture_qa

from cuhkx import action_generic_nested as base
from cuhkx.action_window_decoding import compile_questions
from cuhkx.action_window_head import FEATURE_DIMENSION
from scripts.run_action_window_nested import fit_scope


def test_actual_fit_and_decoding_ignore_held_out_labels(tmp_path):
    torch.set_num_threads(1)
    qa = fixture_qa(subjects=(1, 2, 3))
    prepared = base.prepare(qa, training=True)
    poison = qa.assign(answer="POISON", target="POISON", prediction="POISON")
    poisoned = base.prepare(poison, training=True)
    compiled = compile_questions(qa, prepared)
    assert compiled == compile_questions(poison, poisoned)
    units = sorted(set(prepared.frame.unit))
    matrix = np.random.default_rng(56).normal(size=(len(units), FEATURE_DIMENSION)).astype(np.float32)
    matrix[:, -16:] = 1
    visual = {"X": matrix, "units": np.asarray(units), "feature_names": np.asarray([str(i) for i in range(FEATURE_DIMENSION)]),
              "index": {unit: i for i, unit in enumerate(units)}}
    config = {"presence": {"std_floor": .001, "device": "cpu", "lr": .001, "weight_decay": .01,
                            "epochs": 2, "batch_size": 4, "temperature": .5, "gradient_clip": 5.},
              "frequency": {"n_estimators": 10, "min_samples_leaf": 1, "max_depth": 3},
              "decoders": [{"name": "visual", "branch": "visual", "likelihood": "marginal", "visual_weight": 1., "option_weight": .05},
                           {"name": "prior", "branch": "prior", "likelihood": "minimal_set", "visual_weight": 1., "option_weight": .05}]}
    original, scope, predictions = fit_scope(qa, prepared, poisoned, compiled, visual, ["user1", "user2"], ["user3"],
                                              config, 123, tmp_path / "original", [])
    changed = qa.copy()
    changed.loc[changed.path.str.contains("user3"), "answer"] = "FORBIDDEN_VALIDATION_TARGET"
    replay, other_scope, other_predictions = fit_scope(changed, prepared, poisoned, compiled, visual, ["user1", "user2"], ["user3"],
                                                        config, 123, tmp_path / "poison", [])
    assert scope["visual_matrix_and_targets_sha256"] == other_scope["visual_matrix_and_targets_sha256"]
    assert original["training_subjects"] == ["user1", "user2"]
    for key in original["presence"].state:
        np.testing.assert_array_equal(original["presence"].state[key], replay["presence"].state[key])
    for key in predictions:
        pd.testing.assert_frame_equal(predictions[key], other_predictions[key])
