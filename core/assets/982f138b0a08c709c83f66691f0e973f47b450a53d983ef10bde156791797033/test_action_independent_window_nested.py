"""Independent option fitting reuses only exactly matched visual supervision."""

import numpy as np
import torch
from test_action_generic_nested import fixture_qa

from cuhkx import action_generic_nested as base
from cuhkx.action_explicit_layout import prepare
from cuhkx.action_window_decoding import compile_questions
from cuhkx.action_window_head import FEATURE_DIMENSION
from scripts.run_action_independent_window_nested import fit_scope as independent_fit
from scripts.run_action_window_nested import fit_scope


def test_independent_refit_reuses_head_but_changes_option_inputs(tmp_path):
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
    original, scope, _ = fit_scope(qa, prepared, poisoned, compiled, visual, ["user1", "user2"], ["user3"],
                                     config, 123, tmp_path / "parent" / "fit", [])
    independent = prepare(qa, training=True)
    independent_poison = prepare(poison, training=True)
    assert not np.array_equal(prepared.option_matrix, independent.option_matrix)
    config.update(parent_run=str(tmp_path / "parent"), runtime_output=str(tmp_path / "independent"))
    changed = qa.copy()
    changed.loc[changed.path.str.contains("user3"), "answer"] = "FORBIDDEN_VALIDATION_TARGET"
    replay, other_scope, predictions = independent_fit(changed, independent, independent_poison, compiled, visual,
                                                       ["user1", "user2"], ["user3"], config, 123,
                                                       tmp_path / "independent" / "fit", [])
    assert scope["visual_matrix_and_targets_sha256"] == other_scope["visual_matrix_and_targets_sha256"]
    assert scope["option_matrix_and_targets_sha256"] != other_scope["option_matrix_and_targets_sha256"]
    for key in original["presence"].state:
        np.testing.assert_array_equal(original["presence"].state[key], replay["presence"].state[key])
    assert set(predictions) == {"visual", "prior"}
    assert all(len(prediction) == 9 for prediction in predictions.values())
