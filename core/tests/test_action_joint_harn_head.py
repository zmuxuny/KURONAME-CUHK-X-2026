"""Joint supervision preserves its parent at zero weight and excludes held-out HARn labels."""

import numpy as np
import pytest
import torch

from cuhkx.action_joint_harn_head import HarnWindowBank, JointHarnWindowClassifier
from cuhkx.action_window_head import FEATURE_DIMENSION, WindowPartialActionClassifier


def test_zero_auxiliary_weight_exactly_reproduces_parent_training():
    torch.set_num_threads(1)
    random = np.random.default_rng(71)
    matrix = random.normal(size=(8, FEATURE_DIMENSION)).astype(np.float32)
    matrix[:, -16:] = 1
    targets = np.full((8, 40), -1, dtype=np.int8)
    targets[:, :2] = np.asarray([[0, 1], [1, 0]] * 4)
    auxiliary = (random.normal(size=(6, 1, 2, 768)).astype(np.float32), np.ones((6, 1, 2), dtype=bool),
                 np.asarray([0, 1] * 3, dtype=np.int64), {"training_subjects": ["user1"]})
    settings = {"std_floor": .001, "device": "cpu", "lr": .001, "weight_decay": .01,
                "epochs": 2, "batch_size": 4, "temperature": .5, "gradient_clip": 5., "auxiliary_weight": 0.}
    parent = WindowPartialActionClassifier(settings, 123).fit(matrix, targets)
    joint = JointHarnWindowClassifier(settings, 123, auxiliary).fit(matrix, targets)
    for name in parent.state:
        np.testing.assert_array_equal(parent.state[name], joint.state[name])
    np.testing.assert_array_equal(parent.predict(matrix), joint.predict(matrix))
    assert not hasattr(joint, "auxiliary")


def test_auxiliary_scope_never_parses_excluded_person_action_labels():
    bank = HarnWindowBank.__new__(HarnWindowBank)
    bank.units = np.asarray(["Training/HARn/6_drinking/user1/a", "Training/HARn/7_eating/user1/b",
                             "Training/HARn/40_other/user1/c", "Training/HARn/FORBIDDEN_LABEL/user2/a"])
    bank.subjects = np.asarray(["user1", "user1", "user1", "user2"])
    bank.features = np.ones((4, 1, 2, 768), dtype=np.float32)
    bank.mask = np.ones((4, 1, 2), dtype=bool)
    bank.manifest_path, bank.manifest_sha256 = "fixture.json", "fixture"
    features, _, labels, scope = bank.scope(["user1"], ["user2"])
    assert len(features) == 2 and labels.tolist() == [6, 7]
    assert scope["training_subjects"] == ["user1"]
    with pytest.raises(ValueError, match="overlap"):
        bank.scope(["user1"], ["user1"])
