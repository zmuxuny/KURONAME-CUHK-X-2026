"""Explicit groups preserve the known contract; opaque independent IDs need no triplets."""

import numpy as np
import pandas as pd
from test_action_generic_nested import fixture_qa

from cuhkx import action_generic_nested as base
from cuhkx.action_explicit_layout import prepare
from cuhkx.clip_constraints import repeated_recording_groups


def test_explicit_groups_match_original_training_preparation():
    qa = fixture_qa()
    expected = base.prepare(qa, training=True)
    actual = prepare(qa, training=True, groups=repeated_recording_groups(qa))
    pd.testing.assert_frame_equal(actual.frame, expected.frame)
    pd.testing.assert_frame_equal(actual.option_metadata, expected.option_metadata)
    np.testing.assert_array_equal(actual.option_matrix, expected.option_matrix)
    for left, right in zip(actual.clips, expected.clips, strict=True):
        assert set(left) == set(right)
        for key in left:
            if isinstance(left[key], np.ndarray):
                np.testing.assert_array_equal(left[key], right[key])
            else:
                assert left[key] == right[key]


def test_opaque_singleton_and_partial_pair_ignore_target_columns():
    qa = fixture_qa(subjects=(1,)).iloc[:6].copy()
    paths = dict(zip(qa.path.unique(), ["opaque/alpha", "opaque/omega"], strict=True))
    qa.path = qa.path.map(paths)
    independent = prepare(qa, training=False)
    assert len({clip["repeat_group"] for clip in independent.clips}) == 2
    groups = {path: "declared_scene" for path in paths.values()}
    paired = prepare(qa, training=False, groups=groups)
    poisoned = prepare(qa.assign(answer="POISON", target="POISON", prediction="POISON"), training=False, groups=groups)
    pd.testing.assert_frame_equal(paired.frame, poisoned.frame)
    np.testing.assert_array_equal(paired.option_matrix, poisoned.option_matrix)
    assert len({clip["repeat_group"] for clip in paired.clips}) == 1
    assert not np.array_equal(independent.option_matrix, paired.option_matrix)
    singleton = prepare(qa.iloc[:3], training=False)
    assert len(singleton.clips) == 1
