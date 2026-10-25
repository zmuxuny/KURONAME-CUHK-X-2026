"""Anonymous identity, cross-recording isolation and descriptor contracts."""
import numpy as np
import pandas as pd
import pytest
from test_action_generic_nested import fixture_qa

from cuhkx.action_independent_visual_inference import KIND, normalize_metadata, predict
from cuhkx.action_joint_harn_head import JointHarnWindowClassifier
from cuhkx.action_window_head import FEATURE_DIMENSION
from cuhkx.metadata import feature_unit


class Frequency:
    n_features_in_ = 224
    classes_ = np.asarray([0, 1])

    def predict_proba(self, values):
        # Each row depends on its same-recording option features.
        p = .25 + .5 / (1 + np.exp(-values[:, 0]))
        return np.stack([1 - p, p], axis=1)


def setup(monkeypatch):
    qa = fixture_qa(subjects=(1,)).iloc[:6].copy()
    monkeypatch.setattr(JointHarnWindowClassifier, 'predict',
                        lambda self, x: np.full((len(x), 40), .6))
    names = np.asarray([str(i) for i in range(FEATURE_DIMENSION)])
    model = {'kind': KIND, 'layout': 'independent_recordings', 'presence': object.__new__(JointHarnWindowClassifier),
             'training_subjects': ['user1'], 'validation_subjects': [], 'known_action_classes': list(range(40)),
             'prior': np.full(40, .5), 'frequency': Frequency(), 'feature_names': names,
             'selected_decoder': {'name': 'fixed', 'branch': 'visual', 'likelihood': 'marginal',
                                  'visual_weight': .25, 'option_weight': 1.}}
    bank = {'X': np.zeros((2, FEATURE_DIMENSION)), 'feature_names': names,
            'units': np.asarray([feature_unit(path, training=False) for path in qa.path.unique()])}
    bank["X"][:, -16::2] = 1
    return qa, model, bank


def test_identity_is_lookup_only_and_targets_are_ignored(monkeypatch):
    qa, model, bank = setup(monkeypatch)
    before, raw = predict(model, qa, bank)
    renamed = qa.assign(path=qa.path.map(dict(zip(qa.path.unique(), ['opaque/099', 'user999/003'], strict=True))),
                        answer='POISON', prediction='POISON', group='fake_scene')
    changed = {**bank, 'units': np.asarray([feature_unit(path, training=False) for path in renamed.path.unique()])}
    after, other = predict(model, renamed, changed)
    pd.testing.assert_frame_equal(before, after)
    np.testing.assert_array_equal(raw['option_matrix'], other['option_matrix'])
    assert raw['scene_count'] == other['scene_count'] == 2
    normalized, _ = normalize_metadata(qa)
    assert not normalized.path.str.contains('user1').any()


def test_single_recording_matches_batch_and_requires_exact_features(monkeypatch):
    qa, model, bank = setup(monkeypatch)
    batch, _ = predict(model, qa, bank)
    one = qa.iloc[:3]
    subset = {**bank, 'X': bank['X'][:1], 'units': bank['units'][:1]}
    single, _ = predict(model, one, subset)
    pd.testing.assert_frame_equal(single, batch.iloc[:3])
    with pytest.raises(ValueError, match='exactly'):
        predict(model, one, bank)
    with pytest.raises(ValueError, match='eight-window'):
        predict({**model, 'presence': object()}, one, subset)


def test224_layout_keeps_visual_identity_and_visible_bits(monkeypatch):
    qa, model, bank = setup(monkeypatch)
    _, raw = predict(model, qa, bank)
    assert raw['option_matrix'].shape[1] == 224
    assert (raw['option_matrix'][:, 210:213] == 1).all()
    assert (raw['option_matrix'][:, 213] == 0).all()
    np.testing.assert_array_equal(raw['option_matrix'][:, 220:224], [[1, 0, 1, 0]] * len(raw['option_matrix']))
    reduced = qa.loc[qa.category.eq('multi')]
    _, fewer = predict(model, reduced, bank)
    np.testing.assert_array_equal(fewer['option_matrix'][:, 210:214], [[0, 1, 0, 0]] * len(fewer['option_matrix']))
    original_lookup = dict(zip(zip(raw['option_qa_ids'], raw['option_letters']), raw['option_matrix'][:, 214:], strict=True))
    for q, letter, row in zip(fewer['option_qa_ids'], fewer['option_letters'], fewer['option_matrix'], strict=True):
        np.testing.assert_array_equal(row[214:], original_lookup[(q, letter)])
