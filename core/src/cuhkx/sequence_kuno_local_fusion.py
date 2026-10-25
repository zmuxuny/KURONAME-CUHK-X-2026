"""Conditional K-KUNO control versus concatenated scoped SSv2 temporal evidence."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from cuhkx.pairwise_temporal import PAIR_INDICES, decode_pairwise_order, pair_curve_features
from cuhkx.repeated_temporal import share_repeated_pair_probabilities
from cuhkx.sequence_skeleton_nested import array_digest
from cuhkx.sequence_videomae import training_arrays
from cuhkx.temporal import softmax

MODES = ('control', 'concat')


def read_kuno(paths, temperature):
    sources = []
    for path, segments in zip(paths, (8, 16), strict=True):
        with np.load(path, allow_pickle=False) as archive:
            units = archive['units'].astype(str)
            logits, weights = archive['logits'], archive['weights'].astype(np.float64)
            np.testing.assert_array_equal(archive['classes'], np.arange(40))
            if logits.shape != (len(weights), len(units), segments, 40) or not np.isfinite(logits).all():
                raise ValueError('KUNO temporal archive layout changed')
        if len(set(units)) != len(units) or any(not unit.startswith('Training/HAU/') for unit in units):
            raise ValueError('requires unique Train-only KUNO units')
        weights /= weights.sum()
        probabilities = np.tensordot(weights, softmax(logits, temperature), axes=(0, 0)).astype(np.float32)
        sources.append(dict(zip(units, probabilities, strict=True)))
    return sources


def pair_inputs(frame, sources, curves):
    parts = {mode: {'forward': [], 'reverse': []} for mode in MODES}
    for row, curve in zip(frame.itertuples(), curves, strict=True):
        for a, b in PAIR_INDICES:
            for direction, left, right in [('forward', a, b), ('reverse', b, a)]:
                classes = row.classes[left], row.classes[right]
                old = pair_curve_features([source[row.unit] for source in sources], *classes)
                new = pair_curve_features([curve], *classes)
                np.testing.assert_array_equal(old[-80:], new[-80:])
                parts['control'][direction].append(old)
                # Preserve exactly one class identity block, always at the end.
                parts['concat'][direction].append(np.concatenate([old[:-80], new]))
    return {mode: {name: np.asarray(value, dtype=np.float32) for name, value in values.items()}
            for mode, values in parts.items()}


def forest_digest(model):
    digest = hashlib.sha256()
    digest.update(array_digest(model.classes_, np.asarray([model.n_features_in_])).encode())
    for estimator in model.estimators_:
        state = estimator.tree_.__getstate__()
        digest.update(str((state['max_depth'], state['node_count'])).encode())
        for name in state['nodes'].dtype.names:
            digest.update(name.encode())
            digest.update(array_digest(state['nodes'][name]).encode())
        digest.update(array_digest(state['values']).encode())
    return digest.hexdigest()


def canonical_padding(model):
    """Clear unused C struct padding only; preserve every learned tree field exactly."""
    before = forest_digest(model)
    for estimator in model.estimators_:
        state = estimator.tree_.__getstate__()
        nodes = np.zeros(state['nodes'].shape, dtype=state['nodes'].dtype)
        for name in nodes.dtype.names:
            nodes[name] = state['nodes'][name]
        estimator.tree_.__setstate__({**state, 'nodes': nodes})
    if forest_digest(model) != before:
        raise AssertionError('serialization changed a learned tree value')
    return before


def fit(qa, frame, pairs, config, seed):
    models, hashes, scope = {}, {}, {}
    counts = None
    for mode in MODES:
        matrix, target, prior = training_arrays(qa, frame, pairs[mode])
        if counts is not None and prior != counts:
            raise AssertionError('control and fusion training labels differ')
        counts = prior
        models[mode], hashes[mode] = [], []
        for index, family in enumerate(config['families']):
            model = ExtraTreesClassifier(**config['forest'], min_samples_leaf=family['min_samples_leaf'],
                                        max_features=family['max_features'], n_jobs=3,
                                        random_state=seed + index * 1000).fit(matrix, target)
            hashes[mode].append(canonical_padding(model))
            models[mode].append(model)
        scope[mode] = {'matrix_shape': list(matrix.shape), 'actual_features_targets_sha256': array_digest(matrix, target)}
    bundle = {'kind': 'cuhkx_kuno_local_concat_conditional_v1', 'models': models, 'counts': counts,
              'training_subjects': sorted(set(frame.subject)), 'config': config, 'forest_hashes': hashes}
    return bundle, scope


def evidence(bundle, frame, pairs):
    if set(frame.subject) & set(bundle['training_subjects']):
        raise ValueError('validation person reached fitted forest')
    evidence = {}
    for mode in MODES:
        raw = []
        for model in bundle['models'][mode]:
            column = list(model.classes_).index(1)
            x, y = pairs[mode]['forward'], pairs[mode]['reverse']
            raw.append((.5 * (model.predict_proba(x)[:, column] + 1 - model.predict_proba(y)[:, column])).reshape(-1, 6))
        raw = np.asarray(raw)
        combined, _ = share_repeated_pair_probabilities(raw, np.asarray(frame.classes.tolist()),
                        frame.repeat_group.tolist(), family_indices=[0, 1], sibling_weight=bundle['config']['sibling_weight'])
        evidence[mode] = combined
        evidence[mode + '_families'] = raw
    return evidence


def predict(bundle, frame, values, mode):
    result = []
    for row, probability in zip(frame.itertuples(), values[mode], strict=True):
        answer, margin = decode_pairwise_order(probability, row.classes, bundle['counts'],
                            prior_weight=bundle['config']['prior_weight'], prior_smoothing=1.5)
        result.append({'qa_id': row.qa_id, 'prediction': answer, 'margin': margin})
    return pd.DataFrame(result)
