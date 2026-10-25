"""Privileged full-scene teacher with fixed decoding and joint-state soft targets."""
from __future__ import annotations

import numpy as np
from scipy.special import softmax

from cuhkx import emotion_fusion_nested as fusion
from cuhkx import emotion_nested as legacy
from cuhkx.emotion_scene_robustness import inference_frame

DECODER = {'name': 'fixed_current_projected_option_log_sum',
           'fusion_mode': 'projected_option_log_sum', 'lexical': .25,
           'context': 4., 'sensor': 1., 'use_position': True}
SENSOR_WEIGHTS = {'imu': 1., 'skeleton': 2., 'radar': .2, 'visual': 1.}


def validate_scope(train_subjects, target_subjects, forbidden_subjects):
    train, target, forbidden = map(set, (train_subjects, target_subjects, forbidden_subjects))
    if not train or not target or train & (target | forbidden) or target & forbidden:
        raise ValueError('teacher must exclude target people and outer validation people')


def joint_marginals(frame, evidence, *, temperature):
    """No answer or truth_word is accessed; priors/position belong to the teacher."""
    safe = inference_frame(frame)
    output, states, hard = {}, [], {}
    for group_id, group in safe.groupby('group', sort=False):
        assignments = legacy.feasible_assignments(group)
        rows = list(group.itertuples())
        values = []
        for words in assignments:
            value = 0.
            for row, word in zip(rows, words, strict=True):
                item = evidence[row.qa_id][word]
                value += (DECODER['lexical'] * item['lexical'] + DECODER['context'] * item['context'])[row.position]
                value += DECODER['sensor'] * item['sensor']
            values.append(float(value))
        probabilities = softmax(np.asarray(values) / temperature)
        best = assignments[int(np.argmax(values))]
        for index, row in enumerate(rows):
            marginal = np.zeros(4, dtype=np.float64)
            for words, probability in zip(assignments, probabilities, strict=True):
                marginal['ABCD'.index(row.letters[row.options.index(words[index])])] += probability
            if not np.isclose(marginal.sum(), 1):
                raise ValueError('teacher marginal normalization failed')
            output[row.qa_id] = marginal.astype(np.float32)
            hard[row.qa_id] = row.letters[row.options.index(best[index])]
        states.append({'group': str(group_id), 'qa_ids': [row.qa_id for row in rows],
                       'assignments': assignments, 'energies': values, 'temperature': temperature,
                       'state_probabilities': probabilities.tolist()})
    return output, hard, states


def teacher_outputs(bundle, frame, contexts, matrices, masks, rows, *, temperature):
    safe = inference_frame(frame)
    inference = {**bundle, 'config': {**bundle['config'], 'sensor_weights': SENSOR_WEIGHTS}}
    evidence = fusion.base_evidence(inference, safe, contexts, matrices, masks, rows)
    adapted = fusion.adapted_evidence(evidence, DECODER)
    return joint_marginals(safe.iloc[rows], adapted, temperature=temperature), evidence
