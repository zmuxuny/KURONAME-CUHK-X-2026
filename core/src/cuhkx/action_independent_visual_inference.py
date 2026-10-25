"""Independent224-column option inference with fixed cross-fitted visual summaries."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from cuhkx import action_generic_nested as base
from cuhkx.action_explicit_layout import prepare
from cuhkx.action_joint_harn_head import JointHarnWindowClassifier
from cuhkx.action_question_dropout import bits_for_options
from cuhkx.action_visual_option_dropout import option_visual_features
from cuhkx.action_window_decoding import compile_questions, evidence_from_probabilities
from cuhkx.action_window_head import FEATURE_DIMENSION, unpack
from cuhkx.metadata import clip_key, feature_unit

KIND = 'cuhkx_independent_visual_options_action_inference_v1'


def normalize_metadata(qa):
    """Use recording IDs only as equality/feature lookup keys, never as subjects/scenes."""
    metadata = qa.loc[qa.source.eq('HAU'), base.METADATA].copy()
    clips = metadata.path.map(clip_key)
    aliases = {clip: 'anonymous/clip_' + hashlib.sha256(clip.encode()).hexdigest()
               for clip in clips.unique()}
    original_units = {feature_unit(alias, training=False): feature_unit(clip, training=False)
                      for clip, alias in aliases.items()}
    metadata['path'] = clips.map(aliases)
    return metadata, original_units


def predict(model, qa, bank):
    if model['kind'] != KIND or type(model['presence']) is not JointHarnWindowClassifier:
        raise ValueError('requires the exported independent linear eight-window model')
    if model['layout'] != 'independent_recordings' or model['validation_subjects']:
        raise ValueError('requires the final independently trained inference bundle')
    metadata, aliases = normalize_metadata(qa)
    prepared = prepare(metadata, training=False, groups=None)
    if len({clip['repeat_group'] for clip in prepared.clips}) != len(prepared.clips):
        raise AssertionError('cross-recording group sharing reached independent inference')
    compiled = compile_questions(metadata, prepared)
    np.testing.assert_array_equal(bank['feature_names'], model['feature_names'])
    units = prepared.frame.drop_duplicates('unit').unit.tolist()
    requested = [aliases[unit] for unit in units]
    available = bank['units'].astype(str).tolist()
    if len(set(available)) != len(available) or set(available) != set(requested):
        raise ValueError('descriptors must cover exactly the requested independent recordings')
    if bank['X'].shape != (len(available), FEATURE_DIMENSION) or not np.isfinite(bank['X']).all():
        raise ValueError('requires finite matched eight-window descriptors')
    lookup = {unit: index for index, unit in enumerate(available)}
    values = bank['X'][[lookup[unit] for unit in requested]]
    # Single-recording forward calls make outputs invariant to BLAS batch packing.
    recipe = model['selected_decoder']
    visual = np.concatenate([model['presence'].predict(row[None]) for row in values])
    prior = np.repeat(model['prior'][None], len(values), axis=0)
    probabilities = visual if recipe['branch'] == 'visual' else prior
    _, mask = unpack(values)
    availability = mask.mean(axis=1)
    index = {unit: i for i, unit in enumerate(units)}
    by_qa = prepared.frame.set_index('qa_id')
    summaries = []
    for row in prepared.option_metadata.itertuples():
        question = by_qa.loc[row.qa_id]
        position = index[question.unit]
        summaries.append(option_visual_features(question[row.letter], visual[position],
            model['known_action_classes'], availability[position]))
    prepared.option_matrix = np.column_stack((prepared.option_matrix,
        bits_for_options(metadata, prepared.option_metadata), np.asarray(summaries)))
    if prepared.option_matrix.shape[1] != 224 or model['frequency'].n_features_in_ != 224:
        raise ValueError('requires fixed210counts+4visiblebits+10visualsummary model')
    evidence = evidence_from_probabilities(model, prepared, compiled, units, probabilities,
                                          prepared.frame.qa_id.tolist())
    result = base.predict(prepared, evidence[recipe['likelihood']], recipe)
    if result.qa_id.tolist() != prepared.frame.qa_id.tolist():
        raise AssertionError('inference changed QA order')
    states = []
    for item in evidence[recipe['likelihood']]['records']:
        clip = item['clip']
        action = recipe['visual_weight'] * item['action_scores']
        option = recipe['option_weight'] * item['option_scores']
        total = np.round(action + option, 12)
        states.append({'qa_ids': clip['qa_ids'], 'answers': clip['answers'],
            'action': item['action_scores'].tolist(), 'option': item['option_scores'].tolist(),
            'action_contribution': action.tolist(), 'option_contribution': option.tolist(),
            'total_rounded12': total.tolist(), 'selected_state': int(np.argsort(-total, kind='stable')[0])})
    return result, {'units': np.asarray(requested), 'probabilities': probabilities, 'visual': visual, 'prior': prior,
        'scene_count': len(prepared.clips), 'option_matrix': prepared.option_matrix,
        'option_qa_ids': prepared.option_metadata.qa_id.to_numpy(dtype=str),
        'option_letters': prepared.option_metadata.letter.to_numpy(dtype=str),
        'option_probability': model['frequency'].predict_proba(prepared.option_matrix),
        'state_records_json': np.asarray(json.dumps(states))}
