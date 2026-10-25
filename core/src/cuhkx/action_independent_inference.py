"""Independent-recording inference for the audited linear eight-window HAU head."""
from __future__ import annotations

import hashlib

import numpy as np

from cuhkx import action_generic_nested as base
from cuhkx.action_explicit_layout import prepare
from cuhkx.action_joint_harn_head import JointHarnWindowClassifier
from cuhkx.action_window_decoding import compile_questions, evidence_from_probabilities
from cuhkx.action_window_head import FEATURE_DIMENSION
from cuhkx.metadata import clip_key, feature_unit

KIND = 'cuhkx_independent_linear_eight_window_action_inference_v1'
PARENT_KIND = 'cuhkx_action_independent_window_nested_v1'


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
    if recipe['branch'] == 'visual':
        probabilities = np.concatenate([model['presence'].predict(row[None]) for row in values])
    else:
        probabilities = np.repeat(model['prior'][None], len(values), axis=0)
    evidence = evidence_from_probabilities(model, prepared, compiled, units, probabilities,
                                          prepared.frame.qa_id.tolist())
    result = base.predict(prepared, evidence[recipe['likelihood']], recipe)
    if result.qa_id.tolist() != prepared.frame.qa_id.tolist():
        raise AssertionError('inference changed QA order')
    return result, {'units': np.asarray(requested), 'probabilities': probabilities,
                    'scene_count': len(prepared.clips), 'option_matrix': prepared.option_matrix}
