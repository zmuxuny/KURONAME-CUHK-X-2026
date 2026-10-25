"""Bind a preselected external R34 candidate to its source and fusion recipe."""
from __future__ import annotations

import json


def selected_option_target_recipe(config, read):
    selected = json.loads(read(config['selection_record'], config['selection_sha256']).read_text())
    if selected['selected'] != 'option_BCE_RGB_cosine' or config['selected_arm'] != selected['selected']:
        raise ValueError('Candidate differs from the selected option-space experiment')
    external = config['external_source']
    if external['arm'] != 'depth' or external['sha256'] != selected['source_sha256']:
        raise ValueError('Option-space candidate source differs')
    read(external['path'], external['sha256'])
    return 'depth', 'half'


def selected_paired_target_recipe(config, read):
    """Use the Train-selected paired-depth target model and fixed half fusion."""
    selected = json.loads(read(config['selection_record'], config['selection_sha256']).read_text())
    if (selected['selected'] != config['selected_arm']
            or selected['source_name'] != 'BCE_RGB_cosine'
            or selected['fusion'] != 'half' or not selected['temporal_linear']):
        raise ValueError('Candidate differs from the Train-selected paired target recipe')
    external = config['external_source']
    if external['arm'] != 'depth' or external['sha256'] != selected['source_sha256']:
        raise ValueError('Paired target source differs from the selected source')
    read(external['path'], external['sha256'])
    return 'depth', 'half'


def selected_external_recipe(config, read):
    recipes = {
        'E_RGB_replace': ('rgb', 'replace'),
        'E_RGB_half': ('rgb', 'half'),
        'E_Depth_replace': ('depth', 'replace'),
        'E_Depth_half': ('depth', 'half'),
    }
    selected = json.loads(read(config['selection_record'], config['selection_sha256']).read_text())
    arm = selected['selected_single_exploratory_arm']
    if arm not in recipes or arm != config['selected_arm']:
        raise ValueError('Candidate differs from the preselected external-source recipe')
    source, fusion = recipes[arm]
    external = config['external_source']
    if external['arm'] != source:
        raise ValueError('Selected source differs from declared source checkpoint')
    read(external['path'], external['sha256'])
    for path, digest in selected['evidence_sha256'].items():
        read(path, digest)
    return source, fusion
