#!/usr/bin/env python3
"""Bind exact packaged A/SAT assets and run on caller-supplied current inputs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import pandas as pd

from cuhkx.submission import validate_submission_files


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def bind(runtime):
    manifest_path = ROOT / 'package_manifest.json'
    if sha(manifest_path) != (ROOT / 'package_manifest.sha256').read_text().strip():
        raise ValueError('Package manifest changed')
    package = json.loads(manifest_path.read_text())
    if package['kind'] != 'cuhkx_relocatable_current_decision_v1':
        raise ValueError('Unsupported package')
    for entry in package['files']:
        path = (ROOT / entry['relative']).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != entry['sha256']:
            raise ValueError('Packaged file changed: ' + str(path))
    locations = {key: ROOT / value for key, value in package['locations'].items()}

    def rebind(record):
        path = locations[record['path']]
        if sha(path) != record['sha256']:
            raise ValueError('Model identity differs')
        return {**record, 'path': str(path)}

    descriptor = json.loads(locations[package['descriptor']].read_text())
    original = copy.deepcopy(descriptor)
    policy = json.loads(locations[descriptor['parent_policy']['path']].read_text())
    policy['source_manifest'] = rebind(policy['source_manifest'])
    for key in ['checkpoint', 'scope']:
        policy['source_model'][key] = rebind(policy['source_model'][key])
    write(runtime / 'parent_policy.json', policy)
    descriptor['parent_policy'] = {'path': str(runtime / 'parent_policy.json'),
                                   'sha256': sha(runtime / 'parent_policy.json'),
                                   'bytes': (runtime / 'parent_policy.json').stat().st_size}
    for key in ['normalizer', 'student_scope']:
        descriptor[key] = rebind(descriptor[key])
    descriptor['students'] = {key: rebind(value) for key, value in descriptor['students'].items()}
    descriptor['sources'] = [rebind(value) for value in descriptor['sources']]
    write(runtime / 'descriptor.json', descriptor)
    sat = json.loads(locations[package['sat_profile']].read_text())
    by_sha = {entry['sha256']: ROOT / entry['relative'] for entry in package['files']}
    sat['models'] = [{**record, 'path': str(by_sha[record['sha256']])} for record in sat['models']]
    write(runtime / 'sat_profile.json', sat)
    write(runtime / 'binding.json', {'package_manifest_sha256': sha(manifest_path),
                                     'original_descriptor_sha256': sha(locations[package['descriptor']]),
                                     'runtime_descriptor_sha256': sha(runtime / 'descriptor.json'),
                                     'weights': [original['parent_weight'], original['student_weight']],
                                     'learned_states_modified': False,
                                     'change': 'Verified descriptor paths and descriptor hashes only; immutable states/source unchanged.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sat-python', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    runtime = out / 'runtime'
    runtime.mkdir()
    bind(runtime)
    inputs = json.loads(args.inputs.read_text())
    qa_path, sample_path, parent_path, acquisition = [Path(inputs[name]) for name in ['qa', 'sample', 'parent', 'acquisition']]
    input_paths = [args.inputs, qa_path, sample_path, parent_path, acquisition, *map(Path, inputs['features'].values())]
    hashes = {str(path.resolve()): sha(path) for path in input_paths}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    commands = []

    def run(python, script, parameters, name):
        command = [str(python), '-I', str(ROOT / 'scripts/run_relocated_python.py'), str(out / (name + '_reads.json')),
                   str(ROOT / 'scripts' / script), *map(str, parameters)]
        commands.append(command)
        with (out / (name + '.log')).open('x') as handle:
            subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)

    qa = pd.read_csv(qa_path, dtype=str, keep_default_na=False)
    parent = pd.read_csv(parent_path, dtype=str, keep_default_na=False)
    expected = set(qa.loc[qa.source.eq('HAU') & qa.category.eq('emotion'), 'qa_id'])
    if expected:
        parameters = ['--descriptor', runtime / 'descriptor.json', '--qa-csv', qa_path, '--scene-manifest', acquisition,
                      '--arm', 'A', '--output-dir', out / 'emotion_A']
        for name in ['imu', 'skeleton', 'radar', 'visual']:
            parameters += ['--' + name, inputs['features'][name]]
        run(sys.executable, 'infer_local_emotion_full_fusion_v2.py', parameters, 'emotion_A')
        emotion = pd.read_csv(out / 'emotion_A/test_predictions.csv', dtype=str, keep_default_na=False)
        if not emotion.qa_id.is_unique or set(emotion.qa_id) != expected:
            raise ValueError('A must exactly cover current emotion questions')
        selected = parent.qa_id.isin(expected)
        parent.loc[selected, 'prediction'] = parent.loc[selected, 'qa_id'].map(emotion.set_index('qa_id').prediction)
    parent.to_csv(out / 'composed_parent.csv', index=False)
    validate_submission_files(qa_path, sample_path, out / 'composed_parent.csv')
    groups = pd.read_csv(acquisition, dtype=str).rename(columns={'scene': 'group'})
    groups[['clip', 'group']].to_csv(out / 'groups.csv', index=False)
    run(args.sat_python, 'apply_explicit_group_sat.py', ['--qa-csv', qa_path, '--parent-csv', out / 'composed_parent.csv',
        '--profile', runtime / 'sat_profile.json', '--output-dir', out / 'sat', '--scene-manifest', out / 'groups.csv'], 'sat')
    (out / 'submission.csv').write_bytes((out / 'sat/submission.csv').read_bytes())
    validation = validate_submission_files(qa_path, sample_path, out / 'submission.csv').to_dict()
    for path, digest in hashes.items():
        if sha(path) != digest:
            raise ValueError('Caller input changed')
    write(out / 'report.json', {'status': 'complete_current_A_SAT_package_inference', 'validation': validation,
                               'commands': commands, 'inputs_sha256': hashes, 'fits': 0,
                               'emotion_A_applied': bool(expected),
                               'scope': 'Relocated current A/SAT component on supplied parent and features. Full original-media/four-expert relocation remains unverified.'})


if __name__ == '__main__':
    main()
