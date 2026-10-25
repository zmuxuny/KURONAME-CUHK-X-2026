#!/usr/bin/env python3
"""Route actual experts by explicitly supplied scene information and QA types."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import numpy as np
import pandas as pd

from cuhkx.acquisition_v2 import assemble, expert_questions, read_batch
from cuhkx.metadata import clip_key, feature_unit
from scripts.research_write_guard import install

KIND = 'cuhkx_explicit_condition_four_experts_v1'


def entry(path):
    path = Path(path).resolve()
    with path.open('rb') as handle:
        sha = hashlib.file_digest(handle, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': sha, 'bytes': path.stat().st_size}


def resolve(path):
    return (ROOT / path).resolve()


def commands(config, plan, input_paths, work, workers):
    profile = config['independent']
    checkpoints = {name: resolve(profile['checkpoints'][name]) for name in plan.experts}
    features = profile['features']
    result = {}
    for name in plan.experts:
        qa = input_paths[name]
        destination = work / name
        if name == 'action':
            command = ['infer_action_independent.py', 'infer', '--bundle', checkpoints[name],
                '--qa-csv', qa, '--features', qa.with_name('descriptors.npz'), '--output-dir', destination]
        elif name == 'emotion':
            command = ['infer_emotion_independent.py', 'infer', '--bundle', checkpoints[name], '--qa-csv', qa,
                '--output', destination / 'test_predictions.csv']
            for key in ['imu', 'skeleton', 'radar', 'visual']:
                command += ['--' + key, resolve(features['emotion_visual' if key == 'visual' else key])]
        elif name == 'sequence':
            command = ['temporal_checkpoint.py', 'infer', '--checkpoint', checkpoints[name], '--qa-csv', qa,
                '--logits', resolve(features['logits8']), '--logits', resolve(features['logits16']),
                '--groups-json', qa.with_name('independent_groups.json'), '--output-dir', destination,
                '--workers', workers]
        else:
            command = ['run_harn_seed_ensemble.py', 'infer', '--checkpoint', checkpoints[name], '--qa-csv', qa,
                '--skeleton', resolve(features['skeleton']), '--visual', resolve(features['harn_middle']),
                '--output-dir', destination]
        result[name] = [sys.executable, str(ROOT / 'scripts/research_write_guard.py'),
            str(ROOT / 'scripts' / command[0]), *map(str, command[1:])]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--scene-manifest', type=Path)
    parser.add_argument('--workers', type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    install(Path('/root/workspace/CUHK-X'))
    config = json.loads(args.config.read_text())
    if config.get('kind') != KIND:
        raise ValueError('explicit-condition profile required')
    qa, sample, plan = read_batch(args.data_dir)
    output = args.output.resolve()
    report_path = output.with_suffix('.routing.json')
    if any(path.exists() for path in [output, output.with_suffix('.work'), report_path]):
        raise FileExistsError('new output paths required')
    output.parent.mkdir(parents=True, exist_ok=True)
    grouped = args.scene_manifest is not None
    routing = {'kind': KIND, 'status': 'running', 'condition': 'explicit_scene' if grouped else 'independent_recordings',
        'active_experts': list(plan.experts), 'selection': 'Only explicit scene argument; no adjacent file or recording ID interpretation.',
        'config': entry(args.config), 'qa': entry(args.data_dir / 'test_qa.csv'),
        'scene': entry(args.scene_manifest) if grouped else None,
        'sources': [entry(Path(__file__)), entry(ROOT / 'scripts/infer_grouped_experts_research.py')],
        'new_fits': 0, 'pseudo_dependency': False, 'scope': 'From supplied keyed features; raw extraction is upstream.',
        'limits': ['Corresponding per-expert Train metrics have different protocols; no summed unified OOF claim.',
            'Independent temporal model retains K-KUNO with unknown backbone fitting people.',
            'Missing same-clip action questions have not been separately validated for the independent action model.']}

    def save():
        report_path.write_text(json.dumps(routing, indent=2) + '\n')

    save()
    environment = os.environ.copy()
    environment.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS=str(args.workers), MKL_NUM_THREADS=str(args.workers))
    try:
        if grouped:
            command = [sys.executable, str(ROOT / 'scripts/infer_grouped_experts_research.py'),
                str(args.data_dir.resolve()), str(output), '--config', str(resolve(config['grouped_profile'])),
                '--scene-manifest', str(args.scene_manifest.resolve()), '--workers', str(args.workers)]
            with output.with_suffix('.grouped.log').open('x') as log:
                subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        else:
            work = output.with_suffix('.work')
            work.mkdir()
            inputs = work / 'inputs'
            inputs.mkdir()
            paths = {}
            for name in plan.experts:
                directory = inputs / name
                directory.mkdir()
                frame = expert_questions(qa, name)
                path = directory / 'test_qa.csv'
                frame.to_csv(path, index=False)
                paths[name] = path
                if name == 'action':
                    feature_path = resolve(config['independent']['features']['action_visual8'])
                    with np.load(feature_path, allow_pickle=False) as archive:
                        units, values, names = [archive[key] for key in ['units', 'X', 'feature_names']]
                    requested = sorted({feature_unit(value, training=False) for value in frame.path})
                    lookup = {str(unit): index for index, unit in enumerate(units)}
                    if len(lookup) != len(units) or not set(requested) <= set(lookup):
                        raise ValueError('action feature keys must be unique and cover this batch')
                    np.savez_compressed(directory / 'descriptors.npz', units=np.asarray(requested),
                        X=values[[lookup[unit] for unit in requested]], feature_names=names)
                    routing['action_feature_input'] = entry(feature_path)
                if name == 'sequence':
                    clips = sorted(set(frame.path.map(clip_key)))
                    (directory / 'independent_groups.json').write_text(json.dumps({'schema_version': 1,
                        'groups': {clip: 'independent_' + str(index) for index, clip in enumerate(clips)}}) + '\n')
            frames = {}
            for name, command in commands(config, plan, paths, work, args.workers).items():
                print('expert=' + name, flush=True)
                with (work / (name + '.log')).open('x') as log:
                    subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
                prediction_name = 'predictions.csv' if name == 'action' else 'test_predictions.csv'
                frames[name] = pd.read_csv(work / name / prediction_name, dtype=str, keep_default_na=False)
            submission, decisions = assemble(qa, sample, frames)
            submission.to_csv(output, index=False)
            decisions.to_csv(output.with_suffix('.decisions.csv'), index=False)
        routing.update(status='complete', rows=len(qa), output=entry(output))
        save()
    except BaseException as error:
        routing.update(status='failed_preserved', error=repr(error))
        save()
        raise
    print(json.dumps({'status': routing['status'], 'condition': routing['condition'], 'rows': len(qa)}), flush=True)


if __name__ == '__main__':
    main()
