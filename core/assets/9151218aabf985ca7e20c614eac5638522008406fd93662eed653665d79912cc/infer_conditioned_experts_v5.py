#!/usr/bin/env python3
"""Route full-scene reference or independently trained visual-option action224 expert."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import numpy as np
import pandas as pd

from cuhkx.acquisition_v2 import (
    assemble,
    expert_acquisition,
    expert_questions,
    read_acquisition,
    read_batch,
)
from cuhkx.metadata import clip_key, feature_unit
from scripts.infer_conditioned_experts_v3 import entry, resolve
from scripts.research_write_guard import install

KIND = 'cuhkx_conditioned_visual_option_four_experts_v1'


def command(name, profile, qa, work, grouped, workers):
    checkpoint = resolve(profile['checkpoints'][name])
    features = profile['features']
    output = work / name
    if name == 'harn':
        args = ['infer_harn_three_family_v2.py', '--checkpoint', checkpoint, '--qa-csv', qa,
            '--skeleton', resolve(features['skeleton']), '--visual', resolve(features['harn_middle']),
            '--r34', resolve(features['harn_r34']), '--videomae', resolve(features['harn_videomae']),
            '--workers', workers, '--output-dir', output]
    elif name == 'sequence':
        args = ['temporal_checkpoint.py', 'infer', '--checkpoint', checkpoint, '--qa-csv', qa,
            '--logits', resolve(features['logits8']), '--logits', resolve(features['logits16']),
            '--groups-json', qa.with_name('groups.json'), '--workers', workers, '--output-dir', output]
    elif name == 'action' and grouped:
        args = ['hau_action_acquisition.py', 'infer', '--checkpoint', checkpoint, '--qa-csv', qa,
            '--logits', resolve(features['logits8']), resolve(features['logits16']),
            '--window-features', *[resolve(features[key]) for key in ['windows16', 'windows32', 'windows48']],
            '--scene-manifest', qa.with_name('acquisition.csv'), '--workers', workers, '--output-dir', output]
    elif name == 'action':
        args = ['infer_action_independent_visual.py', 'infer', '--bundle', checkpoint, '--qa-csv', qa,
            '--features', qa.with_name('descriptors.npz'), '--output-dir', output]
    elif grouped:
        args = ['run_emotion_recipe_nested.py', 'infer', '--policy', checkpoint, '--test-csv', qa,
            '--scene-manifest', qa.with_name('acquisition.csv'), '--output-dir', output]
        for key in ['imu', 'skeleton', 'radar', 'visual']:
            args += ['--' + key, resolve(features['emotion_visual' if key == 'visual' else key])]
    else:
        args = ['infer_emotion_independent.py', 'infer', '--bundle', checkpoint, '--qa-csv', qa,
            '--output', output / 'test_predictions.csv']
        for key in ['imu', 'skeleton', 'radar', 'visual']:
            args += ['--' + key, resolve(features['emotion_visual' if key == 'visual' else key])]
    return [sys.executable, str(ROOT / 'scripts/research_write_guard.py'),
        str(ROOT / 'scripts' / args[0]), *map(str, args[1:])]


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
    if config['kind'] != KIND:
        raise ValueError('expected explicit three-family condition profile')
    qa, sample, plan = read_batch(args.data_dir)
    grouped = args.scene_manifest is not None
    acquisition = read_acquisition(qa, args.scene_manifest) if grouped else None
    profile = config['grouped' if grouped else 'independent']
    output = args.output.resolve()
    work, report_path = output.with_suffix('.work'), output.with_suffix('.routing.json')
    if any(path.exists() for path in [output, work, report_path, output.with_suffix('.decisions.csv')]):
        raise FileExistsError('new output required')
    work.mkdir(parents=True)
    inputs = work / 'inputs'
    inputs.mkdir()
    report = {'kind': KIND, 'status': 'running', 'condition': 'explicit_scene' if grouped else 'independent_recordings',
        'active_experts': list(plan.experts), 'config': entry(args.config), 'qa': entry(args.data_dir / 'test_qa.csv'),
        'scene': entry(args.scene_manifest) if grouped else None, 'sources': [entry(Path(__file__)),
            entry(ROOT / 'scripts/infer_conditioned_experts_v3.py'),
            entry(ROOT / 'src/cuhkx/action_independent_visual_inference.py')], 'stages': {},
        'selection': 'Explicit metadata availability only. Fixed three-family HARn in both conditions; no ID or adjacent-file interpretation.',
        'new_fits': 0, 'pseudo_dependency': False, 'scope': 'Supplied features to complete answers; raw extraction upstream.',
        'limits': ['Per-expert Train protocols differ; no summed whole-system OOF.',
            'HAU K-KUNO backbone fitting subjects remain unknown.',
            'Independent action Train: full2183/2408, nosequence2061/2408, single649/809, multi372/809, combination671/790; multi-only remains weak. All18 people historically observed.']}

    def save():
        report_path.write_text(json.dumps(report, indent=2) + '\n')

    save()
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS=str(args.workers),
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS=str(args.workers))
    frames = {}
    try:
        for name in plan.experts:
            directory = inputs / name
            directory.mkdir()
            frame = expert_questions(qa, name)
            path = directory / 'test_qa.csv'
            frame.to_csv(path, index=False)
            if grouped and name != 'harn':
                subset = expert_acquisition(frame, acquisition)
                subset.scenes.to_csv(directory / 'acquisition.csv', index=False)
                (directory / 'groups.json').write_text(json.dumps(subset.temporal_manifest()) + '\n')
            elif name == 'sequence':
                clips = sorted(set(frame.path.map(clip_key)))
                (directory / 'groups.json').write_text(json.dumps({'schema_version': 1,
                    'groups': {clip: 'independent_' + str(i) for i, clip in enumerate(clips)}}) + '\n')
            elif name == 'action':
                feature_path = resolve(profile['features']['action_visual8'])
                with np.load(feature_path, allow_pickle=False) as archive:
                    units, values, names = [archive[key] for key in ['units', 'X', 'feature_names']]
                requested = sorted({feature_unit(value, training=False) for value in frame.path})
                lookup = {str(unit): index for index, unit in enumerate(units)}
                if len(lookup) != len(units) or not set(requested) <= set(lookup):
                    raise ValueError('action feature keys must cover exactly this batch after slicing')
                np.savez_compressed(directory / 'descriptors.npz', units=np.asarray(requested),
                    X=values[[lookup[unit] for unit in requested]], feature_names=names)
                report['action_feature_input'] = entry(feature_path)
            args_list = command(name, profile, path, work, grouped, args.workers)
            print('expert=' + name, flush=True)
            with (work / (name + '.log')).open('x') as log:
                subprocess.run(args_list, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            filename = 'predictions.csv' if name == 'action' and not grouped else 'test_predictions.csv'
            prediction = work / name / filename
            frames[name] = pd.read_csv(prediction, dtype=str, keep_default_na=False)
            report['stages'][name] = {'command': args_list, 'prediction': entry(prediction), 'input': entry(path)}
            save()
        submission, decisions = assemble(qa, sample, frames)
        submission.to_csv(output, index=False)
        decisions.to_csv(output.with_suffix('.decisions.csv'), index=False)
        report.update(status='complete', rows=len(qa), output=entry(output))
        save()
    except BaseException as error:
        report.update(status='failed_preserved', error=repr(error))
        save()
        raise
    print(json.dumps({'status': 'complete', 'condition': report['condition'], 'rows': len(qa)}), flush=True)


if __name__ == '__main__':
    main()
