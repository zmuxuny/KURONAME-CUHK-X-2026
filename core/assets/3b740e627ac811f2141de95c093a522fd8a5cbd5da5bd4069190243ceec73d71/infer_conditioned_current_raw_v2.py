#!/usr/bin/env python3
"""Run frozen conditioned experts, grouped A and SAT from current raw producers."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import pandas as pd

from cuhkx.submission import validate_submission_files
from scripts import run_finalist_pipeline as base


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sat-python', type=Path, required=True)
    p.add_argument('--sat-profile', type=Path)
    p.add_argument('--emotion-descriptor', type=Path, default=ROOT / 'runs/20260912_local_emotion_final_component_v2/descriptor.json')
    p.add_argument('--cuda-slot', action='store_true', help='Run HARn tails within an exclusively scheduled GPU slot')
    args = p.parse_args()
    raw = args.raw_run.resolve()
    contract = base.read_contract(raw)
    state = json.loads((raw / 'state.json').read_text())
    if contract['kind'] != 'cuhkx_current_raw_conditioned_inputs_v1' or state['status'] != 'complete_raw_inputs':
        raise ValueError('Complete current-batch raw producer required')
    if set(state['stages']) != set(contract['stages']):
        raise ValueError('Raw stage coverage differs')
    base.verify(contract['sources'] + contract['dependencies'] + contract['inputs'] + contract['raw'])
    for entry in state['stages'].values():
        if entry['status'] != 'complete' or base.sha(Path(entry['primary_output'])) != entry['sha256']:
            raise ValueError('Actual raw producer output changed')
    profile_path = raw / 'inference_profile.json'
    profile = json.loads(profile_path.read_text())
    grouped = contract['condition'] == 'explicit'
    features = profile['grouped' if grouped else 'independent']['features']
    if features != state['features']:
        raise ValueError('Inference must use the completed current-batch feature paths')
    args.sat_profile = args.sat_profile or ROOT / 'configs' / ('sat_min_edit_explicit_v1.json' if grouped else 'sat_independent_general_v1.json')
    if json.loads(args.sat_profile.read_text())['layout'] != 'explicit_or_independent':
        raise ValueError('New inputs require the explicit-or-independent SAT contract')
    needs_cuda = 'harn' in contract['batch_plan']['experts']
    if needs_cuda:
        if not args.cuda_slot or os.environ.get('CUDA_VISIBLE_DEVICES') == '':
            raise ValueError('Frozen HARn R18/R34 tails require CUDA; schedule an exclusive slot and pass --cuda-slot')
        occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
        if occupied:
            raise RuntimeError('GPU slot is occupied: ' + occupied)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / 'source_snapshot'
    snapshot.mkdir()
    (snapshot / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    subprocess.run(['git', 'archive', '--format=tar.gz', '--output=' + str(snapshot / 'committed.tar.gz'), 'HEAD'], cwd=ROOT, check=True)
    (snapshot / 'pip_freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze', '--all'], text=True))
    qa_dir = Path(contract['qa_dir'])
    qa = pd.read_csv(qa_dir / 'test_qa.csv', dtype=str, keep_default_na=False)
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    if not needs_cuda:
        env['CUDA_VISIBLE_DEVICES'] = ''
    report = {'status': 'running', 'raw_contract': base.entry(raw / 'contract.json'),
        'raw_state': base.entry(raw / 'state.json'), 'source': base.entry(Path(__file__)),
        'profile': base.entry(profile_path), 'sat_profile': base.entry(args.sat_profile),
        'condition': contract['condition'], 'stages': {}, 'new_fits': 0, 'pseudo_dependency': False,
        'argv': sys.argv, 'python': sys.executable,
        'runtime_environment': {key: env.get(key) for key in ['CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']},
        'cuda_required_by_frozen_harn_tails': needs_cuda,
        'scope': 'Actual current raw features to four experts, A for explicit groups with emotion, and legal SAT output. Does not claim a new accuracy score.'}
    def save():
        base.write(out / 'report.json', report)
    def run(name, command):
        started = time.perf_counter()
        report['stages'][name] = {'command': list(map(str, command)), 'status': 'running'}
        save()
        with (out / (name + '.log')).open('x') as log:
            subprocess.run(list(map(str, command)), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        report['stages'][name]['status'] = 'complete'
        report['stages'][name]['elapsed_seconds'] = time.perf_counter() - started
        save()
    save()
    try:
        parent = out / 'four_expert_parent.csv'
        command = [sys.executable, ROOT / 'scripts/infer_conditioned_experts_v5.py', qa_dir, parent,
                   '--config', profile_path, '--workers', '2']
        if grouped:
            command += ['--scene-manifest', contract['scene_manifest']]
        run('four_experts', command)
        composed = out / 'composed_parent.csv'
        frame = pd.read_csv(parent, dtype=str, keep_default_na=False)
        if grouped and 'emotion' in contract['batch_plan']['experts']:
            component = out / 'emotion_A'
            command = [sys.executable, ROOT / 'scripts/infer_local_emotion_full_fusion_v2.py',
                '--descriptor', args.emotion_descriptor.resolve(), '--qa-csv', qa_dir / 'test_qa.csv',
                '--scene-manifest', contract['scene_manifest'], '--arm', 'A', '--output-dir', component]
            for name in ['imu', 'skeleton', 'radar', 'visual']:
                command += ['--' + name, features['emotion_visual' if name == 'visual' else name]]
            run('emotion_A', command)
            emotion = pd.read_csv(component / 'test_predictions.csv', dtype=str, keep_default_na=False)
            expected = set(qa.loc[qa.source.eq('HAU') & qa.category.eq('emotion'), 'qa_id'])
            if not emotion.qa_id.is_unique or set(emotion.qa_id) != expected:
                raise ValueError('A output does not exactly cover current emotion questions')
            selected = frame.qa_id.isin(expected)
            frame.loc[selected, 'prediction'] = frame.loc[selected, 'qa_id'].map(emotion.set_index('qa_id').prediction)
        frame.to_csv(composed, index=False)
        validate_submission_files(qa_dir / 'test_qa.csv', qa_dir / 'sample_submission.csv', composed)
        sat_script = 'apply_explicit_group_sat.py' if grouped and json.loads(args.sat_profile.read_text())['mode'] == 'sat_min_edit' else 'apply_scored_sat_v3.py'
        command = [args.sat_python.absolute(), ROOT / 'scripts' / sat_script,
            '--qa-csv', qa_dir / 'test_qa.csv', '--parent-csv', composed, '--profile', args.sat_profile.resolve(),
            '--output-dir', out / 'sat']
        if grouped:
            mapping = pd.read_csv(contract['scene_manifest'], dtype=str)
            if 'group' not in mapping:
                mapping = mapping.rename(columns={'scene': 'group'})
            mapping[['clip', 'group']].to_csv(out / 'sat_groups.csv', index=False)
            command += ['--scene-manifest', out / 'sat_groups.csv']
        run('sat', command)
        result = out / 'submission.csv'
        result.write_bytes((out / 'sat/submission.csv').read_bytes())
        validation = validate_submission_files(qa_dir / 'test_qa.csv', qa_dir / 'sample_submission.csv', result).to_dict()
        base.verify(contract['raw'])
        report.update(status='complete', output=base.entry(result), validation=validation,
                      raw_files_rehashed=len(contract['raw']), completed_utc=base.now())
    except BaseException as error:
        report.update(status='failed_preserved', error=repr(error))
        raise
    finally:
        save()
    print(json.dumps({'status': report['status'], 'output': report['output']}))


if __name__ == '__main__':
    main()
