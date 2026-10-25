#!/usr/bin/env python3
"""Run the relocated conditioned experts and current decision chain from features."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import pandas as pd

from scripts import infer_conditioned_experts_v5 as route
from scripts.bind_conditioned_descriptors import bind
from scripts.infer_conditioned_experts_v5 import main as infer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sat-python', type=Path, required=True)
    parser.add_argument('--cuda-slot', action='store_true')
    args = parser.parse_args()
    inputs = json.loads(args.inputs.read_text())
    questions, _, batch = route.read_batch(Path(inputs['qa_dir']))
    if 'harn' in batch.experts:
        if not args.cuda_slot or os.environ.get('CUDA_VISIBLE_DEVICES') == '':
            raise ValueError('Schedule a CUDA slot for frozen HARn inference')
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip():
            raise RuntimeError('GPU slot occupied')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    profile = bind(ROOT, out / 'bound')
    condition = 'grouped' if inputs.get('acquisition') else 'independent'
    profile[condition]['features'] = inputs['features']
    config = out / 'profile.json'
    config.write_text(json.dumps(profile, indent=2) + '\n')
    original_command = route.command

    def command(name, profile, qa, work, grouped, workers):
        original = original_command(name, profile, qa, work, grouped, workers)
        return [original[0], '-I', str(ROOT / 'scripts/run_relocated_python.py'),
                str(work / (name + '_read_audit.json')), *original[1:]]

    route.command = command
    parent = out / 'four_expert_parent.csv'
    sys.argv = [str(ROOT / 'scripts/infer_conditioned_experts_v5.py'), inputs['qa_dir'], str(parent), '--config', str(config), '--workers', '2']
    if condition == 'grouped':
        sys.argv += ['--scene-manifest', inputs['acquisition']]
    infer()
    d10 = json.loads((ROOT / 'conditioned_package.json').read_text()).get('d10')
    d10_applied = bool(d10 and 'sequence' in batch.experts and condition in d10['conditions'])
    if d10_applied:
        command = [sys.executable, '-I', ROOT / 'scripts/run_relocated_python.py', out / 'd10_read_audit.json',
                   ROOT / 'scripts/infer_conditioned_d10.py', '--descriptor', ROOT / d10['descriptor'],
                   '--qa-csv', Path(inputs['qa_dir']) / 'test_qa.csv', '--output', out / 'd10']
        for name in ['logits8', 'logits16']:
            command += ['--logits', inputs['features'][name]]
        if condition == 'grouped':
            command += ['--acquisition', inputs['acquisition']]
        with (out / 'd10.log').open('x') as handle:
            subprocess.run(list(map(str, command)), cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
        temporal = pd.read_csv(out / 'd10/predictions.csv', dtype=str)
        expected = set(questions.loc[questions.source.eq('HAU') & questions.category.eq('sequence'), 'qa_id'])
        if not temporal.qa_id.is_unique or set(temporal.qa_id) != expected:
            raise ValueError('D10 must cover exactly the current sequence questions')
        legacy = out / 'legacy_four_expert_parent.csv'
        parent.rename(legacy)
        frame = pd.read_csv(legacy, dtype=str)
        selected = frame.qa_id.isin(expected)
        frame.loc[selected, 'prediction'] = frame.loc[selected, 'qa_id'].map(temporal.set_index('qa_id').prediction)
        frame.to_csv(parent, index=False)
    if condition == 'grouped':
        decision = {'qa': str(Path(inputs['qa_dir']) / 'test_qa.csv'), 'sample': str(Path(inputs['qa_dir']) / 'sample_submission.csv'),
            'parent': str(parent), 'acquisition': inputs['acquisition'],
            'features': {name: inputs['features']['emotion_visual' if name == 'visual' else name]
                         for name in ['imu', 'skeleton', 'radar', 'visual']} if 'emotion' in batch.experts else {}}
        decision_path = out / 'decision_inputs.json'
        decision_path.write_text(json.dumps(decision, indent=2) + '\n')
        command = [sys.executable, '-I', ROOT / 'scripts/run_relocated_python.py', out / 'decision_read_audit.json',
            ROOT / 'scripts/run_current_decision_package.py', '--inputs', decision_path, '--output', out / 'decision', '--sat-python', args.sat_python]
    else:
        command = [args.sat_python, '-I', ROOT / 'scripts/run_relocated_python.py', out / 'sat_read_audit.json',
            ROOT / 'scripts/apply_scored_sat_v3.py', '--qa-csv', Path(inputs['qa_dir']) / 'test_qa.csv',
            '--parent-csv', parent, '--profile', ROOT / 'configs/sat_independent_general_v1.json', '--output-dir', out / 'decision']
    with (out / 'decision.log').open('x') as handle:
        subprocess.run(list(map(str, command)), cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
    (out / 'submission.csv').write_bytes((out / 'decision/submission.csv').read_bytes())
    (out / 'report.json').write_text(json.dumps({'status': 'complete_conditioned_feature_package', 'condition': condition,
        'fits': 0, 'rows': len(questions), 'active_experts': list(batch.experts), 'input_contract': str(args.inputs),
        'D10_applied': d10_applied,
        'scope': 'Requested active conditioned experts and current A/SAT on caller features. Original-media production and clean environment installation are not covered.'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
