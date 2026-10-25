#!/usr/bin/env python3
"""Extract current raw inputs and run both conditioned experts inside the package."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from scripts.bind_conditioned_descriptors import bind, sha


def verify_prepared(args, out):
    receipt = json.loads((out / 'prepare_receipt.json').read_text())
    if receipt['package_sha256'] != sha(ROOT / 'conditioned_package.json'):
        raise ValueError('Prepared inputs belong to a different package')
    if receipt['raw_contract_sha256'] != sha(out / 'raw/contract.json'):
        raise ValueError('Prepared raw contract changed')
    command = receipt['commands'][0]
    for flag, actual in [('--qa-dir', args.qa_dir), ('--raw-media-root', args.media_root),
                         ('--scene-manifest', args.acquisition)]:
        expected = command[command.index(flag) + 1] if flag in command else None
        supplied = str(actual.resolve()) if actual is not None else None
        if supplied != expected:
            raise ValueError('Resume input differs from preparation: ' + flag)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qa-dir', type=Path, required=True)
    parser.add_argument('--media-root', type=Path, required=True)
    parser.add_argument('--acquisition', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sat-python', type=Path, required=True)
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument('--prepare-only', action='store_true')
    stage.add_argument('--resume-prepared', action='store_true')
    parser.add_argument('--cuda-slot', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    if args.resume_prepared:
        verify_prepared(args, out)
    if not args.prepare_only:
        if not args.cuda_slot or os.environ.get('CUDA_VISIBLE_DEVICES') == '':
            raise ValueError('Exclusive visible CUDA slot required for raw visual production')
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip():
            raise RuntimeError('GPU slot occupied')
    if not args.resume_prepared:
        out.mkdir(parents=True, exist_ok=False)
        profile = bind(ROOT, out / 'bound')
        (out / 'profile.json').write_text(json.dumps(profile, indent=2) + '\n')
        (out / 'read_audits').mkdir()
    environment = dict(os.environ, PYTHONPATH=str(ROOT / 'runtime_guard'), CUHKX_RUNTIME_AUDIT_DIR=str(out / 'read_audits'),
        OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    commands = []

    def run(name, command):
        commands.append(list(map(str, command)))
        with (out / (name + '.log')).open('x') as handle:
            subprocess.run(list(map(str, command)), cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=True)

    if not args.resume_prepared:
        command = [sys.executable, ROOT / 'scripts/produce_conditioned_raw_inputs.py', 'prepare', '--qa-dir', args.qa_dir.resolve(),
            '--raw-media-root', args.media_root.resolve(), '--sensor-schema', ROOT / 'configs/current_sensor_schema.json',
            '--profile', out / 'profile.json', '--output', out / 'raw']
        if args.acquisition:
            command += ['--scene-manifest', args.acquisition.resolve()]
        run('prepare', command)
        (out / 'prepare_receipt.json').write_text(json.dumps({'package_sha256': sha(ROOT / 'conditioned_package.json'),
            'commands': commands, 'raw_contract_sha256': sha(out / 'raw/contract.json'), 'scope': 'Raw collection and source/environment capture only.'}, indent=2) + '\n')
    if args.prepare_only:
        return
    run('produce', [sys.executable, ROOT / 'scripts/produce_conditioned_raw_inputs.py', 'run', '--output', out / 'raw'])
    contract = json.loads((out / 'raw/contract.json').read_text())
    state = json.loads((out / 'raw/state.json').read_text())
    if state['status'] != 'complete_raw_inputs':
        raise ValueError('Raw production did not complete')
    inputs = {'qa_dir': contract['qa_dir'], 'features': state['features'], 'acquisition': contract['scene_manifest']}
    (out / 'feature_inputs.json').write_text(json.dumps(inputs, indent=2) + '\n')
    run('infer', [sys.executable, ROOT / 'scripts/run_conditioned_package.py', '--inputs', out / 'feature_inputs.json',
                  '--output', out / 'inference', '--sat-python', args.sat_python, '--cuda-slot'])
    (out / 'submission.csv').write_bytes((out / 'inference/submission.csv').read_bytes())
    (out / 'report.json').write_text(json.dumps({'status': 'complete_relocated_raw_conditioned_chain', 'commands': commands,
        'raw_contract_sha256': sha(out / 'raw/contract.json'), 'raw_state_sha256': sha(out / 'raw/state.json'),
        'output_sha256': sha(out / 'submission.csv'), 'new_fits': 0,
        'scope': 'Current source media through packaged extractors, conditioned experts and A/SAT. Same installed research/SAT environments; no new-collection accuracy claim.'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
