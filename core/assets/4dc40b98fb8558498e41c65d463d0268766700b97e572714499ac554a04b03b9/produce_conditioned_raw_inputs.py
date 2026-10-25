#!/usr/bin/env python3
"""Produce current-batch sensor and visual banks for frozen conditioned experts."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'scripts')]

import numpy as np
import pandas as pd

from cuhkx.acquisition_v2 import read_acquisition, read_batch
from cuhkx.action_videomae_inference_bank import read_window_bank
from scripts import run_finalist_acquisition_v2 as partial
from scripts import run_finalist_pipeline as base
from scripts.extract_r34_midfeatures import collect_inputs
from scripts.produce_conditioned_sensor_inputs import sensor_requirements


def selected_stages(plan, condition):
    visual = ['roi']
    if 'sequence' in plan.experts or ('action' in plan.experts and condition == 'explicit'):
        visual += ['logits8', 'logits16']
    if 'emotion' in plan.experts:
        visual.append('emotion_visual')
    if 'harn' in plan.experts:
        visual += ['harn_middle', 'harn_r34', 'harn_videomae']
    if 'action' in plan.experts and condition == 'independent':
        visual.append('action_visual8')
    return (['sensors'] if sensor_requirements(plan, condition) else []) + visual


def prepare(args):
    qa, sample, plan = read_batch(args.qa_dir)
    condition = 'explicit' if args.scene_manifest else 'independent'
    acquisition = read_acquisition(qa, args.scene_manifest) if args.scene_manifest else None
    run = args.output.resolve()
    if run.exists():
        raise FileExistsError('Use a fresh output directory or the run command to resume')
    config = json.loads((ROOT / 'configs/finalist_acquisition_v2.json').read_text())
    config['r34_config'] = 'configs/r34_midfeatures_v1.json'
    config['videomae_config'] = 'configs/videomae_ssv2_harn_test_whole_v1.json'
    raw, modalities = partial.collect_raw(args.raw_media_root.resolve(), config, plan)
    # The legacy partial collector only knew HARn's Depth/IR R18 pair.
    # The additional R34/VideoMAE families also consume supplied Depth_Color.
    existing = {(entry['kind'], entry['relative']) for entry in raw}
    visual_root = args.raw_media_root.resolve() / config['layout']['visual']
    for unit in plan.harn_units:
        relative = str(Path(unit) / 'Depth_Color/Depth_Color.mp4')
        path = visual_root / relative
        if path.is_file() and ('visual', relative) not in existing:
            raw.append({'kind': 'visual', 'relative': relative, **base.raw_entry(path)})
    sources = base.source_closure([str(Path(__file__)), *base.EXTRACTOR_SOURCES,
        'scripts/extract_r34_midfeatures.py', 'scripts/extract_videomae_windows.py',
        'scripts/produce_conditioned_sensor_inputs.py'])
    dependencies = partial.extraction_dependencies(config, plan)
    for name in ['r34_config', 'videomae_config']:
        if 'harn' in plan.experts:
            dependencies.append(base.entry(base.resolve(config[name])))
    if 'action' in plan.experts and condition == 'independent':
        dependencies.append(base.entry(ROOT / 'configs/videomae_ssv2_hau_test_windows_v1.json'))
    run.mkdir(parents=True)
    (run / 'inputs/qa').mkdir(parents=True)
    # Restore only current question text. The metadata reader deliberately drops targets.
    raw_qa = pd.read_csv(args.qa_dir / 'test_qa.csv', dtype=str, keep_default_na=False)
    if 'question' in raw_qa:
        qa['question'] = qa.qa_id.map(raw_qa.set_index('qa_id').question)
    qa.to_csv(run / 'inputs/qa/test_qa.csv', index=False)
    sample.to_csv(run / 'inputs/qa/sample_submission.csv', index=False)
    for entry in raw:
        target = run / 'raw_view' / entry['kind'] / entry['relative']
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(entry['path'], target)
    (run / 'raw_view/sensors').mkdir(parents=True, exist_ok=True)
    pd.DataFrame(modalities).to_csv(run / 'raw_view/sensors/manifest_nonvisual.csv', index=False)
    scene = None
    if acquisition is not None:
        scene = run / 'inputs/acquisition.csv'
        acquisition.scenes.to_csv(scene, index=False)
    shutil.copyfile(args.sensor_schema, run / 'inputs/sensor_schema.json')
    profile = json.loads(args.profile.read_text())
    for branch in ['grouped', 'independent']:
        profile[branch]['features'] = {}
    generic = json.loads(base.resolve(config['generic_config']).read_text())
    for key in ['train_csv', 'test_csv', 'expected_training_units', 'expected_training_qa_units']:
        generic.pop(key, None)
    generic.update(media_root=str(run / 'raw_view/visual'),
                   weights_path=str(base.resolve(config['generic_weights'])), person_window_caches=[])
    snapshot = run / 'source_snapshot'
    snapshot.mkdir()
    for path in sources:
        target = snapshot / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    subprocess.run(['git', 'archive', '--format=tar.gz', '--output=' + str(snapshot / 'committed.tar.gz'), 'HEAD'], cwd=ROOT, check=True)
    (snapshot / 'pip_freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze', '--all'], text=True))
    inputs = [args.profile, args.sensor_schema, args.qa_dir / 'test_qa.csv', args.qa_dir / 'sample_submission.csv',
              run / 'inputs/qa/test_qa.csv', run / 'inputs/qa/sample_submission.csv', run / 'inputs/sensor_schema.json']
    if scene:
        inputs += [args.scene_manifest, scene]
    contract = {'kind': 'cuhkx_current_raw_conditioned_inputs_v1', 'created_utc': base.now(),
        'condition': condition, 'config': config, 'profile': profile, 'generic_config': generic,
        'units': list(plan.units), 'hau_units': list(plan.hau_units), 'harn_units': list(plan.harn_units),
        'qa_dir': str(run / 'inputs/qa'), 'scene_manifest': str(scene) if scene else None,
        'batch_plan': plan.to_dict(), 'stages': selected_stages(plan, condition),
        'sources': [base.entry(p) for p in sources], 'dependencies': dependencies,
        'inputs': [base.entry(p) for p in inputs], 'raw': raw, 'python': sys.executable,
        'command': sys.argv, 'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'scope': 'Current raw inputs only; no historical feature vectors, targets, new fits or candidate choice'}
    base.write(run / 'contract.json', contract)
    (run / 'contract.sha256').write_text(base.sha(run / 'contract.json') + '\n')
    base.write(run / 'state.json', {'status': 'prepared', 'stages': {}})
    print(json.dumps({'status': 'prepared', 'stages': contract['stages'], 'raw_files': len(raw)}))


def videomae_stage(run, name, attempt, contract, state):
    action = name == 'action_visual8'
    units = contract['hau_units' if action else 'harn_units']
    crops = json.loads(base.stage_output(run, state, 'roi').read_text())['windows']
    collected = collect_inputs(run / 'raw_view/visual',
        [{'unit': u, 'source': 'HAU' if action else 'HARn'} for u in units], crops)
    records = []
    for original in collected['records']:
        record = {k: original[k] for k in ['unit', 'modality', 'available', 'path', 'raw', 'metadata', 'roi']}
        record['alignment_ok'] = collected['alignment'][record['unit']]['alignment_ok']
        records.append(record)
    record_path = attempt / 'raw_records.json'
    base.write(record_path, {'records': records})
    template = 'configs/videomae_ssv2_hau_test_windows_v1.json' if action else contract['config']['videomae_config']
    config = json.loads(base.resolve(template).read_text())
    config.update(records_file=base.entry(record_path), expected_records=len(records), expected_units=len(units), declared_utc=base.now())
    if not action:
        config.update(window_seconds=max(23.6, max(r['metadata']['duration_seconds'] for r in records if r['available']) + 1), windows=1)
    for key in ['D_available_bytes_before', 'next_training_hypothesis', 'whole_sampling_verification']:
        config.pop(key, None)
    config_path = attempt / 'extraction.json'
    base.write(config_path, config)
    subprocess.run([sys.executable, str(ROOT / 'scripts/extract_videomae_windows.py'),
        '--config', str(config_path), '--output-dir', str(attempt / 'bank')], cwd=ROOT, check=True)
    output = attempt / 'bank/manifest.json'
    if action:
        bank = read_window_bank(output.parent)
        output = attempt / 'descriptors.npz'
        np.savez_compressed(output, **{key: bank[key] for key in ['X', 'units', 'feature_names']})
    return output


def stage(run, name):
    contract = base.read_contract(run)
    state = json.loads((run / 'state.json').read_text())
    if name not in contract['stages']:
        raise ValueError('Stage not requested by this batch')
    base.verify(contract['sources'] + contract['dependencies'] + contract['inputs'] + contract['raw'])
    attempt = run / 'stages' / name
    attempt.mkdir(parents=True, exist_ok=False)
    if name == 'sensors':
        output = attempt / 'sensors/report.json'
        subprocess.run([sys.executable, str(ROOT / 'scripts/produce_conditioned_sensor_inputs.py'), 'extract',
            '--qa-dir', contract['qa_dir'], '--sensor-root', str(run / 'raw_view/sensors'),
            '--schema', str(run / 'inputs/sensor_schema.json'), '--condition', contract['condition'],
            '--output', str(output.parent)], cwd=ROOT, check=True)
    elif name in ['harn_videomae', 'action_visual8']:
        output = videomae_stage(run, name, attempt, contract, state)
    elif name == 'harn_r34':
        output = attempt / 'middle/manifest.json'
        roi = base.stage_output(run, state, 'roi')
        subprocess.run([sys.executable, str(ROOT / 'scripts/extract_r34_midfeatures.py'), 'prepare',
            '--config', str(base.resolve(contract['config']['r34_config'])), '--media-root', str(run / 'raw_view/visual'),
            '--output-dir', str(output.parent), '--sources', 'HARn', '--test-qa', str(Path(contract['qa_dir']) / 'test_qa.csv'),
            '--roi-cache', str(roi), '--roi-sha256', base.sha(roi)], cwd=ROOT, check=True)
        subprocess.run([sys.executable, str(ROOT / 'scripts/extract_r34_midfeatures.py'), 'extract',
            '--run-dir', str(output.parent)], cwd=ROOT, check=True)
    elif name == 'harn_middle':
        output, receipt = partial.middle_stage(run, attempt, contract, state)
        base.write(attempt / 'result.json', {**receipt, 'primary_output': str(output)})
    else:
        base.execute_stage(run, name, attempt)
        output = Path(json.loads((attempt / 'result.json').read_text())['primary_output'])
    base.verify(contract['raw'])
    state['stages'][name] = {'status': 'complete', 'primary_output': str(output), 'sha256': base.sha(output), 'utc': base.now()}
    base.write(run / 'state.json', state)


def run_stages(run, stop_after):
    contract = base.read_contract(run)
    for name in contract['stages']:
        state = json.loads((run / 'state.json').read_text())
        if name in state['stages']:
            record = state['stages'][name]
            if record['status'] != 'complete' or base.sha(Path(record['primary_output'])) != record['sha256']:
                raise ValueError('Completed stage identity changed')
        else:
            log_path = run / (name + '.log')
            with log_path.open('x') as log:
                subprocess.run([sys.executable, str(Path(__file__)), '_stage', '--output', str(run), '--stage', name],
                    cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2'),
                    stdout=log, stderr=subprocess.STDOUT, check=True)
        if name == stop_after:
            return
    state = json.loads((run / 'state.json').read_text())
    features = {}
    for name, receipt in state['stages'].items():
        path = Path(receipt['primary_output'])
        if name == 'sensors':
            features.update({key: str(path.parent / (key + '.npz')) for key in json.loads(path.read_text())['stages']})
        elif name != 'roi':
            features[name] = str(path)
    profile = contract['profile']
    profile['grouped' if contract['condition'] == 'explicit' else 'independent']['features'] = features
    base.write(run / 'inference_profile.json', profile)
    state.update(status='complete_raw_inputs', completed_utc=base.now(), features=features,
                 remaining='Actual conditioned expert inference, grouped A if applicable, and SAT')
    base.write(run / 'state.json', state)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('prepare')
    for key in ['qa-dir', 'raw-media-root', 'sensor-schema', 'profile']:
        q.add_argument('--' + key, type=Path, required=True)
    q.add_argument('--scene-manifest', type=Path)
    for parser in [q, sub.add_parser('run'), sub.add_parser('_stage')]:
        parser.add_argument('--output', type=Path, required=True)
        if parser is not q:
            parser.add_argument('--stage')
    args = p.parse_args()
    if args.command == 'prepare':
        prepare(args)
    elif args.command == '_stage':
        stage(args.output.resolve(), args.stage)
    else:
        run_stages(args.output.resolve(), args.stage)


if __name__ == '__main__':
    main()
