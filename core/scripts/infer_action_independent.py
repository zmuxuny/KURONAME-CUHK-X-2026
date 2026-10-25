#!/usr/bin/env python3
"""Export or infer the audited independent-recording linear eight-window action expert."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from cuhkx import action_generic_nested as base
from cuhkx.action_independent_inference import KIND, PARENT_KIND, predict
from cuhkx.action_joint_harn_head import JointHarnWindowClassifier
from cuhkx.action_window_head import FEATURE_DIMENSION, WINDOWS
from cuhkx.emotion_nested import OFFICIAL_SUBJECTS
from cuhkx.validation import sha256_file
from scripts.research_write_guard import install


def identity(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha256_file(path), 'bytes': path.stat().st_size}


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def export(run, audit, output):
    run, audit = run.resolve(), audit.resolve()
    frozen = json.loads((run / 'frozen_config.json').read_text())
    report = json.loads((run / 'report.json').read_text())
    audited = json.loads((audit / 'report.json').read_text())
    if (frozen['config']['kind'] != PARENT_KIND or json.loads((run / 'state.json').read_text())['status'] != 'complete'
            or audited['status'] != 'completed_independent_actual_model_and_selection_replay'
            or len(audited['audits']) != 76 or len(audited['selection_replays']) != 19
            or sum(row['prediction_replays'] for row in audited['audits']) != 975):
        raise ValueError('requires completed independent training and all-scope replay')
    if any(value['overall']['correct_rows'] != 2137 or value['overall']['rows'] != 2408
           for value in audited['metrics'].values()):
        raise ValueError('independent audit score does not match the declared parent')
    code = {}
    for item in frozen['source_manifest']:
        path = Path(item['path'])
        if sha256_file(path) != item['sha256']:
            raise ValueError('frozen training source changed')
        relative = path.relative_to(Path('/root/workspace/CUHK-X')) if path.is_relative_to('/root/workspace/CUHK-X') else path.relative_to(ROOT)
        if sha256_file(ROOT / relative) != item['sha256']:
            raise ValueError('inference dependency differs from audited training source')
        code[str(relative)] = item['sha256']
    for name in ['scripts/infer_action_independent.py', 'src/cuhkx/action_independent_inference.py',
                 'scripts/research_write_guard.py']:
        code[name] = sha256_file(ROOT / name)
    checkpoint = run / 'model.joblib'
    if sha256_file(checkpoint) != (run / 'model.sha256').read_text().strip():
        raise ValueError('final checkpoint changed')
    model = joblib.load(checkpoint)
    selected = json.loads((run / 'final_selection/selection.json').read_text())['selected_decoder']
    if (model['kind'] != PARENT_KIND or model['layout'] != 'independent_recordings'
            or type(model['presence']) is not JointHarnWindowClassifier
            or set(model['training_subjects']) != OFFICIAL_SUBJECTS or model['validation_subjects']
            or model['selected_decoder'] != selected or report['selected_final_decoder'] != selected):
        raise ValueError('parent architecture, condition, final subjects or decoder mismatch')
    head = copy.deepcopy(model['presence'])
    prior = head.predict_prior(np.empty((1, 0)))[0]
    del head.training_targets
    if hasattr(head, 'auxiliary_scope'):
        del head.auxiliary_scope
    exported = {key: model[key] for key in ['frequency', 'training_subjects', 'validation_subjects',
                                           'known_action_classes', 'feature_names', 'selected_decoder']}
    exported.update(kind=KIND, layout='independent_recordings', presence=head, prior=prior)
    output.mkdir(parents=True, exist_ok=False)
    joblib.dump(exported, output / 'model.joblib', compress=3)
    reloaded = joblib.load(output / 'model.joblib')
    for name in head.state:
        np.testing.assert_array_equal(head.state[name], reloaded['presence'].state[name])
    provenance = [identity(path) for path in [checkpoint, run / 'final_selection/selection.json',
                  run / 'report.json', run / 'frozen_config.json', audit / 'report.json']]
    write(output / 'manifest.json', {'kind': KIND, 'model': identity(output / 'model.joblib'),
        'selected_decoder': selected, 'sources': code, 'provenance': provenance,
        'feature_spec': {'windows': WINDOWS, 'dimension': FEATURE_DIMENSION,
                         'layout': '8 windows x (Depth_Color, IR) x 768 plus 16 availability flags',
                         'training_bank': frozen['config']['window_bank']},
        'validation': {'scope': 'Independent recordings, all same-clip QA context retained',
                       'correct_rows': 2137, 'rows': 2408, 'subjects': 18, 'historically_observed': True,
                       'matched_visual_and_harn_upstream': True},
        'input_policy': 'No acquisition argument or file discovery. Recording IDs are equality/descriptor lookup keys only. Every recording is its own group.',
        'limits': 'HAU action expert only; missing same-clip questions and new raw acquisitions are not validated. This is not best whole-system OOF.'})


def infer(bundle, qa_path, features, output):
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if manifest['kind'] != KIND or sha256_file(bundle / 'model.joblib') != manifest['model']['sha256']:
        raise ValueError('export manifest or checkpoint changed')
    for path, digest in manifest['sources'].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError('exported inference source changed')
    allowed = {path.resolve() for path in [bundle / 'model.joblib', qa_path, features]}
    reads = set()

    def guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1:]
        if (mode and any(value in mode for value in 'wax+')) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
            return
        if path.suffix in {'.csv', '.npy', '.npz', '.joblib'}:
            if path not in allowed and not path.is_relative_to(output.resolve()):
                raise PermissionError(f'unlisted inference data read: {path}')
            reads.add(str(path))

    sys.addaudithook(guard)
    model = joblib.load(bundle / 'model.joblib')
    model['frequency'].n_jobs = 1
    qa = pd.read_csv(qa_path, usecols=base.METADATA, dtype=str, keep_default_na=False)
    with np.load(features, allow_pickle=False) as archive:
        bank = {name: archive[name] for name in ['units', 'X', 'feature_names']}
    predictions, evidence = predict(model, qa, bank)
    output.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(output / 'predictions.csv', index=False)
    np.savez_compressed(output / 'evidence.npz', **evidence)
    write(output / 'report.json', {'status': 'complete', 'rows': len(predictions),
        'recordings': len(evidence['units']), 'condition': 'independent_recordings',
        'scene_count': evidence['scene_count'], 'decoder': model['selected_decoder'],
        'qa': identity(qa_path), 'features': identity(features), 'model': identity(bundle / 'model.joblib'),
        'prediction': identity(output / 'predictions.csv'), 'target_columns_read': False,
        'observed_data_reads': sorted(reads), 'input_allowlist': sorted(map(str, allowed)),
        'acquisition_read': False, 'cross_recording_sharing': False,
        'scope': 'Actual model inference from supplied eight-window descriptors. No fit, extraction or upload.'})


def main():
    install(Path('/root/workspace/CUHK-X'))
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    exp = sub.add_parser('export')
    exp.add_argument('--run', required=True, type=Path)
    exp.add_argument('--audit', required=True, type=Path)
    exp.add_argument('--output-dir', required=True, type=Path)
    inf = sub.add_parser('infer')
    inf.add_argument('--bundle', required=True, type=Path)
    inf.add_argument('--qa-csv', required=True, type=Path)
    inf.add_argument('--features', required=True, type=Path)
    inf.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.command == 'export':
            export(args.run, args.audit, args.output_dir)
        else:
            infer(args.bundle, args.qa_csv, args.features, args.output_dir)


if __name__ == '__main__':
    main()
