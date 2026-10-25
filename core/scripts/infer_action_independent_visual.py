#!/usr/bin/env python3
"""Export or infer the fixed224-column independent action expert; no fitting."""
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
from cuhkx.action_independent_visual_inference import KIND, predict
from cuhkx.action_joint_harn_head import JointHarnWindowClassifier
from cuhkx.sequence_kuno_local_fusion import forest_digest
from cuhkx.validation import sha256_file
from scripts.infer_action_independent import identity, write
from scripts.research_write_guard import install


def export(config_path, output):
    config = json.loads(config_path.read_text())
    old = ROOT / config['old_bundle']
    manifest = json.loads((old / 'manifest.json').read_text())
    assert identity(old / 'model.joblib')['sha256'] == manifest['model']['sha256']
    for path, digest in manifest['sources'].items():
        assert sha256_file(ROOT / path) == digest
    model = joblib.load(old / 'model.joblib')
    checkpoint = ROOT / config['forest']
    forest = joblib.load(checkpoint)
    parent = Path(forest['parent_checkpoint']['path'])
    assert identity(parent)['sha256'] == forest['parent_checkpoint']['sha256']
    original = joblib.load(parent)
    assert type(model['presence']) is JointHarnWindowClassifier
    for name, value in original['presence'].state.items():
        np.testing.assert_array_equal(value, model['presence'].state[name])
    np.testing.assert_array_equal(original['presence'].predict_prior(np.empty((1, 0)))[0], model['prior'])
    assert forest['training_subjects'] == model['training_subjects'] and not forest['validation_subjects']
    assert forest['selected_decoder'] == model['selected_decoder']
    assert forest['frequency'].n_features_in_ == 224 and forest_digest(forest['frequency']) == forest['forest_sha256']
    summary = json.loads((ROOT / config['train_summary']).read_text())
    amendment = json.loads((ROOT / config['feature_amendment']).read_text())
    assert summary['forest_input_dimension'] == 224 and amendment['actual_bank_view_order'] == ['Depth_Color', 'IR']
    assert summary['metrics']['versus_original_independent']['full_same_clip']['overall']['correct_rows'] == 2183
    output.mkdir(parents=True)
    exported = copy.deepcopy(model)
    exported.update(kind=KIND, frequency=forest['frequency'], option_dimension=224,
        visual_feature_names=amendment['corrected_feature_names'])
    exported['frequency'].n_jobs = 1
    joblib.dump(exported, output / 'model.joblib', compress=('xz', 3))
    restored = joblib.load(output / 'model.joblib')
    assert forest_digest(restored['frequency']) == forest['forest_sha256']
    for name, value in exported['presence'].state.items():
        np.testing.assert_array_equal(value, restored['presence'].state[name])
    sources = dict(manifest['sources'])
    for item in config['source_files']:
        sources[item] = sha256_file(ROOT / item)
    provenance = [identity(path) for path in [old / 'manifest.json', old / 'model.joblib',
        checkpoint, parent, ROOT / config['train_summary'], ROOT / config['feature_amendment'], config_path]]
    write(output / 'manifest.json', {'kind': KIND, 'model': identity(output / 'model.joblib'),
        'sources': sources, 'provenance': provenance, 'selected_decoder': model['selected_decoder'],
        'feature_spec': {'dimension': 224, 'order': '210 independentcounts +4 visible single/multi/combination/sequence bits +10 visualsummaries',
            'visual_feature_names': amendment['corrected_feature_names'], 'view_order': ['Depth_Color', 'IR']},
        'validation': summary['metrics'], 'training_scope': '19 optionforests with75 subject-excluded linear8 heads; officialTrain only. Finalforest uses3 inner heads; inference uses18personfinalhead.',
        'limits': 'Multi-only372/809 remains weaker than old387. All18people historically observed; no full-system OOF. Selection usesTrain only.',
        'new_fits': 0, 'pseudo_dependency': False})


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
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    exp = sub.add_parser('export')
    exp.add_argument('--config', type=Path, required=True)
    exp.add_argument('--output-dir', type=Path, required=True)
    inf = sub.add_parser('infer')
    inf.add_argument('--bundle', type=Path, required=True)
    inf.add_argument('--qa-csv', type=Path, required=True)
    inf.add_argument('--features', type=Path, required=True)
    inf.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.command == 'export':
            export(args.config, args.output_dir)
        else:
            infer(args.bundle, args.qa_csv, args.features, args.output_dir)


if __name__ == '__main__':
    main()
