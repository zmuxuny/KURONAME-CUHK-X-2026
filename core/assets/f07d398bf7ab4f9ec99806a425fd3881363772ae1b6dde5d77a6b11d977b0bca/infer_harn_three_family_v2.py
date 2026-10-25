#!/usr/bin/env python3
"""Infer fixed equal HARn families with four actual R18/R18/R34/VideoMAE models."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from cuhkx.harn_nested import QA_COLUMNS, decode_questions, predict_evidence
from cuhkx.harn_seed_ensemble import NAMES, check_final_models, equal_evidence
from cuhkx.harn_three_family import three_family_evidence
from cuhkx.harn_videomae_inference import InferenceVideoMAEBank
from cuhkx.metadata import feature_unit
from cuhkx.r34_tail import R34PairedBank
from cuhkx.validation import sha256_file
from cuhkx.video_midfeatures import MidFeatureBank
from scripts.research_write_guard import install
from scripts.run_harn_nested_fusion import harn_rows, read_qa, save_evidence, write_json

KIND = 'cuhkx_harn_three_equal_families_v1'
FAMILY_WEIGHTS = [1 / 3] * 3
MODEL_WEIGHTS = [1 / 6, 1 / 6, 1 / 3, 1 / 3]


def resolve(path):
    return (ROOT / path).resolve()


def verify_entries(entries):
    for item in entries:
        if sha256_file(resolve(item['path'])) != item['sha256']:
            raise ValueError('frozen three-family input/source hash changed')


def validate_contract(descriptor, plan, report):
    if descriptor['kind'] != KIND or descriptor['family_weights'] != FAMILY_WEIGHTS or descriptor['model_weights'] != MODEL_WEIGHTS:
        raise ValueError('expected fixed three equal families and four model weights')
    if plan['kind'] != 'cuhkx_harn_three_equal_families_plan_v1' or plan['family_weights'] != FAMILY_WEIGHTS or plan['model_weights'] != MODEL_WEIGHTS:
        raise ValueError('descriptor differs from predeclared family recipe')
    if report['status'] != 'complete_nested_three_family_comparison' or report['plan'] != plan or report['test_qa_read']:
        raise ValueError('expected completed Train-only matched-family nested report')
    if len(report['heldout_scopes']) != 75 or len(report['selections']) != 19:
        raise ValueError('nested report does not cover all declared heldout scopes/selections')
    metric = report['metrics']['overall']
    if (metric['correct_rows'], metric['rows']) != (530, 562):
        raise ValueError('descriptor does not reference the declared 530/562 experiment')
    weight = descriptor['selected_skeleton_weight']
    if weight != report['final_skeleton_weight'] or weight != report['selections'][-1]['selected_skeleton_weight'] or weight not in plan['skeleton_weights']:
        raise ValueError('Skeleton weight differs from Train-only final selection')
    if set(descriptor['feature_specs']) != {'r2_middle', 'r34_middle', 'videomae'}:
        raise ValueError('missing exact three-family feature specifications')
    if any(plan['pseudo_dependency'].values()):
        raise ValueError('this independent candidate requires pseudo-free plan choices')


def validate_extra_models(models, r34, videomae, descriptor):
    check_final_models(models)
    if r34['kind'] != 'cuhkx-harn-r34-last-block-nested-v1' or videomae['kind'] != 'cuhkx_harn_videomae_nested_v1':
        raise ValueError('wrong additional visual model architecture')
    reference = models['original']
    for bundle, specification in [(r34, 'r34_middle'), (videomae, 'videomae')]:
        if bundle['train_subjects'] != reference['train_subjects']:
            raise ValueError('four models have different fitting subjects')
        for key in ['action_mapping', 'object_mapping', 'prior']:
            if bundle[key] != reference[key]:
                raise ValueError('four models have different Train dictionaries or prior')
        for key in ['skeleton_names', 'skeleton_classes', 'visual_classes']:
            np.testing.assert_array_equal(bundle[key], reference[key])
        if bundle['visual_spec'] != descriptor['feature_specs'][specification]:
            raise ValueError('model visual specification differs from descriptor')
    if reference['visual_spec'] != descriptor['feature_specs']['r2_middle']:
        raise ValueError('R18 model visual specification differs from descriptor')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint', 'qa-csv', 'skeleton', 'visual', 'r34', 'videomae', 'output-dir']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2, choices=[1, 2])
    args = parser.parse_args()
    install(Path('/root/workspace/CUHK-X'))
    descriptor = json.loads(args.checkpoint.read_text())
    entries = [descriptor[key] for key in ['parent_checkpoint', 'r34_checkpoint', 'videomae_model', 'nested_report', 'plan']]
    entries += descriptor['sources']
    verify_entries(entries)
    plan = json.loads(resolve(descriptor['plan']['path']).read_text())
    report = json.loads(resolve(descriptor['nested_report']['path']).read_text())
    validate_contract(descriptor, plan, report)
    verify_entries(plan['sources'])
    parent = json.loads(resolve(descriptor['parent_checkpoint']['path']).read_text())
    if (parent.get('kind') != 'cuhkx_harn_equal_seed_ensemble_v1' or parent.get('visual_weights') != [.5, .5]
            or set(parent.get('classifiers', {})) != set(NAMES)):
        raise ValueError('expected frozen dual-seed R18 parent')
    parent_entries = [*parent['sources'], *parent['classifiers'].values()]
    verify_entries(parent_entries)
    models = {name: joblib.load(resolve(parent['classifiers'][name]['path'])) for name in NAMES}
    r34 = joblib.load(resolve(descriptor['r34_checkpoint']['path']))
    videomae = joblib.load(resolve(descriptor['videomae_model']['path']))
    validate_extra_models(models, r34, videomae, descriptor)
    banks = {'r2_middle': MidFeatureBank.load(args.skeleton, args.visual),
             'r34_middle': R34PairedBank.load(args.skeleton, args.r34),
             'videomae': InferenceVideoMAEBank.load(args.skeleton, args.videomae)}
    for name, bank in banks.items():
        for axis in [bank.skeleton_units, bank.visual_units]:
            if not np.char.startswith(axis.astype(str), 'Testing/').all():
                raise ValueError('inference banks must contain Testing-only units')
        if bank.visual_spec != descriptor['feature_specs'][name]:
            raise ValueError('inference feature specification differs from frozen model')
    qa = harn_rows(read_qa(args.qa_csv, labeled=False))[QA_COLUMNS]
    if not all(feature_unit(path, training=False).startswith('Testing/') for path in qa.path):
        raise ValueError('only Testing question metadata is allowed')
    torch.set_num_threads(args.workers)
    for bundle in [*models.values(), r34, videomae]:
        for forest in bundle['models'].values():
            forest.n_jobs = 1
    with threadpool_limits(limits=args.workers):
        raw = {name: predict_evidence(bundle, banks['r2_middle'], qa, training_paths=False) for name, bundle in models.items()}
        raw['r34'] = predict_evidence(r34, banks['r34_middle'], qa, training_paths=False)
        raw['videomae'] = predict_evidence(videomae, banks['videomae'], qa, training_paths=False)
        combined = three_family_evidence(equal_evidence(raw['original'], raw['repeated']), raw['r34'], raw['videomae'])
        predictions = decode_questions(qa, combined, models['original'], descriptor['selected_skeleton_weight'], training_paths=False)
    verify_entries(entries + parent_entries + plan['sources'])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(args.output_dir / 'test_predictions.csv', index=False)
    for name, evidence in raw.items():
        save_evidence(args.output_dir / (name + '_probabilities.npz'), evidence)
    save_evidence(args.output_dir / 'test_probabilities.npz', combined)
    write_json(args.output_dir / 'report.json', {
        'status': 'actual_four_models_three_equal_families_Testing_inference_complete', 'rows': len(predictions),
        'checkpoint_sha256': sha256_file(args.checkpoint), 'prediction_sha256': sha256_file(args.output_dir / 'test_predictions.csv'),
        'family_weights': FAMILY_WEIGHTS, 'model_weights_when_all_families_available': MODEL_WEIGHTS,
        'missing_family_policy': 'Renormalize equally across available families; R18 retains fixed two-seed mean.',
        'skeleton_evidence': 'Retained dual-R18 shared Skeleton probability; no new Skeleton model fit.',
        'selected_skeleton_weight': descriptor['selected_skeleton_weight'], 'nested_report': descriptor['nested_report'],
        'source_inputs': entries + parent_entries, 'new_fits': 0, 'training_qa_reads': 0, 'validation_artifacts': 'Bound by frozen plan/report; checked before inference, not reread by Test inference.',
        'pseudo_dependency': plan['pseudo_dependency'],
        'outputs': [{'path': str(path), 'sha256': sha256_file(path)} for path in sorted(args.output_dir.glob('*.npz'))],
    })


if __name__ == '__main__':
    main()
