#!/usr/bin/env python3
"""Evaluate equal R18/R34/VideoMAE families on all matched nested Train scopes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import numpy as np
import pandas as pd

from cuhkx.harn_nested import QA_COLUMNS, decode_questions, evaluate_predictions, select_weight
from cuhkx.harn_seed_ensemble import validate_evidence
from cuhkx.harn_three_family import three_family_evidence
from cuhkx.metadata import subject_id
from cuhkx.validation import prepare_qa
from scripts.assess_action_window_nested import compare
from scripts.run_action_generic_nested import verify_manifest
from scripts.run_harn_nested_fusion import harn_rows, manifest_entry, read_qa, write_json
from scripts.run_harn_seed_ensemble import compare_frame, load_evidence
from scripts.run_harn_support_fusion import CachedScopes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--videomae-audit', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    assert plan['family_weights'] == [1 / 3] * 3 and plan['skeleton_weights'] == [0., .5, .75, 1.]
    assert plan['model_weights'] == [1 / 6, 1 / 6, 1 / 3, 1 / 3]
    verify_manifest(plan['frozen_inputs'])
    verify_manifest(plan['sources'])
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    used = {}

    def register(path):
        path = Path(path).resolve()
        if str(path) not in used:
            used[str(path)] = manifest_entry(path)
        return path

    register(args.plan)
    sources = [Path(__file__), ROOT / 'src/cuhkx/harn_three_family.py', ROOT / 'scripts/run_harn_support_fusion.py',
               ROOT / 'src/cuhkx/harn_seed_ensemble.py', ROOT / 'src/cuhkx/harn_nested.py', ROOT / 'src/cuhkx/harn.py']
    for path in sources:
        register(path)
    parent = ROOT / 'runs/20260905_harn_seed_ensemble_nested_v1'
    new = ROOT / plan['models'][3]
    r34 = ROOT / plan['models'][2]
    r34_inventory = json.loads(register(ROOT / plan['r34_inventory']).read_text())
    r34_frozen = {str(Path(e['path']).resolve()): e['sha256'] for e in r34_inventory}
    for item in plan['frozen_inputs'] + plan['sources']:
        register(ROOT / item['path'])
    assert manifest_entry(ROOT / 'configs/harn_videomae_nested_v1.json')['sha256'] == plan['training_config_sha256']
    audit = json.loads(register(args.videomae_audit).read_text())
    assert audit['status'] == 'complete_actual_model_scope_probability_decision_replay'
    assert len(audit['scopes']) == 76 and len(audit['inner_selections']) == 19 and audit['heldout_qa_rows_replayed'] == 10678
    audited = {entry['fit_id']: entry for entry in audit['scopes']}
    for path in [new / 'frozen_config.json', parent / 'input_manifest.json', parent / 'frozen_config.json']:
        register(path)
    qa = harn_rows(read_qa(register(ROOT / 'data/raw/kaggle/training_qa.csv'), labeled=True))
    cache = CachedScopes(parent, qa)
    assert {str(p.resolve()) for p in cache.parents.values()} == {str((ROOT / x).resolve()) for x in plan['models'][:2]}
    scopes = []

    def evidence(fit_id, train, target):
        original, dictionary = cache.load(fit_id, train, target)
        scope_path = register(new / 'fits' / (fit_id + '.json'))
        scope = json.loads(scope_path.read_text())
        expected = qa.loc[qa.path.map(subject_id).isin(train)]
        assert scope['train_subjects'] == sorted(train) and scope['training_qa_ids'] == expected.qa_id.tolist()
        assert not set(target.path.map(subject_id)) & set(train)
        model = register(new / 'fits' / (fit_id + '.models.joblib'))
        assert used[str(model)]['sha256'] == scope['visual_head']['checkpoint_sha256'] == audited[fit_id]['checkpoint_sha256']
        extra = load_evidence(register(new / 'fits' / (fit_id + '.saved_evidence.npz')))
        validate_evidence(extra)
        np.testing.assert_array_equal(extra['units'], original['units'])
        r34_scope_path = register(r34 / 'fits' / (fit_id + '.json'))
        assert used[str(r34_scope_path)]['sha256'] == r34_frozen[str(r34_scope_path)]
        r34_scope = json.loads(r34_scope_path.read_text())
        assert r34_scope['train_subjects'] == sorted(train) and r34_scope['training_qa_ids'] == expected.qa_id.tolist()
        for family in ('skeleton', 'visual'):
            units = r34_scope['features'][family]['units']
            assert all(u.startswith('Training/HARn/') and subject_id(u) in train for u in units)
        r34_model = register(r34 / 'fits' / (fit_id + '.models.joblib'))
        assert used[str(r34_model)]['sha256'] == r34_frozen[str(r34_model)] == r34_scope['adapted_visual']['checkpoint_sha256']
        r34_saved = register(r34 / 'fits' / (fit_id + '.saved_evidence.npz'))
        assert used[str(r34_saved)]['sha256'] == r34_frozen[str(r34_saved)]
        r34_evidence = load_evidence(r34_saved)
        np.testing.assert_array_equal(r34_evidence['visual_classes'], np.unique(r34_scope['features']['visual']['labels']))
        mixed = three_family_evidence(original, r34_evidence, extra)
        scopes.append({'fit_id': fit_id, 'train_subjects': sorted(train), 'target_qa_ids': target.qa_id.tolist(),
                       'videomae_checkpoint_sha256': audited[fit_id]['checkpoint_sha256'],
                       'r34_checkpoint_sha256': used[str(r34_model)]['sha256'],
                       'r34_scope_sha256': used[str(r34_scope_path)]['sha256']})
        return original, mixed, dictionary

    selections, outer_predictions, baseline_predictions = [], [], []
    for name in ['outer_' + s for s in plan_subjects(parent)] + ['final_all_subjects']:
        path = register(parent / 'selections' / (name + '.json'))
        declaration = json.loads(path.read_text())
        fitting = qa if name == 'final_all_subjects' else qa.loc[qa.path.map(subject_id).ne(name.removeprefix('outer_'))]
        parts = {mode: {w: [] for w in plan['skeleton_weights']} for mode in ['baseline', 'three_family_equal']}
        for index, fold in enumerate(declaration['inner_folds']):
            train, valid = set(fold['train_subjects']), set(fold['validation_subjects'])
            assert not train & valid and train | valid == set(fitting.path.map(subject_id))
            target = fitting.loc[fitting.qa_id.isin(fold['prediction_qa_ids'])]
            assert target.qa_id.tolist() == fold['prediction_qa_ids']
            original, mixed, dictionary = evidence(fold['fold_id'], train, target)
            for mode, probability in [('baseline', original), ('three_family_equal', mixed)]:
                for weight in plan['skeleton_weights']:
                    frame = decode_questions(target[QA_COLUMNS], probability, dictionary, weight, training_paths=True)
                    pd.testing.assert_frame_equal(frame, decode_questions(target.assign(answer='POISON', prediction='POISON'),
                                                                         probability, dictionary, weight, training_paths=True))
                    frame['inner_fold'] = index
                    parts[mode][weight].append(frame)
        reports = {mode: [{'skeleton_weight': w, 'metrics': evaluate_predictions(fitting, pd.concat(frames))}
                          for w, frames in weights.items()] for mode, weights in parts.items()}
        reference = register(parent / 'selections' / ('ensemble_' + name + '_inner_predictions.csv'))
        compare_frame(pd.concat([pd.concat(x) for x in parts['baseline'].values()], ignore_index=True), reference)
        assert reports['baseline'] == declaration['candidate_reports']
        assert select_weight(reports['baseline'], plan['weight_tie_order']) == declaration['selected_skeleton_weight']
        weight = select_weight(reports['three_family_equal'], plan['weight_tie_order'])
        record = {'scope': name, 'selected_skeleton_weight': weight, 'candidate_reports': reports['three_family_equal'],
                  'baseline_replay_exact': True, 'inner_folds': declaration['inner_folds']}
        write_json(output / (name + '.json'), record)
        selections.append(record)
        if name != 'final_all_subjects':
            target = qa.loc[qa.path.map(subject_id).eq(name.removeprefix('outer_'))]
            original, mixed, dictionary = evidence(name + '_final', set(fitting.path.map(subject_id)), target)
            outer_predictions.append(decode_questions(target[QA_COLUMNS], mixed, dictionary, weight, training_paths=True))
            baseline_predictions.append(decode_questions(target[QA_COLUMNS], original, dictionary,
                                                         declaration['selected_skeleton_weight'], training_paths=True))
        print(json.dumps({'scope': name, 'selected_skeleton_weight': weight}), flush=True)
    assert len(scopes) == 75
    predicted = pd.concat(outer_predictions).set_index('qa_id').loc[qa.qa_id].reset_index()
    baseline = pd.concat(baseline_predictions).set_index('qa_id').loc[qa.qa_id].reset_index()
    reference = pd.read_csv(register(parent / 'oof_predictions.csv'), dtype={'qa_id': str, 'prediction': str})
    pd.testing.assert_series_equal(baseline.set_index('qa_id').prediction, reference.set_index('qa_id').loc[qa.qa_id].prediction, check_names=False)
    predicted.to_csv(output / 'oof_predictions.csv', index=False)
    target = prepare_qa(qa, ['HARn:single', 'HARn:object_interaction'])
    metrics = compare(target, predicted, baseline)
    for path, entry in cache.used.items():
        used[path] = entry
    verify_manifest(list(used.values()))
    write_json(output / 'input_manifest.json', list(used.values()))
    write_json(output / 'report.json', {'status': 'complete_nested_three_family_comparison', 'metrics': metrics,
               'previously_audited_models_reused': 304, 'heldout_model_scopes_reused': 75, 'heldout_scopes': scopes, 'selections': selections,
               'final_skeleton_weight': selections[-1]['selected_skeleton_weight'], 'test_qa_read': False,
               'new_fits': 0, 'uploads': 0, 'plan': plan,
               'limits': 'Train-only cached probability fusion using all matched source audits; no new neural forward or Test inference in this stage. Four models, three equally weighted families; all18people historicallyexposed. No pseudo-label training, parameter or selection dependency.'})
    print(json.dumps(metrics['overall']), flush=True)


def plan_subjects(parent):
    return json.loads((parent / 'frozen_config.json').read_text())['config']['subjects']


if __name__ == '__main__':
    main()
