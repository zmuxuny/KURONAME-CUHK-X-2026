#!/usr/bin/env python3
"""Evaluate predeclared parent/A/B joint-state mixtures; no fitting or Test data."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.ensemble._forest import BaseForest
from threadpoolctl import threadpool_limits

from cuhkx import emotion_fusion_nested as fusion
from cuhkx import emotion_recipe_nested as recipe
from cuhkx import emotion_template_nested as template
from cuhkx.emotion_nested import OFFICIAL_SUBJECTS, feasible_assignments
from cuhkx.emotion_scene_robustness import inference_frame
from cuhkx.emotion_visual_nested import verify_training_content
from cuhkx.validation import evaluation_report, sha256_file
from scripts.audit_emotion_nested_provenance import audit_word_priors
from scripts.research_write_guard import install
from scripts.run_local_emotion_distillation import (
    check_resources,
    deny_teacher_fit,
    freeze,
    load_inputs,
    write,
)


def state_fusion(frame, evidence, policy, student_probabilities):
    safe = inference_frame(frame)
    records, answers = [], {name: {} for name in ('parent_control', 'parent_half_A', 'parent_half_B')}
    for group_id, group in safe.groupby('group', sort=False):
        decoder = template.policy_decoder(policy, group)
        local = fusion.adapted_evidence(evidence[decoder['recipe']], decoder)
        assignments = feasible_assignments(group)
        rows = list(group.itertuples())
        energies = []
        for words in assignments:
            energy = 0.
            for row, word in zip(rows, words, strict=True):
                item = local[row.qa_id][word]
                energy += (decoder['lexical'] * item['lexical'] + decoder['context'] * item['context'])[row.position]
                energy += decoder['sensor'] * item['sensor']
            energies.append(float(energy))
        parent = softmax(np.asarray(energies))
        state_arrays = {'parent_control': parent}
        for arm in ('A', 'B'):
            log_values = [sum(float(np.log(max(student_probabilities[arm][row.qa_id][
                'ABCD'.index(row.letters[row.options.index(word)])], 1e-12)))
                for row, word in zip(rows, words, strict=True)) for words in assignments]
            projected = softmax(np.asarray(log_values))
            state_arrays['student_' + arm] = projected
            state_arrays['parent_half_' + arm] = (parent + projected) / 2
        for name, answer_map in answers.items():
            best = assignments[int(np.argmax(state_arrays[name]))]
            for row, word in zip(rows, best, strict=True):
                answer_map[row.qa_id] = row.letters[row.options.index(word)]
        records.append({'group': str(group_id), 'qa_ids': group.qa_id.tolist(), 'decoder': decoder,
                        'assignments': assignments, 'parent_energies': energies,
                        'state_probabilities': {name: value.tolist() for name, value in state_arrays.items()}})
    return answers, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/local_emotion_full_fusion_v1.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--replay-of', type=Path)
    args = parser.parse_args()
    install(Path('/root/workspace/CUHK-X'))
    BaseForest.fit = deny_teacher_fit
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    started = time.perf_counter()
    config = json.loads(args.config.read_text())
    if config['parent_weight'] != .5 or config['student_weight'] != .5 or config['arms'] != ['parent_control', 'parent_half_A', 'parent_half_B']:
        raise ValueError('undeclared fusion choice')
    parent, students = ROOT / config['parent'], ROOT / config['students']
    student_config = json.loads((ROOT / config['student_protocol']).read_text())
    report = json.loads((students / 'report.json').read_text())
    if report['status'] != 'completed_fixed_QA_vs_crossfit_distillation':
        raise ValueError('student fits not complete')
    dependencies = [args.config, Path(__file__), ROOT / config['student_protocol'], students / 'report.json',
                    students / 'completion_freeze.json', parent / 'model_source_manifest.json', parent / 'inference_policy.json', parent / 'oof_predictions.csv']
    dependencies += [Path(item['path']) for item in json.loads((students / 'protection.json').read_text())['assets']]
    for item in json.loads((students / 'completion_freeze.json').read_text())['files']:
        if sha256_file(Path(item['path'])) != item['sha256']:
            raise ValueError('frozen student output changed')
    for user in sorted(OFFICIAL_SUBJECTS):
        dependencies += [parent / 'folds' / user / 'inner_selection.json', parent / 'folds' / user / 'oof_predictions.csv',
                         students / 'scopes' / user / 'scope.json']
        dependencies += [students / 'scopes' / user / (arm + '_predictions.npz') for arm in ('A', 'B')]
    hashes = {str(path.resolve()): sha256_file(path) for path in dependencies}
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'protection.json', {'config': config, 'assets': [{'path': path, 'sha256': digest,
        'isolation': 'read-only frozen model/data/source; separate output', 'recovery_check': 'sha256sum ' + path}
        for path, digest in hashes.items()]})
    check_resources(out, config)
    _, manifest, frame, contexts, _, matrices, masks, _, _ = load_inputs(student_config)
    outputs, scope_receipts = {name: {} for name in config['arms']}, []
    with threadpool_limits(limits=2):
        for subject in sorted(OFFICIAL_SUBJECTS):
            item = manifest['models']['outer/' + subject]
            bundle = joblib.load(item['checkpoint']['path'])
            scope = json.loads(Path(item['scope']['path']).read_text())
            verify_training_content(bundle, scope, frame, matrices, masks)
            audit_word_priors(bundle, frame, contexts, set(scope['train_subjects']))
            for model in bundle['models'].values():
                model.n_jobs = 1
            rows = np.flatnonzero(frame.subject.eq(subject))
            local_frame = frame.iloc[rows]
            policy = json.loads((parent / 'folds' / subject / 'inner_selection.json').read_text())['selected']
            recipe_config = json.loads((parent / 'inference_policy.json').read_text())['recipe_config']
            evidence = recipe.recipe_evidence(bundle, inference_frame(frame), contexts, matrices, masks, rows, recipe_config)
            probabilities = {}
            student_scope = json.loads((students / 'scopes' / subject / 'scope.json').read_text())
            if student_scope['train_subjects'] != scope['train_subjects'] or student_scope['validation_subjects'] != scope['validation_subjects']:
                raise ValueError('student and parent outer scopes differ')
            for arm in ('A', 'B'):
                with np.load(students / 'scopes' / subject / f'{arm}_predictions.npz') as archive:
                    if not np.array_equal(rows, archive['rows']) or archive['qa_ids'].tolist() != local_frame.qa_id.tolist():
                        raise ValueError('student heldout probability row order differs')
                    probabilities[arm] = dict(zip(archive['qa_ids'], archive['probabilities'], strict=True))
            answers, states = state_fusion(local_frame, evidence, policy, probabilities)
            poisoned, poisoned_states = state_fusion(inference_frame(local_frame, poison=True), evidence, policy, probabilities)
            if answers != poisoned or joblib.hash(states) != joblib.hash(poisoned_states):
                raise ValueError('fusion reads target labels')
            expected = pd.read_csv(parent / 'folds' / subject / 'oof_predictions.csv', dtype=str).set_index('qa_id')
            if any(expected.loc[key, 'prediction'] != value for key, value in answers['parent_control'].items()):
                raise ValueError('original full-scene parent was not exactly reproduced')
            write(out / f'{subject}_states.json', states)
            for name, answer_map in outputs.items():
                answer_map.update(answers[name])
            scope_receipts.append({'subject': subject, 'rows': len(rows), 'states_digest': joblib.hash(states),
                                   'parent_answers_exact': True, 'target_poison_exact': True})
            print(f'local_emotion_fusion={subject} completed={len(scope_receipts)}/18', flush=True)
    reports = {}
    for name, answers in outputs.items():
        evaluated = frame[['qa_id', 'source', 'category', 'subject', 'answer']].copy()
        evaluated['prediction'] = evaluated.qa_id.map(answers)
        evaluated['baseline_prediction'] = evaluated.qa_id.map(outputs['parent_control'])
        if evaluated.prediction.isna().any():
            raise ValueError('fusion coverage incomplete')
        evaluated.to_csv(out / f'{name}_predictions.csv', index=False)
        reports[name] = evaluation_report(evaluated, samples=10000, seed=20260912, confidence=.95)
    if reports['parent_control']['overall']['correct_rows'] != 749:
        raise ValueError('matched parent not749')
    if args.replay_of:
        original = json.loads((args.replay_of / 'report.json').read_text())
        if original['scope_receipts'] != scope_receipts or original['conditions'] != reports:
            raise ValueError('independent state fusion replay differs')
    for path, digest in hashes.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError('protected input changed')
    write(out / 'report.json', {'status': 'passed_fixed_full_scene_state_fusion', 'conditions': reports,
        'scope_receipts': scope_receipts, 'fits': 0, 'protected_asset_posthash': len(hashes),
        'replay_of': str(args.replay_of) if args.replay_of else None, 'limits': config['limits'],
        'Test_rows': 0, 'pseudo_rows': 0, 'wall_seconds': time.perf_counter() - started, 'resources': check_resources(out, config)})
    freeze(out)


if __name__ == '__main__':
    main()
