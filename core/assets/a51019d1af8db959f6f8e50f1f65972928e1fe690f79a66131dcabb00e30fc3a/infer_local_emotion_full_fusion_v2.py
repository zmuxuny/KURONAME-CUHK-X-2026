#!/usr/bin/env python3
"""Load local A/B students and explicit full-scene teacher for fixed state fusion."""
from __future__ import annotations

import argparse
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
from sklearn.ensemble._forest import BaseForest
from threadpoolctl import threadpool_limits

from cuhkx import emotion_recipe_nested as recipe
from cuhkx import local_emotion_student as student
from cuhkx.acquisition_v2 import expert_acquisition, expert_questions, read_acquisition
from cuhkx.emotion_nested import OFFICIAL_SUBJECTS, prepare, read_sensor, sensor_matrix
from cuhkx.emotion_scene_robustness import inference_frame
from cuhkx.inference import question_metadata
from cuhkx.validation import sha256_file
from scripts.assess_local_emotion_full_fusion_v2 import state_fusion
from scripts.research_write_guard import install
from scripts.run_local_emotion_distillation import deny_teacher_fit, freeze, write


def verified(item):
    path = Path(item['path'])
    if sha256_file(path) != item['sha256']:
        raise ValueError('frozen dependency differs: ' + str(path))
    return path


def local_question_context(raw):
    # Historical metadata helpers intentionally omit natural-language text.
    # Restore only the current question, keyed by its ID; never restore targets.
    if 'question' not in raw:
        raise ValueError('local student requires original current-question text')
    qa = question_metadata(raw)
    visible = expert_questions(qa, 'emotion')
    visible['question'] = visible.qa_id.map(raw.set_index('qa_id')['question'])
    if visible.question.isna().any():
        raise ValueError('question text lookup incomplete')
    return qa, visible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--descriptor', type=Path, required=True)
    parser.add_argument('--qa-csv', type=Path, required=True)
    parser.add_argument('--scene-manifest', type=Path, required=True)
    parser.add_argument('--arm', choices=['A', 'B'], default='A')
    for name in student.MODALITIES:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--replay-of', type=Path)
    args = parser.parse_args()
    install(Path('/root/workspace/CUHK-X'))
    BaseForest.fit = deny_teacher_fit
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    descriptor = json.loads(args.descriptor.read_text())
    if descriptor['kind'] != 'cuhkx_local_emotion_full_state_fusion_final_v1':
        raise ValueError('unexpected final descriptor')
    if descriptor['parent_weight'] != .5 or descriptor['student_weight'] != .5:
        raise ValueError('this implementation freezes equal state probabilities')
    dependencies = [args.descriptor, args.qa_csv, args.scene_manifest,
                    *(getattr(args, name) for name in student.MODALITIES)]
    dependencies += [verified(item) for item in descriptor['sources']]
    policy_path = verified(descriptor['parent_policy'])
    dependencies.append(policy_path)
    policy = json.loads(policy_path.read_text())
    for name, digest in policy['recipe_source_hashes'].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError('historical parent implementation changed: ' + name)
    recipe.validate_config(policy['recipe_config'], policy['original_fit_config'])
    recipe.check_policy(policy['selected_policy'], policy['recipe_config'])
    model_record = policy['source_model']
    parent_path = verified(model_record['checkpoint'])
    scope_path = verified(model_record['scope'])
    dependencies += [parent_path, scope_path, verified(policy['source_manifest'])]
    parent = joblib.load(parent_path)
    parent_scope = json.loads(scope_path.read_text())
    if parent['training_subjects'] != sorted(OFFICIAL_SUBJECTS) or parent_scope['validation_subjects']:
        raise ValueError('parent must be final allTrain model')
    if parent['config'] != policy['original_fit_config'] or parent['source_hashes'] != policy['original_source_hashes']:
        raise ValueError('parent configuration/provenance differs')
    student_scope_path = verified(descriptor['student_scope'])
    dependencies.append(student_scope_path)
    student_scope = json.loads(student_scope_path.read_text())
    if student_scope['train_subjects'] != sorted(OFFICIAL_SUBJECTS) or student_scope['validation_subjects']:
        raise ValueError('student scope must be final allTrain')
    normalizer_path = verified(descriptor['normalizer'])
    dependencies.append(normalizer_path)
    normalizer = joblib.load(normalizer_path)
    checkpoints = {}
    for arm in ('A', 'B'):
        path = verified(descriptor['students'][arm])
        dependencies.append(path)
        checkpoints[arm] = torch.load(path, map_location='cpu', weights_only=False)
        item = checkpoints[arm]
        if item['training_subjects'] != sorted(OFFICIAL_SUBJECTS) or item['validation_subjects'] or item['arm'] != arm:
            raise ValueError('student must be matching allTrain final arm')
        if item['normalizer_sha256'] != descriptor['normalizer']['sha256']:
            raise ValueError('student normalizer identity differs')
        if student.state_digest(item['state_dict']) != item['final_state_digest']:
            raise ValueError('student learned values differ')
    if checkpoints['A']['initial_state_digest'] != checkpoints['B']['initial_state_digest'] or checkpoints['A']['steps'] != checkpoints['B']['steps']:
        raise ValueError('student A/B matched-control identity differs')
    hashes = {str(path.resolve()): sha256_file(path) for path in dependencies}
    qa, visible = local_question_context(pd.read_csv(args.qa_csv, dtype=str, keep_default_na=False))
    acquisition = read_acquisition(qa, args.scene_manifest)
    scenes = expert_acquisition(visible, acquisition).scenes
    frame, contexts = prepare(visible, training=False, manifest=scenes)
    banks, matrices, masks = {}, {}, {}
    for name in student.MODALITIES:
        bank = read_sensor(getattr(args, name), parent['config']['views'][name])
        if not np.array_equal(bank['names'], parent['feature_names'][name]):
            raise ValueError('feature schema differs: ' + name)
        if len(bank['names']) != len(normalizer[name]['mean']):
            raise ValueError('student normalization schema differs')
        banks[name] = bank
        matrices[name], masks[name] = sensor_matrix(frame, bank)
    inputs, input_ids = student.local_inputs(visible, banks, training=False)
    if set(input_ids) != set(frame.qa_id):
        raise ValueError('teacher/student question scopes differ')
    tensor_batch = student.tensors(inputs, normalizer)
    probabilities, models = {}, {}
    with threadpool_limits(limits=2):
        for arm in ('A', 'B'):
            checkpoint = checkpoints[arm]
            model = student.LocalEmotionStudent(checkpoint['media_width'], checkpoint['config']['hidden'], checkpoint['config']['dropout'])
            model.load_state_dict(checkpoint['state_dict'])
            local_probability = student.infer(model, tensor_batch, np.arange(len(input_ids)))
            probabilities[arm] = dict(zip(input_ids, local_probability, strict=True))
            models[arm] = model
        for model in parent['models'].values():
            model.n_jobs = 1
        evidence = recipe.recipe_evidence(parent, inference_frame(frame), contexts, matrices, masks, np.arange(len(frame)), policy['recipe_config'])
        answers, states = state_fusion(frame, evidence, policy['selected_policy'], probabilities)
        poisoned, poison_states = state_fusion(inference_frame(frame, poison=True), evidence, policy['selected_policy'], probabilities)
        if answers != poisoned or joblib.hash(states) != joblib.hash(poison_states):
            raise ValueError('full fusion depends on target fields')
        # Student cannot see actual other questions, acquisition or target metadata.
        local_only = visible.loc[visible.category.eq('emotion')].assign(answer='POISON', group='POISON', position=99)
        poison_inputs, poison_ids = student.local_inputs(local_only, banks, training=False)
        if poison_ids != input_ids or joblib.hash(poison_inputs) != joblib.hash(inputs):
            raise ValueError('student input depends on nonlocal fields')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'protection.json', {'assets': [{'path': path, 'sha256': digest, 'isolation': 'read-only final/parent/cache; new output only',
        'recovery_check': 'sha256sum ' + path} for path, digest in hashes.items()], 'pseudo_dependency': False})
    outputs = {}
    for name, mapping in answers.items():
        output = pd.DataFrame({'qa_id': input_ids, 'prediction': [mapping[qa_id] for qa_id in input_ids]})
        outputs[name] = output
        output.to_csv(out / f'{name}_predictions.csv', index=False)
    outputs['parent_half_' + args.arm].to_csv(out / 'test_predictions.csv', index=False)
    np.savez_compressed(out / 'student_probabilities.npz', qa_ids=np.asarray(input_ids),
        A=np.stack([probabilities['A'][key] for key in input_ids]), B=np.stack([probabilities['B'][key] for key in input_ids]))
    write(out / 'joint_states.json', states)
    if args.replay_of:
        with np.load(args.replay_of / 'student_probabilities.npz') as previous, np.load(out / 'student_probabilities.npz') as current:
            if previous.files != current.files or any(not np.array_equal(previous[name], current[name]) for name in previous.files):
                raise ValueError('independent student reload probabilities differ')
        if json.loads((args.replay_of / 'joint_states.json').read_text()) != json.loads(json.dumps(states)):
            raise ValueError('independent full-state reload differs')
        for name, output in outputs.items():
            expected = pd.read_csv(args.replay_of / f'{name}_predictions.csv', dtype=str)
            if not expected.equals(output):
                raise ValueError('independent fusion answers differ')
    for path, digest in hashes.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError('protected model/source/input changed')
    write(out / 'report.json', {'status': 'passed_final_local_emotion_full_state_fusion', 'rows': len(frame),
        'arm': args.arm, 'fits': 0, 'pseudo_dependency': {'training': False, 'tuning': False, 'selection': False},
        'explicit_scene': str(args.scene_manifest), 'student_current_question_only': 'all rows exact after other question removal and target/group/position poison',
        'state_digest': joblib.hash(states), 'replay_of': str(args.replay_of) if args.replay_of else None,
        'component_sha256': {name: sha256_file(out / f'{name}_predictions.csv') for name in answers},
        'sources': hashes, 'limits': 'Full-scene fusion requires supplied valid scene metadata; student uses only current recording/question. Teacher and student trained on historically observed18Train people.'})
    freeze(out)
    print(json.dumps({'rows': len(frame), 'report': str(out / 'report.json'), 'sha256': sha256_file(out / 'report.json')}), flush=True)


if __name__ == '__main__':
    main()
