#!/usr/bin/env python3
"""Train or independently reload fixed single-question QA versus crossfit distillation."""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble._forest import BaseForest
from threadpoolctl import threadpool_limits

from cuhkx import local_emotion_student as student
from cuhkx import local_emotion_teacher as teacher
from cuhkx.emotion_nested import OFFICIAL_SUBJECTS, prepare, read_sensor, sensor_matrix
from cuhkx.emotion_scene_robustness import inference_frame
from cuhkx.emotion_visual_nested import array_digest, verify_training_content
from cuhkx.validation import evaluation_report, sha256_file
from scripts.audit_emotion_nested_provenance import audit_word_priors
from scripts.research_write_guard import install


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def deny_teacher_fit(*args, **kwargs):
    raise RuntimeError('Teacher training is forbidden; use retained scoped checkpoints')


def load_inputs(config):
    qa = pd.read_csv(ROOT / config['train_qa'], dtype=str, keep_default_na=False)
    manifest = json.loads((ROOT / config['teacher_manifest']).read_text())
    frame, contexts = prepare(qa, training=True)
    if len(frame) != 809 or set(frame.subject) != OFFICIAL_SUBJECTS:
        raise ValueError('expected all809 officialTrain emotion questions')
    banks = {name: read_sensor(ROOT / path, manifest['original_fit_config']['views'][name])
             for name, path in config['feature_paths'].items()}
    matrices, masks = {}, {}
    for name, bank in banks.items():
        matrices[name], masks[name] = sensor_matrix(frame, bank)
    local, ids = student.local_inputs(qa, banks, training=True)
    if ids != frame.qa_id.tolist():
        raise ValueError('teacher/student question order differs')
    labels = np.asarray([student.LETTERS.index(letter) for letter in frame.answer], dtype=np.int64)
    return qa, manifest, frame, contexts, banks, matrices, masks, local, labels


def assert_local_invariance(qa, banks, original):
    safe = qa.loc[qa.source.eq('HAU') & qa.category.eq('emotion')].copy()
    safe['answer'] = 'DO_NOT_READ'
    safe['position'] = 99
    safe['group'] = 'DO_NOT_READ'
    safe['subject'] = 'DO_NOT_READ'
    safe['qa_id'] = [f'anonymous_{i}' for i in range(len(safe))]
    poisoned, _ = student.local_inputs(safe, banks, training=True)
    if joblib.hash(poisoned) != joblib.hash(original):
        raise ValueError('student inputs depend on labels, IDs, groups or other questions')
    # Anonymous input paths require the same explicit bank-key relabeling.
    renamed = safe.copy()
    original_units = [student.feature_unit(path, training=True) for path in safe.path]
    renamed['path'] = [f'HAU/anonymous/{i}' for i in range(len(safe))]
    renamed_units = [student.feature_unit(path, training=True) for path in renamed.path]
    renamed_banks = {name: {**bank, 'index': {new: bank['index'][old] for old, new in
        zip(original_units, renamed_units, strict=True) if old in bank['index']}} for name, bank in banks.items()}
    changed, _ = student.local_inputs(renamed, renamed_banks, training=True)
    if joblib.hash(changed) != joblib.hash(original):
        raise ValueError('student features encode recording path or subject identity')
    return {'all809_other_questions_removed_target_position_group_subject_ID_poison': 'exact',
            'all809_explicit_bank_key_anonymous_paths': 'exact'}


def freeze_inputs(config, config_path, manifest, out):
    paths = [config_path, ROOT / config['train_qa'], ROOT / config['teacher_manifest'],
             *(ROOT / path for path in config['feature_paths'].values()),
             ROOT / 'runs/20260909_raw_training_sensor_rebuild_v1/report.json', Path(__file__),
             ROOT / 'src/cuhkx/local_emotion_student.py', ROOT / 'src/cuhkx/local_emotion_teacher.py',
             ROOT / 'scripts/research_write_guard.py', ROOT / 'scripts/audit_emotion_nested_provenance.py']
    paths += [ROOT / path for path in manifest['original_source_hashes']]
    for name, digest in manifest['original_source_hashes'].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError('old teacher source changed: ' + name)
    for key, item in manifest['models'].items():
        if key == 'final':
            continue  # Never use the allTrain teacher to label its own fitting rows.
        for kind in ('checkpoint', 'scope'):
            path = Path(item[kind]['path'])
            if sha256_file(path) != item[kind]['sha256']:
                raise ValueError('old teacher source identity differs: ' + key)
            paths.append(path)
    assets = {str(path.resolve()): {'path': str(path.resolve()), 'sha256': sha256_file(path),
              'bytes': path.stat().st_size, 'isolation': 'read-only old asset; separate new worktree/run',
              'recovery': 'retain original asset at listed path; isolate only this new experiment output',
              'recovery_check': 'sha256sum ' + str(path.resolve())} for path in paths}
    write(out / 'protection.json', {'assets': list(assets.values()), 'config': config,
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'created_utc': datetime.now(UTC).isoformat(), 'D_host_free_bytes': shutil.disk_usage('/mnt/d').free,
        'environment': {'python': sys.executable, 'torch': torch.__version__, 'numpy': np.__version__,
                        'joblib': joblib.__version__, 'cpu_affinity': sorted(os.sched_getaffinity(0))}})
    sources = out / 'sources'
    sources.mkdir()
    for path in [config_path, Path(__file__), ROOT / 'src/cuhkx/local_emotion_student.py', ROOT / 'src/cuhkx/local_emotion_teacher.py']:
        (sources / path.name).write_bytes(path.read_bytes())
    return assets


def check_resources(out, config):
    size = sum(path.stat().st_size for path in out.rglob('*') if path.is_file())
    free = shutil.disk_usage('/mnt/d').free
    if size > config['budget_bytes'] or free < config['host_reserve_bytes']:
        raise RuntimeError(f'resource limit: new={size}, actual_D_free={free}')
    return {'new_bytes': size, 'actual_D_free_bytes': free}


def generate_teacher(key, outer_forbidden, manifest, frame, contexts, matrices, masks, directory, temperature):
    item = manifest['models'][key]
    path = Path(item['checkpoint']['path'])
    if sha256_file(path) != item['checkpoint']['sha256']:
        raise ValueError('teacher model changed')
    scope = json.loads(Path(item['scope']['path']).read_text())
    train, valid = set(item['train_subjects']), set(item['validation_subjects'])
    teacher.validate_scope(train, valid, outer_forbidden)
    if set(scope['train_subjects']) != train or set(scope['validation_subjects']) != valid:
        raise ValueError('teacher scope record differs')
    if scope['source_hashes'] != manifest['original_source_hashes'] or scope['input_hashes'] != manifest['original_input_hashes']:
        raise ValueError('teacher source/input provenance differs')
    bundle = joblib.load(path)
    if set(bundle['training_subjects']) != train or bundle['config'] != manifest['original_fit_config']:
        raise ValueError('actual teacher bundle scope/config differs')
    verify_training_content(bundle, scope, frame, matrices, masks)
    audit_word_priors(bundle, frame, contexts, train)
    if bundle['fitted_units'] != scope['training_units_by_sensor']:
        raise ValueError('actual teacher fitting unit list differs')
    for model in bundle['models'].values():
        if not np.array_equal(model.classes_, [0, 1, 2]):
            raise ValueError('teacher manner class axis differs')
        model.n_jobs = 1
    rows = np.flatnonzero(frame.subject.isin(valid))
    safe = inference_frame(frame)
    result, evidence = teacher.teacher_outputs(bundle, safe, contexts, matrices, masks, rows, temperature=temperature)
    poison, poison_evidence = teacher.teacher_outputs(bundle, frame.assign(answer='DO_NOT_READ', truth_word='DO_NOT_READ'),
        contexts, matrices, masks, rows, temperature=temperature)
    if joblib.hash(result) != joblib.hash(poison) or joblib.hash(evidence) != joblib.hash(poison_evidence):
        raise ValueError('teacher soft labels depend on heldout answers')
    probabilities, hard, states = result
    soft = np.stack([probabilities[qa_id] for qa_id in frame.iloc[rows].qa_id])
    directory.mkdir(parents=True)
    np.savez_compressed(directory / 'teacher.npz', rows=rows, qa_ids=frame.iloc[rows].qa_id.to_numpy(dtype=str),
        probabilities=soft, map_answers=np.asarray([hard[qa_id] for qa_id in frame.iloc[rows].qa_id]))
    write(directory / 'joint_states.json', states)
    receipt = {'key': key, 'checkpoint': item['checkpoint'], 'scope': item['scope'],
        'train_subjects': sorted(train), 'target_subjects': sorted(valid), 'outer_forbidden': sorted(outer_forbidden),
        'rows': len(rows), 'qa_ids': frame.iloc[rows].qa_id.tolist(), 'target_answers_removed': True,
        'word_priors_exact': True, 'training_content_exact': True, 'poison_exact': True,
        'soft_array_sha256': array_digest(rows, soft), 'evidence_digest': joblib.hash(evidence),
        'decoder': teacher.DECODER, 'temperature': temperature}
    write(directory / 'receipt.json', receipt)
    return rows, soft, hard, receipt


def fit_student(initial, batch, labels, soft, train_rows, options, seed, arm):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = student.LocalEmotionStudent(batch['media'].shape[1], options['hidden'], options['dropout'])
    model.load_state_dict(copy.deepcopy(initial))
    optimizer = torch.optim.AdamW(model.parameters(), lr=options['learning_rate'], weight_decay=options['weight_decay'])
    generator = np.random.default_rng(seed)
    history, step_count, order_digest = [], 0, __import__('hashlib').sha256()
    target = torch.from_numpy(labels)
    teacher_soft = torch.from_numpy(soft) if arm == 'B' else None
    for epoch in range(options['epochs']):
        order = generator.permutation(train_rows)
        order_digest.update(order.tobytes())
        sums = np.zeros(3, dtype=np.float64)
        model.train()
        for start in range(0, len(order), options['batch_size']):
            rows = order[start:start + options['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            logits = model(**{key: value[rows] for key, value in batch.items()})
            loss, ce, kl = student.objective(logits, target[rows], teacher_soft[rows] if teacher_soft is not None else None,
                temperature=2., distill_weight=options['distillation_weight'] if arm == 'B' else 0.)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite student objective')
            loss.backward()
            optimizer.step()
            sums += np.asarray([float(loss.detach()), float(ce.detach()), float(kl.detach())]) * len(rows)
            step_count += 1
        history.append({'epoch': epoch + 1, 'loss': sums[0] / len(order), 'qa_ce': sums[1] / len(order),
                        'distillation_kl_T2': sums[2] / len(order), 'steps': step_count})
    return model, optimizer, history, step_count, order_digest.hexdigest()


def train(args, config):
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    data = load_inputs(config)
    qa, manifest, frame, contexts, banks, matrices, masks, local, labels = data
    assets = freeze_inputs(config, args.config, manifest, out)
    local_checks = assert_local_invariance(qa, banks, local)
    input_digest = joblib.hash(local)
    scopes, predictions = [], []
    teacher_count = 0
    with threadpool_limits(limits=config['cpu_threads']):
        for index, heldout in enumerate([*sorted(OFFICIAL_SUBJECTS), None]):
            name = heldout or 'final'
            directory = out / 'scopes' / name
            directory.mkdir(parents=True)
            train_users = OFFICIAL_SUBJECTS - ({heldout} if heldout else set())
            train_rows = np.flatnonzero(frame.subject.isin(train_users))
            valid_rows = np.flatnonzero(frame.subject.eq(heldout)) if heldout else np.arange(len(frame))
            inner_keys = next(fold['inner_keys'] for fold in manifest['outer_folds'] if fold['subject'] == heldout) if heldout else manifest['final_inner_keys']
            soft = np.full((len(frame), 4), np.nan, dtype=np.float32)
            coverage, teacher_records = set(), []
            for inner_index, key in enumerate(inner_keys):
                rows, values, _, receipt = generate_teacher(key, {heldout} if heldout else set(), manifest,
                    frame, contexts, matrices, masks, directory / f'teacher_inner_{inner_index}', config['teacher']['temperature'])
                if coverage & set(rows) or not set(rows) <= set(train_rows):
                    raise ValueError('crossfit teacher coverage overlaps or escapes student fitting scope')
                soft[rows] = values
                coverage.update(rows)
                teacher_records.append(receipt)
                teacher_count += 1
            if coverage != set(train_rows) or not np.isfinite(soft[train_rows]).all():
                raise ValueError('crossfit teacher does not cover every fitting QA exactly once')
            heldout_teacher = None
            if heldout:
                rows, _, heldout_teacher, receipt = generate_teacher('outer/' + heldout, set(), manifest,
                    frame, contexts, matrices, masks, directory / 'teacher_outer', config['teacher']['temperature'])
                if not np.array_equal(rows, valid_rows):
                    raise ValueError('heldout teacher row order differs')
                teacher_records.append(receipt)
                teacher_count += 1
            normalizer = student.fit_normalizer(local, train_rows)
            joblib.dump(normalizer, directory / 'normalizer.joblib', compress=3)
            batch = student.tensors(local, normalizer)
            seed = config['seed'] + index * 10000
            torch.manual_seed(seed)
            model = student.LocalEmotionStudent(batch['media'].shape[1], config['student']['hidden'], config['student']['dropout'])
            initial = copy.deepcopy(model.state_dict())
            initial_digest = student.state_digest(initial)
            torch.save({'state_dict': initial, 'seed': seed, 'media_width': batch['media'].shape[1]}, directory / 'initial.pt')
            scope_record = {'name': name, 'train_subjects': sorted(train_users), 'validation_subjects': [heldout] if heldout else [],
                'training_qa_ids': frame.iloc[train_rows].qa_id.tolist(), 'seed': seed, 'teacher_sources': teacher_records,
                'student_input_digest': input_digest, 'initial_state_digest': initial_digest,
                'actual_fit_tensor_sha256': array_digest(*[value[train_rows].numpy() for value in batch.values()], labels[train_rows]),
                'soft_training_sha256': array_digest(soft[train_rows]), 'arms': {}}
            for arm in ('A', 'B'):
                trained, optimizer, history, steps, order_hash = fit_student(initial, batch, labels, soft, train_rows, config['student'], seed, arm)
                state = copy.deepcopy(trained.state_dict())
                checkpoint = {'state_dict': state, 'optimizer': optimizer.state_dict(), 'arm': arm,
                    'initial_state_digest': initial_digest, 'final_state_digest': student.state_digest(state),
                    'batch_order_digest': order_hash, 'steps': steps, 'seed': seed, 'config': config['student'],
                    'media_width': batch['media'].shape[1], 'training_subjects': sorted(train_users),
                    'validation_subjects': [heldout] if heldout else [], 'normalizer_sha256': sha256_file(directory / 'normalizer.joblib')}
                torch.save(checkpoint, directory / f'{arm}.pt')
                write(directory / f'{arm}_history.json', history)
                probability = student.infer(trained, batch, valid_rows)
                # Reload the actual saved weights before accepting each completed fit.
                loaded = torch.load(directory / f'{arm}.pt', map_location='cpu', weights_only=False)
                replay_model = student.LocalEmotionStudent(batch['media'].shape[1], config['student']['hidden'], config['student']['dropout'])
                replay_model.load_state_dict(loaded['state_dict'])
                if not np.array_equal(probability, student.infer(replay_model, batch, valid_rows)):
                    raise ValueError('saved model probabilities changed after reload')
                np.savez_compressed(directory / f'{arm}_predictions.npz', rows=valid_rows,
                    qa_ids=frame.iloc[valid_rows].qa_id.to_numpy(dtype=str), probabilities=probability)
                difference = np.sqrt(sum(float(torch.sum((state[key] - initial[key]) ** 2)) for key in state))
                scope_record['arms'][arm] = {'steps': steps, 'batch_order_digest': order_hash,
                    'checkpoint_sha256': sha256_file(directory / f'{arm}.pt'), 'final_state_digest': checkpoint['final_state_digest'],
                    'weight_update_l2': difference, 'reload_probabilities_exact': True}
                if heldout:
                    predictions.append(pd.DataFrame({'qa_id': frame.iloc[valid_rows].qa_id.to_numpy(),
                        'subject': heldout, 'arm': arm, 'prediction': np.asarray(student.LETTERS)[probability.argmax(axis=1)],
                        'teacher_prediction': [heldout_teacher[qa_id] for qa_id in frame.iloc[valid_rows].qa_id]}))
            if scope_record['arms']['A']['steps'] != scope_record['arms']['B']['steps'] or scope_record['arms']['A']['batch_order_digest'] != scope_record['arms']['B']['batch_order_digest']:
                raise ValueError('A/B optimization steps or minibatch order differ')
            write(directory / 'scope.json', scope_record)
            scopes.append(scope_record)
            resources = check_resources(out, config)
            # Progress contains coverage only; no outer-score based decisions.
            print(f'local_emotion_scope={name} completed={len(scopes)}/19 teacher={teacher_count}/75 steps_each={steps}', flush=True)
            (out / 'state.json').write_text(json.dumps({'status': 'running', 'scopes': len(scopes), 'teachers': teacher_count, 'pid': os.getpid(), 'resources': resources}) + '\n')
    if teacher_count != 75 or len(scopes) != 19 or joblib.hash(local) != input_digest:
        raise ValueError('scope/teacher counts or immutable inputs differ')
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_csv(out / 'outer_predictions.csv', index=False)
    reports = {}
    baseline = predictions.loc[predictions.arm.eq('A')].set_index('qa_id').prediction
    for arm in ('A', 'B'):
        selected = predictions.loc[predictions.arm.eq(arm)].set_index('qa_id').loc[frame.qa_id]
        evaluated = frame[['qa_id', 'source', 'category', 'subject', 'answer']].copy()
        evaluated['prediction'] = selected.prediction.to_numpy()
        evaluated['baseline_prediction'] = frame.qa_id.map(baseline).to_numpy()
        reports[arm] = evaluation_report(evaluated, samples=10000, seed=config['seed'], confidence=.95)
        reports[arm]['modality_slices'] = {
            name: {state: {'rows': int(selected_mask.sum()),
                           'correct': int(evaluated.loc[selected_mask, 'prediction'].eq(evaluated.loc[selected_mask, 'answer']).sum())}
                   for state, selected_mask in [('available', local['masks'][:, index]),
                                                ('missing', ~local['masks'][:, index])]}
            for index, name in enumerate(student.MODALITIES)}
    teacher_evaluation = frame[['qa_id', 'source', 'category', 'subject', 'answer']].copy()
    teacher_evaluation['prediction'] = predictions.loc[predictions.arm.eq('A')].set_index('qa_id').loc[frame.qa_id].teacher_prediction.to_numpy()
    reports['fixed_full_scene_teacher'] = evaluation_report(teacher_evaluation, samples=10000, seed=config['seed'], confidence=.95)
    for path, item in assets.items():
        if sha256_file(Path(path)) != item['sha256']:
            raise ValueError('protected old/source/input asset changed: ' + path)
    write(out / 'report.json', {'status': 'completed_fixed_QA_vs_crossfit_distillation', 'conditions': reports,
        'student_scopes': 19, 'student_fits': 38, 'teacher_checkpoints_reused': 75, 'teacher_fits': 0,
        'local_invariance': local_checks, 'protected_asset_posthash': len(assets),
        'limits': config['validation'], 'wall_seconds': time.perf_counter() - started,
        'Test_rows': 0, 'pseudo_rows': 0, 'uploads': 0, 'resources': check_resources(out, config)})
    (out / 'state.json').write_text(json.dumps({'status': 'completed', 'student_scopes': 19, 'teachers': 75, 'pid': os.getpid()}) + '\n')
    freeze(out)


def freeze(out):
    files = sorted(path for path in out.rglob('*') if path.is_file())
    write(out / 'completion_freeze.json', {'files': [{'path': str(path.resolve()), 'sha256': sha256_file(path),
        'bytes': path.stat().st_size} for path in files], 'status': 'closed'})


def replay(args, config):
    source, out = args.source_run.resolve(), args.output_dir.resolve()
    original_report = json.loads((source / 'report.json').read_text())
    if original_report['status'] != 'completed_fixed_QA_vs_crossfit_distillation':
        raise ValueError('fitting run not complete')
    out.mkdir(parents=True, exist_ok=False)
    for item in json.loads((source / 'completion_freeze.json').read_text())['files']:
        if sha256_file(Path(item['path'])) != item['sha256']:
            raise ValueError('frozen fitting output changed')
    qa, _, frame, _, banks, _, _, local, _ = load_inputs(config)
    checks = assert_local_invariance(qa, banks, local)
    records = []
    with threadpool_limits(limits=config['cpu_threads']):
        for directory in sorted((source / 'scopes').iterdir()):
            scope = json.loads((directory / 'scope.json').read_text())
            normalizer = joblib.load(directory / 'normalizer.joblib')
            expected = student.fit_normalizer(local, np.flatnonzero(frame.subject.isin(scope['train_subjects'])))
            if joblib.hash(normalizer) != joblib.hash(expected):
                raise ValueError('normalizer fitting scope differs')
            batch = student.tensors(local, normalizer)
            for arm in ('A', 'B'):
                checkpoint = torch.load(directory / f'{arm}.pt', map_location='cpu', weights_only=False)
                if student.state_digest(checkpoint['state_dict']) != scope['arms'][arm]['final_state_digest']:
                    raise ValueError('learned weights changed')
                model = student.LocalEmotionStudent(checkpoint['media_width'], config['student']['hidden'], config['student']['dropout'])
                model.load_state_dict(checkpoint['state_dict'])
                with np.load(directory / f'{arm}_predictions.npz') as archive:
                    rows, expected_probability = archive['rows'], archive['probabilities']
                actual = student.infer(model, batch, rows)
                if not np.array_equal(actual, expected_probability):
                    raise ValueError('independent model reload probabilities differ')
                # Each prediction was one current question. Also reorder whole requested batch.
                if not np.array_equal(student.infer(model, batch, rows[::-1])[::-1], actual):
                    raise ValueError('batch ordering affects student')
                records.append({'scope': scope['name'], 'arm': arm, 'rows': len(rows), 'probabilities_exact': True,
                                'reversed_batch_exact': True, 'actual_probability_sha256': array_digest(actual)})
                np.savez_compressed(out / f'{scope["name"]}_{arm}.npz', rows=rows, probabilities=actual)
            print(f'local_emotion_reload={scope["name"]} models={len(records)}/38', flush=True)
    for item in json.loads((source / 'protection.json').read_text())['assets']:
        if sha256_file(Path(item['path'])) != item['sha256']:
            raise ValueError('protected old asset changed during independent replay')
    write(out / 'report.json', {'status': 'passed_independent_local_student_reload', 'models': len(records),
        'records': records, 'local_invariance': checks, 'source_report_sha256': sha256_file(source / 'report.json'),
        'fits': 0, 'Test_rows': 0, 'pseudo_rows': 0, 'resources': check_resources(out, config)})
    freeze(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['fit', 'replay'])
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/local_emotion_distillation_v1.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-run', type=Path)
    args = parser.parse_args()
    install(Path('/root/workspace/CUHK-X'))
    config = json.loads(args.config.read_text())
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:config['cpu_threads']])
    torch.set_num_threads(config['cpu_threads'])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    BaseForest.fit = deny_teacher_fit
    if args.mode == 'fit':
        train(args, config)
    else:
        replay(args, config)


if __name__ == '__main__':
    main()
