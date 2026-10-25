#!/usr/bin/env python3
"""Export and audit action224 on CPU; separately schedule real four-expert CUDA inference."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
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
from cuhkx.action_independent_visual_inference import predict
from cuhkx.action_question_dropout import CONDITIONS, condition_frame
from cuhkx.metadata import clip_key, feature_unit
from scripts.infer_action_independent import identity, write
from scripts.prepare_r34_complete_inference_v2 import GUARD
from scripts.research_write_guard import install
from scripts.run_action_generic_nested import verify_manifest


def resources(config):
    output = ROOT / config['output']
    size = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    free = shutil.disk_usage('/mnt/d').free
    assert size <= config['budget_bytes'] and free >= config['reserve_bytes'], (size, free)
    return {'new_bytes': size, 'D_free_bytes': free}


def call(script, arguments, log, environment=None):
    with log.open('x') as stream:
        subprocess.run([sys.executable, str(ROOT / 'scripts/research_write_guard.py'),
            str(ROOT / 'scripts' / script), *map(str, arguments)], cwd=ROOT,
            stdout=stream, stderr=subprocess.STDOUT, check=True, env=environment)


def load_raw(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def compare_raw(current, expected):
    index = {unit: i for i, unit in enumerate(expected['units'])}
    for key in ['probabilities', 'visual', 'prior']:
        np.testing.assert_array_equal(current[key], expected[key][[index[u] for u in current['units']]])
    index = {tuple(pair): i for i, pair in enumerate(zip(expected['option_qa_ids'], expected['option_letters'], strict=True))}
    rows = [index[tuple(pair)] for pair in zip(current['option_qa_ids'], current['option_letters'], strict=True)]
    for key in ['option_matrix', 'option_probability']:
        np.testing.assert_array_equal(current[key], expected[key][rows])
    expected_states = {tuple(row['qa_ids']): row for row in json.loads(str(expected['state_records_json']))}
    for row in json.loads(str(current['state_records_json'])):
        assert row == expected_states[tuple(row['qa_ids'])]


def subset_bank(qa, bank):
    units = np.asarray(list(dict.fromkeys(feature_unit(p, training=False) for p in qa.path)))
    index = {unit: i for i, unit in enumerate(bank['units'])}
    return {'units': units, 'X': bank['X'][[index[u] for u in units]], 'feature_names': bank['feature_names']}


def cpu(config, config_path):
    output = ROOT / config['output']
    output.mkdir()
    write(output / 'protection.json', {'assets': config['assets'], 'config': identity(config_path),
        'resources': resources(config), 'protocol': config['protocol'],
        'restore': 'All listed models, caches, logs, sources readonly. Abandon only this new output; verify original SHA. Environment unchanged.'})
    for relative in config['source_files']:
        target = output / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    work = output / 'cpu'
    work.mkdir()
    call('infer_action_independent_visual.py', ['export', '--config', config_path, '--output-dir', work / 'bundle'], work / 'export.log')
    qa = pd.read_csv(ROOT / config['action_qa'], usecols=base.METADATA, dtype=str, keep_default_na=False)
    bank = load_raw(ROOT / config['action_features'])
    assert qa.path.map(clip_key).nunique() == 144
    inputs = work / 'inputs'
    inputs.mkdir()
    (inputs / 'acquisition.csv').write_text('FORBIDDEN_ADJACENT_SCENE_FILE\n')
    model = joblib.load(work / 'bundle/model.joblib')
    model['frequency'].n_jobs = 1
    audits = []
    for condition in CONDITIONS:
        visible = condition_frame(qa, condition)
        selected_bank = subset_bank(visible, bank)
        qa_path = inputs / (condition + '.csv')
        features = inputs / (condition + '.npz')
        visible.to_csv(qa_path, index=False)
        np.savez_compressed(features, **selected_bank)
        folder = work / condition
        call('infer_action_independent_visual.py', ['infer', '--bundle', work / 'bundle',
            '--qa-csv', qa_path, '--features', features, '--output-dir', folder], work / (condition + '.log'))
        reference = pd.read_csv(folder / 'predictions.csv', dtype={'qa_id': str, 'prediction': str})
        raw = load_raw(folder / 'evidence.npz')
        actual, evidence = predict(model, visible, selected_bank)
        pd.testing.assert_frame_equal(actual, reference, check_dtype=False, atol=1e-12, rtol=0)
        compare_raw(evidence, raw)
        clips = visible.path.map(clip_key)
        ordered = list(clips.unique())
        records = []
        for size in [1, 7]:
            parts = []
            for start in range(0, len(ordered), size):
                subset = visible.loc[clips.isin(ordered[start:start + size])]
                prediction, evidence = predict(model, subset, subset_bank(subset, selected_bank))
                expected = reference.set_index('qa_id').loc[prediction.qa_id].reset_index()
                pd.testing.assert_frame_equal(prediction, expected, check_dtype=False, atol=1e-12, rtol=0)
                compare_raw(evidence, raw)
                parts.append(prediction)
                records.append({'batch_size': size, 'qa_ids': prediction.qa_id.tolist(), 'exact_probabilities_options_states_answers': True})
                if size == 1 and (start + 1) % 48 == 0:
                    print(json.dumps({'condition': condition, 'single_recordings_checked': start + 1}), flush=True)
            joined = pd.concat(parts).set_index('qa_id').loc[reference.qa_id].reset_index()
            pd.testing.assert_frame_equal(joined, reference, check_dtype=False, atol=1e-12, rtol=0)
            joined.to_csv(folder / f'partition_{size}.csv', index=False)
        write(folder / 'isolation_audit.json', {'condition': condition, 'batches': records,
            'recordings': len(ordered), 'action_rows': len(reference), 'target_accuracy': 'unknown; this is inference consistency only'})
        audits.append({'condition': condition, 'recordings': len(ordered), 'action_rows': len(reference), 'batches': len(records),
            'independent_process_reload_exact': True, 'probabilities_options_states_answers_exact': True})
        print(json.dumps({'completed_condition': condition, 'resources': resources(config)}), flush=True)
    # Opaque recording/question aliases and poisoned unused target fields.
    renamed = qa.copy()
    qmap = {q: f'opaque_question_{i:04}' for i, q in enumerate(qa.qa_id)}
    clips = qa.path.map(clip_key)
    cmap = {clip: f'opaque/user999/recording_{i:04}' for i, clip in enumerate(clips.unique())}
    umap = {feature_unit(k, training=False): feature_unit(v, training=False) for k, v in cmap.items()}
    renamed['qa_id'] = qa.qa_id.map(qmap)
    renamed['path'] = clips.map(cmap)
    renamed = renamed.assign(answer='POISON', target='POISON', prediction='POISON', group='SHARED_POISON')
    renamed.to_csv(inputs / 'renamed.csv', index=False)
    np.savez_compressed(inputs / 'renamed.npz', X=bank['X'], feature_names=bank['feature_names'], units=np.asarray([umap[u] for u in bank['units']]))
    write(inputs / 'rename_restore.json', {'qa': qmap, 'clips': cmap})
    call('infer_action_independent_visual.py', ['infer', '--bundle', work / 'bundle', '--qa-csv', inputs / 'renamed.csv',
        '--features', inputs / 'renamed.npz', '--output-dir', work / 'renamed'], work / 'renamed.log')
    reference = pd.read_csv(work / 'full_same_clip/predictions.csv', dtype=str)
    renamed_prediction = pd.read_csv(work / 'renamed/predictions.csv', dtype=str)
    renamed_prediction.qa_id = renamed_prediction.qa_id.map({v: k for k, v in qmap.items()})
    pd.testing.assert_frame_equal(reference, renamed_prediction)
    raw = load_raw(work / 'renamed/evidence.npz')
    raw['units'] = np.asarray([{v: k for k, v in umap.items()}[u] for u in raw['units']])
    raw['option_qa_ids'] = np.asarray([{v: k for k, v in qmap.items()}[q] for q in raw['option_qa_ids']])
    states = json.loads(str(raw['state_records_json']))
    for row in states:
        row['qa_ids'] = [{v: k for k, v in qmap.items()}[q] for q in row['qa_ids']]
    raw['state_records_json'] = np.asarray(json.dumps(states))
    compare_raw(raw, load_raw(work / 'full_same_clip/evidence.npz'))
    verify_manifest(config['assets'])
    write(work / 'report.json', {'status': 'complete', 'conditions': audits, 'opaque_and_target_poison_exact': True,
        'target_columns_read': False, 'cross_recording_context': False, 'new_fits': 0, 'GPU_used': False,
        'pseudo_dependency': False, 'resources': resources(config), 'protected_assets_reverified': len(config['assets'])})
    write(work / 'completion_freeze.json', {'status': 'complete', 'files': [identity(p) for p in sorted(work.rglob('*')) if p.is_file()]})


def complete(config):
    output = ROOT / config['output']
    assert json.loads((output / 'cpu/report.json').read_text())['status'] == 'complete'
    folder = output / 'complete'
    folder.mkdir()
    old = ROOT / config['v4_run']
    for name in ['inputs', 'guard', 'read_audit']:
        (output / name).mkdir()
    for p in (old / 'inputs').iterdir():
        shutil.copyfile(p, output / 'inputs' / p.name)
    (output / 'guard/sitecustomize.py').write_text(GUARD)
    shutil.copyfile(old / 'guard_policy.json', output / 'guard_policy.json')
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(output / 'guard'),
        CUHKX_RESEARCH_INFERENCE_RUN=str(output), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')
    for condition in ['grouped', 'independent']:
        destination = folder / condition / 'submission.csv'
        arguments = [output / 'inputs', destination, '--config', ROOT / config['profile'], '--workers', '2']
        if condition == 'grouped':
            arguments += ['--scene-manifest', output / 'inputs/historical_release_scene_fixture.csv']
        call('infer_conditioned_experts_v5.py', arguments, folder / (condition + '.log'), environment)
        assert len(pd.read_csv(destination)) == 682
        resources(config)
    assert identity(folder / 'grouped/submission.csv')['sha256'] == config['expected_grouped_sha256']
    independent = pd.read_csv(folder / 'independent/submission.csv', dtype=str)
    former = pd.read_csv(old / 'independent/submission.csv', dtype=str)
    action = pd.read_csv(output / 'cpu/full_same_clip/predictions.csv', dtype=str)
    ids = set(action.qa_id)
    pd.testing.assert_frame_equal(independent.loc[~independent.qa_id.isin(ids)].reset_index(drop=True),
        former.loc[~former.qa_id.isin(ids)].reset_index(drop=True))
    pd.testing.assert_frame_equal(independent.set_index('qa_id').loc[action.qa_id].reset_index(), action[['qa_id', 'prediction']])
    compare_raw(load_raw(folder / 'independent/submission.work/action/evidence.npz'), load_raw(output / 'cpu/full_same_clip/evidence.npz'))
    audits = [json.loads(p.read_text()) for p in sorted((output / 'read_audit').glob('*.json'))]
    assert audits and not any(item['blocked'] for item in audits)
    verify_manifest(config['assets'])
    write(folder / 'report.json', {'status': 'complete', 'grouped_rows': 682, 'independent_rows': 682,
        'grouped_exact_reference_sha256': config['expected_grouped_sha256'], 'new_action_rows_exact': 427,
        'other_experts_exact_v4': True, 'action_raw_state_replay_exact': True, 'new_fits': 0, 'uploads': 0,
        'pseudo_dependency': False, 'CUDA_used_for_HARn_tail_inference': True, 'read_audit_processes': len(audits),
        'blocked_data_reads': 0, 'resources': resources(config)})
    write(output / 'completion_freeze.json', {'status': 'complete_all_freezable',
        'assets_reverified': config['assets'], 'files': [identity(p) for p in sorted(output.rglob('*')) if p.is_file()],
        'resources': resources(config)})


def main():
    install(Path('/root/workspace/CUHK-X'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['cpu', 'complete'])
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    verify_manifest(config['assets'])
    resources(config)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    torch.set_num_threads(2)
    threadpool_limits(limits=2)
    if args.mode == 'cpu':
        cpu(config, args.config.resolve())
    else:
        complete(config)
    print(json.dumps({'status': 'complete', 'mode': args.mode, 'resources': resources(config)}), flush=True)


if __name__ == '__main__':
    main()
