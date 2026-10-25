#!/usr/bin/env python3
"""Assemble one preselected external R34 expert replacement or half-fusion D10 candidate."""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import joblib
import numpy as np
import pandas as pd

from cuhkx.external_candidate_contract import selected_external_recipe
from cuhkx.harn_nested import QA_COLUMNS, decode_questions, harn_rows
from cuhkx.harn_seed_ensemble import equal_evidence
from cuhkx.harn_three_family import three_family_evidence
from cuhkx.spatial_training import write_json
from cuhkx.submission import validate_submission_frames
from cuhkx.validation import sha256_file
from scripts.evaluate_r34_expert_slot import half_slot, replace_slot


def run(args, recipe_validator=selected_external_recipe,
        raw_script='infer_r34_external_harn.py', scope_note=None, visual_fusion=None):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(args.config.read_text())
    protected = {}

    def read(path, expected=None):
        path = Path(path).resolve()
        digest = sha256_file(path)
        assert expected is None or digest == expected, str(path)
        protected[str(path)] = digest
        return path

    read(args.config)
    source, fusion = recipe_validator(cfg, read)
    qa_path = read(args.data_dir / 'test_qa.csv')
    sample_path = read(args.data_dir / 'sample_submission.csv')
    parent_path = read(cfg['D10_parent'], cfg['D10_parent_sha256'])
    base = Path(cfg['base_harn'])
    read(base / 'test_probabilities.npz', cfg['base_harn_sha256'])
    descriptor = json.loads(read(cfg['parent_descriptor'], cfg['parent_descriptor_sha256']).read_text())
    parent_descriptor = json.loads(read(ROOT / descriptor['parent_checkpoint']['path'], descriptor['parent_checkpoint']['sha256']).read_text())
    entry = parent_descriptor['classifiers']['original']
    dictionary = joblib.load(read(ROOT / entry['path'], entry['sha256']))
    commands = []

    def execute(stage, command):
        commands.append({'stage': stage, 'command': list(map(str, command))})
        write_json(out / 'commands.json', commands)
        with (out / (stage + '.log')).open('x') as log:
            subprocess.run(list(map(str, command)), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)

    def sat(stage, source):
        execute(stage, ['/root/cuhkx-runtime/sat_replay_20260912_v1/bin/python', ROOT / 'scripts/apply_scored_sat.py',
                        '--qa-csv', qa_path, '--parent-csv', source, '--profile', ROOT / cfg['sat_profile'], '--output-dir', out / stage])

    sat('baseline_sat', parent_path)
    assert sha256_file(out / 'baseline_sat/submission.csv') == cfg['D10_scored_sha256']
    execute('raw_R', [sys.executable, ROOT / 'scripts' / raw_script, '--config', args.config.resolve(),
                     '--qa-csv', qa_path, '--output', out / 'r34'])
    def evidence(path):
        with np.load(read(path)) as z:
            return {k: z[k].copy() for k in z.files}

    raw = {name: evidence(base / (name + '_probabilities.npz')) for name in ['original', 'repeated', 'r34', 'videomae']}
    r18 = equal_evidence(raw['original'], raw['repeated'])
    control = three_family_evidence(r18, raw['r34'], raw['videomae'])
    baseline = evidence(base / 'test_probabilities.npz')
    for key in baseline:
        np.testing.assert_array_equal(control[key], baseline[key])
    new = evidence(out / 'r34/probabilities.npz')
    available = new['present']
    r34 = replace_slot(raw['r34'], new['units'][available], new['probabilities'][available], new['classes'])
    slot = half_slot(raw['r34'], r34) if fusion == 'half' else r34
    joint = three_family_evidence(r18, slot, raw['videomae'])
    for key in ['skeleton_probabilities', 'skeleton_present', 'skeleton_classes']:
        np.testing.assert_array_equal(joint[key], control[key])
    qa = pd.read_csv(qa_path, dtype=str, keep_default_na=False)
    target = harn_rows(qa)[QA_COLUMNS]
    weight = descriptor['selected_skeleton_weight']
    old = decode_questions(target, control, dictionary, weight, training_paths=False)
    expected = pd.read_csv(read(base / 'test_predictions.csv'), dtype=str)
    pd.testing.assert_series_equal(old.set_index('qa_id').prediction, expected.set_index('qa_id').prediction)
    def decode_target(frame):
        vectors = (visual_fusion(frame, r18, raw['r34'], r34, raw['videomae'], dictionary)
                   if visual_fusion is not None else None)
        return decode_questions(frame, joint, dictionary, weight, training_paths=False,
                                visual_option_scores=vectors)

    pred = decode_target(target)
    pd.testing.assert_frame_equal(pred, decode_target(target.assign(answer='POISON')))
    keyed = target.set_index('qa_id')
    words = {r.qa_id: keyed.loc[r.qa_id, r.prediction] for r in pred.itertuples()}
    for permutation in itertools.permutations('ABCD'):
        shifted = target.copy()
        for letter, previous in zip('ABCD', permutation):
            shifted[letter] = target[previous]
        changed = decode_target(shifted)
        keyed = shifted.set_index('qa_id')
        assert all(keyed.loc[r.qa_id, r.prediction] == words[r.qa_id] for r in changed.itertuples())
    pred.to_csv(out / 'harn_predictions.csv', index=False)
    old.to_csv(out / 'harn_control.csv', index=False)
    np.savez_compressed(out / 'combined_evidence.npz', **joint)
    computed = pd.read_csv(parent_path, dtype=str).set_index('qa_id')
    assert computed.index.is_unique and set(computed.index) == set(qa.qa_id)
    assert pred.qa_id.is_unique and set(pred.qa_id) == set(target.qa_id)
    computed.loc[pred.qa_id, 'prediction'] = pred.set_index('qa_id').prediction
    computed.loc[qa.qa_id].reset_index().to_csv(out / 'computed_parent.csv', index=False)
    sat('sat', out / 'computed_parent.csv')
    final = pd.read_csv(out / 'sat/submission.csv', dtype=str)
    validate_submission_frames(qa, pd.read_csv(sample_path, dtype=str), final)
    (out / 'submission.csv').write_bytes((out / 'sat/submission.csv').read_bytes())
    for path, digest in protected.items():
        assert sha256_file(Path(path)) == digest, path
    parent = pd.read_csv(out / 'baseline_sat/submission.csv', dtype=str).set_index('qa_id').prediction
    write_json(out / 'report.json', {'status': 'complete_external_R34_expert_D10_candidate', 'rows': len(final),
        'sha256': sha256_file(out / 'submission.csv'), 'changed_vs_D10': int(final.set_index('qa_id').prediction.ne(parent).sum()),
        'input_sha256': protected, 'selection_sha256': cfg['selection_sha256'], 'option_checks': len(target) * 24,
        'selected_arm': cfg['selected_arm'], 'external_source_arm': source,
        'fusion': fusion, 'new_fits': 0, 'Test_truth_read': False, 'upload_performed': False,
        'scope': scope_note or 'Actual raw external-source final R34, preselected replace or fixed half in the existing R34 family, unchanged R18/VideoMAE/sensors/dictionary/weights, complete D10/SAT. Original experts reused from the supplied verified reference; separate pipeline records whether freshly recomputed. No additional IMU variant is introduced.'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['config', 'data-dir', 'output']:
        p.add_argument('--' + name, required=True, type=Path)
    run(p.parse_args())


if __name__ == '__main__':
    main()
