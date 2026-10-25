"""Infer the fixed temporal residual from declared D1 banks and full parent."""
from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    inputs = config['inputs']
    for item in inputs.values():
        assert sha(item['path']) == item['sha256'], item['path']
    paths = {key: Path(item['path']) for key, item in inputs.items()}
    # Keep the recorded producer's joblib classes; append the new consumer only.
    sys.path[:0] = [config['D1_source'], str(ROOT)]
    import cuhkx
    cuhkx.__path__.append(str(ROOT / 'src/cuhkx'))
    from cuhkx.server_temporal_d1_head import MODALITIES, read_d1_bank, unpack_d1
    from cuhkx.server_temporal_d1_sequence import read_d1_centers
    from cuhkx.sequence_skeleton_nested import prepare_sequences
    from cuhkx.server_temporal_d6_residual import PAIRS, features
    from cuhkx.server_temporal_d10_backoff import predict as prior_predict
    from cuhkx.temporal_consumer_residual import predict

    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    qa = pd.read_csv(paths['qa_csv'], dtype=str, keep_default_na=False)
    qa = qa[['qa_id', 'source', 'path', 'category', 'A', 'B', 'C', 'D']]
    frame = prepare_sequences(qa, training=False)
    with np.load(paths['pair_evidence']) as archive:
        a = {k: archive[k].copy() for k in archive.files}
    assert set(a['qa_ids']) == set(frame.qa_id)
    frame = frame.set_index('qa_id').loc[a['qa_ids']]
    classes = np.asarray(frame.classes.tolist())
    np.testing.assert_array_equal(classes, a['classes'])
    bank = read_d1_bank(Path(config['bank']), 'test', 'coarse')
    centers = read_d1_centers(Path(config['bank']), 'test', 'coarse')
    bundle = joblib.load(paths['D1_model'])
    receipt = json.loads(paths['residual_receipt'].read_text())
    prior = json.loads(paths['D10_model'].read_text())
    assert sorted(bundle['training_subjects']) == sorted(receipt['training_people'])
    assert sorted(prior['training_subjects']) == sorted(receipt['training_people'])
    assert len(receipt['training_people']) == 18
    tree = ast.parse(paths['curve_source'].read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'view_curves')
    env = dict(np=np, torch=torch, expit=expit, unpack_d1=unpack_d1, MODALITIES=MODALITIES)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(paths['curve_source']), 'exec'), env)
    matrix = bank['X'][[bank['index'][unit] for unit in frame.unit]]
    curves, mask = env['view_curves'](bundle['presence'], matrix, frame.unit.tolist(), centers)
    probability, logits = a['probability'], a['prior_logits']
    parent, offset, _, _ = prior_predict(prior, probability, logits, classes)
    np.testing.assert_array_equal(parent, a['D10_prediction'])
    np.testing.assert_allclose(offset, a['D10_score'], rtol=0, atol=1e-12)
    raw = features(probability, classes, curves, mask)
    with np.load(paths['residual_model']) as archive:
        model = {k: archive[k].copy() for k in archive.files}
    assert str(model['arm']) == 'linear'
    result, scores, delta = predict(model, raw, probability, offset)
    reverse = predict(model, raw[::-1].copy(), probability[::-1].copy(), offset[::-1].copy())
    np.testing.assert_array_equal(result, reverse[0][::-1])
    np.testing.assert_allclose(scores, reverse[1][::-1], rtol=0, atol=1e-12)
    # Recompute actual features after relabeling all four options, not just delta.
    pm = np.zeros((len(frame), 4, 4))
    lm = np.zeros_like(pm)
    for j, (left, right) in enumerate(PAIRS):
        pm[:, left, right], pm[:, right, left] = probability[:, j], 1 - probability[:, j]
        lm[:, left, right], lm[:, right, left] = logits[:, j], -logits[:, j]
    physical = [tuple(c['ABCD'.index(letter)] for letter in answer)
                for c, answer in zip(classes, result, strict=True)]
    for perm in itertools.permutations(range(4)):
        p = np.stack([pm[:, perm[l], perm[r]] for l, r in PAIRS], axis=1)
        l = np.stack([lm[:, perm[l], perm[r]] for l, r in PAIRS], axis=1)
        c = classes[:, perm]
        _, off, _, _ = prior_predict(prior, p, l, c)
        answer, energies, _ = predict(model, features(p, c, curves, mask), p, off)
        actual = [tuple(row['ABCD'.index(letter)] for letter in label)
                  for row, label in zip(c, answer, strict=True)]
        assert actual == physical
        np.testing.assert_allclose(np.sort(energies - energies.mean(1, keepdims=True)),
                                   np.sort(scores - scores.mean(1, keepdims=True)), rtol=0, atol=1e-10)
    pd.DataFrame(dict(qa_id=a['qa_ids'], prediction=result)).to_csv(args.output / 'temporal.csv', index=False)
    np.savez_compressed(args.output / 'evidence.npz', qa_ids=a['qa_ids'], curves=curves,
                        mask=mask, features=raw, prediction=result, scores=scores, delta=delta)
    full = pd.read_csv(paths['base_parent'], dtype=str).set_index('qa_id')
    np.testing.assert_array_equal(full.loc[a['qa_ids'], 'prediction'], parent)
    full.loc[a['qa_ids'], 'prediction'] = result
    full = full.loc[qa.qa_id].reset_index()[['qa_id', 'prediction']]
    full.to_csv(args.output / 'computed_parent.csv', index=False)
    cmd = [config['sat_python'], str(ROOT / 'scripts/apply_scored_sat.py'),
           '--qa-csv', str(paths['qa_csv']), '--parent-csv', str(args.output / 'computed_parent.csv'),
           '--profile', str(paths['sat_profile']), '--output-dir', str(args.output / 'sat')]
    (args.output / 'sat_command.json').write_text(json.dumps(cmd, indent=2) + '\n')
    with (args.output / 'sat.log').open('x') as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT,
                       env=dict(os.environ, PYTHONHASHSEED='0'))
    final = pd.read_csv(args.output / 'sat/submission.csv', dtype=str).set_index('qa_id').loc[qa.qa_id]
    assert len(final) == len(qa) == 682 and final.index.is_unique
    final.reset_index().to_csv(args.output / 'submission.csv', index=False)
    base = pd.read_csv(paths['base_scored'], dtype=str).set_index('qa_id').loc[qa.qa_id]
    for item in inputs.values():
        assert sha(item['path']) == item['sha256']
    report = dict(status='complete_fixed_joint_candidate_requires_dedup', rows=682,
                  temporal_rows=len(frame), changed_temporal=int((result != parent).sum()),
                  changed_full=int((final.prediction != base.prediction).sum()),
                  option_permutations_verified=len(frame) * 24, reversed_rows_verified=True,
                  csv_sha256=sha(args.output / 'submission.csv'), config_sha256=sha(args.config),
                  source_sha256=sha(__file__), limits=config['limits'], Test_truth_read=False)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
