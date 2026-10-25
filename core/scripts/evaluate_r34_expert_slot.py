#!/usr/bin/env python3
"""Keep the strong HARn fusion and replace only its R34 visual family."""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
import pandas as pd

from cuhkx.harn_nested import QA_COLUMNS, decode_questions, harn_rows
from cuhkx.harn_seed_ensemble import validate_evidence
from cuhkx.harn_three_family import three_family_evidence
from cuhkx.metadata import subject_id
from cuhkx.spatial_training import write_json
from cuhkx.validation import sha256_file
from scripts.run_harn_seed_ensemble import load_evidence
from scripts.run_harn_support_fusion import CachedScopes


def replace_slot(old, units, probability, classes):
    result = {k: v.copy() for k, v in old.items()}
    lookup = {u: i for i, u in enumerate(units)}
    if len(lookup) != len(units):
        raise ValueError('Duplicate new raw recording')
    present = np.asarray([u in lookup for u in old['units']])
    values = np.zeros((len(present), 44), dtype=float)
    values[present] = probability[[lookup[u] for u in old['units'][present]]]
    mass = values[present].sum(1, keepdims=True)
    np.testing.assert_allclose(mass, 1., rtol=0, atol=4 * np.finfo(np.float32).eps)
    values[present] /= mass
    result.update(visual_probabilities=values, visual_present=present, visual_classes=np.asarray(classes))
    validate_evidence(result)
    return result


def half_slot(old, new):
    out = {k: v.copy() for k, v in old.items()}
    denominator = old['visual_present'].astype(int) + new['visual_present'].astype(int)
    values = old['visual_probabilities'] + new['visual_probabilities']
    out.update(visual_probabilities=np.divide(values, denominator[:, None], out=np.zeros_like(values), where=denominator[:, None] > 0),
               visual_present=denominator > 0, visual_classes=np.union1d(old['visual_classes'], new['visual_classes']))
    validate_evidence(out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / 'source_snapshot'
    snapshot.mkdir()
    (snapshot / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    subprocess.run(['git', 'archive', '--format=tar.gz', '--output=' + str(snapshot.resolve() / 'source.tar.gz'), 'HEAD'], cwd=ROOT, check=True)
    write_json(args.output / 'invocation.json', {'argv': sys.argv, 'cwd': str(Path.cwd()), 'python': sys.executable,
               'numpy': np.__version__, 'pandas': pd.__version__, 'source_sha256': sha256_file(Path(__file__))})
    main = args.workspace
    plan = json.loads(args.plan.read_text())
    if plan['arms'] != ['control', 'C_replace', 'R_replace', 'C_half', 'R_half']:
        raise ValueError('Fixed five-arm contract differs')
    protected = {}
    def read(path, expected=None):
        path = Path(path).resolve()
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError('Protected input changed: ' + str(path))
        protected[str(path)] = digest
        return path
    read(args.plan)
    progress = json.loads(read(args.evaluation / 'progress.json').read_text())
    if not progress['complete'] or len(progress['completed_people']) != 18:
        raise ValueError('All18 raw evaluations required')
    qa_path = read(main / 'data/raw/kaggle/training_qa.csv')
    allqa = pd.read_csv(qa_path, keep_default_na=False)
    qa = harn_rows(allqa)
    people = set(qa.path.map(subject_id))
    if people != set(progress['required_people']):
        raise ValueError('Person coverage differs')
    baseline = main / 'runs/20260909_harn_three_family_nested_v1'
    old = pd.read_csv(read(baseline / 'oof_predictions.csv')).set_index('qa_id')
    cache_root = main / 'runs/20260905_harn_seed_ensemble_nested_v1'
    cache = CachedScopes(cache_root, qa)
    # Resolve private parents against the supplied workspace, including source replay.
    config = json.loads(read(cache_root / 'frozen_config.json').read_text())['config']
    cache.parents = {k: main / v['run'] for k, v in config['parents'].items()}
    folders = [main / 'runs/20260906_r34_tail_nested_v1', main / 'runs/20260908_harn_videomae_nested_v1']
    outputs = {arm: [] for arm in plan['arms']}
    evidence_dir = args.output / 'evidence'
    evidence_dir.mkdir()
    checks, permutations = [], 0
    for person in sorted(people):
        target = qa.loc[qa.path.map(subject_id).eq(person)]
        training = sorted(people - {person})
        fitid = 'outer_' + person + '_final'
        weight = json.loads(read(baseline / ('outer_' + person + '.json')).read_text())['selected_skeleton_weight']
        r18, dictionary = cache.load(fitid, set(training), target)
        extras = []
        for folder in folders:
            scope = json.loads(read(folder / 'fits' / (fitid + '.json')).read_text())
            if scope['train_subjects'] != training or set(scope['training_qa_ids']) != set(qa.loc[qa.path.map(subject_id).isin(training), 'qa_id']):
                raise ValueError('Frozen expert person scope differs')
            extras.append(load_evidence(read(folder / 'fits' / (fitid + '.saved_evidence.npz'))))
        old_r34, vm = extras
        variants = {'control': three_family_evidence(r18, old_r34, vm)}
        for arm in ['C', 'R']:
            path = args.models / 'models' / (person + '_' + arm + '.pt')
            receipt = json.loads(read(path.with_suffix('.json')).read_text())
            read(path, receipt['sha256'])
            raw = json.loads(read(args.evaluation / person / 'report.json').read_text())
            if receipt['training_subjects'] != training or any(subject_id(u) == person for u in receipt['training_units']):
                raise ValueError('New R34 scope differs')
            if raw['arms'][arm]['model_sha256'] != receipt['sha256']:
                raise ValueError('New probabilities came from another model')
            with np.load(read(args.evaluation / person / (arm + '_harn.npz'))) as bank:
                new = replace_slot(old_r34, bank['units'].astype(str), bank['probabilities'], receipt['supported_classes'])
            variants[arm + '_replace'] = three_family_evidence(r18, new, vm)
            variants[arm + '_half'] = three_family_evidence(r18, half_slot(old_r34, new), vm)
            checks.append({'person': person, 'arm': arm, 'old_present': int(old_r34['visual_present'].sum()),
                           'new_present': int(new['visual_present'].sum()), 'model_sha256': receipt['sha256']})
        for arm, evidence in variants.items():
            for name in ['skeleton_probabilities', 'skeleton_present', 'skeleton_classes']:
                np.testing.assert_array_equal(evidence[name], variants['control'][name])
            pred = decode_questions(target[QA_COLUMNS], evidence, dictionary, weight, training_paths=True)
            if arm == 'control':
                np.testing.assert_array_equal(pred.prediction, old.loc[pred.qa_id, 'prediction'])
            pd.testing.assert_frame_equal(pred, decode_questions(target.assign(answer='POISON')[QA_COLUMNS], evidence, dictionary, weight, training_paths=True))
            words = {r.qa_id: target.set_index('qa_id').loc[r.qa_id, r.prediction] for r in pred.itertuples()}
            for perm in itertools.permutations('ABCD'):
                shifted = target.copy()
                for new_letter, old_letter in zip('ABCD', perm):
                    shifted[new_letter] = target[old_letter]
                got = decode_questions(shifted[QA_COLUMNS], evidence, dictionary, weight, training_paths=True)
                keyed = shifted.set_index('qa_id')
                if any(keyed.loc[r.qa_id, r.prediction] != words[r.qa_id] for r in got.itertuples()):
                    raise ValueError('Option permutation changed physical answer')
                permutations += len(target)
            pred['answer'] = pred.qa_id.map(qa.set_index('qa_id').answer)
            outputs[arm].append(pred)
            np.savez_compressed(evidence_dir / (person + '_' + arm + '.npz'), **evidence)
        print('person', person, 'complete', flush=True)
    frames = {arm: pd.concat(parts).set_index('qa_id').loc[qa.qa_id] for arm, parts in outputs.items()}
    base_correct = frames['control'].prediction.eq(frames['control'].answer)
    metrics = {}
    for arm, frame in frames.items():
        if len(frame) != 562 or not frame.index.is_unique:
            raise ValueError('Full562 scope differs')
        frame.reset_index().to_csv(args.output / (arm + '_oof.csv'), index=False)
        correct = frame.prediction.eq(frame.answer)
        metrics[arm] = {'correct': int(correct.sum()), 'fixes': int((~base_correct & correct).sum()), 'harms': int((base_correct & ~correct).sum())}
    if int(base_correct.sum()) != 530:
        raise ValueError('Actual strong baseline differs')
    reference = main / 'runs/20260913_final_strategy_external_review_v1/d9d10_complete_train_v2'
    truth = allqa.set_index('qa_id').answer
    sat_results = {}
    for parent in ['current_D10', 'P0_D10']:
        source = pd.read_csv(read(reference / parent / 'computed_parent.csv'), dtype=str).set_index('qa_id')
        oldsat = pd.read_csv(read(reference / parent / 'sat/submission.csv'), dtype=str).set_index('qa_id').prediction.reindex(truth.index)
        np.testing.assert_array_equal(source.loc[qa.qa_id, 'prediction'], frames['control'].prediction)
        for arm, frame in frames.items():
            folder = args.output / (parent + '_' + arm)
            folder.mkdir()
            candidate = source.copy()
            candidate.loc[frame.index, 'prediction'] = frame.prediction
            candidate.reset_index().to_csv(folder / 'computed_parent.csv', index=False)
            command = ['/root/cuhkx-runtime/sat_replay_20260912_v1/bin/python', str(main / 'scripts/apply_sat_train_oof.py'), '--qa', str(qa_path), '--parent', str(folder / 'computed_parent.csv'), '--frequency-run', str(reference / 'frequency'), '--output-dir', str(folder / 'sat')]
            write_json(folder / 'command.json', command)
            with (folder / 'sat.log').open('x') as handle:
                subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
            got = pd.read_csv(folder / 'sat/submission.csv', dtype=str).set_index('qa_id').prediction
            if len(got) != 4087 or set(got.index) != set(truth.index) or not got.index.is_unique:
                raise ValueError('Full Train output differs')
            got = got.reindex(truth.index)
            if arm == 'control':
                pd.testing.assert_series_equal(got, oldsat)
            a, b = oldsat.eq(truth), got.eq(truth)
            sat_results[parent + '_' + arm] = {'correct': int(b.sum()), 'parent_correct': int(a.sum()),
                'fixes': int((~a & b).sum()), 'harms': int((a & ~b).sum()), 'sha256': sha256_file(folder / 'sat/submission.csv')}
            print(parent, arm, int(b.sum()), flush=True)
    for path, entry in cache.used.items():
        read(path, entry['sha256'])
    for path, digest in protected.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError('Read-only experiment input changed')
    write_json(args.output / 'report.json', {'status': 'complete_full18_R34_expert_slot', 'QA': metrics, 'fullTrain': sat_results,
        'input_sha256': protected, 'new_models': checks, 'option_permutations_checked': permutations,
        'Test_read': False, 'new_fits': 0, 'scope': plan, 'skeleton_weights': 'Original per-outer choices frozen; not reselected for new R34'})
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
