"""Streaming all-recording evaluation of the frozen S0/S1 contract."""

from __future__ import annotations

import itertools
import json
import time

import numpy as np
import pandas as pd
import torch

from cuhkx.pairwise_temporal import PAIR_INDICES, before_counts, decode_pairwise_order
from cuhkx.repeated_temporal import share_repeated_pair_probabilities
from cuhkx.sequence_skeleton_nested import prepare_sequences, temporal_before_probability
from cuhkx.spatial_training import check_space, encode, inputs, load_head, runtime, write_json
from cuhkx.validation import sha256_file
from cuhkx.videomae_windows import read_windows


def order_prior(qa, sequence, training_people):
    selected = sequence.loc[sequence.subject.isin(training_people)]
    truth = qa.loc[qa.qa_id.isin(selected.qa_id)].set_index('qa_id').answer
    orders = [[r.classes['ABCD'.index(letter)] for letter in truth[r.qa_id]]
              for r in selected.itertuples()]
    return before_counts(orders)


def decode(frame, curves, axes, supported, counts):
    raw = []
    for row, prob, axis in zip(frame.itertuples(), curves, axes, strict=True):
        times = np.unique(axis)
        values = np.stack([prob[axis == v].mean(0) for v in times])
        raw.append([temporal_before_probability(times, values[:, row.classes[a]],
                                               values[:, row.classes[b]], mode='mass_before')
                    if supported[[row.classes[a], row.classes[b]]].all() else .5
                    for a, b in PAIR_INDICES])
    raw = np.asarray(raw)
    pooled, _ = share_repeated_pair_probabilities(
        raw[None], np.asarray(frame.classes.tolist()), frame.repeat_group.tolist(),
        family_indices=[0], sibling_weight=.75)
    result = []
    for row, probability in zip(frame.itertuples(), pooled, strict=True):
        prediction, margin = decode_pairwise_order(probability, row.classes, counts,
                                                   prior_weight=1., prior_smoothing=1.5)
        plain, _ = decode_pairwise_order(probability, row.classes, counts,
                                         prior_weight=0., prior_smoothing=1.5)
        result.append({'qa_id': row.qa_id, 'subject': row.subject, 'unit': row.unit,
                       'prediction': prediction, 'no_prior': plain, 'margin': margin})
    return pd.DataFrame(result), raw, pooled


def predict(config, output):
    records, examples = inputs(config)
    subjects = sorted(examples.subject.unique())
    models = output / 'models'
    if len(list(models.glob('*.json'))) != 38:
        raise ValueError('All 38 completed, reloaded heads are required')
    backbone, processor = runtime(config)
    cache = np.load(output / 'trusted_tokens.npy', mmap_mode='r')
    cache_rows = {(r.hau_unit, int(r.window_index)): i for i, r in examples.iterrows()}
    destination = output / 'dense_oof'
    destination.mkdir(exist_ok=True)
    started = time.monotonic()
    count = 0
    cache_checked = False
    final = {arm: load_head(models / f'final_{arm}.pt') for arm in ['mean', 'spatial']}
    for subject in subjects:
        heads = {arm: load_head(models / f'{subject}_{arm}.pt') for arm in final}
        for record in records:
            if record['subject'] != subject:
                continue
            target = destination / (record['unit'].replace('/', '__') + '.npz')
            if target.exists():
                count += 1
                continue
            check_space(config)
            windows = read_windows(record, config['runtime'])
            values = {name: [] for name in ['mean', 'spatial', 'spatial_uniform',
                                           'final_mean', 'final_spatial']}
            for window_index, frames in enumerate(windows):
                key = (record['unit'], window_index)
                if key in cache_rows:
                    tokens = torch.from_numpy(np.array(cache[cache_rows[key]])).cuda()
                    if not cache_checked:
                        replay = encode(backbone, processor, frames)
                        torch.testing.assert_close(tokens, replay, rtol=0, atol=0)
                        cache_checked = True
                else:
                    tokens = encode(backbone, processor, frames)
                x = tokens.float()[None]
                with torch.inference_mode():
                    for arm, head in heads.items():
                        logits = head(x, arm=arm)[0]
                        values[arm].append(logits.softmax(-1).cpu().numpy())
                        values['final_' + arm].append(final[arm](x, arm=arm)[0].softmax(-1).cpu().numpy())
                    values['spatial_uniform'].append(heads['spatial'](
                        x, uniform=True)[0].softmax(-1).cpu().numpy())
                    if count == 0 and window_index == 0:
                        for arm, head in heads.items():
                            torch.testing.assert_close(head(x, arm=arm),
                                head(x.flip(-2), arm=arm), rtol=2e-5, atol=2e-5)
            axes = np.asarray(record['frame_indices']).reshape(16, 8, 2).mean(-1)
            axes = axes.reshape(128) / max(record['metadata']['frame_count'] - 1, 1)
            temporary = target.with_suffix('.tmp')
            with temporary.open('wb') as handle:
                np.savez_compressed(handle, axes=axes,
                                    **{key: np.concatenate(value) for key, value in values.items()})
            temporary.replace(target)
            count += 1
            if count % 20 == 0:
                print(f'dense OOF {count}/809 recordings; {time.monotonic() - started:.1f}s', flush=True)
    write_json(output / 'dense_report.json', {
        'complete': True, 'recordings': count, 'uniform_windows_per_recording': 16,
        'bins_per_window': 8, 'models_reloaded': 38,
        'spatial_permutation_probe': True, 'elapsed_seconds': time.monotonic() - started,
        'canonical_cache_raw_replay_exact': cache_checked,
        'final_outputs_scope': 'in-sample deployment replay only; excluded from OOF scores',
        'peak_cuda_bytes': torch.cuda.max_memory_allocated()})


def evaluate(config, output):
    records, examples = inputs(config)
    for outer in sorted(examples.subject.unique()) + ['final']:
        receipts = [json.loads((output / 'models' / f'{outer}_{arm}.json').read_text())
                    for arm in ['mean', 'spatial']]
        for key in ['initial_sha256', 'interval_order_sha256', 'bin_order_sha256',
                    'steps', 'training_rows', 'training_subjects']:
            if receipts[0][key] != receipts[1][key]:
                raise ValueError('Paired fitting contract differs: ' + key)
    qa = pd.read_csv(config['training_qa'], keep_default_na=False)
    sequence = prepare_sequences(qa, training=True)
    pd.testing.assert_frame_equal(sequence, prepare_sequences(
        qa.assign(answer='POISON', prediction='POISON'), training=True))
    truth = qa.set_index('qa_id').answer
    index = {record['unit']: record for record in records}
    dense = {}
    for unit in index:
        path = output / 'dense_oof' / (unit.replace('/', '__') + '.npz')
        with np.load(path) as archive:
            dense[unit] = {key: archive[key] for key in archive.files}
    predictions, locals_, scopes = [], [], []
    permutation_checks = 0
    for subject in sorted(examples.subject.unique()):
        target = sequence.loc[sequence.subject.eq(subject)].reset_index(drop=True)
        people = sorted(set(examples.subject) - {subject})
        counts = order_prior(qa, sequence, people)
        altered = qa.copy()
        altered.loc[altered.qa_id.isin(target.qa_id), 'answer'] = 'POISON'
        if counts != order_prior(altered, sequence, people):
            raise ValueError('Outer answers influenced order prior')
        axes = np.stack([dense[u]['axes'] for u in target.unit])
        for arm in ['mean', 'spatial', 'spatial_uniform']:
            model_arm = arm.split('_')[0]
            path = output / 'models' / f'{subject}_{model_arm}.pt'
            payload = torch.load(path, map_location='cpu', weights_only=True)
            if payload['training_subjects'] != people:
                raise ValueError('Incorrect outer head scope')
            supported = np.zeros(40, dtype=bool)
            supported[payload['supported_classes']] = True
            curves = np.stack([dense[u][arm] for u in target.unit])
            pred, raw, pooled = decode(target, curves, axes, supported, counts)
            # Full option permutations on each outer subset also exercise sibling alignment.
            for permutation in itertools.permutations(range(4)):
                changed = target.copy()
                changed['classes'] = [tuple(c[i] for i in permutation) for c in target.classes]
                permuted, _, _ = decode(changed, curves, axes, supported, counts)
                restored = permuted.prediction.map(lambda text, permutation=permutation: ''.join(
                    'ABCD'[permutation['ABCD'.index(letter)]] for letter in text))
                if restored.tolist() != pred.prediction.tolist():
                    raise ValueError('Option order affected decoded action sequence')
                permutation_checks += len(target)
            pred['answer'] = truth.reindex(pred.qa_id).to_numpy()
            pred['arm'] = arm
            predictions.append(pred)
            np.savez_compressed(output / f'{subject}_{arm}_sequence.npz',
                                qa_ids=target.qa_id.to_numpy(str), raw=raw, pooled=pooled,
                                curves=curves, axes=axes, classes=np.asarray(target.classes.tolist()))
            scopes.append({'subject': subject, 'arm': arm, 'training_subjects': people,
                           'model_sha256': sha256_file(path), 'prior_poison_passed': True})
            for row_index, row in examples.loc[examples.subject.eq(subject)].iterrows():
                probability = dense[row.hau_unit][arm].reshape(16, 8, 40)[int(row.window_index)].mean(0)
                label = int(row['class'])
                final_probability = dense[row.hau_unit]['final_' + model_arm].reshape(16, 8, 40)[int(row.window_index)].mean(0)
                locals_.append({'row': row_index, 'subject': subject, 'arm': arm,
                                'label': label, 'prediction': int(probability.argmax()),
                                'label_probability': float(probability[label]),
                                'final_fit_prediction': int(final_probability.argmax()),
                                'final_fit_label_probability': float(final_probability[label])})
    allpred = pd.concat(predictions, ignore_index=True)
    local = pd.DataFrame(locals_)
    allpred.to_csv(output / 'oof.csv', index=False)
    local.to_csv(output / 'local_primary_oof.csv', index=False)
    class_metrics = []
    for (arm, label), group in local.groupby(['arm', 'label']):
        class_metrics.append({'arm': arm, 'class': label, 'rows': len(group),
                              'oof_correct': int(group.prediction.eq(group.label).sum()),
                              'final_fit_correct': int(group.final_fit_prediction.eq(group.label).sum()),
                              'oof_nll': float(-np.log(group.label_probability.clip(lower=1e-12)).mean()),
                              'final_fit_nll': float(-np.log(group.final_fit_label_probability.clip(lower=1e-12)).mean())})
    pd.DataFrame(class_metrics).to_csv(output / 'primary_by_class.csv', index=False)
    write_json(output / 'evaluation_scopes.json', scopes)
    paired = {arm: group.set_index('qa_id').sort_index() for arm, group in allpred.groupby('arm')}
    a, b = paired['mean'], paired['spatial']
    ac, bc = a.prediction.eq(a.answer), b.prediction.eq(b.answer)
    report = {'status': 'complete_18_historical_LOSO_plus_final', 'fits': 38,
              'recordings': len(dense), 'sequence_qa': len(a),
              'sequence_correct': {arm: int(g.prediction.eq(g.answer).sum()) for arm, g in paired.items()},
              'no_prior_correct': {arm: int(g.no_prior.eq(g.answer).sum()) for arm, g in paired.items()},
              'spatial_fixes_mean': int((bc & ~ac).sum()),
              'spatial_harms_mean': int((ac & ~bc).sum()),
              'changed': int(a.prediction.ne(b.prediction).sum()),
              'local_primary': {arm: {'rows': len(g), 'correct': int(g.prediction.eq(g.label).sum()),
                               'nll': float(-np.log(g.label_probability.clip(lower=1e-12)).mean()),
                               'final_fit_correct': int(g.final_fit_prediction.eq(g.label).sum()),
                               'final_fit_nll': float(-np.log(g.final_fit_label_probability.clip(lower=1e-12)).mean())}
                                for arm, g in local.groupby('arm')},
              'option_permutation_checks': permutation_checks,
              'Test_QA_read': False, 'fullchain_candidate': False,
              'limitations': ['Historical subjects repeatedly observed; not a new blind evaluation',
                              'Primary class proxy labels are not exhaustive action localization labels',
                              'Uniform attention is a contribution diagnostic, not causal localization proof']}
    write_json(output / 'report.json', report)
    print(json.dumps(report, indent=2), flush=True)
