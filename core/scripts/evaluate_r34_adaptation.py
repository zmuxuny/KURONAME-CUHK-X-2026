#!/usr/bin/env python3
"""Raw-input HARn and full uniform HAU inference for each excluded R34 person."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import pandas as pd
import torch

from cuhkx.harn import action_class
from cuhkx.harn_tail import configure_torch
from cuhkx.metadata import subject_id
from cuhkx.r34_adaptation import batch_views, load_records, restore
from cuhkx.r34_video import metadata_arrays, sample_preprocessed
from cuhkx.sequence_r34_transfer import select_views
from cuhkx.sequence_skeleton_nested import prepare_sequences
from cuhkx.spatial_evaluation import decode, order_prior
from cuhkx.spatial_training import write_json
from cuhkx.validation import sha256_file


@torch.inference_mode()
def probabilities(model, x, supported):
    values = []
    for start in range(0, len(x), 2):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x[start:start + 2].cuda()).float()
        result = torch.zeros_like(logits)
        result[:, supported] = logits[:, supported].softmax(1)
        values.append(result.cpu().numpy())
    return np.concatenate(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--outer', help='Evaluate a completed pair; final reporting still requires all 18 people')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.output / 'source_snapshot'
    sources = [Path(__file__), ROOT / 'src/cuhkx/r34_adaptation.py',
               ROOT / 'src/cuhkx/r34_video.py', ROOT / 'src/cuhkx/sequence_r34_transfer.py',
               ROOT / 'src/cuhkx/spatial_evaluation.py']
    if source.exists():
        for item in json.loads((source / 'manifest.json').read_text()):
            if sha256_file(Path(item['path'])) != item['sha256']:
                raise ValueError('Evaluation implementation changed; version the output explicitly')
    else:
        for path in sources:
            target = source / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
        write_json(source / 'manifest.json', [{'path': str(p), 'sha256': sha256_file(p)} for p in sources])
        write_json(source / 'config.json', config)
    configure_torch()
    grouped = load_records(config, args.run)
    path = Path(config['source_manifest'])
    if sha256_file(path) != config['source_sha256']:
        raise ValueError('Original raw input manifest changed')
    raw = json.loads(path.read_text())
    metadata = metadata_arrays(raw)
    lookup = {(str(u), str(m), int(w)): i for i, (u, m, w) in enumerate(zip(
        metadata['units'], metadata['modalities'], metadata['window_indices'], strict=True))}
    recordings = {}
    for record in raw['records']:
        if record['source'] == 'HAU' and record['unit'].startswith('Training/'):
            record = dict(record)
            record['path'] = str(Path(config['data_root']) / record['unit'] / record['modality'] / (record['modality'] + '.mp4'))
            recordings.setdefault(record['unit'], {})[record['modality']] = record
    if len(recordings) != 809:
        raise ValueError('HAU coverage differs from 809 Train recordings')
    qa = pd.read_csv(config['training_qa'], keep_default_na=False)
    seq = prepare_sequences(qa, training=True)
    pd.testing.assert_frame_equal(seq, prepare_sequences(qa.assign(answer='POISON'), training=True))
    people = [args.outer] if args.outer else config['subjects']
    for outer in people:
        if outer not in config['subjects']:
            raise ValueError('Unknown outer person')
        destination = args.output / outer
        if (destination / 'report.json').exists():
            continue
        destination.mkdir(exist_ok=False)
        expected = sorted(set(config['subjects']) - {outer})
        receipts = [json.loads((args.run / 'models' / f'{outer}_{arm}.json').read_text()) for arm in ['C', 'R']]
        for key in ['initial_state_sha256', 'sample_order_sha256', 'training_units', 'training_subjects', 'steps']:
            if receipts[0][key] != receipts[1][key]:
                raise ValueError('Shallow/deep matched fitting contract differs: ' + key)
        all_harn, all_seq, arms = [], [], {}
        target = seq.loc[seq.subject.eq(outer)].reset_index(drop=True)
        counts = order_prior(qa, seq, expected)
        poisoned = qa.copy()
        poisoned.loc[poisoned.qa_id.isin(target.qa_id), 'answer'] = 'POISON'
        if counts != order_prior(poisoned, seq, expected):
            raise ValueError('Outer prior contamination')
        for arm in ['C', 'R']:
            path = args.run / 'models' / f'{outer}_{arm}.pt'
            receipt = json.loads(path.with_suffix('.json').read_text())
            if sha256_file(path) != receipt['sha256']:
                raise ValueError('Adaptation checkpoint changed')
            model, payload = restore(config, path)
            if payload['training_subjects'] != expected or any(subject_id(u) == outer for u in payload['training_units']):
                raise ValueError('Wrong R34 outer training scope')
            supported = payload['supported_classes']
            units = [u for u in grouped if subject_id(u) == outer]
            pieces = []
            for start in range(0, len(units), 16):
                chosen = units[start:start + 16]
                x, owners = batch_views(chosen, grouped)
                values = probabilities(model, x, supported)
                pieces.extend([values[owners == i].mean(0) for i in range(len(chosen))])
            matrix = np.stack(pieces)
            labels = np.asarray([action_class(u) for u in units])
            harn = pd.DataFrame({'unit': units, 'subject': outer, 'arm': arm, 'label': labels,
                                 'prediction': matrix.argmax(1), 'label_probability': matrix[np.arange(len(units)), labels]})
            all_harn.append(harn)
            np.savez_compressed(destination / f'{arm}_harn.npz', units=np.asarray(units), probabilities=matrix)
            curves, axes, reasons, view_counts = {}, {}, {}, {}
            for unit, views in recordings.items():
                if subject_id(unit) != outer:
                    continue
                # Short clips may have fewer distinct windows. Decode averages
                # duplicated physical times instead of discarding their evidence.
                _, mask, times, reason = select_views(metadata, lookup, unit, allow_duplicate_windows=True)
                if mask is None:
                    curves[unit], axes[unit], reasons[unit] = np.zeros((8, 44)), np.linspace(0, 1, 8), reason
                    continue
                selected = [m for m, use in zip(['Depth_Color', 'IR'], mask[0], strict=True) if use]
                values = []
                for modality in selected:
                    record = views[modality]
                    if sha256_file(Path(record['path'])) != record['raw']['sha256']:
                        raise ValueError('HAU raw recording changed')
                    values.append(probabilities(model, sample_preprocessed(record), supported))
                curves[unit], axes[unit] = np.stack(values).mean(0), times[:, 2]
                view_counts[reason] = view_counts.get(reason, 0) + 1
            order_curves = np.stack([curves[u] for u in target.unit])
            order_axes = np.stack([axes[u] for u in target.unit])
            mask = np.zeros(44, dtype=bool)
            mask[supported] = True
            prediction, raw_pairs, pooled = decode(target, order_curves, order_axes, mask, counts)
            prediction['answer'] = prediction.qa_id.map(qa.set_index('qa_id').answer)
            prediction['arm'] = arm
            all_seq.append(prediction)
            np.savez_compressed(destination / f'{arm}_hau.npz', units=np.asarray(list(curves)),
                                probabilities=np.stack(list(curves.values())), axes=np.stack(list(axes.values())))
            np.savez_compressed(destination / f'{arm}_sequence.npz', qa_ids=target.qa_id.to_numpy(str),
                                raw=raw_pairs, pooled=pooled, classes=np.asarray(target.classes.tolist()))
            arms[arm] = {'HARn_units': len(harn), 'HARn_correct': int(harn.label.eq(harn.prediction).sum()),
                         'HARn_nll': float(-np.log(harn.label_probability.clip(lower=1e-12)).mean()),
                         'HAU_recordings': len(curves), 'sequence_qa': len(prediction),
                         'sequence_correct': int(prediction.prediction.eq(prediction.answer).sum()),
                         'view_counts': view_counts, 'fallback_reasons': reasons,
                         'model_sha256': receipt['sha256'], 'training_people': expected}
            del model, payload
            torch.cuda.empty_cache()
        pd.concat(all_harn).to_csv(destination / 'harn_oof.csv', index=False)
        pd.concat(all_seq).to_csv(destination / 'sequence_oof.csv', index=False)
        write_json(destination / 'report.json', {'outer': outer, 'arms': arms, 'paired_contract_checked': True})
        print(outer, json.dumps(arms), flush=True)
    complete = [p for p in config['subjects'] if (args.output / p / 'report.json').exists()]
    write_json(args.output / 'progress.json', {'completed_people': complete, 'required_people': config['subjects'],
               'complete': len(complete) == 18, 'scope': 'Raw uniform HAU inference and lossless raw-crop HARn OOF; no Test'})
    if len(complete) == 18:
        for name in ['harn', 'sequence']:
            pd.concat([pd.read_csv(args.output / p / f'{name}_oof.csv') for p in complete]).to_csv(
                args.output / f'{name}_oof.csv', index=False)


if __name__ == '__main__':
    main()
