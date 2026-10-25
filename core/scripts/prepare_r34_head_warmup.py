#!/usr/bin/env python3
"""Prepare generic pooled features and excluded-person classifier warm-up."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch

from cuhkx.harn import action_class
from cuhkx.harn_tail import configure_torch
from cuhkx.metadata import subject_id
from cuhkx.r34_adaptation import batch_views, load_records, network
from cuhkx.sequence_skeleton_nested import array_digest
from cuhkx.spatial_training import write_json
from cuhkx.validation import sha256_file


def extract(config, raw_run, output, model=None, producer='generic'):
    configure_torch()
    grouped = load_records(config, raw_run)
    model = network(config, 'C') if model is None else model
    model = model.eval().requires_grad_(False).cuda()
    model.fc = torch.nn.Identity()
    values, records = [], []
    units = list(grouped)
    with torch.inference_mode():
        for start in range(0, len(units), 16):
            chosen = units[start:start + 16]
            x, _ = batch_views(chosen, grouped)
            for position in range(0, len(x), 2):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    result = model(x[position:position + 2].cuda()).float().cpu()
                    if start == 0 and position == 0:
                        replay = model(x[position:position + 2].cuda()).float().cpu()
                        torch.testing.assert_close(result, replay, rtol=0, atol=0)
                if result.shape[1] != 512 or not torch.isfinite(result).all():
                    raise ValueError('Unexpected generic pooled feature shape/value')
                values.append(result.numpy())
            records.extend({'unit': unit, 'modality': row['modality'], 'crop_sha256': row['sha256']}
                           for unit in chosen for row in grouped[unit])
            if start % 320 == 0:
                print(f'{producer} frozen pool {min(start + 16, len(units))}/{len(units)}', flush=True)
    features = np.concatenate(values)
    if len(features) != len(records):
        raise ValueError('Pooled rows and canonical input identities disagree')
    np.save(output / 'features.npy', features)
    write_json(output / 'rows.json', records)
    write_json(output / 'extraction_report.json', {'views': len(records), 'units': len(units),
               'feature_sha256': sha256_file(output / 'features.npy'), 'rows_sha256': sha256_file(output / 'rows.json'),
               'source_cache_report_sha256': sha256_file(raw_run / 'cache_report.json'),
               'precision': f'{producer} BF16 frozen backbone, exact float32 storage of pooled512 output',
               'fits': 0, 'raw_crop_repeat_exact': True, 'source': 'official Train HARn only'})


def warmup(config, output, producer='generic'):
    torch.set_num_threads(2)
    report = json.loads((output / 'extraction_report.json').read_text())
    for filename, field in [('features.npy', 'feature_sha256'), ('rows.json', 'rows_sha256')]:
        if sha256_file(output / filename) != report[field]:
            raise ValueError('Generic warm-up source changed')
    features = torch.from_numpy(np.load(output / 'features.npy'))
    records = json.loads((output / 'rows.json').read_text())
    by_unit = {}
    for i, row in enumerate(records):
        by_unit.setdefault(row['unit'], []).append(i)
    destination = output / 'heads'
    destination.mkdir(exist_ok=False)
    cfg = config['training']
    summaries = []
    for outer in config['subjects'] + ['final']:
        units = [u for u in by_unit if subject_id(u) != outer]
        labels = np.asarray([action_class(u) for u in units])
        counts = np.bincount(labels, minlength=44)
        supported = np.flatnonzero(counts)
        class_weights = np.zeros(44, dtype=np.float32)
        class_weights[supported] = len(units) / (len(supported) * counts[supported])
        class_weights = torch.from_numpy(class_weights)
        torch.manual_seed(cfg['seed'])
        head = torch.nn.Linear(512, 44)
        initial = array_digest(*[p.detach().numpy() for p in head.parameters()])
        optimizer = torch.optim.AdamW(head.parameters(), lr=cfg['head_lr'], weight_decay=cfg['weight_decay'])
        epochs = 20
        steps = epochs * ((len(units) + cfg['batch_units'] - 1) // cfg['batch_units'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        rng = np.random.default_rng(cfg['seed'] + 1410)
        orders = [rng.permutation(len(units)) for _ in range(epochs)]
        history = []
        for epoch, order in enumerate(orders):
            total = 0.
            for start in range(0, len(order), cfg['batch_units']):
                selected = order[start:start + cfg['batch_units']]
                rows, targets, weights = [], [], []
                for index in selected:
                    indices = by_unit[units[index]]
                    rows.extend(indices)
                    targets.extend([labels[index]] * len(indices))
                    weights.extend([1. / len(indices)] * len(indices))
                optimizer.zero_grad(set_to_none=True)
                losses = torch.nn.functional.cross_entropy(head(features[rows]), torch.tensor(targets),
                                                           weight=class_weights, reduction='none')
                loss = (losses * torch.tensor(weights)).sum() / len(selected)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                total += float(loss.detach()) * len(selected)
            history.append({'epoch': epoch + 1, 'loss': total / len(units)})
        payload = {'state_dict': head.state_dict(), 'outer': outer, 'training_units': units,
                   'training_subjects': sorted({subject_id(u) for u in units}), 'epochs': epochs,
                   'steps': steps, 'history': history, 'initial_head_sha256': initial,
                   'sample_order_sha256': array_digest(*orders), 'feature_sha256': report['feature_sha256'],
                   'scope': f'Classifier-only CPUfloat32 warm-up on frozen {producer} pooled features; no outer labels or backbone update; no validation score'}
        path = destination / f'{outer}.pt'
        torch.save(payload, path)
        reloaded = torch.nn.Linear(512, 44)
        reloaded.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['state_dict'], strict=True)
        with torch.inference_mode():
            torch.testing.assert_close(head(features[:2]), reloaded(features[:2]), rtol=0, atol=0)
        receipt = {k: v for k, v in payload.items() if k != 'state_dict'}
        receipt.update(sha256=sha256_file(path), reload_exact=True)
        write_json(path.with_suffix('.json'), receipt)
        summaries.append({'outer': outer, 'initial_loss': history[0]['loss'], 'final_loss': history[-1]['loss'], 'sha256': receipt['sha256']})
        print(outer, summaries[-1], flush=True)
    write_json(output / 'warmup_report.json', {'status': 'complete19classifierwarmups', 'heads': summaries,
               'proposed_mechanism': 'Isolate random-head cold start before assessing backbone update stability',
               'validation_score': None, 'full_backbone_fits': 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--raw-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['extract', 'warmup'], required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = args.output / ('source_' + args.stage)
    snapshot.mkdir(exist_ok=False)
    sources = [Path(__file__), ROOT / 'src/cuhkx/r34_adaptation.py', ROOT / 'src/cuhkx/r34_video.py']
    for path in sources:
        target = snapshot / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    write_json(snapshot / 'config.json', config)
    write_json(snapshot / 'manifest.json', [{'path': str(p), 'sha256': sha256_file(p)} for p in sources])
    if args.stage == 'extract':
        extract(config, args.raw_run, args.output)
    else:
        warmup(config, args.output)


if __name__ == '__main__':
    main()
