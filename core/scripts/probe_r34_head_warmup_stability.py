#!/usr/bin/env python3
"""Training-only cold/warm-head R34 stability comparison on identical batches."""
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
from cuhkx.r34_adaptation import (
    batch_views,
    buffers_digest,
    load_records,
    network,
    optimizer_for,
    restore,
    update,
)
from cuhkx.sequence_skeleton_nested import array_digest
from cuhkx.spatial_training import write_json
from cuhkx.validation import sha256_file


def activation(model, inputs, labels):
    captures, probabilities = [], []
    def hook(_module, _inputs, output):
        captures.append(output.detach().float().cpu().flatten(1))
    handle = model.avgpool.register_forward_hook(hook)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for start in range(0, len(inputs), 2):
            probabilities.append(model(inputs[start:start + 2].cuda()).float().softmax(-1).cpu())
    handle.remove()
    values, probability = torch.cat(captures), torch.cat(probabilities)
    return {'positive_fraction': float((values > 0).float().mean()),
            'zero_views': int(values.abs().amax(1).eq(0).sum()),
            'pooled_abs_mean': float(values.abs().mean()),
            'label_probability': probability[torch.arange(len(labels)), labels].tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--raw-run', type=Path, required=True)
    parser.add_argument('--warmup', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backbone-lr', type=float,
                        help='Explicit training-only step-size control; stored in the effective config')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    original_backbone_lr = config['training']['backbone_lr']
    if args.backbone_lr is not None:
        if args.backbone_lr <= 0:
            raise ValueError('backbone-lr must be positive')
        config['training']['backbone_lr'] = args.backbone_lr
    sources = [Path(__file__), ROOT / 'src/cuhkx/r34_adaptation.py', ROOT / 'src/cuhkx/r34_video.py']
    for path in sources:
        target = args.output / 'source_snapshot' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    write_json(args.output / 'config.json', config)
    write_json(args.output / 'command.json', {'argv': sys.argv, 'cwd': str(Path.cwd()),
               'original_backbone_lr': original_backbone_lr,
               'effective_backbone_lr': config['training']['backbone_lr']})
    write_json(args.output / 'sources.json', [{'path': str(p), 'sha256': sha256_file(p)} for p in sources])
    configure_torch()
    grouped = load_records(config, args.raw_run)
    outer = 'user1'
    units = [u for u in grouped if subject_id(u) != outer]
    labels = np.asarray([action_class(u) for u in units])
    counts = np.bincount(labels, minlength=44)
    supported = np.flatnonzero(counts)
    weights = np.zeros(44, dtype=np.float32)
    weights[supported] = len(units) / (len(supported) * counts[supported])
    class_weights = torch.from_numpy(weights).cuda()
    cfg = config['training']
    order = np.random.default_rng(cfg['seed']).permutation(len(units))
    chosen = [next(u for u in units if action_class(u) == label) for label in [0, 11, 22, 33]]
    monitor, owners = batch_views(chosen, grouped)
    monitor_labels = torch.tensor([action_class(chosen[i]) for i in owners])
    warm_path = args.warmup / 'heads/user1.pt'
    if sha256_file(warm_path) != json.loads(warm_path.with_suffix('.json').read_text())['sha256']:
        raise ValueError('Warm-up model changed')
    warm = torch.load(warm_path, map_location='cpu', weights_only=True)
    if warm['training_units'] != units or outer in warm['training_subjects']:
        raise ValueError('Warm-up scope differs from the controlled target batches')
    reports, initial_backbone = [], None
    for arm in ['cold', 'warm']:
        model = network(config, 'R')
        backbone = array_digest(*[v.numpy() for k, v in model.state_dict().items() if not k.startswith('fc.')])
        if initial_backbone is None:
            initial_backbone = backbone
        elif backbone != initial_backbone:
            raise ValueError('Cold/warm initial backbones differ')
        if arm == 'warm':
            model.fc.load_state_dict(warm['state_dict'], strict=True)
        model.cuda()
        frozen = buffers_digest(model)
        optimizer = optimizer_for(model, config)
        steps = cfg['epochs'] * ((len(units) + cfg['batch_units'] - 1) // cfg['batch_units'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        history = [{'step': 0, **activation(model, monitor, monitor_labels)}]
        losses = []
        for step in range(40):
            selected = order[step * cfg['batch_units']:(step + 1) * cfg['batch_units']]
            x, batch_owners = batch_views([units[i] for i in selected], grouped)
            loss, grad = update(model, optimizer, x, batch_owners, labels[selected], class_weights, cfg)
            scheduler.step()
            losses.append(loss)
            if (step + 1) % 10 == 0:
                history.append({'step': step + 1, 'last10_train_loss': float(np.mean(losses[-10:])),
                                'gradient_norm': grad, **activation(model, monitor, monitor_labels)})
                write_json(args.output / f'{arm}_progress.json', history)
                print(arm, history[-1], flush=True)
        if buffers_digest(model) != frozen:
            raise ValueError('BN buffer policy changed during stability probe')
        payload = {'arm': 'R', 'learned': {n: p.detach().half().cpu() for n, p in model.named_parameters() if p.requires_grad},
                   'initialization': arm, 'steps': 40, 'training_units': units,
                   'sample_order_sha256': array_digest(order), 'initial_backbone_sha256': backbone,
                   'warm_head_sha256': sha256_file(warm_path) if arm == 'warm' else None}
        path = args.output / f'{arm}.pt'
        torch.save(payload, path)
        state = model.state_dict()
        state.update(payload['learned'])
        model.load_state_dict(state, strict=True)
        reloaded, _ = restore(config, path)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            torch.testing.assert_close(model(monitor[:2].cuda()), reloaded(monitor[:2].cuda()), rtol=0, atol=0)
        reports.append({'arm': arm, 'history': history, 'model_sha256': sha256_file(path),
                        'reload_exact': True, 'bn_buffers_unchanged': True})
        del model, optimizer, reloaded, payload
        torch.cuda.empty_cache()
    write_json(args.output / 'report.json', {'status': 'completed_training_dynamics_comparison',
               'arms': reports, 'backbone_updates_each': 40,
               'heldout_or_Test_scores_used': False, 'monitor_training_units': chosen,
               'scope': 'Cold versus classifier-warmed R on the same first40trainingbatches and original4epochLRschedule; not18personvalidation or evidence of generalization'})


if __name__ == '__main__':
    main()
