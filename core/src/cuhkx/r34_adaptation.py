"""Matched shallow/deep R34 adaptation with fixed BN statistics and raw crops."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from cuhkx.generic_video_windows import KINETICS_MEAN, KINETICS_STD
from cuhkx.harn import action_class
from cuhkx.harn_tail import configure_torch
from cuhkx.metadata import subject_id
from cuhkx.r34_video import pretrained_model
from cuhkx.sequence_skeleton_nested import array_digest
from cuhkx.spatial_training import check_space, write_json
from cuhkx.validation import sha256_file


def load_records(config, output):
    records = json.loads(Path(config['records']).read_text())['records']
    report = json.loads((output / 'cache_report.json').read_text())
    if report['records_sha256'] != sha256_file(Path(config['records'])):
        raise ValueError('Cache records differ from the declared fitting data')
    files = {(r['unit'], r['modality']): r for r in report['files']}
    grouped = {}
    for record in records:
        cached = files[(record['unit'], record['modality'])]
        if sha256_file(Path(cached['path'])) != cached['sha256']:
            raise ValueError('Canonical crop cache changed')
        grouped.setdefault(record['unit'], []).append(cached)
    return grouped


def network(config, arm):
    model = pretrained_model(config['weights'])
    torch.manual_seed(config['training']['seed'])
    model.fc = torch.nn.Linear(512, 44)
    for name, parameter in model.named_parameters():
        learn = name.startswith(('layer4.2.', 'fc.')) if arm == 'C' else name.startswith(('layer3.', 'layer4.', 'fc.'))
        parameter.requires_grad_(learn)
    # eval mode fixes every BN buffer while selected affine weights can still learn.
    return model.eval()


def batch_views(units, grouped):
    frames, owners = [], []
    for owner, unit in enumerate(units):
        for item in grouped[unit]:
            with np.load(item['path']) as archive:
                value = archive['frames']
            if value.shape[-1] == 1:
                value = np.repeat(value, 3, axis=-1)
            if value.dtype != np.uint8 or value.shape != (32, 112, 112, 3):
                raise ValueError('Unexpected cached raw crop')
            frames.append(value)
            owners.append(owner)
    x = torch.from_numpy(np.stack(frames)).permute(0, 4, 1, 2, 3).float() / 255.
    mean = torch.tensor(KINETICS_MEAN)[None, :, None, None, None]
    std = torch.tensor(KINETICS_STD)[None, :, None, None, None]
    return (x - mean) / std, np.asarray(owners)


def buffers_digest(model):
    return array_digest(*[value.detach().cpu().numpy() for _, value in model.named_buffers()])


def optimizer_for(model, config):
    cfg = config['training']
    return torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('fc.')],
         'lr': cfg['backbone_lr']},
        {'params': list(model.fc.parameters()), 'lr': cfg['head_lr']}], weight_decay=cfg['weight_decay'])


def update(model, optimizer, x, owners, labels, class_weights, cfg):
    """Accumulate identical unit-weighted loss despite variable available views."""
    optimizer.zero_grad(set_to_none=True)
    counts = np.bincount(owners, minlength=len(labels))
    total_loss = 0.
    for start in range(0, len(x), cfg['microbatch_views']):
        stop = start + cfg['microbatch_views']
        owner = owners[start:stop]
        target = torch.as_tensor(labels[owner], device='cuda')
        view_weight = torch.as_tensor(1. / counts[owner], device='cuda', dtype=torch.float32)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x[start:stop].cuda())
            losses = torch.nn.functional.cross_entropy(logits, target, weight=class_weights, reduction='none')
            loss = (losses * view_weight).sum() / len(labels)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite R34 loss')
        loss.backward()
        total_loss += float(loss.detach())
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
    optimizer.step()
    return total_loss, float(norm)


def restore(config, path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    model = network(config, payload['arm'])
    expected = {name for name, p in model.named_parameters() if p.requires_grad}
    if set(payload['learned']) != expected:
        raise ValueError('Saved R34 learning scope differs')
    state = model.state_dict()
    state.update(payload['learned'])
    model.load_state_dict(state, strict=True)
    return model.eval().cuda(), payload


def probe(config, output):
    configure_torch()
    if not torch.cuda.is_bf16_supported():
        raise ValueError('Declared BF16 training unavailable')
    grouped = load_records(config, output)
    chosen = list(grouped)[:2]
    x, owners = batch_views(chosen, grouped)
    labels = np.asarray([action_class(u) for u in chosen])
    results, initial = [], None
    cfg = config['training']
    for arm in ['C', 'R']:
        model = network(config, arm).cuda()
        frozen = buffers_digest(model)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x[:2].cuda()).float().cpu()
        if initial is None:
            initial = logits
        else:
            torch.testing.assert_close(logits, initial, rtol=0, atol=0)
        optimizer = optimizer_for(model, config)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        loss, grad = update(model, optimizer, x, owners, labels, torch.ones(44, device='cuda'), cfg)
        torch.cuda.synchronize()
        if buffers_digest(model) != frozen:
            raise ValueError('A BN running statistic changed')
        payload = {'arm': arm, 'learned': {n: p.detach().half().cpu()
                                          for n, p in model.named_parameters() if p.requires_grad}}
        path = output / f'probe_{arm}.pt'
        torch.save(payload, path)
        state = model.state_dict()
        state.update(payload['learned'])
        model.load_state_dict(state, strict=True)
        reloaded, _ = restore(config, path)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            torch.testing.assert_close(model(x[:2].cuda()), reloaded(x[:2].cuda()), rtol=0, atol=0)
        results.append({'arm': arm, 'loss': loss, 'gradient_norm': grad,
                        'seconds_including_export_reload': time.monotonic() - started,
                        'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
                        'parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                        'optimizer_steps': 1, 'bn_buffers_unchanged': True, 'reload_exact': True,
                        'initial_logits_matched': True, 'model_sha256': sha256_file(path)})
        del reloaded, model, optimizer
        torch.cuda.empty_cache()
    write_json(output / 'probe_report.json', {'status': 'passed', 'units': chosen,
               'results': results, 'scope': 'Two real HARn training units; resource/optimizer proof only'})


def fit(config, output, max_models=None):
    configure_torch()
    if not (output / 'probe_report.json').exists():
        raise ValueError('Run actual two-arm optimizer/resource acceptance first')
    grouped = load_records(config, output)
    cfg = config['training']
    subjects = config['subjects']
    destination = output / 'models'
    destination.mkdir(exist_ok=True)
    completed_this_session = 0
    for outer in subjects + ['final']:
        units = [u for u in grouped if subject_id(u) != outer]
        people = sorted({subject_id(u) for u in units})
        labels = np.asarray([action_class(u) for u in units])
        counts = np.bincount(labels, minlength=44)
        supported = np.flatnonzero(counts)
        weights = np.zeros(44, dtype=np.float32)
        weights[supported] = len(units) / (len(supported) * counts[supported])
        class_weights = torch.from_numpy(weights).cuda()
        rng = np.random.default_rng(cfg['seed'])
        orders = [rng.permutation(len(units)) for _ in range(cfg['epochs'])]
        for arm in ['C', 'R']:
            path = destination / f'{outer}_{arm}.pt'
            receipt = path.with_suffix('.json')
            if receipt.exists():
                if sha256_file(path) != json.loads(receipt.read_text())['sha256']:
                    raise ValueError('Completed model changed')
                continue
            if path.exists():
                raise ValueError('Partial model retained without completed reload; inspect before restart')
            check_space(config)
            model = network(config, arm).cuda()
            frozen = buffers_digest(model)
            initial = array_digest(*[p.detach().cpu().numpy() for p in model.state_dict().values()])
            optimizer = optimizer_for(model, config)
            steps = cfg['epochs'] * ((len(units) + cfg['batch_units'] - 1) // cfg['batch_units'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
            history, started = [], time.monotonic()
            for epoch, order in enumerate(orders):
                total = 0.
                for start in range(0, len(order), cfg['batch_units']):
                    selection = order[start:start + cfg['batch_units']]
                    x, owners = batch_views([units[i] for i in selection], grouped)
                    loss, _ = update(model, optimizer, x, owners, labels[selection], class_weights, cfg)
                    scheduler.step()
                    total += loss * len(selection)
                history.append({'epoch': epoch + 1, 'loss': total / len(units),
                                'elapsed_seconds': time.monotonic() - started})
                print(f'R34 {outer} {arm} epoch {epoch + 1}/{cfg["epochs"]} '
                      f'loss={total / len(units):.5f}; {time.monotonic() - started:.1f}s', flush=True)
                write_json(output / 'fit_progress.json', {'outer': outer, 'arm': arm,
                                                          'completed_epochs': history})
            if buffers_digest(model) != frozen:
                raise ValueError('Fixed BN policy violated')
            payload = {'arm': arm, 'outer': outer, 'training_subjects': people,
                       'training_units': units, 'config': copy.deepcopy(cfg),
                       'source_weights_sha256': config['weights_sha256'],
                       'learned': {n: p.detach().half().cpu() for n, p in model.named_parameters() if p.requires_grad},
                       'supported_classes': supported.tolist(), 'steps': steps, 'history': history,
                       'initial_state_sha256': initial, 'sample_order_sha256': array_digest(*orders),
                       'bn_buffer_sha256': frozen}
            temp = path.with_suffix('.tmp')
            torch.save(payload, temp)
            temp.replace(path)
            state = model.state_dict()
            state.update(payload['learned'])
            model.load_state_dict(state, strict=True)
            reloaded, _ = restore(config, path)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                torch.testing.assert_close(model(x[:2].cuda()), reloaded(x[:2].cuda()), rtol=0, atol=0)
            write_json(receipt, {'sha256': sha256_file(path), 'retained_fp16_reload_exact': True,
                                **{k: v for k, v in payload.items() if k != 'learned'}})
            del model, optimizer, reloaded, payload
            torch.cuda.empty_cache()
            completed_this_session += 1
            if max_models is not None and completed_this_session >= max_models:
                return
