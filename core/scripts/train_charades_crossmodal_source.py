#!/usr/bin/env python3
"""Matched source-only depth continuation with frozen paired RGB features."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch
import torch.nn.functional as F
from train_charades_r34_source import masked_bce, monitor_features, network

from cuhkx.charades_packed_source import PackedSourceWindows, verify_cache
from cuhkx.harn_tail import configure_torch
from cuhkx.r34_adaptation import buffers_digest
from cuhkx.sequence_skeleton_nested import array_digest
from cuhkx.spatial_training import write_json
from cuhkx.validation import sha256_file


def cosine_alignment(student, teacher):
    return 1 - F.cosine_similarity(student.float(), teacher.detach().float(), dim=1)


def load_source(folder, expected_sha):
    path = folder / 'model.pt'
    report = json.loads((folder / 'report.json').read_text())
    if sha256_file(path) != expected_sha or report['model_sha256'] != expected_sha:
        raise ValueError('Source model identity changed')
    source = torch.load(path, map_location='cpu', weights_only=True)
    if source['source_only'] != 'Charades official Train' or not report['reload_exact']:
        raise ValueError('Expected verified source-only model')
    model = network(source['config'])
    if set(source['learned']) != {n for n, p in model.named_parameters() if p.requires_grad}:
        raise ValueError('Source learned parameter scope differs')
    state = model.state_dict()
    state.update(source['learned'])
    model.load_state_dict(state, strict=True)
    if buffers_digest(model) != source['bn_buffer_sha256']:
        raise ValueError('Source BN differs from original frozen buffers')
    return model.cuda(), source


def space(path, reserve):
    disk = os.statvfs(path)
    free = disk.f_bavail * disk.f_frsize
    if free < reserve:
        raise RuntimeError(f'Storage reserve reached: {free}')


def dataset(cache, arm, rows):
    return PackedSourceWindows(cache, arm, [(a, b, None, None, None) for a, b in rows])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--packed-cache', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=['teacher', 'BCE_control', 'BCE_RGB_cosine'], required=True)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    configure_torch()
    source_root, out = args.source_root, args.output
    out.mkdir(parents=True, exist_ok=True)
    target = out / args.stage
    target.mkdir(exist_ok=False)
    write_json(target / 'plan.json', plan)
    write_json(target / 'environment.json', {'python': sys.version, 'torch': torch.__version__,
               'numpy': np.__version__, 'cuda': torch.version.cuda, 'argv': sys.argv,
               'interpreter': sys.executable})
    write_json(target / 'code.json', {'script_sha256': sha256_file(Path(__file__)),
               'source_trainer_sha256': sha256_file(ROOT / 'scripts/train_charades_r34_source.py'),
               'packed_reader_sha256': sha256_file(ROOT / 'src/cuhkx/charades_packed_source.py')})
    rows = json.loads((source_root / 'depth/source_rows.json').read_text())
    if rows != json.loads((source_root / 'rgb/source_rows.json').read_text()) or len(rows) != 14075:
        raise ValueError('Paired source row identities differ')
    with np.load(source_root / 'depth/supervision.npz') as z:
        supervision = {k: z[k] for k in z.files}
    with np.load(source_root / 'rgb/supervision.npz') as z:
        for key, value in supervision.items():
            np.testing.assert_array_equal(value, z[key])
    reserve = plan['storage']['remote_reserve_bytes']
    space(out, reserve)
    if args.stage == 'teacher':
        model, source = load_source(source_root / 'rgb', plan['sources']['rgb']['sha256'])
        model.requires_grad_(False)
        identities = source['videos']
        checked = {arm: verify_cache(args.packed_cache, args.inputs, arm, identities)
                   for arm in ['rgb', 'depth']}
        for arm in ['rgb', 'depth']:
            if sha256_file(args.inputs / (arm + '_report.json')) != source['source_' + arm + '_sha256']:
                raise ValueError('Original source input report changed')
        write_json(target / 'input_verification.json', {'cache_reports': checked,
                   'source_rows_sha256': sha256_file(source_root / 'depth/source_rows.json'),
                   'supervision_sha256': sha256_file(source_root / 'depth/supervision.npz'),
                   'plan_sha256': sha256_file(args.plan), 'rows': len(rows)})
        data = dataset(args.packed_cache, 'rgb', rows)
        loader = torch.utils.data.DataLoader(data, batch_size=16, num_workers=2, pin_memory=True)
        features = np.lib.format.open_memmap(target / 'features.npy', mode='w+', dtype=np.float32,
                                            shape=(len(rows), 512))
        captures = []
        handle = model.avgpool.register_forward_hook(
            lambda _m, _i, v: captures.append(v.detach().float().flatten(1).cpu().numpy()))
        cursor = 0
        started = time.monotonic()
        with torch.inference_mode():
            for x, y, known in loader:
                np.testing.assert_array_equal(y.numpy(), supervision['labels'][cursor:cursor + len(x)])
                np.testing.assert_array_equal(known.numpy(), supervision['masks'][cursor:cursor + len(x)])
                for start in range(0, len(x), 2):
                    captures.clear()
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        model(x[start:start + 2].cuda(non_blocking=True))
                    value = captures.pop()
                    if not np.isfinite(value).all() or np.any(np.linalg.norm(value, axis=1) == 0):
                        raise ValueError('Invalid paired RGB feature')
                    features[cursor + start:cursor + start + len(value)] = value
                cursor += len(x)
                if cursor % 320 == 0:
                    write_json(target / 'progress.json', {'windows': cursor, 'total': len(rows),
                               'seconds': time.monotonic() - started})
        features.flush()
        handle.remove()
        del model
        torch.cuda.empty_cache()
        reloaded, _ = load_source(source_root / 'rgb', plan['sources']['rgb']['sha256'])
        captures = []
        handle = reloaded.avgpool.register_forward_hook(
            lambda _m, _i, v: captures.append(v.detach().float().flatten(1).cpu().numpy()))
        probes = [0, (len(rows) // 2) // 2 * 2, (len(rows) - 1) // 2 * 2]
        with torch.inference_mode():
            for start in probes:
                x = torch.stack([data[i][0] for i in range(start, min(start + 2, len(rows)))])
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    reloaded(x.cuda())
                np.testing.assert_array_equal(features[start:start + len(x)], captures.pop())
        handle.remove()
        write_json(target / 'report.json', {'status': 'complete_paired_RGB_teacher_features',
                   'teacher_sha256': plan['sources']['rgb']['sha256'], 'rows': cursor,
                   'features_sha256': sha256_file(target / 'features.npy'), 'reload_probe_starts': probes,
                   'reload_exact': True, 'target_Train_or_Test_read': False,
                   'seconds': time.monotonic() - started})
        return

    teacher_report = json.loads((out / 'teacher/report.json').read_text())
    if teacher_report['teacher_sha256'] != plan['sources']['rgb']['sha256']:
        raise ValueError('Wrong paired RGB teacher')
    if sha256_file(out / 'teacher/features.npy') != teacher_report['features_sha256']:
        raise ValueError('Teacher features changed')
    receipt = json.loads((out / 'teacher/input_verification.json').read_text())
    if receipt['plan_sha256'] != sha256_file(args.plan):
        raise ValueError('Teacher extraction used a different plan')
    for name, key in [('source_rows.json', 'source_rows_sha256'), ('supervision.npz', 'supervision_sha256')]:
        if sha256_file(source_root / 'depth' / name) != receipt[key]:
            raise ValueError('Source supervision changed after teacher preparation')
    teacher = torch.from_numpy(np.load(out / 'teacher/features.npy')).cuda()
    model, source = load_source(source_root / 'depth', plan['sources']['depth']['sha256'])
    initial = array_digest(*[v.detach().cpu().numpy() for v in model.state_dict().values()])
    frozen = buffers_digest(model)
    recipe = plan['recipe']
    rng = np.random.default_rng(recipe['seed'])
    for original in supervision['orders']:
        np.testing.assert_array_equal(original, rng.choice(len(rows), len(rows), replace=True,
                                                         p=supervision['sample_probability']))
    order = rng.choice(len(rows), len(rows), replace=True, p=supervision['sample_probability'])
    write_json(target / 'source_rows.json', rows)
    updated_supervision = {**supervision, 'orders': order[None, :]}
    np.savez_compressed(target / 'supervision.npz', **updated_supervision)
    data = dataset(args.packed_cache, 'depth', rows)
    loader = torch.utils.data.DataLoader(data, batch_size=16, sampler=order.tolist(), num_workers=2,
                                         pin_memory=True)
    optimizer = torch.optim.AdamW([
        {'params': [v for n, v in model.named_parameters() if v.requires_grad and not n.startswith('fc.')],
         'lr': recipe['backbone_lr']},
        {'params': list(model.fc.parameters()), 'lr': recipe['head_lr']}], weight_decay=recipe['weight_decay'])
    steps = (len(order) + 15) // 16
    assert steps == recipe['steps'] == 880
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    pos_weight = torch.from_numpy(supervision['pos_weight']).cuda()
    selected = np.linspace(0, len(rows) - 1, 4, dtype=int).tolist()
    monitor_batch = tuple(torch.stack(v) for v in zip(*[data[i] for i in selected], strict=True))
    monitors = [{'step': 0, **monitor_features(model, monitor_batch, pos_weight)}]
    captures = []
    handle = model.avgpool.register_forward_hook(lambda _m, _i, v: captures.append(v.flatten(1)))
    cursor, ce_total, align_total, started = 0, 0., 0., time.monotonic()
    alignment_enabled = args.stage == 'BCE_RGB_cosine'
    for step, (x, y, known) in enumerate(loader, 1):
        space(out, reserve)
        indices = order[cursor:cursor + len(x)]
        np.testing.assert_array_equal(y.numpy(), supervision['labels'][indices])
        np.testing.assert_array_equal(known.numpy(), supervision['masks'][indices])
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(x), 2):
            captures.clear()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x[start:start + 2].cuda(non_blocking=True))
            pooled = captures.pop()
            ce = masked_bce(logits, y[start:start + 2].cuda(), known[start:start + 2].cuda(), pos_weight)
            aligned = cosine_alignment(pooled, teacher[indices[start:start + 2]])
            loss = (ce + aligned if alignment_enabled else ce).sum() / len(x)
            if not torch.isfinite(loss) or not torch.isfinite(aligned).all():
                raise ValueError('Nonfinite source objective')
            loss.backward()
            ce_total += float(ce.detach().sum())
            align_total += float(aligned.detach().sum())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        cursor += len(x)
        if step % 20 == 0 or step == steps:
            monitors.append({'step': step, **monitor_features(model, monitor_batch, pos_weight)})
            captures.clear()
            write_json(target / 'activation_progress.json', monitors)
            write_json(target / 'progress.json', {'step': step, 'steps': steps, 'windows': cursor,
                       'BCE_mean': ce_total / cursor, 'cosine_loss_mean': align_total / cursor,
                       'seconds': time.monotonic() - started})
            print(args.stage, step, ce_total / cursor, align_total / cursor, flush=True)
    handle.remove()
    if buffers_digest(model) != frozen:
        raise ValueError('Frozen BN changed')
    config = copy.deepcopy(source['config'])
    config.update(epochs=1, source_continuation=plan, alignment_enabled=alignment_enabled)
    history = [{'epoch': 1, 'loss': ce_total / cursor, 'cosine_loss': align_total / cursor,
                'seconds': time.monotonic() - started}]
    payload = {k: copy.deepcopy(v) for k, v in source.items() if k not in ['learned', 'history', 'config']}
    payload.update(learned={n: v.detach().float().cpu() for n, v in model.named_parameters() if v.requires_grad},
                   config=config, history=history, steps=steps, initial_sha256=initial,
                   sample_order_sha256=array_digest(order),
                   supervision_sha256=sha256_file(target / 'supervision.npz'),
                   parent_source_sha256=plan['sources']['depth']['sha256'],
                   teacher_source_sha256=plan['sources']['rgb']['sha256'],
                   teacher_feature_sha256=teacher_report['features_sha256'],
                   alignment_enabled=alignment_enabled, prior_source_steps=source['steps'])
    torch.save(payload, target / 'model.pt')
    torch.save({'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'actual_steps': steps, 'model_sha256': sha256_file(target / 'model.pt')},
               target / 'optimizer_final.pt')
    reloaded = network(config).cuda()
    state = reloaded.state_dict()
    state.update(torch.load(target / 'model.pt', map_location='cpu', weights_only=True)['learned'])
    reloaded.load_state_dict(state, strict=True)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        torch.testing.assert_close(model(x[:2].cuda()), reloaded(x[:2].cuda()), rtol=0, atol=0)
    write_json(target / 'config.json', config)
    write_json(target / 'report.json', {'status': 'source_pretraining_complete',
               'model_sha256': sha256_file(target / 'model.pt'), 'reload_exact': True,
               'retained_precision': 'float32', 'windows': len(rows), 'videos': len(source['videos']),
               'people': len(source['source_people']), 'target_Train_or_Test_read': False,
               'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
               **{k: v for k, v in payload.items() if k != 'learned'}})


if __name__ == '__main__':
    main()
