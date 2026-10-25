#!/usr/bin/env python3
"""Matched RGB/synthetic-depth R34 preadaptation on external Train only."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch

from cuhkx.charades_packed_source import PackedSourceWindows, verify_cache
from cuhkx.generic_video_windows import KINETICS_MEAN, KINETICS_STD
from cuhkx.harn_tail import configure_torch
from cuhkx.r34_adaptation import buffers_digest
from cuhkx.r34_video import pretrained_model
from cuhkx.sequence_skeleton_nested import array_digest
from cuhkx.source_depth_encoding import encode_frames, encoding_contract
from cuhkx.spatial_training import check_space, write_json
from cuhkx.validation import sha256_file


class SourceWindows(torch.utils.data.Dataset):
    def __init__(self, path, arm, rows, encoding='identity'):
        self.path, self.arm, self.rows = path, arm, rows
        encoding_contract(encoding, arm)
        self.encoding = encoding

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        video, window, _, _, _ = self.rows[index]
        with np.load(self.path / self.arm / (video + '.npz')) as z:
            raw, label, mask = z['frames'][window], z['labels'][window], z['mask'][window]
        raw = encode_frames(raw, self.encoding)
        if raw.shape[-1] == 1:
            raw = np.repeat(raw, 3, axis=-1)
        x = torch.from_numpy(raw).permute(3, 0, 1, 2).float() / 255.
        x = ((x - torch.tensor(KINETICS_MEAN)[:, None, None, None])
             / torch.tensor(KINETICS_STD)[:, None, None, None])
        return x, torch.from_numpy(label), torch.from_numpy(mask)


def network(config):
    model = pretrained_model(config['weights'])
    torch.manual_seed(config['seed'])
    model.fc = torch.nn.Linear(512, 157)
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith(('layer3.', 'layer4.', 'fc.')))
    return model.eval()


def masked_bce(logits, labels, known, pos_weight):
    values = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.float(), labels.float(), pos_weight=pos_weight, reduction='none')
    count = known.sum(1)
    if (count == 0).any():
        raise ValueError('Source window has no known action axes')
    return (values * known).sum(1) / count


def constant_bce_baseline(labels, masks, probability, pos_weight):
    known_count = masks.sum(1)
    if (known_count == 0).any():
        raise ValueError('Source window has no known action axes')
    effective_weight = probability / known_count
    positive = (labels * masks * effective_weight[:, None]).sum(0)
    negative = ((1 - labels) * masks * effective_weight[:, None]).sum(0)
    denominator = negative + pos_weight * positive
    prior = np.divide(pos_weight * positive, denominator,
                      out=np.full_like(positive, .5), where=denominator > 0)
    prior = np.clip(prior, 1e-7, 1 - 1e-7)
    logits = torch.from_numpy(np.log(prior / (1 - prior))).float()
    losses = masked_bce(logits.expand(len(labels), -1), torch.from_numpy(labels),
                         torch.from_numpy(masks), torch.from_numpy(pos_weight)).numpy()
    return prior, losses


def unique_video_mass(files, raw_identity):
    """Equal person/unique-clip mass, shared by its annotation records."""
    unique = defaultdict(set)
    repetitions = Counter()
    for row in files:
        key = (row['subject'], raw_identity[row['id']])
        unique[row['subject']].add(key[1])
        repetitions[key] += 1
    return {row['id']: 1. / (len(unique[row['subject']])
                            * repetitions[(row['subject'], raw_identity[row['id']])])
            for row in files}


@torch.inference_mode()
def monitor_features(model, batch, pos_weight):
    """Fixed external-source windows reveal feature collapse hidden by sparse BCE."""
    x, y, known = batch
    captures, losses = [], []
    handle = model.avgpool.register_forward_hook(
        lambda _module, _input, value: captures.append(value.detach().float().flatten(1).cpu()))
    try:
        for start in range(0, len(x), 2):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x[start:start + 2].cuda())
            losses.append(masked_bce(logits, y[start:start + 2].cuda(),
                                     known[start:start + 2].cuda(), pos_weight).cpu())
    finally:
        handle.remove()
    pooled = torch.cat(captures)
    return {'pooled_abs_mean': float(pooled.abs().mean()),
            'positive_fraction': float(pooled.gt(0).float().mean()),
            'zero_views': int(pooled.abs().amax(1).eq(0).sum()),
            'fixed_window_bce': torch.cat(losses).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=['rgb', 'depth'], required=True)
    parser.add_argument('--packed-cache', type=Path,
                        help='Verified lossless physical-frame cache; keeps the original training recipe')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    encoding = config.get('source_encoding', 'identity')
    contract = encoding_contract(encoding, args.arm)
    destination = args.output / args.arm
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'source.py').write_bytes(Path(__file__).read_bytes())
    write_json(destination / 'config.json', config)
    write_json(destination / 'input_encoding.json', contract)
    sources = [Path(__file__).resolve(), ROOT / 'src/cuhkx/r34_adaptation.py',
               ROOT / 'src/cuhkx/r34_video.py', ROOT / 'src/cuhkx/generic_video_windows.py',
               ROOT / 'src/cuhkx/harn_tail.py', ROOT / 'src/cuhkx/spatial_training.py',
               ROOT / 'src/cuhkx/sequence_skeleton_nested.py', ROOT / 'src/cuhkx/validation.py',
               ROOT / 'src/cuhkx/charades_packed_source.py',
               ROOT / 'src/cuhkx/source_depth_encoding.py']
    for path in sources:
        target = destination / 'source_snapshot' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    write_json(destination / 'source_manifest.json', [{'path': str(p), 'sha256': sha256_file(p)} for p in sources])
    write_json(destination / 'environment.json', {'python': sys.version, 'torch': torch.__version__,
               'numpy': np.__version__, 'CUDA': torch.version.cuda,
               'argv': sys.argv, 'cwd': str(Path.cwd()), 'interpreter': sys.executable})
    rgb = json.loads((args.inputs / 'rgb_report.json').read_text())
    depth = json.loads((args.inputs / 'depth_report.json').read_text())
    depth_files = {r['id']: r for r in depth['files']}
    if config.get('expected_source_videos', len(rgb['files'])) != len(rgb['files']):
        raise ValueError('Source video count differs from declared training plan')
    people = Counter(r['subject'] for r in rgb['files'])
    if config.get('sampling_unit') == 'unique_raw_sha256_per_person':
        identities, metadata_sha = {}, {}
        for row in rgb['files']:
            path = args.inputs / 'metadata' / (row['id'] + '.json')
            meta = json.loads(path.read_text())
            if meta['sha256'] != row['sha256'] or meta['subject'] != row['subject']:
                raise ValueError('Raw identity metadata and RGB production disagree')
            identities[row['id']] = meta['raw_sha256']
            metadata_sha[row['id']] = sha256_file(path)
        video_mass = unique_video_mass(rgb['files'], identities)
        write_json(destination / 'sampling_identity.json', {'raw_sha256': identities,
                   'metadata_sha256': metadata_sha, 'unnormalized_record_mass': video_mass,
                   'unique_source_media': len(set(identities.values())),
                   'policy': 'Equal person then unique raw clip; share its mass equally across annotation records, then windows. Retain every annotation; no majority truth selection.'})
    else:
        video_mass = {r['id']: 1. / people[r['subject']] for r in rgb['files']}
    rows = []
    if args.packed_cache is not None:
        cache_sha = verify_cache(args.packed_cache, args.inputs, args.arm,
                                 [r['id'] for r in rgb['files']])
        write_json(destination / 'packed_cache.json', {'path': str(args.packed_cache),
                   'arm_report_sha256': cache_sha, 'policy': 'Exact original window tensors and sampling; storage only'})
    for entry in rgb['files']:
        video = entry['id']
        if sha256_file(args.inputs / 'rgb' / (video + '.npz')) != entry['sha256']:
            raise ValueError('RGB fitting source changed')
        if sha256_file(args.inputs / 'depth' / (video + '.npz')) != depth_files[video]['sha256']:
            raise ValueError('Depth fitting source changed')
        with np.load(args.inputs / 'rgb' / (video + '.npz')) as left, np.load(args.inputs / 'depth' / (video + '.npz')) as right:
            for key in ['labels', 'mask', 'frame_indices', 'pts_seconds', 'bounds']:
                np.testing.assert_array_equal(left[key], right[key])
            n = len(left['labels'])
            for window in range(n):
                rows.append((video, window, video_mass[video] / n, left['labels'][window], left['mask'][window]))
    probability = np.asarray([r[2] for r in rows])
    if config.get('expected_source_windows', len(rows)) != len(rows):
        raise ValueError('Source window count differs from declared training plan')
    probability /= probability.sum()
    labels = np.stack([r[3] for r in rows])
    masks = np.stack([r[4] for r in rows])
    positive = (labels * masks * probability[:, None]).sum(0)
    negative = ((1 - labels) * masks * probability[:, None]).sum(0)
    pos_weight = np.clip(np.sqrt(negative / np.maximum(positive, 1e-12)), 1., 20.).astype(np.float32)
    # Weighted BCE has a low constant-prediction optimum on sparse multi-label data.
    # Comparing against it prevents interpreting prediction of prevalence as learning motion.
    prior, baseline = constant_bce_baseline(labels, masks, probability, pos_weight)
    rng = np.random.default_rng(config['seed'])
    orders = [rng.choice(len(rows), len(rows), replace=True, p=probability) for _ in range(config['epochs'])]
    np.savez_compressed(destination / 'supervision.npz', labels=labels, masks=masks,
                        sample_probability=probability, pos_weight=pos_weight, orders=np.stack(orders),
                        constant_prior=prior, constant_bce_by_window=baseline)
    write_json(destination / 'source_rows.json', [[r[0], r[1]] for r in rows])
    configure_torch()
    model = network(config).cuda()
    initial = array_digest(*[p.detach().cpu().numpy() for p in model.state_dict().values()])
    frozen = buffers_digest(model)
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('fc.')], 'lr': config['backbone_lr']},
        {'params': list(model.fc.parameters()), 'lr': config['head_lr']}], weight_decay=config['weight_decay'])
    steps = config['epochs'] * ((len(rows) + config['batch_windows'] - 1) // config['batch_windows'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    dataset = (SourceWindows(args.inputs, args.arm, rows, encoding) if args.packed_cache is None else
               PackedSourceWindows(args.packed_cache, args.arm, rows, encoding))
    positive_weight = torch.from_numpy(pos_weight).cuda()
    monitor_indices = np.linspace(0, len(rows) - 1, 4, dtype=int).tolist()
    monitor_batch = tuple(torch.stack(values) for values in zip(
        *[dataset[i] for i in monitor_indices], strict=True))
    monitors = [{'step': 0, **monitor_features(model, monitor_batch, positive_weight)}]
    write_json(destination / 'activation_progress.json', monitors)
    write_json(destination / 'monitor_scope.json', {'row_indices': monitor_indices,
               'windows': [[rows[i][0], rows[i][1]] for i in monitor_indices],
               'selection': 'Four uniformly spaced source row indices; identical RGB/depth; no target information',
               'full_source_weighted_constant_bce': float(baseline @ probability),
               'fixed_windows_constant_bce': baseline[monitor_indices].tolist()})
    actual_steps = 0
    history, started = [], time.monotonic()
    for epoch, order in enumerate(orders):
        loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_windows'], sampler=order.tolist(),
                                            num_workers=2, pin_memory=True)
        loss_sum = 0.
        for x, y, known in loader:
            check_space({'min_host_free_bytes': config.get('min_host_free_bytes', 8 * 2**30)})
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(x), 2):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(x[start:start + 2].cuda(non_blocking=True))
                loss = masked_bce(logits, y[start:start + 2].cuda(), known[start:start + 2].cuda(), positive_weight).sum() / len(x)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite source loss')
                loss.backward()
                loss_sum += float(loss.detach()) * len(x)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            actual_steps += 1
            if actual_steps % 20 == 0:
                monitors.append({'step': actual_steps, **monitor_features(model, monitor_batch, positive_weight)})
                write_json(destination / 'activation_progress.json', monitors)
                print(args.arm, 'training_monitor', monitors[-1], flush=True)
        history.append({'epoch': epoch + 1, 'loss': loss_sum / len(order),
                        'same_draws_constant_bce': float(baseline[order].mean()),
                        'seconds': time.monotonic() - started})
        write_json(destination / 'progress.json', history)
        print(args.arm, history[-1], flush=True)
    if buffers_digest(model) != frozen:
        raise ValueError('Source preadaptation changed frozen BN statistics')
    if monitors[-1]['step'] != actual_steps:
        monitors.append({'step': actual_steps, **monitor_features(model, monitor_batch, positive_weight)})
        write_json(destination / 'activation_progress.json', monitors)
    # Preserve fp32 source master weights so E0/source target initialization
    # differs through preadaptation, without an extra source-only rounding step.
    payload = {'learned': {n: p.detach().float().cpu() for n, p in model.named_parameters() if p.requires_grad},
               'arm': args.arm, 'config': copy.deepcopy(config), 'source_only': 'Charades official Train',
               'source_people': sorted(people), 'videos': [r['id'] for r in rgb['files']],
               'initial_sha256': initial, 'sample_order_sha256': array_digest(*orders),
               'supervision_sha256': sha256_file(destination / 'supervision.npz'),
               'source_rgb_sha256': sha256_file(args.inputs / 'rgb_report.json'),
               'source_depth_sha256': sha256_file(args.inputs / 'depth_report.json'),
               'history': history, 'steps': steps, 'bn_buffer_sha256': frozen}
    path = destination / 'model.pt'
    torch.save(payload, path)
    if args.packed_cache is not None:
        torch.save({'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'actual_steps': actual_steps, 'model_sha256': sha256_file(path)},
                   destination / 'optimizer_final.pt')
    state = model.state_dict()
    state.update(payload['learned'])
    model.load_state_dict(state, strict=True)
    reloaded = network(config).cuda()
    saved = torch.load(path, map_location='cpu', weights_only=True)
    state = reloaded.state_dict()
    state.update(saved['learned'])
    reloaded.load_state_dict(state, strict=True)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        torch.testing.assert_close(model(x[:2].cuda()), reloaded(x[:2].cuda()), rtol=0, atol=0)
    write_json(destination / 'report.json', {'status': 'source_pretraining_complete', 'model_sha256': sha256_file(path),
               'windows': len(rows), 'videos': len(rgb['files']), 'people': len(people),
               'positive_sampled_classes': np.flatnonzero(positive > 0).tolist(),
               'peak_cuda_bytes': torch.cuda.max_memory_allocated(), 'reload_exact': True,
               'retained_precision': 'float32',
               'target_Train_or_Test_read': False, **{k: v for k, v in payload.items() if k != 'learned'}})


if __name__ == '__main__':
    main()
