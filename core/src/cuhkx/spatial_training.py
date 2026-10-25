"""Canonical frozen IR token production and subject-excluded paired readouts."""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cuhkx.metadata import subject_id
from cuhkx.sequence_skeleton_nested import array_digest
from cuhkx.spatial_readout import SpatialReadout
from cuhkx.validation import sha256_file
from cuhkx.video import temporal_window_indices
from cuhkx.videomae_legacy_weights import load_runtime
from cuhkx.videomae_windows import read_windows


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def check_space(config):
    disk = os.statvfs('/mnt/d')
    free = disk.f_bavail * disk.f_frsize
    if free < config['min_host_free_bytes']:
        raise RuntimeError(f'Windows host free space below reserve: {free}')
    return free


def inputs(config):
    records = json.loads(Path(config['records']).read_text())['records']
    records = [copy.deepcopy(r) for r in records
               if r['split'] == 'train' and r['modality'] == 'IR' and r['available']]
    for record in records:
        record['path'] = str(Path(config['data_root']) / record['unit'] / 'IR/IR.mp4')
        record['subject'] = subject_id(record['unit'])
        meta = record['metadata']
        indices = temporal_window_indices(meta['frame_count'], meta['fps'], 16, 16, 1.)
        np.testing.assert_array_equal(indices, record['frame_indices'])
    lookup = {r['unit']: r for r in records}
    examples = pd.read_csv(config['paired_examples'])
    if len(examples) != 1448 or len(records) != 809:
        raise ValueError('Canonical S0/S1 input coverage changed')
    if examples.duplicated(['hau_unit', 'window_index']).any():
        raise ValueError('Trusted supervision windows are not unique')
    for row in examples.itertuples():
        record = lookup[row.hau_unit]
        frames = np.asarray(record['frame_indices'][row.window_index])
        if (record['raw']['sha256'] != row.parent_sha256
                or row.subject != record['subject']
                or frames.min() < row.interval_start or frames.max() > row.interval_end):
            raise ValueError('Supervision provenance/time/person mismatch')
    return records, examples


def runtime(config):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if config.get('producer') == 'vjepa21_vitb':
        from cuhkx.jepa_tokens import load_encoder

        torch.cuda.set_per_process_memory_fraction(config['runtime']['gpu_memory_fraction'])
        torch.cuda.reset_peak_memory_stats()
        backbone, processor, audit = load_encoder(config['jepa'])
        backbone.cuhkx_load_audit = audit
        backbone.cuhkx_producer = 'vjepa21_vitb'
        return backbone, processor
    from transformers import VideoMAEImageProcessorPil

    _, backbone, _ = load_runtime(config['runtime'])
    backbone.requires_grad_(False)
    processor = VideoMAEImageProcessorPil.from_pretrained(
        config['runtime']['model_directory'], local_files_only=True)
    return backbone, processor


@torch.inference_mode()
def encode(backbone, processor, frames):
    if getattr(backbone, 'cuhkx_producer', None) == 'vjepa21_vitb':
        from cuhkx.jepa_tokens import encode as encode_jepa

        tokens, _ = encode_jepa(backbone, processor, frames)
        # Identical fp16 rounding before fitting, cached replay, and uncached inference.
        tokens = tokens[0].half()
        if not torch.isfinite(tokens).all():
            raise ValueError('JEPA tokens exceed the declared fp16 cache range')
        return tokens
    pixels = processor(list(frames), return_tensors='pt')['pixel_values']
    if pixels.shape != (1, 16, 3, 224, 224) or not torch.isfinite(pixels).all():
        raise ValueError('Unexpected canonical pixel tensor')
    hidden = backbone.videomae(pixels.to('cuda', torch.float16)).last_hidden_state
    tokens = backbone.fc_norm(hidden).reshape(8, 196, 768)
    if not torch.isfinite(tokens).all():
        raise ValueError('Nonfinite frozen tokens')
    return tokens


def extract(config, output):
    """Cache only trusted training windows; atomic progress supports interruption."""
    records, examples = inputs(config)
    cache = output / 'trusted_tokens.npy'
    done_path = output / 'extraction_progress.json'
    spatial_tokens = 576 if config.get('producer') == 'vjepa21_vitb' else 196
    if done_path.exists():
        done = set(json.loads(done_path.read_text())['completed_units'])
        tokens = np.lib.format.open_memmap(cache, mode='r+')
    else:
        if cache.exists():
            raise ValueError('Unowned partial cache: inspect before restoring')
        required = len(examples) * 8 * spatial_tokens * 768 * 2
        if check_space(config) - required < config['min_host_free_bytes']:
            raise RuntimeError('Insufficient host space for declared token cache plus reserve')
        tokens = np.lib.format.open_memmap(cache, mode='w+', dtype=np.float16,
                                         shape=(len(examples), 8, spatial_tokens, 768))
        done = set()
        write_json(done_path, {'completed_units': [], 'complete': False})
    backbone, processor = runtime(config)
    started = time.monotonic()
    groups = {unit: group for unit, group in examples.groupby('hau_unit', sort=False)}
    for record in records:
        unit = record['unit']
        if unit not in groups or unit in done:
            continue
        check_space(config)
        windows = read_windows(record, config['runtime'])
        for index, row in groups[unit].iterrows():
            value = encode(backbone, processor, windows[int(row.window_index)])
            tokens[index] = value.cpu().numpy()
            if not done and index == groups[unit].index[0]:
                replay = encode(backbone, processor, windows[int(row.window_index)])
                torch.testing.assert_close(value, replay, rtol=0, atol=0)
        tokens.flush()
        done.add(unit)
        write_json(done_path, {'completed_units': sorted(done), 'complete': False,
                              'elapsed_this_session_seconds': time.monotonic() - started})
        if len(done) % 20 == 0:
            print(f'extract {len(done)}/{len(groups)} recordings; '
                  f'{time.monotonic() - started:.1f}s', flush=True)
    examples.to_csv(output / 'trusted_examples.csv', index=False)
    write_json(output / 'extraction_report.json', {
        'complete': True, 'windows': len(examples), 'recordings': len(done),
        'shape': list(tokens.shape), 'dtype': str(tokens.dtype),
        'cache_sha256': sha256_file(cache), 'backbone_loading': backbone.cuhkx_load_audit,
        'processor': type(processor).__module__ + '.' + type(processor).__name__,
        'first_window_raw_repeat_exact': True,
        'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
        'elapsed_this_session_seconds': time.monotonic() - started})
    write_json(done_path, {'completed_units': sorted(done), 'complete': True})


def sampling_plan(labels, rows, seed, epochs):
    """All label access is restricted to explicitly selected training rows."""
    selected = np.asarray(labels)[rows]
    counts = np.bincount(selected, minlength=40)
    weights = 1. / counts[selected]
    weights /= weights.sum()
    rng, bins_rng = np.random.default_rng(seed), np.random.default_rng(seed + 818)
    return [(rng.choice(rows, len(rows), replace=True, p=weights),
             bins_rng.integers(0, 8, len(rows))) for _ in range(epochs)]


def fit(config, output):
    if not (output / 'extraction_report.json').exists():
        raise ValueError('Complete canonical token extraction first')
    _, examples = inputs(config)
    tokens = np.load(output / 'trusted_tokens.npy', mmap_mode='r')
    labels = examples['class'].to_numpy(dtype=np.int64)
    subjects = sorted(examples.subject.unique())
    if len(subjects) != 18:
        raise ValueError('Expected all 18 training people')
    cfg = config['training']
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(config['runtime']['gpu_memory_fraction'])
    model_dir = output / 'models'
    model_dir.mkdir(exist_ok=True)
    for outer in subjects + ['final']:
        rows = np.flatnonzero(examples.subject.to_numpy() != outer)
        people = sorted(examples.iloc[rows].subject.unique())
        if outer in people:
            raise ValueError('Outer person entered fitting rows')
        plan = sampling_plan(labels, rows, cfg['seed'], cfg['epochs'])
        torch.manual_seed(cfg['seed'] + 1000)
        initial = SpatialReadout(projection=cfg['projection'])
        initial_digest = array_digest(*[v.numpy() for v in initial.state_dict().values()])
        interval_digest = array_digest(*[indices for indices, _ in plan])
        bin_digest = array_digest(*[bins for _, bins in plan])
        for arm in ['mean', 'spatial']:
            path = model_dir / f'{outer}_{arm}.pt'
            if path.exists() and path.with_suffix('.json').exists():
                saved = json.loads(path.with_suffix('.json').read_text())
                if saved['sha256'] != sha256_file(path):
                    raise ValueError('Completed checkpoint changed')
                continue
            if path.exists():
                raise ValueError('Partial checkpoint without reload receipt; inspect before resuming')
            check_space(config)
            head = copy.deepcopy(initial).cuda().train()
            optimizer = torch.optim.AdamW(head.parameters(), lr=cfg['lr'],
                                          weight_decay=cfg['weight_decay'])
            steps_per_epoch = (len(rows) + cfg['batch_size'] - 1) // cfg['batch_size']
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg['epochs'] * steps_per_epoch)
            history, started = [], time.monotonic()
            for epoch, (indices, bins) in enumerate(plan):
                total = 0.
                for start in range(0, len(rows), cfg['batch_size']):
                    selected, temporal = (x[start:start + cfg['batch_size']]
                                          for x in (indices, bins))
                    x = torch.from_numpy(np.array(tokens[selected, temporal])).cuda().float()[:, None]
                    y = torch.from_numpy(labels[selected]).cuda()
                    optimizer.zero_grad(set_to_none=True)
                    logits = head(x, arm=arm)[:, 0]
                    loss = torch.nn.functional.cross_entropy(logits, y)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite paired readout loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 5.)
                    optimizer.step()
                    scheduler.step()
                    total += float(loss.detach()) * len(selected)
                history.append({'epoch': epoch + 1, 'loss': total / len(rows)})
                if (epoch + 1) % 10 == 0:
                    print(f'train {outer} {arm} epoch {epoch + 1}/{cfg["epochs"]} '
                          f'loss={total / len(rows):.5f}', flush=True)
            head.eval()
            payload = {'state_dict': {k: v.cpu() for k, v in head.state_dict().items()},
                       'arm': arm, 'outer': outer, 'training_subjects': people,
                       'training_rows': rows.tolist(), 'config': cfg,
                       'supported_classes': sorted(set(labels[rows].tolist())),
                       'initial_sha256': initial_digest,
                       'interval_order_sha256': interval_digest, 'bin_order_sha256': bin_digest,
                       'steps': cfg['epochs'] * steps_per_epoch, 'history': history}
            temporary = path.with_suffix('.tmp')
            torch.save(payload, temporary)
            temporary.replace(path)
            reloaded = load_head(path)
            with torch.inference_mode():
                expected = head(x, arm=arm)
                torch.testing.assert_close(expected, reloaded(x, arm=arm), rtol=0, atol=0)
            write_json(path.with_suffix('.json'), {
                'sha256': sha256_file(path), 'reloaded_exact': True,
                'elapsed_seconds': time.monotonic() - started,
                **{k: v for k, v in payload.items() if k != 'state_dict'}})
            del head, reloaded, optimizer


def load_head(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    head = SpatialReadout(projection=payload['config']['projection']).cuda().eval()
    head.load_state_dict(payload['state_dict'], strict=True)
    return head
