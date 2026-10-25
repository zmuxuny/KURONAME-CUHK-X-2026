#!/usr/bin/env python3
"""Run the fixed external target recipe with ordered prefetch and deferred loss logging."""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import numpy as np
import torch

from cuhkx.metadata import subject_id
from scripts import fit_r34_external_target as original


def update(model, optimizer, x, owners, labels, class_weights, cfg):
    """Preserve every microbatch computation; collect CPU loss values once per step."""
    optimizer.zero_grad(set_to_none=True)
    counts = np.bincount(owners, minlength=len(labels))
    x = x.cuda()
    target = torch.as_tensor(labels[owners], device='cuda')
    weights = torch.as_tensor(1. / counts[owners], device='cuda', dtype=torch.float32)
    loss_log = []
    for start in range(0, len(x), cfg['microbatch_views']):
        stop = start + cfg['microbatch_views']
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x[start:stop])
            losses = torch.nn.functional.cross_entropy(
                logits, target[start:stop], weight=class_weights, reduction='none')
            loss = (losses * weights[start:stop]).sum() / len(labels)
        loss.backward()
        loss_log.append(loss.detach())
    values = torch.stack(loss_log).cpu().tolist()
    if not all(np.isfinite(v) for v in values):
        raise ValueError('Nonfinite R34 loss')
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
    optimizer.step()
    return sum(values), float(norm)


class OrderedPrefetch:
    """Speculate only the next deterministic CPU batch; caller order is authoritative."""

    def __init__(self, config):
        self.config = config
        self.load = original.load_records
        self.batch = original.batch_views
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.next_units = {}
        self.pending = None
        self.pending_units = None
        self.hits = 0
        self.misses = 0

    def load_records(self, config, output):
        grouped = self.load(config, output)
        cfg = config['training']
        # If an identical batch has different successors in two scopes, do not speculate.
        ambiguous = set()
        for outer in config['subjects'] + ['final']:
            units = [u for u in grouped if subject_id(u) != outer]
            rng = np.random.default_rng(cfg['seed'])
            batches = []
            for _ in range(cfg['epochs']):
                order = rng.permutation(len(units))
                batches.extend(tuple(units[i] for i in order[start:start + cfg['batch_units']])
                               for start in range(0, len(order), cfg['batch_units']))
            for before, after in itertools.pairwise(batches):
                if before in self.next_units and self.next_units[before] != after:
                    ambiguous.add(before)
                self.next_units[before] = after
        for key in ambiguous:
            self.next_units.pop(key, None)
        return grouped

    def batch_views(self, units, grouped):
        key = tuple(units)
        if self.pending is not None and self.pending_units == key:
            result = self.pending.result()
            self.hits += 1
        else:
            if self.pending is not None:
                # Finish an already running read before reusing this one-slot pipeline.
                self.pending.result()
            result = self.batch(units, grouped)
            self.misses += 1
        self.pending = None
        self.pending_units = self.next_units.get(key)
        if self.pending_units is not None:
            self.pending = self.pool.submit(self.batch, self.pending_units, grouped)
        return result

    def close(self):
        self.pool.shutdown(wait=True)


def main():
    # The original parser, initialization, training loop, schedule and reload remain in use.
    argv = sys.argv[1:]
    if '--help' in argv or '-h' in argv:
        original.main()
        return
    config = json.loads(Path(argv[argv.index('--base-config') + 1]).read_text())
    output = Path(argv[argv.index('--output') + 1])
    output.mkdir(parents=True, exist_ok=True)
    source_files = [Path(__file__), Path(original.__file__)]
    evidence = {'implementation': 'same_microbatch_deferred_sync_and_ordered_CPU_prefetch_v1',
                'source_sha256': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in source_files},
                'microbatch_views': config['training']['microbatch_views'],
                'training_contract': 'Unchanged original data, normalization, scopes, order, '
                                     'initialization, precision, loss, schedule and update count.'}
    manifest = output / 'performance_implementation.json'
    if manifest.exists() and json.loads(manifest.read_text()) != evidence:
        raise ValueError('Performance implementation changed on resume')
    manifest.write_text(json.dumps(evidence, indent=2) + '\n')
    for path in source_files:
        target = output / 'performance_source' / path.name
        target.parent.mkdir(exist_ok=True)
        if target.exists() and target.read_bytes() != path.read_bytes():
            raise ValueError('Preserved performance source changed')
        target.write_bytes(path.read_bytes())
    prefetch = OrderedPrefetch(config)
    original.load_records = prefetch.load_records
    original.batch_views = prefetch.batch_views
    original.update = update
    # Original source_fit snapshot now explicitly pins this active entrypoint.
    original.__file__ = __file__
    try:
        original.main()
    finally:
        prefetch.close()
        (output / 'prefetch_report.json').write_text(json.dumps(
            {'hits': prefetch.hits, 'misses': prefetch.misses}, indent=2) + '\n')


if __name__ == '__main__':
    main()
