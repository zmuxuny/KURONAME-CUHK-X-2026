#!/usr/bin/env python3
"""Apply the matched warmed-head target R recipe after external source fitting."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

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
from cuhkx.spatial_training import check_space, write_json
from cuhkx.validation import sha256_file
from scripts.probe_r34_head_warmup_stability import activation


def initialize(config, source, device='cuda'):
    model = network(config, 'R')
    original_head = array_digest(*[p.detach().numpy() for p in model.fc.parameters()])
    learned = {k: v for k, v in source['learned'].items() if not k.startswith('fc.')}
    expected = {k for k, p in model.named_parameters() if p.requires_grad and not k.startswith('fc.')}
    if set(learned) != expected:
        raise ValueError('External source backbone learning scope differs from target R')
    state = model.state_dict()
    state.update(learned)
    model.load_state_dict(state, strict=True)
    if original_head != array_digest(*[p.detach().numpy() for p in model.fc.parameters()]):
        raise ValueError('Source 157-axis head entered the new target 44-axis head')
    if buffers_digest(model) != source['bn_buffer_sha256']:
        raise ValueError('External source changed the fixed generic BN buffers')
    return model.to(device), original_head


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-config', type=Path, required=True)
    parser.add_argument('--raw-run', type=Path, required=True)
    parser.add_argument('--source-checkpoint', type=Path, required=True)
    parser.add_argument('--warmup', type=Path, required=True)
    parser.add_argument('--control-run', type=Path, required=True,
                        help='Completed E0 warm-target run with the matched R checkpoints')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-models', type=int)
    args = parser.parse_args()
    if args.max_models is not None and args.max_models < 1:
        raise ValueError('max-models must be positive')
    configure_torch()
    config = json.loads(args.base_config.read_text())
    declared = json.loads((args.control_run / 'source_fit/config.json').read_text())
    if config != declared:
        raise ValueError('Target fitting must match the actual E0 R recipe')
    report = json.loads(args.source_checkpoint.with_name('report.json').read_text())
    digest = sha256_file(args.source_checkpoint)
    if report['model_sha256'] != digest or not report['reload_exact'] or report['retained_precision'] != 'float32':
        raise ValueError('External source has no completed exact-reload receipt')
    source = torch.load(args.source_checkpoint, map_location='cpu', weights_only=True)
    if source['source_only'] != 'Charades official Train':
        raise ValueError('Unsupported external training source')
    external = json.loads((args.warmup / 'external_source.json').read_text())
    if external['sha256'] != digest or external['arm'] != source['arm']:
        raise ValueError('Target warm-up did not use this source backbone')
    warm_report = json.loads((args.warmup / 'warmup_report.json').read_text())
    if warm_report['status'] != 'complete19classifierwarmups':
        raise ValueError('All19 source-specific target warm-ups are required')
    if json.loads((args.warmup / 'source_warmup/config.json').read_text()) != config:
        raise ValueError('Warm-up target contract changed')
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = {'base_config': config, 'source_model': str(args.source_checkpoint.resolve()),
                'source_model_sha256': digest, 'source_arm': source['arm'],
                'source_report_sha256': sha256_file(args.source_checkpoint.with_name('report.json')),
                'source_subjects': source['source_people'], 'source_videos': source['videos'],
                'warmup_report_sha256': sha256_file(args.warmup / 'warmup_report.json'),
                'control_run': str(args.control_run.resolve()),
                'control_fit_report_sha256': sha256_file(args.control_run / 'fit_report.json'),
                'raw_cache_report_sha256': sha256_file(args.raw_run / 'cache_report.json'),
                'target_training': copy.deepcopy(config['training']),
                'comparison': 'E0/RGB/Depth share target units, views, initial random44head,20epoch head-warmup algorithm, class weights, orders,4epoch adaptation steps and fixed BN. Each warmed head uses its own backbone features; learned warm-head values need not be identical.'}
    identity = args.output / 'input.json'
    if identity.exists() and json.loads(identity.read_text()) != evidence:
        raise ValueError('External source or target contract changed on resume')
    if not identity.exists():
        write_json(identity, evidence)
    sources = [Path(__file__), ROOT / 'src/cuhkx/r34_adaptation.py', ROOT / 'src/cuhkx/r34_video.py',
               ROOT / 'src/cuhkx/harn_tail.py', ROOT / 'src/cuhkx/spatial_training.py',
               ROOT / 'scripts/probe_r34_head_warmup_stability.py']
    snapshot = args.output / 'source_fit'
    if snapshot.exists():
        for entry in json.loads((snapshot / 'manifest.json').read_text()):
            if sha256_file(Path(entry['path'])) != entry['sha256']:
                raise ValueError('Target fitting source changed; explicitly version before resuming')
    else:
        for path in sources:
            target = snapshot / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
        write_json(snapshot / 'manifest.json', [{'path': str(p), 'sha256': sha256_file(p)} for p in sources])
        write_json(snapshot / 'config.json', config)
        subprocess.run(['git', 'archive', '--format=tar.gz',
                        '--output=' + str(snapshot.resolve() / 'committed_dependencies.tar.gz'), 'HEAD'],
                       cwd=ROOT, check=True)
        with (snapshot / 'pip_freeze.txt').open('x') as handle:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze', '--all'], stdout=handle, check=True)
        (args.output / 'cache_report.json').write_bytes((args.raw_run / 'cache_report.json').read_bytes())
    invocations = args.output / 'invocations'
    invocations.mkdir(exist_ok=True)
    write_json(invocations / f'{time.time_ns()}.json', {'argv': sys.argv, 'cwd': str(Path.cwd()),
               'python': sys.executable, 'python_version': sys.version,
               'torch': torch.__version__, 'CUDA': torch.version.cuda, 'effective_config': config})
    grouped = load_records(config, args.raw_run)
    cfg = config['training']
    models = args.output / 'models'
    models.mkdir(exist_ok=True)
    diagnostics = args.output / 'training_dynamics'
    diagnostics.mkdir(exist_ok=True)
    completed = 0
    for outer in config['subjects'] + ['final']:
        path = models / f'{outer}_R.pt'
        receipt = path.with_suffix('.json')
        head_path = args.warmup / 'heads' / (outer + '.pt')
        head_sha = sha256_file(head_path)
        head_receipt = json.loads(head_path.with_suffix('.json').read_text())
        if head_receipt['sha256'] != head_sha or not head_receipt['reload_exact']:
            raise ValueError('Warm-up head changed or lacks exact-reload evidence')
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if (sha256_file(path) != saved['sha256'] or saved['warm_head_sha256'] != head_sha
                    or saved['external_source_sha256'] != digest):
                raise ValueError('Completed target checkpoint changed')
            continue
        if path.exists() or path.with_suffix('.tmp').exists():
            raise ValueError('Retain and inspect partial target checkpoint before resume')
        check_space(config)
        units = [u for u in grouped if subject_id(u) != outer]
        people = sorted({subject_id(u) for u in units})
        head = torch.load(head_path, map_location='cpu', weights_only=True)
        if head['training_units'] != units or head['training_subjects'] != people:
            raise ValueError('External warm-up used wrong target people or units')
        control = json.loads((args.control_run / 'models' / f'{outer}_R.json').read_text())
        if sha256_file(args.control_run / 'models' / f'{outer}_R.pt') != control['sha256']:
            raise ValueError('Matched E0 model changed')
        control_head = json.loads((Path(config['head_initialization']['warmup']) / 'heads' / (outer + '.json')).read_text())
        for key in ['training_units', 'training_subjects', 'epochs', 'steps',
                    'initial_head_sha256', 'sample_order_sha256']:
            if head_receipt[key] != control_head[key]:
                raise ValueError('Source/E0 target head warm-up differs: ' + key)
        labels = np.asarray([action_class(u) for u in units])
        counts = np.bincount(labels, minlength=44)
        supported = np.flatnonzero(counts)
        weights = np.zeros(44, dtype=np.float32)
        weights[supported] = len(units) / (len(supported) * counts[supported])
        class_weights = torch.from_numpy(weights).cuda()
        rng = np.random.default_rng(cfg['seed'])
        orders = [rng.permutation(len(units)) for _ in range(cfg['epochs'])]
        model, head_digest = initialize(config, source)
        if head_digest != head['initial_head_sha256']:
            raise ValueError('Target random head differs from the warm-up initialization')
        model.fc.load_state_dict(head['state_dict'], strict=True)
        frozen = buffers_digest(model)
        initial = array_digest(*[p.detach().cpu().numpy() for p in model.state_dict().values()])
        optimizer = optimizer_for(model, config)
        steps = cfg['epochs'] * ((len(units) + cfg['batch_units'] - 1) // cfg['batch_units'])
        matched = {'training_units': units, 'training_subjects': people, 'steps': steps,
                   'sample_order_sha256': array_digest(*orders), 'config': cfg}
        for key, value in matched.items():
            if control[key] != value:
                raise ValueError('Source/E0 target backbone fitting differs: ' + key)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        monitor_units = [next(u for u in units if action_class(u) == label) for label in [0, 11, 22, 33]]
        monitor, owners = batch_views(monitor_units, grouped)
        monitor_labels = torch.tensor([action_class(monitor_units[i]) for i in owners])
        dynamics = [{'step': 0, **activation(model, monitor, monitor_labels)}]
        write_json(diagnostics / (outer + '.json'), {'training_units': monitor_units, 'measurements': dynamics})
        actual_steps = 0
        history, started = [], time.monotonic()
        for epoch, order in enumerate(orders):
            total = 0.
            for start in range(0, len(order), cfg['batch_units']):
                chosen = order[start:start + cfg['batch_units']]
                x, owners = batch_views([units[i] for i in chosen], grouped)
                loss, _ = update(model, optimizer, x, owners, labels[chosen], class_weights, cfg)
                scheduler.step()
                actual_steps += 1
                total += loss * len(chosen)
                if actual_steps % 50 == 0:
                    dynamics.append({'step': actual_steps, **activation(model, monitor, monitor_labels)})
                    write_json(diagnostics / (outer + '.json'), {'training_units': monitor_units, 'measurements': dynamics})
                    print(source['arm'], outer, 'training_monitor', dynamics[-1], flush=True)
            history.append({'epoch': epoch + 1, 'loss': total / len(units), 'elapsed_seconds': time.monotonic() - started})
            write_json(args.output / 'progress.json', {'outer': outer, 'history': history})
            print(source['arm'], outer, history[-1], flush=True)
        if buffers_digest(model) != frozen:
            raise ValueError('Target adaptation changed fixed BN statistics')
        dynamics.append({'step': actual_steps, **activation(model, monitor, monitor_labels)})
        write_json(diagnostics / (outer + '.json'), {'training_units': monitor_units, 'measurements': dynamics})
        payload = {'arm': 'R', 'outer': outer, 'training_subjects': people, 'training_units': units,
                   'config': copy.deepcopy(cfg), 'source_weights_sha256': config['weights_sha256'],
                   'external_source_sha256': digest, 'external_source_arm': source['arm'],
                   'warm_head_sha256': head_sha, 'warmup_feature_sha256': head['feature_sha256'],
                   'training_dynamics': dynamics, 'control_model_sha256': control['sha256'],
                   'learned': {n: p.detach().half().cpu() for n, p in model.named_parameters() if p.requires_grad},
                   'supported_classes': supported.tolist(), 'steps': steps, 'history': history,
                   'initial_state_sha256': initial, 'initial_target_head_sha256': head_digest,
                   'sample_order_sha256': array_digest(*orders), 'bn_buffer_sha256': frozen}
        temporary = path.with_suffix('.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)
        state = model.state_dict()
        state.update(payload['learned'])
        model.load_state_dict(state, strict=True)
        # Source and target change exactly the same parameter scope. The
        # retained complete R state therefore replays with the generic prefix.
        reloaded, _ = restore(config, path)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            torch.testing.assert_close(model(x[:2].cuda()), reloaded(x[:2].cuda()), rtol=0, atol=0)
        write_json(receipt, {'sha256': sha256_file(path), 'retained_fp16_reload_exact': True,
                            **{k: v for k, v in payload.items() if k != 'learned'}})
        del model, optimizer, reloaded, payload
        torch.cuda.empty_cache()
        completed += 1
        if args.max_models is not None and completed >= args.max_models:
            return
    write_json(args.output / 'fit_report.json', {'status': 'complete19external_warm_target_models',
               'external_source_sha256': digest, 'arm': source['arm'],
               'models': [{'path': str(p), 'sha256': sha256_file(p)} for p in sorted(models.glob('*.pt'))],
               'evaluation': 'Full18 raw-input evaluation remains separate'})


if __name__ == '__main__':
    main()
