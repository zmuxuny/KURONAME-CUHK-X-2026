#!/usr/bin/env python3
"""Protect inputs and freeze a Train-selected three-family full inference run."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from cuhkx.harn_videomae import SPEC
from scripts.prepare_r34_complete_inference_v2 import GUARD
from scripts.research_write_guard import install

OUT = ROOT / 'runs/20260909_three_family_complete_candidate_v1'
CONFIG = ROOT / 'configs/checkpoint_inference_three_family_harn_v1.json'


def entry(path):
    path = (ROOT / path).resolve()
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def main():
    install(Path('/root/workspace/CUHK-X'))
    if OUT.exists() or CONFIG.exists():
        raise ValueError('new output and config required')
    plan_path = ROOT / 'configs/harn_three_family_plan_v1.json'
    plan = json.loads(plan_path.read_text())
    previous = json.loads((ROOT / 'runs/20260909_r34_harn_complete_candidate_v3/protection.json').read_text())
    assets = {item['path']: item for item in previous['assets']}
    for item in plan['sources'] + plan['frozen_inputs']:
        actual = entry(item['path'])
        if actual['sha256'] != item['sha256']:
            raise ValueError('Train preflight input/source changed')
        assets[actual['path']] = actual
    old_config = json.loads((ROOT / 'configs/checkpoint_inference_r34_harn_comparison_v1.json').read_text())
    old_descriptor = json.loads((ROOT / old_config['checkpoints']['harn']).read_text())
    descriptor = {
        'kind': 'cuhkx_harn_three_equal_families_v1',
        'parent_checkpoint': old_descriptor['parent_checkpoint'],
        'r34_checkpoint': old_descriptor['r34_checkpoint'],
        'videomae_model': entry('runs/20260908_harn_videomae_nested_v1/models.joblib'),
        'nested_report': entry('runs/20260909_harn_three_family_nested_v1/report.json'),
        'plan': entry(plan_path), 'selected_skeleton_weight': .5,
        'family_weights': plan['family_weights'], 'model_weights': plan['model_weights'],
        'feature_specs': {**old_descriptor['feature_specs'], 'videomae': SPEC},
        'sources': [entry(path) for path in [
            'scripts/infer_harn_three_family_v2.py', 'scripts/infer_three_family_harn_experts_v2.py',
            'src/cuhkx/harn_three_family.py', 'src/cuhkx/harn_videomae_inference.py',
            'scripts/prepare_three_family_complete_inference.py', 'scripts/research_write_guard.py']],
    }
    config = {**old_config, 'profile': 'three_family_harn_equal',
        'checkpoints': {**old_config['checkpoints'], 'harn': str(OUT / 'checkpoint.json')},
        'features': {**old_config['features'], 'harn_videomae': 'artifacts/visual/videomae_ssv2_harn_test_whole_v1/manifest.json'}}
    allowed = []
    for value in config['features'].values():
        path = (ROOT / value).resolve()
        assets[str(path)] = entry(path)
        if path.suffix == '.npz':
            paths = [path]
        else:
            payload = json.loads(path.read_text())
            if 'files' in payload:
                paths = [path.parent / item['path'] for item in payload['files']]
            else:
                names = ['features', 'metadata'] if 'metadata' in payload else ['features', 'units']
                paths = [path.parent / (payload[name]['path'] if isinstance(payload[name], dict) else payload[name]) for name in names]
        for value in paths:
            actual = entry(value)
            assets[actual['path']] = actual
            allowed.append(actual['path'])
    for item in [descriptor[key] for key in ['parent_checkpoint', 'r34_checkpoint', 'videomae_model', 'nested_report', 'plan']] + descriptor['sources']:
        assets[str((ROOT / item['path']).resolve())] = item
    for item in assets.values():
        if entry(item['path'])['sha256'] != item['sha256']:
            raise ValueError('protected old asset changed before execution')
    available = shutil.disk_usage('/mnt/d').free
    if available - 100_000_000 < 8_000_000_000:
        raise ValueError('actual Windows D space below session reserve')
    OUT.mkdir()
    protection = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'assets': list(assets.values()), 'host_D_available_bytes': available,
        'maximum_new_bytes': 100_000_000, 'host_D_reserve_bytes': 8_000_000_000,
        'isolation': 'All previous assets read only; only new run/config/source paths; propagated original-workspace write guard.',
        'restoration': 'Preserve frozen originals; abandon new run and new config. Check every asset against this SHA256 inventory.',
        'environment_changes': False, 'downloads': 0, 'pseudo_dependency': plan['pseudo_dependency']}
    write(OUT / 'protection.json', protection)
    write(OUT / 'checkpoint.json', descriptor)
    write(CONFIG, config)
    for name in ['guard', 'read_audit', 'inputs']:
        (OUT / name).mkdir()
    old_inputs = ROOT / 'runs/20260909_r34_harn_complete_candidate_v3/inputs'
    for name in ['test_qa.csv', 'sample_submission.csv', 'poison_qa.csv', 'poison_restore.csv']:
        shutil.copyfile(old_inputs / name, OUT / 'inputs' / name)
    (OUT / 'guard/sitecustomize.py').write_text(GUARD)
    write(OUT / 'guard_policy.json', {'allowed_data_reads': sorted(set(allowed))})
    write(OUT / 'execution_freeze.json', {'files': [entry(path) for path in
        [CONFIG, OUT / 'checkpoint.json', OUT / 'protection.json', OUT / 'guard/sitecustomize.py',
         OUT / 'guard_policy.json', *sorted((OUT / 'inputs').iterdir())]],
        'preflight_Train_artifacts_checked': True,
        'inference_Train_artifacts_read': False, 'pseudo_dependency': plan['pseudo_dependency']})
    print(json.dumps({'prepared': str(OUT), 'protected_assets': len(assets), 'allowed_arrays': len(set(allowed)), 'D_available': available}))


if __name__ == '__main__':
    main()
