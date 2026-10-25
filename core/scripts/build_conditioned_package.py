#!/usr/bin/env python3
"""Augment the current decision package with the verified two-condition experts."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from scripts import build_current_decision_package as decision
from scripts import run_finalist_pipeline as base


def build(assets, output, d10_config=None):
    path = assets / 'asset_manifest.json'
    if base.sha(path) != (assets / 'asset_manifest.sha256').read_text().strip():
        raise ValueError('Asset manifest changed')
    inventory = json.loads(path.read_text())
    decision.build(output)
    for record in inventory['files']:
        source, target = assets / record['relative'], output / record['relative']
        if base.sha(source) != record['sha256']:
            raise ValueError('Collected model/source changed')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if base.sha(target) != record['sha256']:
                raise ValueError('Conflicting packaged asset')
        else:
            os.link(source, target)
    starts = ['scripts/build_conditioned_package.py', 'scripts/bind_conditioned_descriptors.py',
        'scripts/run_conditioned_package.py', 'scripts/infer_conditioned_experts_v5.py',
        'scripts/infer_harn_three_family_v2.py', 'scripts/temporal_checkpoint.py',
        'scripts/hau_action_acquisition.py', 'scripts/infer_action_independent_visual.py',
        'scripts/run_emotion_recipe_nested.py', 'scripts/infer_emotion_independent.py']
    if d10_config is not None:
        starts.append('scripts/infer_conditioned_d10.py')
    runtime = set(base.source_closure(starts))
    for record in inventory['files']:
        source = Path(record['path'])
        if source.suffix == '.py':
            for prefix in ['src', 'scripts', 'tests']:
                if prefix in source.parts:
                    relative = Path(*source.parts[source.parts.index(prefix):])
                    if (ROOT / relative).is_file():
                        runtime.add(ROOT / relative)
                    break
    # HARn verifies the original plan's relative source entries at runtime.
    top = json.loads(Path(inventory['top_descriptors']['grouped']['harn']['path']).read_text())
    plan = json.loads(Path(top['plan']['path']).read_text())
    for record in plan['sources']:
        source = ROOT / record['path']
        if base.sha(source) != record['sha256']:
            raise ValueError('HARn plan source differs from current runtime')
        runtime.add(source)
    for source in sorted(runtime):
        target = output / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (output / 'configs').mkdir(exist_ok=True)
    shutil.copy2(ROOT / 'configs/sat_independent_general_v1.json', output / 'configs/sat_independent_general_v1.json')
    profile = json.loads((ROOT / 'configs/independent_visual_four_expert_profile_v1.json').read_text())
    for condition in ['grouped', 'independent']:
        profile[condition]['features'] = {name: 'caller_current_feature' for name in profile[condition]['features']}
    manifest = {'kind': 'cuhkx_conditioned_inference_package_v1', 'asset_inventory': inventory,
        'template_profile': profile, 'files': [], 'source_asset_manifest_sha256': base.sha(path),
        'scope': 'Both conditioned four-expert routes plus current A and fixed SAT from caller features. Raw producers collected as assets but raw relocation is not implemented by this entry.'}
    if d10_config is not None:
        descriptor = json.loads(d10_config.read_text())
        if descriptor['kind'] != 'cuhkx_conditioned_D10_v1':
            raise ValueError('Unsupported D10 descriptor')
        for name in ['parent_model', 'prior_model']:
            record = descriptor[name]
            source = Path(record['path'])
            if not source.is_absolute():
                source = d10_config.parent / source
            if base.sha(source) != record['sha256']:
                raise ValueError('D10 source model identity changed')
            relative = Path('assets') / record['sha256'] / source.name
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.link(source, target)
            if base.sha(target) != record['sha256']:
                raise ValueError('D10 packaged model identity changed')
            record['path'] = str(Path('..') / relative)
        base.write(output / 'configs/conditioned_d10.json', descriptor)
        manifest['d10'] = {'descriptor': 'configs/conditioned_d10.json',
                           'source_descriptor_sha256': base.sha(d10_config),
                           'conditions': descriptor['conditions']}
    for item in sorted(output.rglob('*')):
        if item.is_file() and '__pycache__' not in item.parts:
            manifest['files'].append({'relative': str(item.relative_to(output)), 'sha256': base.sha(item), 'bytes': item.stat().st_size})
    base.write(output / 'conditioned_package.json', manifest)
    (output / 'conditioned_package.sha256').write_text(base.sha(output / 'conditioned_package.json') + '\n')
    print(json.dumps({'files': len(manifest['files']), 'bytes': sum(r['bytes'] for r in manifest['files']), 'output': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--d10-config', type=Path)
    args = parser.parse_args()
    build(args.assets.resolve(), args.output.resolve(), args.d10_config)
