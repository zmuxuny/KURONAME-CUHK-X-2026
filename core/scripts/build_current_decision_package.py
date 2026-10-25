#!/usr/bin/env python3
"""Collect immutable A/parent and explicit SAT assets for relocation testing."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from scripts import run_finalist_pipeline as base


def build(output):
    if output.exists():
        raise FileExistsError('Use a new package destination')
    descriptor_path = ROOT / 'runs/20260912_local_emotion_final_component_v2/descriptor.json'
    descriptor = json.loads(descriptor_path.read_text())
    policy_path = Path(descriptor['parent_policy']['path'])
    policy = json.loads(policy_path.read_text())
    sat_path = ROOT / 'configs/sat_min_edit_explicit_v1.json'
    sat = json.loads(sat_path.read_text())
    records, aliases = {}, {}

    def add(path, expected=None):
        record = base.entry(Path(path), expected)
        records[record['path']] = record
        aliases[str(Path(path).absolute())] = record['path']

    add(descriptor_path)
    add(sat_path)
    for record in [descriptor['parent_policy'], descriptor['normalizer'], descriptor['student_scope'],
                   *descriptor['students'].values(), *descriptor['sources'], policy['source_manifest'],
                   policy['source_model']['checkpoint'], policy['source_model']['scope']]:
        add(record['path'], record['sha256'])
    for record in sat['models']:
        add(ROOT / record['path'], record['sha256'])
    starts = ['scripts/infer_local_emotion_full_fusion_v2.py', 'scripts/apply_explicit_group_sat.py',
              'scripts/build_current_decision_package.py', 'scripts/run_current_decision_package.py',
              'scripts/run_relocated_python.py', 'src/cuhkx/__init__.py']
    sources = base.source_closure(starts)
    sources += list((ROOT / 'vendor/server_sat_v1').glob('*.py'))
    for path in sources:
        add(path)
    for name, expected in policy['recipe_source_hashes'].items():
        add(ROOT / name, expected)
    output.mkdir(parents=True)
    files = []
    locations = {}
    for original, record in sorted(records.items()):
        path = Path(original)
        # Exact historical descriptors/provenance stay distinct from runtime source.
        relative = Path('assets') / record['sha256'] / path.name
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        assert base.sha(destination) == record['sha256']
        locations[original] = str(relative)
        files.append({**record, 'relative': str(relative), 'role': 'immutable_dependency'})
    locations.update({alias: locations[canonical] for alias, canonical in aliases.items()})
    for path in sorted(set(sources)):
        relative = path.relative_to(ROOT)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        files.append({**base.entry(path), 'relative': str(relative), 'role': 'runtime_source'})
    for name, expected in policy['recipe_source_hashes'].items():
        destination = output / name
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
            files.append({**base.entry(ROOT / name), 'relative': name, 'role': 'runtime_source'})
        if base.sha(destination) != expected:
            raise ValueError('Current runtime differs from frozen parent source: ' + name)
    manifest = {'kind': 'cuhkx_relocatable_current_decision_v1', 'files': files, 'locations': locations,
                'descriptor': str(descriptor_path.resolve()), 'sat_profile': str(sat_path.resolve()),
                'scope': 'Current A plus fixed weighted SAT component. Requires caller-provided parent predictions and current features. Full raw/four-expert relocation remains a separate required delivery step.',
                'model_states_modified': False, 'data_or_feature_banks_collected': False,
                'environment': 'Caller supplies pinned research Python and separate SAT Python; no environment installation.'}
    base.write(output / 'package_manifest.json', manifest)
    (output / 'package_manifest.sha256').write_text(base.sha(output / 'package_manifest.json') + '\n')
    print(json.dumps({'files': len(files), 'bytes': sum(r['bytes'] for r in files), 'output': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output.resolve())
