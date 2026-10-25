#!/usr/bin/env python3
"""Hash the two conditioned expert model graphs and raw producer dependencies."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from scripts import run_finalist_pipeline as base


def collect(output):
    output.mkdir(parents=True, exist_ok=False)
    files, aliases, descriptors, edges = {}, {}, {}, []

    def add(path, expected=None, role='model_dependency'):
        path = Path(path).absolute()
        if path.suffix.lower() in {'.csv', '.npz', '.npy', '.mp4', '.avi'}:
            raise ValueError('Model closure must not collect observations or prediction tables: ' + str(path))
        record = base.entry(path, expected)
        canonical = record['path']
        aliases[str(path)] = canonical
        if canonical not in files:
            files[canonical] = {**record, 'roles': []}
        if role not in files[canonical]['roles']:
            files[canonical]['roles'].append(role)
        return Path(canonical)

    def reference(record, parent, role, recursive=False):
        value = Path(record['path'])
        if not value.is_absolute():
            value = parent / value
        path = add(value, record['sha256'], role)
        edges.append({'descriptor_parent': str(parent), 'reference': record['path'],
                      'resolved': str(path), 'recursive_descriptor': recursive and path.suffix == '.json'})
        if recursive and path.suffix == '.json':
            descriptor(path)

    def descriptor(path):
        path = add(path, role='runtime_descriptor')
        if str(path) in descriptors:
            return
        payload = json.loads(path.read_text())
        descriptors[str(path)] = payload.get('kind', payload.get('model_kind', 'typed_model_descriptor'))
        for suffix in ['.json.sha256', '.sha256']:
            sidecar = path.with_suffix(suffix)
            if sidecar.is_file():
                add(sidecar, role='descriptor_sidecar')
        for name in ['source_hashes', 'recipe_source_hashes', 'original_source_hashes', 'code']:
            for value, digest in payload.get(name, {}).items():
                add(ROOT / value, digest, 'provenance_source')
        sources = payload.get('sources', [])
        if isinstance(sources, dict):
            for value, digest in sources.items():
                add(ROOT / value, digest, 'provenance_source')
        else:
            for record in sources:
                reference(record, path.parent, 'provenance_source')
        for name in ['models', 'classifiers']:
            records = payload.get(name, {})
            for record in records.values() if isinstance(records, dict) else records:
                reference(record, path.parent, 'fitted_model', recursive=True)
        if 'model' in payload:
            reference(payload['model'], path.parent, 'fitted_model', recursive=True)
        for name in ['parent_checkpoint']:
            if name in payload:
                reference(payload[name], path.parent, 'runtime_descriptor', recursive=True)
        for name in ['r34_checkpoint', 'videomae_model', 'nested_report', 'plan', 'source_manifest']:
            if name in payload:
                reference(payload[name], path.parent, 'required_opaque_provenance' if name in ['nested_report', 'plan', 'source_manifest'] else 'fitted_model')
        for record in payload.get('source_model', {}).values():
            if isinstance(record, dict) and {'path', 'sha256'} <= record.keys():
                reference(record, path.parent, 'final_parent_dependency')

    profile_path = ROOT / 'configs/independent_visual_four_expert_profile_v1.json'
    profile = json.loads(profile_path.read_text())
    add(profile_path, role='source_profile')
    tops = {}
    for condition in ['grouped', 'independent']:
        tops[condition] = {}
        for name, value in profile[condition]['checkpoints'].items():
            path = ROOT / value
            if path.is_dir():
                path = path / 'manifest.json'
            canonical = add(path, role='top_' + condition + '_' + name)
            descriptor(canonical)
            tops[condition][name] = {'path': str(canonical), 'input_is_directory': (ROOT / value).is_dir()}
    raw_path = ROOT / 'runs/20260914_conditioned_raw_inputs_v2/contract.json'
    raw = json.loads(raw_path.read_text())
    for record in raw['dependencies']:
        add(record['path'], record['sha256'], 'raw_extractor_dependency')
    # These newer visual producers declare their checkpoints in stage configs,
    # beyond the legacy top-level extraction dependency list.
    for name in ['configs/r34_midfeatures_v1.json', 'configs/videomae_ssv2_harn_test_whole_v1.json',
                 'configs/videomae_ssv2_hau_test_windows_v1.json']:
        config = json.loads((ROOT / name).read_text())
        if 'weights' in config:
            add(ROOT / config['weights'], config['weights_sha256'], 'raw_nested_pretrained_weights')
        if 'model_directory' in config:
            for record in config['model_files']:
                add(Path(config['model_directory']) / record['name'], record['sha256'], 'raw_nested_pretrained_model')
    add(ROOT / 'runs/20260914_conditioned_sensor_inputs_v1/schema.json', role='sensor_column_schema')
    for record in raw['sources']:
        add(record['path'], record['sha256'], 'raw_extractor_source')
    extra = ['scripts/infer_conditioned_experts_v5.py', 'scripts/infer_conditioned_current_raw_v2.py',
             'scripts/produce_conditioned_raw_inputs.py', 'scripts/collect_conditioned_delivery_inputs.py']
    for path in base.source_closure(extra):
        add(path, role='current_source_closure')
    base.write(output / 'inventory.json', {'status': 'verified_two_condition_dependency_inventory',
        'files': list(files.values()), 'aliases': aliases, 'descriptors': descriptors, 'edges': edges,
        'top_descriptors': tops, 'profile_features': 'Only feature-key schemas; actual caller banks must be generated separately.',
        'source_profile_sha256': base.sha(profile_path), 'raw_contract_sha256': base.sha(raw_path),
        'scope': 'Verified dependency inventory for full conditioned delivery preparation, not yet a runnable relocated four-expert package. Historical provenance manifests treated as opaque; their training inputs are not traversed or collected. No observations/prediction tables or fitting.'})
    base.write(output / 'report.json', {'files': len(files), 'bytes': sum(v['bytes'] for v in files.values()),
        'descriptors': len(descriptors), 'reference_edges': len(edges), 'top_conditions': list(tops),
        'inventory_sha256': base.sha(output / 'inventory.json'), 'new_fits': 0,
        'scope': 'Model/source identity census; alias mapping is retained for subsequent runtime descriptor binding. Full raw/four-expert relocation still pending.'})
    (output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print((output / 'report.json').read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    collect(parser.parse_args().output.resolve())
