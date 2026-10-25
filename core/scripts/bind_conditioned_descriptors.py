"""Bind verified descriptor graphs without accessing original filesystem paths."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def bind(root, runtime):
    manifest_path = root / 'conditioned_package.json'
    if sha(manifest_path) != (root / 'conditioned_package.sha256').read_text().strip():
        raise ValueError('Conditioned package manifest changed')
    package = json.loads(manifest_path.read_text())
    if package['kind'] != 'cuhkx_conditioned_inference_package_v1':
        raise ValueError('Unsupported conditioned package')
    for record in package['files']:
        path = (root / record['relative']).resolve()
        if not path.is_relative_to(root) or sha(path) != record['sha256']:
            raise ValueError('Packaged file changed: ' + str(path))
    inventory = package['asset_inventory']
    records = {record['path']: record for record in inventory['files']}
    by_hash = {record['sha256']: root / record['relative'] for record in inventory['files']}
    descriptors = {records[path]['sha256']: path for path in inventory['descriptors']}
    bound, bindings = {}, []
    runtime.mkdir(parents=True, exist_ok=False)

    def reference(record, recursive=False):
        digest = record['sha256']
        target = descriptor(digest) if recursive and digest in descriptors else by_hash[digest]
        if not (recursive and digest in descriptors) and sha(target) != digest:
            raise ValueError('Model/source identity changed')
        result = {**record, 'path': str(target), 'sha256': sha(target)}
        if 'bytes' in result:
            result['bytes'] = target.stat().st_size
        return result

    def descriptor(digest):
        if digest in bound:
            return bound[digest]
        original = by_hash[digest]
        value = json.loads(original.read_text())
        directory = runtime / digest
        directory.mkdir()
        destination = directory / Path(descriptors[digest]).name
        for name in ['models', 'classifiers']:
            if name in value:
                group = value[name]
                value[name] = {key: reference(record, True) for key, record in group.items()} if isinstance(group, dict) else [reference(record, True) for record in group]
        if 'model' in value:
            model = value['model']
            # The fixed independent action loader uses model.joblib by basename.
            if destination.name == 'manifest.json':
                source = by_hash[model['sha256']]
                shutil.copy2(source, directory / Path(model['path']).name)
            value['model'] = reference(model, True)
        if 'parent_checkpoint' in value:
            value['parent_checkpoint'] = reference(value['parent_checkpoint'], True)
        for name in ['r34_checkpoint', 'videomae_model', 'nested_report', 'plan', 'source_manifest']:
            if name in value:
                value[name] = reference(value[name])
        if isinstance(value.get('sources'), list):
            value['sources'] = [reference(record) for record in value['sources']]
        for name, record in value.get('source_model', {}).items():
            if isinstance(record, dict) and {'path', 'sha256'} <= record.keys():
                value['source_model'][name] = reference(record)
        destination.write_text(json.dumps(value, indent=2) + '\n')
        for suffix in ['.json.sha256', '.sha256']:
            destination.with_suffix(suffix).write_text(sha(destination) + '\n')
        bindings.append({'original_sha256': digest, 'runtime_path': str(destination), 'runtime_sha256': sha(destination),
                         'change': 'Descriptor references and descriptor hashes only; fitted states unchanged.'})
        bound[digest] = destination
        return destination

    profile = copy.deepcopy(package['template_profile'])
    for condition, tops in inventory['top_descriptors'].items():
        for name, info in tops.items():
            target = descriptor(records[info['path']]['sha256'])
            profile[condition]['checkpoints'][name] = str(target.parent if info['input_is_directory'] else target)
    (runtime / 'binding.json').write_text(json.dumps({'package_sha256': sha(manifest_path), 'bindings': bindings,
        'fitted_states_modified': False, 'original_path_reads': False}, indent=2) + '\n')
    return profile
