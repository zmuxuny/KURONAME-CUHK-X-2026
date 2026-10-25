#!/usr/bin/env python3
"""Build a standalone raw/conditioned package with explicit nested model files."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from scripts import build_conditioned_package as feature
from scripts import run_finalist_pipeline as base


def build(assets, output, d10_config=None):
    feature.build(assets, output, d10_config)
    manifest_path = output / 'conditioned_package.json'
    package = json.loads(manifest_path.read_text())
    inventory = package['asset_inventory']
    by_sha = {record['sha256']: output / record['relative'] for record in inventory['files']}
    starts = ['scripts/build_raw_conditioned_package.py', 'scripts/run_raw_conditioned_package.py',
        'scripts/produce_conditioned_raw_inputs.py', 'scripts/produce_conditioned_sensor_inputs.py',
        'scripts/relocated_sitecustomize.py', *base.EXTRACTOR_SOURCES,
        'scripts/extract_r34_midfeatures.py', 'scripts/extract_videomae_windows.py']
    for source in base.source_closure(starts):
        target = output / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    bindings = []

    def link_model(digest, directory, name):
        target = output / directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            os.link(by_sha[digest], target)
        if base.sha(target) != digest:
            raise ValueError('Nested pretrained model identity changed')

    names = ['finalist_acquisition_v2.json', 'generic_hau_r2plus1d_windows8_v1.json',
             'r34_midfeatures_v1.json', 'videomae_ssv2_harn_test_whole_v1.json', 'videomae_ssv2_hau_test_windows_v1.json']
    for name in names:
        source = ROOT / 'configs' / name
        config = json.loads(source.read_text())
        if name == 'finalist_acquisition_v2.json':
            for filename, digest in config['kuno_files'].items():
                link_model(digest, 'pretrained/kuno', filename)
            config['kuno_model_dir'] = 'pretrained/kuno'
            config['generic_weights'] = str(by_sha[config['generic_weights_sha256']].relative_to(output))
            shutil.copy2(ROOT / config['dependencies_lock'], output / config['dependencies_lock'])
        elif name == 'r34_midfeatures_v1.json':
            config['weights'] = str(by_sha[config['weights_sha256']].relative_to(output))
            # R34 preparation verifies the upstream architecture and license
            # records as actual files before extracting any features.
            def provenance(path, expected=None):
                source = base.resolve(path)
                digest = base.sha(source)
                if expected is not None and digest != expected:
                    raise ValueError('R34 architecture provenance identity changed')
                relative = Path('delivery/provenance/r34') / source.name
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                if base.sha(target) != digest:
                    raise ValueError('R34 packaged provenance identity changed')
                return str(relative)

            config['provenance_files'] = [provenance(path) for path in config['provenance_files']]
            architecture = config['architecture_reference']
            architecture['path'] = provenance(architecture['path'], architecture['sha256'])
        if 'model_directory' in config:
            for record in config['model_files']:
                link_model(record['sha256'], 'pretrained/videomae', record['name'])
            config['model_directory'] = 'pretrained/videomae'
        target = output / 'configs' / name
        base.write(target, config)
        bindings.append({'name': name, 'original_sha256': base.sha(source), 'runtime_sha256': base.sha(target),
                         'change': 'Pretrained model and provenance locations only; file bytes and extraction parameters unchanged.'})
    schema = next(record for record in inventory['files'] if 'sensor_column_schema' in record['roles'])
    shutil.copy2(by_sha[schema['sha256']], output / 'configs/current_sensor_schema.json')
    guard = output / 'runtime_guard'
    guard.mkdir()
    shutil.copy2(ROOT / 'scripts/relocated_sitecustomize.py', guard / 'sitecustomize.py')
    for source in [*(ROOT / 'delivery/licenses').glob('*'), ROOT / 'delivery/THIRD_PARTY_NOTICES.md', ROOT / 'delivery/model_sources.json']:
        if source.is_file():
            target = output / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    # Existing producers honestly record a Git source snapshot. Give this
    # generated package its own source-only repository, without model assets.
    (output / '.gitignore').write_text('/*\n!/scripts/\n!/src/\n!/configs/\n!/vendor/\n!/runtime_guard/\n!/delivery/\n!/*.sh\n!/*.lock\n!/.gitignore\n')
    subprocess.run(['git', 'init', '-q', '--initial-branch=package'], cwd=output, check=True)
    subprocess.run(['git', 'add', '.'], cwd=output, check=True)
    subprocess.run(['git', '-c', 'user.name=CUHK-X local package', '-c', 'user.email=local-package@invalid',
                    '-c', 'core.hooksPath=/dev/null', 'commit', '-q', '-m', 'Freeze packaged source and path-bound extraction configs'], cwd=output, check=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=output, text=True).strip()
    package['raw_support'] = {'config_bindings': bindings, 'source_git_commit': commit,
        'source_git_scope': 'Generated source-only package repository; not the original model-training commit.',
        'scope': 'Raw extraction entry included; actual relocated original-media validation still required.'}
    package['files'] = []
    for path in sorted(output.rglob('*')):
        if path.is_file() and '.git' not in path.relative_to(output).parts and '__pycache__' not in path.parts and path.name not in ['conditioned_package.json', 'conditioned_package.sha256']:
            package['files'].append({'relative': str(path.relative_to(output)), 'sha256': base.sha(path), 'bytes': path.stat().st_size})
    base.write(manifest_path, package)
    (output / 'conditioned_package.sha256').write_text(base.sha(manifest_path) + '\n')
    print(json.dumps({'files': len(package['files']), 'logical_bytes': sum(record['bytes'] for record in package['files']), 'source_git_commit': commit}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--d10-config', type=Path)
    args = parser.parse_args()
    build(args.assets.resolve(), args.output.resolve(), args.d10_config)
