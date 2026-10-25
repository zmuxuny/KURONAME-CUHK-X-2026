#!/usr/bin/env python3
"""Extract current-batch sensors with explicit schemas for wholly absent modalities."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'src')]

import numpy as np

from cuhkx.acquisition_v2 import read_batch
from cuhkx.validation import sha256_file
from scripts.reproduce_raw_test_features import SENSOR_MODULES, WINDOWS, write


def sensor_requirements(plan, condition):
    names = [name for name in plan.features if name in SENSOR_MODULES or name in WINDOWS]
    if condition == 'independent':
        names = [name for name in names if name not in WINDOWS]
    return {name: plan.feature_units[name] for name in names}


def extract_stage(name, units, root, schema, output):
    windowed = name in WINDOWS
    modality = 'Skeleton' if windowed else SENSOR_MODULES[name][1]
    module = importlib.import_module('extract_skeleton_windows' if windowed else SENSOR_MODULES[name][0])
    names = schema['stages'][name]['feature_names']
    good, absent, failures, inputs = [], [], [], []
    for unit in units:
        files = sorted(path for path in (root / unit / modality).rglob('*') if path.is_file())
        if not files:
            absent.append(unit)
            continue
        inputs.extend({'path': str(p.resolve()), 'bytes': p.stat().st_size, 'sha256': sha256_file(p)} for p in files)
        task = (str(root), unit, *WINDOWS[name][:2]) if windowed else (str(root), unit)
        record = module.extract_one(task)
        valid = bool(record[1]) if windowed else record[1] is not None
        if record[3] or not valid:
            failures.append({'unit': unit, 'error': record[3] or 'No valid feature rows'})
            continue
        if list(record[2]) != names:
            raise ValueError('Current extractor differs from the exported model-input schema: ' + name)
        good.append(record)
    if windowed:
        flat = [(unit, *window) for unit, rows, _, _ in good for window in rows]
        values = np.stack([r[4] for r in flat]).astype(np.float32) if flat else np.empty((0, len(names)), dtype=np.float32)
        arrays = {'X': values, 'units': np.asarray([r[0] for r in flat], dtype=str),
                  'starts': np.asarray([r[1] for r in flat], dtype=np.int64),
                  'stops': np.asarray([r[2] for r in flat], dtype=np.int64),
                  'centers': np.asarray([r[3] for r in flat], dtype=np.float32),
                  WINDOWS[name][2]: np.asarray(WINDOWS[name][0])}
    else:
        values = np.stack([r[1] for r in good]).astype(np.float32) if good else np.empty((0, len(names)), dtype=np.float32)
        arrays = {'X': values, 'units': np.asarray([r[0] for r in good], dtype=str)}
    if values.shape[1] != len(names) or not np.isfinite(values).all():
        raise ValueError('Sensor feature shape or values differ')
    arrays.update(feature_names=np.asarray(names), failed_units=np.asarray([r['unit'] for r in failures], dtype=str),
                  failure_messages=np.asarray([r['error'] for r in failures], dtype=str))
    path = output / (name + '.npz')
    np.savez_compressed(path, **arrays)
    for entry in inputs:
        if sha256_file(Path(entry['path'])) != entry['sha256']:
            raise ValueError('Raw sensor input changed during extraction')
    result = {'stage': name, 'requested_units': units, 'absent_modality_units': absent,
              'failed_units': failures, 'feature_rows': len(values), 'feature_columns': len(names),
              'schema_only_empty_archive': len(values) == 0 and not failures,
              'inputs': inputs, 'output_sha256': sha256_file(path),
              'missing_semantics': 'Absent units have no feature row; existing model lookup constructs its normal missing-modality mask. No fabricated observed zero vector.'}
    write(output / (name + '_report.json'), result)
    if failures:
        raise ValueError('Present sensor data failed extraction; failures retained: ' + name)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    export = sub.add_parser('export-schema')
    export.add_argument('--profile', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)
    run = sub.add_parser('extract')
    run.add_argument('--qa-dir', type=Path, required=True)
    run.add_argument('--sensor-root', type=Path, required=True)
    run.add_argument('--schema', type=Path, required=True)
    run.add_argument('--condition', choices=['explicit', 'independent'], required=True)
    run.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'export-schema':
        profile = json.loads(args.profile.read_text())['grouped']['features']
        stages = {}
        for name in [*SENSOR_MODULES, *WINDOWS]:
            path = Path(profile[name])
            path = path if path.is_absolute() else ROOT / path
            with np.load(path, allow_pickle=False) as z:
                names = z['feature_names'].astype(str).tolist()
            stages[name] = {'feature_names': names, 'schema_origin': str(path.resolve()),
                            'schema_origin_sha256': sha256_file(path),
                            'read_scope': 'Only feature_names array read; no feature values or predictions'}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as handle:
            json.dump({'kind': 'cuhkx_conditioned_sensor_schema_v1', 'stages': stages,
                       'producer_sha256': sha256_file(Path(__file__)),
                       'profile_sha256': sha256_file(args.profile)}, handle, indent=2)
        return
    _, _, plan = read_batch(args.qa_dir)
    schema = json.loads(args.schema.read_text())
    if schema['kind'] != 'cuhkx_conditioned_sensor_schema_v1':
        raise ValueError('Unexpected sensor schema')
    args.output.mkdir(parents=True, exist_ok=False)
    requirements = sensor_requirements(plan, args.condition)
    source = args.output / 'source_snapshot'
    source.mkdir()
    paths = [Path(__file__), ROOT / 'scripts/reproduce_raw_test_features.py',
             *[ROOT / 'scripts' / (v[0] + '.py') for v in SENSOR_MODULES.values()],
             ROOT / 'scripts/extract_skeleton_windows.py', ROOT / 'src/cuhkx/acquisition_v2.py',
             *[ROOT / 'src/cuhkx' / (name + '.py') for name in ['imu', 'radar', 'skeleton', 'skeleton_windows']]]
    for path in paths:
        (source / path.name).write_bytes(path.read_bytes())
    write(source / 'manifest.json', [{'path': str(p), 'sha256': sha256_file(p)} for p in paths])
    (source / 'schema.json').write_bytes(args.schema.read_bytes())
    write(args.output / 'invocation.json', {'argv': sys.argv, 'cwd': str(Path.cwd()), 'python': sys.executable,
          'python_version': platform.python_version(), 'platform': platform.platform(),
          'numpy': np.__version__, 'packages': {name: importlib.metadata.version(name) for name in
              ['numpy', 'pandas', 'scipy', 'scikit-learn']},
          'thread_environment': {name: os.environ.get(name) for name in
              ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']},
          'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          'git_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
          'schema_sha256': sha256_file(args.schema), 'batch_plan': plan.to_dict(),
          'requirements': requirements, 'condition': args.condition, 'new_fits': 0})
    (source / 'pip_freeze.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze', '--all'], text=True))
    subprocess.run(['git', 'archive', '--format=tar.gz',
                    '--output=' + str(source.resolve() / 'committed_dependencies.tar.gz'), 'HEAD'], cwd=ROOT, check=True)
    reports = {}
    for name, units in requirements.items():
        reports[name] = extract_stage(name, units, args.sensor_root.resolve(), schema, args.output)
        print(name, 'rows', reports[name]['feature_rows'], 'absent', len(reports[name]['absent_modality_units']), flush=True)
    write(args.output / 'report.json', {'status': 'complete_raw_current_batch_sensors', 'stages': reports,
          'source_scope': 'Only supplied current-question units under caller sensor root; no training or historical feature values',
          'remaining': 'Visual producers, fullfour-expert routing and SAT run separately; this is the CPU sensor input stage'})


if __name__ == '__main__':
    main()
