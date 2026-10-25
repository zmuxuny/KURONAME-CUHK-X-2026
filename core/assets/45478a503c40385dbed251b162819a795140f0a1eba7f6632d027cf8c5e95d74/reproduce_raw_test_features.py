#!/usr/bin/env python3
"""Replay existing feature extractors from Testing media without refitting models."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
SENSOR_ROOT = ROOT / 'data/media/nonvisual/LMT_(IMU,Radar,Skeleton)'
MEDIA_ROOT = ROOT / 'data/media/visual'
MODEL_ROOT = ROOT / 'data/external/k-kuno-lb-0-711-r2plus1d-model'
SOURCES = [
    'scripts/reproduce_raw_test_features.py', 'scripts/extract_imu_v2_features.py',
    'scripts/extract_skeleton_features.py', 'scripts/extract_radar_features.py',
    'scripts/extract_skeleton_windows.py', 'scripts/extract_kuno_temporal_logits.py',
    'scripts/extract_kuno_transfer_logits.py', 'scripts/extract_video_embeddings.py',
    'scripts/extract_generic_hau_windows.py', 'scripts/pack_emotion_generic_visual.py',
    'scripts/infer_experts.py', 'scripts/infer_refined_experts.py',
    'src/cuhkx/imu.py', 'src/cuhkx/skeleton.py', 'src/cuhkx/radar.py',
    'src/cuhkx/skeleton_windows.py', 'src/cuhkx/video.py',
    'src/cuhkx/generic_video_windows.py', 'src/cuhkx/emotion_visual_nested.py',
    'src/cuhkx/validation.py', 'src/cuhkx/metadata.py',
]
SENSOR_MODULES = {'imu': ('extract_imu_v2_features', 'IMU'),
                  'skeleton': ('extract_skeleton_features', 'Skeleton'),
                  'radar': ('extract_radar_features', 'Radar')}
WINDOWS = {'windows16': (16, 6, 'window_size'),
           'windows32': (32, 12, 'nominal_window_size'),
           'windows48': (48, 16, 'nominal_window_size')}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    def default(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=default) + '\n')


def now():
    return datetime.now(UTC).isoformat()


def reference_contract(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: {'shape': list(archive[key].shape), 'dtype': str(archive[key].dtype)}
                  for key in archive.files}
        units = sorted(set(archive['units'].astype(str)))
    if not all(unit.startswith('Testing/') for unit in units):
        raise ValueError('Reference archive must be Testing-only')
    return {'path': str(path.resolve()), 'sha256': sha(path), 'bytes': path.stat().st_size,
            'arrays': arrays, 'units': units}


def freeze(run):
    if run.exists():
        raise FileExistsError('New raw-reproduction directory required')
    run.mkdir(parents=True)
    for name in ['features', 'reports', 'logs', 'source_snapshot']:
        (run / name).mkdir()
    config = json.loads((ROOT / 'configs/checkpoint_inference_v2.json').read_text())
    refined = json.loads((ROOT / 'configs/checkpoint_inference_emotion_v6.json').read_text())
    from cuhkx.metadata import feature_unit
    qa_path = ROOT / 'data/raw/kaggle/test_qa.csv'
    qa = pd.read_csv(qa_path, usecols=['qa_id', 'source', 'path'], dtype=str, keep_default_na=False)
    units = sorted({feature_unit(path, training=False) for path in qa.path})
    hau_units = sorted({feature_unit(path, training=False) for path in qa.loc[qa.source.eq('HAU'), 'path']})
    assert len(qa) == 682 and len(units) == 208 and len(hau_units) == 144
    sources = {name: sha(ROOT / name) for name in SOURCES}
    for name in SOURCES:
        target = run / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    weights = [MODEL_ROOT / name for name in ['ensemble_packed.pt', 'ig65m_models.py', 'yolo11n.pt']]
    weights += [Path('/root/.cache/torch/hub/checkpoints') / name
                for name in ['r3d_18-b3b3357e.pth', 'r2plus1d_18-91a641e6.pth']]
    dependencies = {str(path): sha(path) for path in weights}
    raw_files = sorted(path for path in (SENSOR_ROOT / 'Testing').rglob('*') if path.is_file())
    raw_files += sorted(path for path in (MEDIA_ROOT / 'Testing').rglob('*.mp4')
                        if path.parent.name in ['Depth', 'IR', 'Depth_Color'])
    raw = []
    for index, path in enumerate(raw_files, start=1):
        raw.append({'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': sha(path)})
        if index % 5000 == 0 or index == len(raw_files):
            print(f'raw_hashed={index}/{len(raw_files)}', flush=True)
    write(run / 'raw_input_manifest.json', raw)
    fingerprints = {str((ROOT / value).resolve()): sha(ROOT / value)
                    for c in [config, refined] for value in c['checkpoints'].values()}
    fingerprints.update({str(ROOT / name): sha(ROOT / name) for name in
                         ['configs/checkpoint_inference_v2.json', 'configs/checkpoint_inference_emotion_v6.json',
                          'configs/generic_hau_r2plus1d_windows8_v1.json']})
    fingerprints[str(qa_path)] = sha(qa_path)
    fingerprints[str(SENSOR_ROOT / 'manifest_nonvisual.csv')] = sha(SENSOR_ROOT / 'manifest_nonvisual.csv')
    contract = {'created_utc': now(), 'scope': 'Existing raw Testing media to nine checkpoint features; optional tenth generic emotion visual modality',
                'units': units, 'hau_units': hau_units, 'sources': sources, 'weights': dependencies,
                'fixed_inputs': fingerprints, 'raw_manifest_sha256': sha(run / 'raw_input_manifest.json'),
                'reference_features': {name: reference_contract(ROOT / value) for name, value in config['features'].items()},
                'reference_emotion_visual': reference_contract(ROOT / refined['features']['emotion_visual']),
                'base_inference_config': config, 'refined_inference_config': refined,
                'window_specs': WINDOWS, 'detector': {'device': 'cpu', 'probes': 8, 'confidence': 0.25},
                'logits': {'logits8': {'segments': 8, 'window_seconds': 2.5},
                           'logits16': {'segments': 16, 'window_seconds': 1.25}},
                'threads': {'workers': 1, 'torch': 1, 'opencv': 1},
                'comparison': {'always_report_exact_equality': True, 'diagnostic_atol': 1e-6, 'diagnostic_rtol': 1e-5,
                               'string_width': 'report dtype separately; compare actual string values'},
                'estimated_new_bytes_upper_bound': 200_000_000, 'estimated_minutes': [25, 45],
                'versions': {name: importlib.metadata.version(name) for name in
                             ['numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'torch', 'torchvision', 'ultralytics']},
                'restrictions': 'No refits, no downloads, no old-feature links/copies as extraction output. Old archives are comparison inputs only. No training QA is a raw inference input.'}
    write(run / 'frozen_contract.json', contract)
    print(json.dumps({'status': 'contract_frozen', 'raw_files': len(raw), 'raw_bytes': sum(r['bytes'] for r in raw),
                      'sources': len(sources), 'sha256': sha(run / 'frozen_contract.json')}, indent=2))


def check(run):
    contract = json.loads((run / 'frozen_contract.json').read_text())
    for name, digest in contract['sources'].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'Frozen source changed: {name}')
    for group in ['weights', 'fixed_inputs']:
        for path, digest in contract[group].items():
            if sha(path) != digest:
                raise ValueError(f'Frozen dependency changed: {path}')
    return contract


def compare(actual_path, reference):
    results = {}
    with np.load(actual_path, allow_pickle=False) as actual, np.load(reference['path'], allow_pickle=False) as expected:
        if sha(reference['path']) != reference['sha256']:
            raise ValueError('Reference archive changed during reproduction')
        for key in expected.files:
            if key not in actual:
                results[key] = {'missing': True, 'exact_values': False}
                continue
            a, e = actual[key], expected[key]
            same_shape = a.shape == e.shape
            numeric = a.dtype.kind in 'biufc' and e.dtype.kind in 'biufc'
            equal = bool(np.array_equal(a, e, equal_nan=True)) if numeric else bool(np.array_equal(a, e))
            item = {'shape': list(a.shape), 'expected_shape': list(e.shape), 'dtype': str(a.dtype),
                    'expected_dtype': str(e.dtype), 'same_dtype': a.dtype == e.dtype, 'exact_values': equal}
            if numeric and same_shape:
                item['allclose_atol1e6_rtol1e5'] = bool(np.allclose(a, e, atol=1e-6, rtol=1e-5, equal_nan=True))
                if a.size:
                    difference = np.abs(a.astype(np.float64) - e.astype(np.float64))
                    item['max_absolute_difference'] = float(np.nanmax(difference))
                    item['different_elements'] = int(np.count_nonzero(~((a == e) | (np.isnan(a) & np.isnan(e)))))
            results[key] = item
        actual_units = sorted(set(actual['units'].astype(str)))
        extras = sorted(set(actual.files) - set(expected.files))
    return {'actual': str(actual_path), 'actual_sha256': sha(actual_path), 'reference': reference,
            'arrays': results, 'all_reference_arrays_exact_values': all(r['exact_values'] for r in results.values()),
            'same_unit_coverage': actual_units == reference['units'], 'extra_metadata_arrays': extras,
            'missing_reference_units': sorted(set(reference['units']) - set(actual_units)),
            'extra_units': sorted(set(actual_units) - set(reference['units']))}


def sensors(stage, run, contract):
    manifest = pd.read_csv(SENSOR_ROOT / 'manifest_nonvisual.csv', dtype=str, keep_default_na=False)
    modality = SENSOR_MODULES[stage][1] if stage in SENSOR_MODULES else 'Skeleton'
    units = manifest.loc[manifest.unit.str.startswith('Testing/') &
                         manifest.modalities.str.split(';').map(lambda value: modality in value), 'unit'].tolist()
    if not set(units) <= set(contract['units']):
        raise ValueError('Discovered sensor units outside official Testing scope')
    windowed = stage in WINDOWS
    module = importlib.import_module('extract_skeleton_windows' if windowed else SENSOR_MODULES[stage][0])
    records = []
    for index, unit in enumerate(units, start=1):
        task = (str(SENSOR_ROOT), unit, *WINDOWS[stage][:2]) if windowed else (str(SENSOR_ROOT), unit)
        records.append(module.extract_one(task))
        if index % 50 == 0 or index == len(units):
            print(f'{stage} raw_units={index}/{len(units)} failures={sum(bool(r[3]) for r in records)}', flush=True)
    good = [r for r in records if (bool(r[1]) if windowed else r[1] is not None)]
    bad = [{'unit': r[0], 'error': r[3]} for r in records if r[3]]
    if not good or any(r[2] != good[0][2] for r in good):
        raise ValueError('Extraction failed or changed feature schema')
    if windowed:
        flat = [(unit, *window) for unit, rows, _, _ in good for window in rows]
        arrays = {'X': np.stack([r[4] for r in flat]).astype(np.float32), 'units': np.asarray([r[0] for r in flat]),
                  'starts': np.asarray([r[1] for r in flat]), 'stops': np.asarray([r[2] for r in flat]),
                  'centers': np.asarray([r[3] for r in flat], dtype=np.float32),
                  WINDOWS[stage][2]: np.asarray(WINDOWS[stage][0])}
    else:
        arrays = {'X': np.stack([r[1] for r in good]).astype(np.float32), 'units': np.asarray([r[0] for r in good])}
    arrays.update(feature_names=np.asarray(good[0][2]), failed_units=np.asarray([r['unit'] for r in bad]),
                  failure_messages=np.asarray([r['error'] for r in bad]))
    output = run / 'features' / f'{stage}.npz'
    np.savez_compressed(output, **arrays)
    return {'output': str(output), 'raw_units_attempted': len(units), 'failed_units': bad,
            'absent_raw_modality_units': sorted(set(contract['units']) - set(units)),
            'missing_output_units': sorted(set(contract['units']) - {r[0] for r in good}),
            'adapter': 'Original extractor.extract_one on only raw Testing units, single process',
            'timing_note': 'Skeleton windows preserve original frame-index centers, not a claimed cross-sensor physical synchronization.'}


def roi(run, contract):
    import cv2
    import torch
    from extract_kuno_transfer_logits import detect_windows, discover_units
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    records = discover_units(MEDIA_ROOT, ('Testing/',))
    if [r['unit'] for r in records] != contract['units']:
        raise ValueError('Raw IR coverage differs from Testing contract')
    windows, stats = detect_windows(records, MODEL_ROOT / 'yolo11n.pt', 8, contract['detector']['device'])
    output = run / 'person_windows.json'
    write(output, {'units': contract['units'], 'windows': windows, 'stats': stats,
                   'provenance': 'YOLO inference rerun from raw Testing IR frames; no cached crops used'})
    old = json.loads((ROOT / 'artifacts/visual/kuno_test_v1.windows.json').read_text())['windows']
    differences = []
    for record in records:
        unit = record['unit']
        actual, expected = np.asarray(windows[unit]), np.asarray(old[unit])
        if not np.array_equal(actual, expected):
            differences.append({'unit': unit, 'actual': actual, 'reference': expected,
                                'max_abs_difference': float(np.max(np.abs(actual - expected)))})
    return {'output': str(output), 'sha256': sha(output), 'stats': stats,
            'different_normalized_rois': differences, 'all_rois_exact': not differences,
            'comparison_cache_only': str(ROOT / 'artifacts/visual/kuno_test_v1.windows.json')}


def existing_visual(stage, run, contract):
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    def forbidden_download(*args, **kwargs):
        raise RuntimeError('This reproduction permits cached local weights only')
    torch.hub.download_url_to_file = forbidden_download
    output = run / 'features' / f'{stage}.npz'
    if stage in ['logits8', 'logits16']:
        spec = contract['logits'][stage]
        entry = 'extract_kuno_temporal_logits'
        arguments = ['--media-root', str(MEDIA_ROOT), '--model-dir', str(MODEL_ROOT),
                     '--output', str(output), '--prefixes', 'Testing/', '--unit-regex',
                     r'/LM_test_(?:00(?:6[5-9]|[7-9][0-9])|01[0-9]{2}|020[0-8])$',
                     '--frames', '16', '--segments', str(spec['segments']),
                     '--window-seconds', str(spec['window_seconds']), '--batch-size', '2',
                     '--windows-json', str(run / 'person_windows.json'), '--device', 'cuda']
    else:
        entry = 'extract_video_embeddings'
        arguments = ['--media-root', str(MEDIA_ROOT), '--output', str(output),
                     '--prefixes', 'Testing/', '--modalities', 'Depth', 'IR',
                     '--backbone', 'r3d_18', '--frames', '16', '--batch-size', '8',
                     '--workers', '1', '--device', 'cuda', '--windows-json', str(run / 'person_windows.json')]
    sys.argv = [str(ROOT / 'scripts' / (entry + '.py')), *arguments]
    code = importlib.import_module(entry).main()
    if code:
        raise RuntimeError(f'Original visual extractor exited {code}')
    return {'output': str(output), 'original_entrypoint_argv': sys.argv, 'original_exit_code': code,
            'original_report': str(output.with_suffix('.json'))}


def generic_visual(run, contract):
    from extract_generic_hau_windows import archive_arrays, embeddings, setup_runtime

    from cuhkx.generic_video_windows import (
        sample_preprocessed,
        temporal_alignment,
        video_metadata,
        window_timing,
    )
    config = json.loads((ROOT / 'configs/generic_hau_r2plus1d_windows8_v1.json').read_text())
    for key in ['train_csv', 'expected_training_units', 'expected_training_qa_units']:
        del config[key]
    config.update(person_window_caches=[str(run / 'person_windows.json')],
                  output_npz=str(run / 'features/generic_windows.npz'), scope='Testing HAU only')
    crops = json.loads((run / 'person_windows.json').read_text())['windows']
    records, alignment = [], {}
    for unit in contract['hau_units']:
        metadata = {}
        for modality in config['modalities']:
            path = MEDIA_ROOT / unit / modality / (modality + '.mp4')
            meta = video_metadata(path) if path.is_file() else None
            metadata[modality] = meta
            records.append({'unit': unit, 'modality': modality, 'available': meta is not None,
                            'path': str(path) if meta is not None else None, 'roi': crops[unit],
                            'sha256': sha(path) if meta is not None else None,
                            'metadata': meta, 'timing': window_timing(meta, config) if meta is not None else None})
        alignment[unit] = temporal_alignment(metadata['Depth'], metadata['IR'], config)
    inputs = {'units': contract['hau_units'], 'records': records, 'alignment': alignment,
              'qa_required_units': {'test': contract['hau_units']}}
    subrun = run / 'generic_extraction'
    subrun.mkdir()
    write(subrun / 'frozen_config.json', {'config': config, 'created_utc': now(), 'source_hashes': contract['sources']})
    write(subrun / 'input_manifest.json', inputs)
    arrays = archive_arrays(inputs, config)
    model = setup_runtime(config)
    started = time.perf_counter()
    for index, record in enumerate(records):
        if record['available']:
            tensor = sample_preprocessed(record['path'], record['roi'], config)
            arrays['X'][index * 8:(index + 1) * 8] = embeddings(model, tensor, config)
        if (index + 1) % 20 == 0 or index + 1 == len(records):
            print(f'generic_raw_videos={index + 1}/{len(records)}', flush=True)
    np.savez_compressed(config['output_npz'], **arrays)
    write(subrun / 'report.json', {'status': 'Testing_generic_windows_extracted', 'sha256': sha(config['output_npz']),
                                  'output': config['output_npz'], 'shape': list(arrays['X'].shape),
                                  'runtime_seconds': time.perf_counter() - started,
                                  'unaligned_units': [u for u, value in alignment.items() if not value['alignment_ok']]})
    arguments = ['--windows', config['output_npz'], '--extraction-run', str(subrun),
                 '--output', str(run / 'features/emotion_visual.npz'), '--run-dir', str(run / 'generic_aggregation')]
    sys.argv = [str(ROOT / 'scripts/pack_emotion_generic_visual.py'), *arguments]
    importlib.import_module('pack_emotion_generic_visual').main()
    return {'output': str(run / 'features/emotion_visual.npz'), 'extraction_run': str(subrun),
            'aggregation_argv': sys.argv}


def run_stage(run, stage):
    contract = check(run)
    report_path = run / 'reports' / f'{stage}.json'
    if report_path.exists() or (run / 'features' / f'{stage}.npz').exists():
        raise FileExistsError('Refusing to overwrite a prior stage output')
    started, began = time.perf_counter(), now()
    if stage in SENSOR_MODULES or stage in WINDOWS:
        result = sensors(stage, run, contract)
    elif stage == 'roi':
        result = roi(run, contract)
    elif stage == 'emotion_visual':
        result = generic_visual(run, contract)
    else:
        result = existing_visual(stage, run, contract)
    if stage != 'roi':
        reference = contract['reference_emotion_visual'] if stage == 'emotion_visual' else contract['reference_features'][stage]
        result['comparison'] = compare(Path(result['output']), reference)
    check(run)
    result.update(stage=stage, started_utc=began, completed_utc=now(), seconds=time.perf_counter() - started,
                  status='extraction_completed_and_compared', command=[sys.executable, str(Path(__file__).resolve()),
                          'run', '--run-dir', str(run), '--stage', stage], exit_code=0)
    write(report_path, result)
    print(json.dumps({'stage': stage, 'status': result['status'], 'seconds': result['seconds'],
                      'all_arrays_exact': result.get('comparison', {}).get('all_reference_arrays_exact_values'),
                      'report': str(report_path)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze', 'run'])
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--stage', choices=[*SENSOR_MODULES, *WINDOWS, 'roi', 'logits8', 'logits16', 'harn_visual', 'emotion_visual'])
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if args.command == 'freeze':
        freeze(run)
    elif args.stage:
        run_stage(run, args.stage)
    else:
        raise ValueError('run requires a stage')


if __name__ == '__main__':
    main()
