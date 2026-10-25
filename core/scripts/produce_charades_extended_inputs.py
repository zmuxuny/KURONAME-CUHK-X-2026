#!/usr/bin/env python3
"""Produce fixed expanded-source inputs without retaining every dense depth map."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import pandas as pd
from produce_charades_training_inputs import raw_source, rgb_job, save_arrays

from cuhkx.charades_depth import infer, load_model
from cuhkx.charades_source import crops, relative_input
from cuhkx.harn_tail import configure_torch
from cuhkx.validation import sha256_file


def write(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def reserve(output):
    if shutil.disk_usage(output).free < 20 * 2**30:
        raise RuntimeError('Source production reached the 20GiB server reserve')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['rgb', 'depth'], required=True)
    parser.add_argument('--max-videos', type=int, help='Explicit smoke subset; never reported complete2000')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    table = pd.read_csv(args.selection, keep_default_na=False)
    assert len(table) == 2000 and table.id.is_unique
    args.output.mkdir(parents=True, exist_ok=True)
    dependencies = [Path(__file__), ROOT / 'scripts/produce_charades_training_inputs.py',
                    ROOT / 'src/cuhkx/charades_depth.py', ROOT / 'src/cuhkx/charades_source.py',
                    args.config, args.selection]
    identity = {str(p): sha256_file(p) for p in dependencies}
    identity_path = args.output / 'input_identity.json'
    if identity_path.exists():
        assert json.loads(identity_path.read_text()) == identity
    else:
        write(identity_path, identity)
    receipts = args.output / ('receipts_' + args.stage)
    receipts.mkdir(exist_ok=True)
    for name in ['rgb', 'metadata', 'depth', 'raw_depth_checks']:
        (args.output / name).mkdir(exist_ok=True)
    source_identity = {}
    started = time.monotonic()

    def wait_source(row):
        path = args.source / 'receipts' / (row['id'] + '.json')
        while not path.exists():
            state = json.loads((args.source / 'status.json').read_text())
            if state['status'] in ['failed', 'complete2000']:
                raise RuntimeError('Source downloader terminal but a required receipt is missing')
            time.sleep(10)
        receipt = json.loads(path.read_text())
        source_identity[row['id']] = receipt['sha256']
        return {**row, 'media_sha256': receipt['sha256']}

    rows = table.to_dict('records')
    if args.max_videos is not None:
        assert 0 < args.max_videos <= len(rows)
        rows = rows[:args.max_videos]
    results = []

    def accept(row, result):
        write(receipts / (row['id'] + '.json'), result)
        results.append(result)
        write(args.output / (args.stage + '_status.json'), {
            'status': 'producing', 'completed': len(results), 'required': len(rows),
            'seconds': time.monotonic() - started})
        print(args.stage, len(results), '/', len(rows), row['id'], flush=True)

    try:
        if args.stage == 'rgb':
            with ProcessPoolExecutor(max_workers=2) as executor:
                # Two outstanding jobs bound decoded-video memory and can follow downloads.
                for start in range(0, len(rows), 2):
                    batch = []
                    for row in rows[start:start + 2]:
                        reserve(args.output)
                        row = wait_source(row)
                        receipt = receipts / (row['id'] + '.json')
                        if receipt.exists():
                            result = json.loads(receipt.read_text())
                            assert sha256_file(Path(result['path'])) == result['sha256']
                            batch.append((row, result, None))
                        else:
                            future = executor.submit(rgb_job, (
                                row, config, str(args.source), str(args.output)))
                            batch.append((row, None, future))
                    for row, result, future in batch:
                        accept(row, future.result() if future is not None else result)
        else:
            configure_torch()
            model, audit = load_model(config)
            write(args.output / 'depth_model_load.json', audit)
            retained = set(table.id.iloc[[0, len(table) // 2, len(table)-1]])
            for row in rows:
                reserve(args.output)
                row = wait_source(row)
                receipt = receipts / (row['id'] + '.json')
                if receipt.exists():
                    result = json.loads(receipt.read_text())
                    assert sha256_file(Path(result['path'])) == result['sha256']
                    accept(row, result)
                    continue
                rgb_receipt = args.output / 'receipts_rgb' / (row['id'] + '.json')
                while not rgb_receipt.exists():
                    status_path = args.output / 'rgb_status.json'
                    if status_path.exists() and json.loads(status_path.read_text())['status'] in ['failed', 'complete2000']:
                        raise RuntimeError('RGB preparation terminal but a required receipt is missing')
                    time.sleep(10)
                declared = json.loads(rgb_receipt.read_text())
                rgb_path = args.output / 'rgb' / (row['id'] + '.npz')
                assert sha256_file(rgb_path) == declared['sha256']
                frames, pts, _ = raw_source(row, config, args.source)
                depth = infer(model, frames, config)
                raw_digest = hashlib.sha256(memoryview(np.ascontiguousarray(depth))).hexdigest()
                values, scaling = relative_input(depth)
                scaling['scope'] = 'Whole continuous video quantiles before windowing; dense output hashed in RAM'
                with np.load(rgb_path) as z:
                    np.testing.assert_array_equal(z['pts_seconds'], pts)
                    arrays = {k: z[k] for k in ['labels', 'mask', 'frame_indices', 'pts_seconds', 'bounds']}
                path = args.output / 'depth' / (row['id'] + '.npz')
                save_arrays(path, frames=crops(values)[arrays['frame_indices']][..., None], **arrays)
                result = {'id': row['id'], 'path': str(path), 'sha256': sha256_file(path),
                          'bytes': path.stat().st_size, 'normalization': scaling,
                          'raw_depth_array_sha256': raw_digest, 'raw_depth_shape': list(depth.shape),
                          'raw_depth_dtype': str(depth.dtype), 'paired_pts_labels_masks_exact': True,
                          'dense_storage': 'Regenerable from pinned media/model/config; first/middle/last retained',
                          'raw_media_sha256': row['media_sha256']}
                if row['id'] in retained:
                    raw_path = args.output / 'raw_depth_checks' / (row['id'] + '.npz')
                    save_arrays(raw_path, depth=depth, pts_seconds=pts)
                    result.update(raw_depth_path=str(raw_path), raw_depth_sha256=sha256_file(raw_path))
                accept(row, result)
        write(args.output / (args.stage + '_report.json'), {
            'status': 'complete' if len(results) == 2000 else 'smoke_complete',
            'videos': len(results), 'files': results,
            'raw_source_sha256': source_identity, 'seconds': time.monotonic() - started,
            'scope': 'Official Charades selected2000 only; same physical32frames/8secondstride4/157occupancy. '
                     'No target/Test data or training. Keep quantized training inputs; all continuous '
                     'depth arrays hashed before quantization, three predeclared dense checks retained.'})
        write(args.output / (args.stage + '_status.json'), {
            'status': 'complete2000' if len(results) == 2000 else 'smoke_complete', 'completed': len(results)})
    except Exception as error:
        write(args.output / (args.stage + '_status.json'), {
            'status': 'failed', 'completed': len(results), 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
