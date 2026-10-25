#!/usr/bin/env python3
"""Produce paired actual-PTS RGB and continuous synthetic-depth source inputs."""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import cv2
import numpy as np
import pandas as pd
import torch

from cuhkx.charades_depth import infer, load_model, sample_video
from cuhkx.charades_source import crops, occupancy, relative_input, window_plan
from cuhkx.harn_tail import configure_torch
from cuhkx.spatial_training import check_space, write_json
from cuhkx.validation import sha256_file


def save_arrays(path, **arrays):
    temporary = path.with_suffix('.tmp')
    with temporary.open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    with np.load(temporary) as z:
        for key, expected in arrays.items():
            np.testing.assert_array_equal(z[key], expected)
    temporary.replace(path)


def raw_source(row, config, source):
    path = Path(source) / 'media' / (row['id'] + '.mp4')
    if sha256_file(path) != row['media_sha256']:
        raise ValueError('Official downloaded source media changed')
    return sample_video(path, dependency_target=config['dependency_target'],
                        target_fps=config['target_fps'], max_resolution=config['max_resolution'])


def rgb_job(args):
    row, config, source, output = args
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    check_space({'min_host_free_bytes': 9 * 2**30})
    output = Path(output)
    destination = output / 'rgb' / (row['id'] + '.npz')
    frames, pts, metadata = raw_source(row, config, source)
    indices, bounds = window_plan(pts, metadata['actual_end_seconds'])
    labels, mask, anomalies = occupancy(row['actions'], pts, indices, metadata['actual_end_seconds'])
    save_arrays(destination, frames=crops(frames)[indices], labels=labels, mask=mask,
                frame_indices=indices, pts_seconds=pts, bounds=bounds)
    metadata.update(subject=row['subject'], id=row['id'], windows=len(indices), anomalies=anomalies,
                    sha256=sha256_file(destination), bytes=destination.stat().st_size)
    write_json(output / 'metadata' / (row['id'] + '.json'), metadata)
    return {'id': row['id'], 'subject': row['subject'], 'windows': len(indices),
            'path': str(destination), 'sha256': metadata['sha256'], 'bytes': metadata['bytes']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['rgb', 'depth'], required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), ROOT / 'src/cuhkx/charades_source.py', ROOT / 'src/cuhkx/charades_depth.py']
    snapshot = args.output / ('source_' + args.stage)
    snapshot.mkdir(exist_ok=False)
    for path in sources:
        target = snapshot / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    write_json(snapshot / 'config.json', config)
    write_json(snapshot / 'manifest.json', [{'path': str(p), 'sha256': sha256_file(p)} for p in sources])
    table = pd.read_csv(args.source / 'selected_train.csv', keep_default_na=False)
    downloaded = json.loads((args.source / 'download_report.json').read_text())
    lookup = {Path(r['path']).stem: r for r in downloaded['files']}
    rows = [{**r, 'media_sha256': lookup[r['id']]['sha256']} for r in table.to_dict('records')]
    write_json(snapshot / 'source_identity.json', {'selection_sha256': sha256_file(args.source / 'selected_train.csv'),
               'download_report_sha256': sha256_file(args.source / 'download_report.json')})
    started, results = time.monotonic(), []
    if args.stage == 'rgb':
        (args.output / 'rgb').mkdir(exist_ok=False)
        (args.output / 'metadata').mkdir(exist_ok=False)
        with ProcessPoolExecutor(max_workers=2) as executor:
            for result in executor.map(rgb_job, [(r, config, str(args.source), str(args.output)) for r in rows]):
                results.append(result)
                if len(results) % 20 == 0:
                    print(f'RGB source {len(results)}/{len(rows)}; {time.monotonic() - started:.1f}s', flush=True)
    else:
        if json.loads((args.output / 'source_rgb/config.json').read_text()) != config:
            raise ValueError('RGB/depth source sampling configuration differs')
        rgb = json.loads((args.output / 'rgb_report.json').read_text())
        if len(rgb['files']) != len(rows):
            raise ValueError('Complete same-source RGB inputs before depth')
        (args.output / 'depth').mkdir(exist_ok=False)
        (args.output / 'raw_depth').mkdir(exist_ok=False)
        configure_torch()
        model, audit = load_model(config)
        write_json(snapshot / 'model_load.json', audit)
        for row in rows:
            check_space({'min_host_free_bytes': 9 * 2**30})
            frames, pts, _metadata = raw_source(row, config, args.source)
            depth = infer(model, frames, config)
            raw_path = args.output / 'raw_depth' / (row['id'] + '.npz')
            save_arrays(raw_path, depth=depth, pts_seconds=pts)
            values, scaling = relative_input(depth)
            rgb_path = args.output / 'rgb' / (row['id'] + '.npz')
            declared = json.loads((args.output / 'metadata' / (row['id'] + '.json')).read_text())
            if sha256_file(rgb_path) != declared['sha256']:
                raise ValueError('Matched RGB input changed')
            with np.load(rgb_path) as z:
                np.testing.assert_array_equal(z['pts_seconds'], pts)
                arrays = {k: z[k] for k in ['labels', 'mask', 'frame_indices', 'pts_seconds', 'bounds']}
            destination = args.output / 'depth' / (row['id'] + '.npz')
            save_arrays(destination, frames=crops(values)[arrays['frame_indices']][..., None], **arrays)
            result = {'id': row['id'], 'path': str(destination), 'sha256': sha256_file(destination),
                      'bytes': destination.stat().st_size, 'raw_depth_path': str(raw_path),
                      'raw_depth_sha256': sha256_file(raw_path), 'raw_depth_bytes': raw_path.stat().st_size,
                      'normalization': scaling, 'paired_pts_labels_masks_exact': True}
            write_json(destination.with_suffix('.json'), result)
            results.append(result)
            print(f'Depth source {len(results)}/{len(rows)}; {time.monotonic() - started:.1f}s', flush=True)
    write_json(args.output / (args.stage + '_report.json'), {'status': 'complete', 'videos': len(results),
               'files': results, 'seconds': time.monotonic() - started,
               'actual_raw_source': 'official Charades Train; PTS-aware; no target labels or Test',
               'windows': '8 seconds stride4 with edge coverage;32 selected actual frames;157 occupancy soft BCE with invalid-class masks'})


if __name__ == '__main__':
    main()
