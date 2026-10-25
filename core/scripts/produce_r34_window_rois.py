#!/usr/bin/env python3
"""Detect person crops from fixed uniform HAU windows without interval labels."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from cuhkx.sequence_skeleton_nested import array_digest
from cuhkx.spatial_training import write_json
from cuhkx.validation import sha256_file
from cuhkx.video import _decode_video_indices, uniform_frame_indices
from scripts.extract_kuno_transfer_logits import square_window


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, help='Fixed leading records for engineering acceptance only')
    parser.add_argument('--split', choices=['Training', 'Testing'], default='Training')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(args.plan.read_text())
    for key in ['manifest', 'weights']:
        if sha256_file(Path(plan[key])) != plan[key + '_sha256']:
            raise ValueError('Declared input changed: ' + key)
    raw = json.loads(Path(plan['manifest']).read_text())
    records = sorted((r for r in raw['records'] if r['source'] == 'HAU' and r['unit'].startswith(args.split + '/') and r['modality'] == 'IR'), key=lambda r: r['unit'])
    required = len(records)
    if len(records) != required or not all(r['available'] for r in records):
        raise ValueError(f'Expected all{required} available {args.split} IR records')
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError('Limit must be positive')
        records = records[:args.limit]
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model = YOLO(plan['weights'])
    snapshot = args.output / 'source_snapshot'
    snapshot.mkdir()
    (snapshot / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    subprocess.run(['git', 'archive', '--format=tar.gz', '--output=' + str(snapshot.resolve() / 'source.tar.gz'), 'HEAD'], cwd=ROOT, check=True)
    write_json(args.output / 'invocation.json', {'argv': sys.argv, 'cwd': str(Path.cwd()), 'python': sys.executable, 'torch': str(torch.__version__), 'numpy': np.__version__, 'plan': plan, 'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS')})
    output = args.output / 'units'
    output.mkdir()
    started = time.monotonic()
    summary = []
    for record in records:
        path = Path(plan['data_root']) / record['unit'] / 'IR/IR.mp4'
        if sha256_file(path) != record['raw']['sha256']:
            raise ValueError('Raw IR changed: ' + str(path))
        meta = record['metadata']
        schedules = np.asarray(record['timing']['frame_indices'])
        if schedules.shape != (8, 32):
            raise ValueError('Expected original eight32frame R34 windows')
        probes = schedules[:, uniform_frame_indices(32, 8)]
        global_probes = uniform_frame_indices(meta['frame_count'], 8)
        indices = np.concatenate([global_probes[None], probes])
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened() or int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != meta['frame_count']:
            raise ValueError('Original frame metadata changed')
        frames = _decode_video_indices(capture, path, indices)
        height, width = frames.shape[2:4]
        results = []
        for view in frames:
            boxes = []
            for detection in model.predict(list(view), classes=[0], conf=.25, verbose=False, device='cpu'):
                values = detection.boxes
                boxes.append(None if values is None or len(values) == 0 else values.xyxy[int(values.conf.argmax())].cpu().tolist())
            detected = [tuple(b) for b in boxes if b is not None]
            roi = square_window(detected, width, height)
            results.append({'boxes': boxes, 'roi': roi, 'detected_frames': len(detected)})
        # The consumer rounds normalized coordinates to pixels. Retain both
        # representations so runtime differences are measured, never hidden.
        scale = np.asarray([width, height, width, height])
        legacy_pixels = np.rint(np.asarray(record['roi']) * scale).astype(int)
        current_pixels = np.rint(np.asarray(results[0]['roi']) * scale).astype(int)
        pixel_exact = bool(np.array_equal(legacy_pixels, current_pixels))
        for item in results[1:]:
            item['fallback_to_global'] = item['detected_frames'] == 0
            if item['fallback_to_global']:
                item['roi'] = results[0]['roi']
        result = {'unit': record['unit'], 'raw_sha256': record['raw']['sha256'], 'frame_indices': indices.tolist(), 'sampled_pixel_sha256': array_digest(frames), 'original_global_roi': record['roi'], 'global_normalized_exact': results[0]['roi'] == tuple(record['roi']), 'global_crop_pixels_exact': pixel_exact, 'legacy_crop_pixels': legacy_pixels.tolist(), 'current_crop_pixels': current_pixels.tolist(), 'global': results[0], 'windows': results[1:]}
        target = output / (record['unit'].replace('/', '__') + '.json')
        write_json(target, result)
        summary.append({'unit': record['unit'], 'path': str(target), 'sha256': sha256_file(target), 'global_crop_pixels_exact': pixel_exact, 'different_windows': sum(tuple(w['roi']) != tuple(record['roi']) for w in results[1:]), 'fallback_windows': sum(w['fallback_to_global'] for w in results[1:])})
        write_json(args.output / 'progress.json', {'completed': len(summary), 'required': len(records), 'elapsed_seconds': time.monotonic() - started})
        if len(summary) % 10 == 0 or len(summary) == len(records):
            print('window person crops', len(summary), '/', len(records), flush=True)
    write_json(args.output / 'report.json', {'status': 'complete_input_only_window_person_rois', 'split': args.split, 'records': summary, 'seconds': time.monotonic() - started, 'all_original_global_crop_pixels_exact': all(r['global_crop_pixels_exact'] for r in summary), 'scope': 'Original8uniformR34windows; sameYOLO/8probes/conf.25/margin1.4/minside.35; GlobalCPU and legacyROI bothretained; comparewindowCPU againstglobalCPU to isolate temporalextent if any legacycrop pixels differ. Missingwindow detection uses inputderivedglobalcrop. No action labels, intervals, QA, classifier fit or crop selection by prediction. Only the explicitly declared split is decoded.', 'engineering_limit': args.limit})


if __name__ == '__main__':
    main()
