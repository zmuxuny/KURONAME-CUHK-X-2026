#!/usr/bin/env python3
"""Frozen generic/source-only pooled HAU features with verified window ROIs."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import cv2
import numpy as np
import torch

from cuhkx.harn_tail import configure_torch
from cuhkx.r34_adaptation import buffers_digest
from cuhkx.r34_video import metadata_arrays, preprocess, pretrained_model
from cuhkx.sequence_r34_transfer import select_views
from cuhkx.sequence_skeleton_nested import array_digest
from cuhkx.spatial_training import check_space, write_json
from cuhkx.validation import sha256_file
from cuhkx.video import _decode_video_indices, uniform_frame_indices
from scripts.prepare_r34_external_warmup import source_identity

ARMS = ('generic', 'rgb', 'depth')


def models(config, arms=ARMS):
    result, identities = {}, {}
    for arm in arms:
        if arm == 'generic' and config.get('reuse_generic'):
            identities[arm] = {'sha256': sha256_file(Path(config['weights'])), 'target_fit': False}
            continue
        model = pretrained_model(config['weights'])
        if arm != 'generic':
            path = Path(config['sources'][arm])
            identities[arm] = source_identity(path)
            source = torch.load(path, map_location='cpu', weights_only=True)
            if source['source_only'] != 'Charades official Train' or source['arm'] != arm:
                raise ValueError('Requires declared source-only arm')
            learned = {k: v for k, v in source['learned'].items() if not k.startswith('fc.')}
            expected = {k for k, _ in model.named_parameters() if k.startswith(('layer3.', 'layer4.'))}
            if set(learned) != expected or buffers_digest(model) != source['bn_buffer_sha256']:
                raise ValueError('Source backbone scope or generic BN differs')
            state = model.state_dict()
            state.update(learned)
            model.load_state_dict(state, strict=True)
        else:
            identities[arm] = {'sha256': sha256_file(Path(config['weights'])), 'target_fit': False}
        model.fc = torch.nn.Identity()
        result[arm] = model.eval().requires_grad_(False).cuda()
    return result, identities


@torch.inference_mode()
def forward(model, x):
    values = []
    for start in range(0, len(x), 2):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            values.append(model(x[start:start + 2].cuda()).float().cpu().numpy())
    values = np.concatenate(values)
    if values.shape != (len(x), 512) or not np.isfinite(values).all():
        raise ValueError('Invalid pooled512 features')
    return values


def tensors(config, unit, views, mask):
    roi_path = Path(config['rois']) / 'units' / (unit.replace('/', '__') + '.json')
    roi = json.loads(roi_path.read_text())
    ir = views['IR']
    if roi['unit'] != unit or roi['raw_sha256'] != ir['raw']['sha256']:
        raise ValueError('ROI source identity differs')
    probes = np.asarray(ir['timing']['frame_indices'])[:, uniform_frame_indices(32, 8)]
    expected = np.concatenate([uniform_frame_indices(ir['metadata']['frame_count'], 8)[None], probes])
    np.testing.assert_array_equal(expected, roi['frame_indices'])
    samples, inputs = {}, {str(roi_path): sha256_file(roi_path)}
    for v, modality in enumerate(('Depth_Color', 'IR')):
        if not mask[0, v]:
            continue
        record = views[modality]
        path = Path(config['data_root']) / unit / modality / (modality + '.mp4')
        inputs[str(path)] = sha256_file(path)
        if inputs[str(path)] != record['raw']['sha256'] or record['roi'] != roi['original_global_roi']:
            raise ValueError('Raw video or original ROI differs')
        indices = np.asarray(record['timing']['frame_indices'])
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError('Cannot open declared Train media')
        if modality == 'IR':
            decoded = _decode_video_indices(capture, path, np.concatenate([expected[0], indices.reshape(-1)]))
            frames = decoded[8:].reshape(8, 32, *decoded.shape[1:])
            detector = np.concatenate([decoded[:8][None], frames[:, uniform_frame_indices(32, 8)]])
            if array_digest(detector) != roi['sampled_pixel_sha256']:
                raise ValueError('Detection pixel identity differs')
        else:
            frames = _decode_video_indices(capture, path, indices)
        samples[v] = torch.cat([preprocess(frames[i:i+1], r['roi'], modality)
                                for i, r in enumerate(roi['windows'])])
    return samples, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    configure_torch()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    check_space(config)
    path = Path(config['source_manifest'])
    if sha256_file(path) != config['source_sha256']:
        raise ValueError('Input manifest changed')
    raw = json.loads(path.read_text())
    # Filter before constructing any feature metadata; shared manifest also lists Test.
    raw['records'] = [r for r in raw['records'] if r['source'] == 'HAU' and r['unit'].startswith('Training/')]
    metadata = metadata_arrays(raw)
    lookup = {(str(u), str(m), int(w)): i for i, (u, m, w) in enumerate(zip(
        metadata['units'], metadata['modalities'], metadata['window_indices'], strict=True))}
    recordings = {}
    for r in raw['records']:
        recordings.setdefault(r['unit'], {})[r['modality']] = r
    units = sorted(recordings)
    if len(units) != 809:
        raise ValueError('Requires809 official Train HAU recordings')
    if args.limit is not None:
        if not 0 < args.limit <= len(units):
            raise ValueError('Invalid probe limit')
        units = units[:args.limit]
    networks, identities = models(config)
    reused = config.get('reuse_generic')
    old_rows, reused_identity = {}, None
    if reused:
        old_path = Path(reused) / 'manifest.json'
        old = json.loads(old_path.read_text())
        if old['probe'] or old['source_models']['generic'] != identities['generic']:
            raise ValueError('Reused generic encoder identity differs or bank is a probe')
        old_rows = {row['unit']: row for row in old['units']}
        reused_identity = {'manifest': str(old_path), 'sha256': sha256_file(old_path),
                           'policy': 'Reuse generic only after exact raw/ROI/tensor/mask/time checks; new RGB/depth forwards'}
    write_json(args.output / 'invocation.json', {'argv': sys.argv, 'config': config,
               'source_models': identities, 'python': sys.version, 'torch': str(torch.__version__),
               'cuda': torch.version.cuda, 'numpy': np.__version__, 'opencv': cv2.__version__,
               'precision': 'BF16 forward; float32 pooled512 storage; batch2',
               'scope': 'OfficialTrain HAU only. Generic or external Train-only encoders; zero target weight fits.'})
    destination = args.output / 'units'
    destination.mkdir()
    started = time.perf_counter()
    summaries = []
    for i, unit in enumerate(units):
        check_space(config)
        _, mask, times, reason = select_views(metadata, lookup, unit, allow_duplicate_windows=True)
        if mask is None:
            raise ValueError('No HAU view available: ' + unit)
        samples, inputs = tensors(config, unit, recordings[unit], mask)
        features = {a: np.zeros((8, 2, 512), np.float32) for a in ARMS}
        tensor_hashes = {str(v): array_digest(x.numpy()) for v, x in samples.items()}
        if reused:
            old_row = old_rows[unit]
            old_path = Path(reused) / old_row['file']
            if old_row['inputs'] != inputs or old_row['tensor_sha256'] != tensor_hashes:
                raise ValueError('Reused generic raw pixels, ROI or normalized tensors differ')
            if sha256_file(old_path) != old_row['sha256']:
                raise ValueError('Reused generic feature file changed')
            with np.load(old_path) as z:
                np.testing.assert_array_equal(z['mask'], mask)
                np.testing.assert_array_equal(z['times'], times)
                if str(z['unit']) != unit or z['generic'].shape != (8, 2, 512):
                    raise ValueError('Reused generic unit or dimensions differ')
                features['generic'] = z['generic'].copy()
        for view, x in samples.items():
            for arm in networks:
                features[arm][:, view] = forward(networks[arm], x)
                if i == 0:
                    np.testing.assert_array_equal(features[arm][:, view], forward(networks[arm], x))
        path = destination / (unit.replace('/', '__') + '.npz')
        np.savez_compressed(path, **features, mask=mask, times=times, unit=np.asarray(unit))
        summary = {'unit': unit, 'file': str(path.relative_to(args.output)), 'sha256': sha256_file(path),
                   'inputs': inputs, 'tensor_sha256': tensor_hashes,
                   'view_policy': reason, 'features_sha256': {a: array_digest(features[a]) for a in ARMS}}
        write_json(path.with_suffix('.json'), summary)
        summaries.append(summary)
        write_json(args.output / 'status.json', {'status': 'extracting', 'completed': i + 1,
                   'required': len(units), 'seconds': time.perf_counter() - started})
        print(f'HAU source pools {i+1}/{len(units)} {time.perf_counter()-started:.1f}s', flush=True)
    write_json(args.output / 'manifest.json', {'kind': 'cuhkx_r34_source_hau_pools_v1',
               'arms': list(ARMS), 'channels': 512, 'windows': 8, 'target_labels_used': False,
               'source_models': identities, 'units': summaries, 'probe': args.limit is not None,
               'config_sha256': sha256_file(args.config), 'first_recording_repeat_exact': not bool(reused),
               'first_recording_repeated_arms': list(networks), 'reused_generic': reused_identity})
    write_json(args.output / 'status.json', {'status': 'complete_probe' if args.limit else 'complete809',
               'completed': len(units), 'required': len(units), 'seconds': time.perf_counter() - started})


if __name__ == '__main__':
    main()
