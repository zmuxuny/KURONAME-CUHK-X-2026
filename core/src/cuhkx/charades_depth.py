"""PTS-aware external RGB sampling and pinned relative video-depth production."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from cuhkx.validation import sha256_file


def sample_video(path, *, dependency_target, target_fps=4., max_resolution=448):
    sys.path.insert(0, dependency_target)
    import av

    frames, selected_pts, indices, all_pts = [], [], [], []
    next_sample, first_pts, last_end = 0., None, 0.
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate)
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                raise ValueError('Actual frame presentation timestamps are required')
            absolute_time = float(frame.pts * frame.time_base)
            if first_pts is None:
                first_pts = absolute_time
            time = absolute_time - first_pts
            if all_pts and time < all_pts[-1]:
                raise ValueError('Video presentation time moved backward')
            all_pts.append(time)
            duration = float(frame.duration * frame.time_base) if frame.duration else 1. / rate
            last_end = time + duration
            if time + 1e-9 < next_sample:
                continue
            rgb = frame.to_ndarray(format='rgb24')
            h, w = rgb.shape[:2]
            scale = min(1., max_resolution / max(h, w))
            rgb = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            frames.append(rgb)
            selected_pts.append(time)
            indices.append(index)
            next_sample = (np.floor(time * target_fps + 1e-8) + 1) / target_fps
    if not frames or last_end <= 0:
        raise ValueError('Video contains no usable presentation-time support')
    metadata = {'raw_sha256': sha256_file(Path(path)), 'decoded_frames': len(all_pts),
                'nominal_original_fps': rate, 'first_pts_seconds': first_pts,
                'actual_end_seconds': last_end, 'sampling': 'first decoded frame at or after each4Hz grid point',
                'selected_frame_indices': indices, 'selected_pts_seconds': selected_pts,
                'all_pts_seconds': all_pts, 'max_resolution': max_resolution,
                'output_shape': list(np.stack(frames).shape)}
    return np.stack(frames), np.asarray(selected_pts), metadata


def load_model(config, device='cuda'):
    root = Path(config['source_root'])
    for item in config['source_files']:
        if sha256_file(root / item['path']) != item['sha256']:
            raise ValueError('Official VDA source changed')
    if sha256_file(Path(config['weights'])) != config['weights_sha256']:
        raise ValueError('Official VDA-S weight changed')
    sys.path[:0] = [config['dependency_target'], str(root)]
    from video_depth_anything.video_depth import VideoDepthAnything

    model = VideoDepthAnything(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
    state = torch.load(config['weights'], map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    if not all(torch.equal(model.state_dict()[key], value) for key, value in state.items()):
        raise ValueError('VDA state changed during loading')
    audit = {'strict_loading': True, 'source_tensors': len(state),
             'parameters': sum(p.numel() for p in model.parameters()),
             'direction': 'relative inverse-depth-like output; author eval.py aligns to 1/ground_truth_depth',
             'input_size': config['input_size'], 'precision': 'author fp16 autocast with fp32 retained output',
             'temporal_policy': 'unaltered author32-frame/10-overlap/global scale-shift alignment',
             'scope': 'synthetic relative geometry, not measured Depth_Color or IR'}
    return model.eval().requires_grad_(False).to(device), audit


def infer(model, frames, config):
    depth, _ = model.infer_video_depth(frames, config['target_fps'], input_size=config['input_size'],
                                      device='cuda', fp32=False)
    if depth.shape != frames.shape[:3] or not np.isfinite(depth).all():
        raise ValueError('Relative depth and original presentation frames are misaligned or nonfinite')
    return depth


def preview(frames, depth, path):
    from PIL import Image

    chosen = np.linspace(0, len(frames) - 1, 4).round().astype(int)
    low, high = np.quantile(depth, [.01, .99])
    if high <= low:
        raise ValueError('Collapsed relative depth')
    rows = []
    for index in chosen:
        rgb = Image.fromarray(frames[index]).resize((320, 220))
        gray = np.clip((depth[index] - low) / (high - low), 0, 1)
        view = Image.fromarray((gray * 255).astype(np.uint8)).convert('RGB').resize((320, 220))
        rows.append((rgb, view))
    sheet = Image.new('RGB', (4 * 320, 440))
    for index, (rgb, view) in enumerate(rows):
        sheet.paste(rgb, (index * 320, 0))
        sheet.paste(view, (index * 320, 220))
    sheet.save(path)
