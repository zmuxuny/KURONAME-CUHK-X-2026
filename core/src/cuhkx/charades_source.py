"""Matched source windows and partial-label occupancy on actual video PTS."""
from __future__ import annotations

import cv2
import numpy as np


def window_plan(pts, end, seconds=8., stride=4.):
    pts = np.asarray(pts, dtype=np.float64)
    last = max(0., end - seconds)
    starts = list(np.arange(0., last + 1e-8, stride))
    if not np.isclose(starts[-1], last):
        starts.append(last)
    indices, bounds = [], []
    for start in starts:
        stop = min(end, start + seconds)
        available = np.flatnonzero((pts >= start) & (pts < stop))
        if not len(available):
            raise ValueError('Source window has no actual PTS support')
        chosen = available[np.linspace(0, len(available) - 1, 32).round().astype(int)]
        if indices and np.array_equal(indices[-1], chosen):
            continue
        indices.append(chosen)
        bounds.append([start, stop])
    return np.stack(indices), np.asarray(bounds)


def occupancy(actions, pts, indices, end):
    """Unknown class/video annotations remain masked, never inferred negatives."""
    pts = np.asarray(pts)
    active = np.zeros((len(pts), 157), dtype=bool)
    known = np.ones(157, dtype=bool)
    anomalies = []
    for text in str(actions).split(';'):
        if not text:
            continue
        label, start, stop = text.split()
        label, start, stop = int(label[1:]), float(start), float(stop)
        if not 0 <= label < 157 or not np.isfinite([start, stop]).all():
            raise ValueError('Invalid source action identity or time')
        if stop <= start or min(stop, end) <= max(start, 0.):
            known[label] = False
            anomalies.append({'class': label, 'start': start, 'stop': stop,
                              'reason': 'invalid_or_no_actual_media_support'})
            continue
        active[:, label] |= (pts >= max(start, 0.)) & (pts < min(stop, end))
        if start < 0 or stop > end:
            anomalies.append({'class': label, 'start': start, 'stop': stop,
                              'reason': 'intersected_actual_media_support'})
    labels = active[indices].mean(1).astype(np.float32)
    return labels, np.tile(known, (len(indices), 1)), anomalies


def crops(frames):
    # Same full-frame source view in RGB/depth; no target-derived person crop.
    return np.stack([cv2.resize(frame, (171, 128), interpolation=cv2.INTER_AREA)[8:120, 29:141]
                     for frame in frames])


def relative_input(depth):
    low, high = np.quantile(depth, [.01, .99])
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError('Collapsed continuous relative geometry')
    values = np.clip((depth - low) / (high - low), 0., 1.)
    return np.round(values * 255).astype(np.uint8), {'low': float(low), 'high': float(high),
             'scope': 'whole continuous video quantiles before any source windowing; raw float32 retained'}
