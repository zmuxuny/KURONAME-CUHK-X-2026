"""Lossless physical-frame storage for repeated Charades training windows."""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from cuhkx.generic_video_windows import KINETICS_MEAN, KINETICS_STD
from cuhkx.source_depth_encoding import encode_frames, encoding_contract
from cuhkx.validation import sha256_file


def pack_video(source, destination):
    destination.mkdir(parents=True, exist_ok=False)
    with np.load(source) as z:
        raw, physical = z['frames'], z['frame_indices']
        _, first, inverse = np.unique(physical.reshape(-1), return_index=True, return_inverse=True)
        flat = raw.reshape((-1, *raw.shape[2:]))
        packed = flat[first]
        mapping = inverse.reshape(physical.shape)
        np.testing.assert_array_equal(flat, packed[inverse])
        np.save(destination / 'frames.npy', packed, allow_pickle=False)
        np.savez(destination / 'index.npz', mapping=mapping, labels=z['labels'], mask=z['mask'])
        with np.load(destination / 'index.npz') as index:
            decoded = np.load(destination / 'frames.npy', mmap_mode='r')
            np.testing.assert_array_equal(raw, decoded[index['mapping']])
            np.testing.assert_array_equal(z['labels'], index['labels'])
            np.testing.assert_array_equal(z['mask'], index['mask'])
    paths = [destination / name for name in ['frames.npy', 'index.npz']]
    return {'source_sha256': sha256_file(source), 'source': str(source), 'windows': len(raw),
            'unique_frames': len(packed), 'all_window_pixels_exact': True,
            'files': {p.name: {'sha256': sha256_file(p), 'bytes': p.stat().st_size} for p in paths}}


def verify_cache(cache, inputs, arm, identities):
    """Check the complete selected cache once before training workers use it."""
    cache = Path(cache) / arm
    report = json.loads((cache / 'report.json').read_text())
    if report['status'] != 'complete2000' or set(report['videos']) != set(identities):
        raise ValueError('Cache does not cover the complete selected source')
    for identity in identities:
        folder = cache / identity
        if sha256_file(folder / 'receipt.json') != report['receipts_sha256'][identity]:
            raise ValueError('Packed cache receipt changed')
        record = json.loads((folder / 'receipt.json').read_text())
        if record['source_sha256'] != sha256_file(Path(inputs) / arm / (identity + '.npz')):
            raise ValueError('Packed cache was built from different training inputs')
        for name, value in record['files'].items():
            if sha256_file(folder / name) != value['sha256']:
                raise ValueError('Packed training input changed')
    return sha256_file(cache / 'report.json')


class PackedSourceWindows(torch.utils.data.Dataset):
    def __init__(self, path, arm, rows, encoding='identity'):
        self.path, self.arm, self.rows = Path(path), arm, rows
        encoding_contract(encoding, arm)
        self.encoding = encoding
        self._cache = OrderedDict()

    def __len__(self):
        return len(self.rows)

    def _load(self, identity):
        if identity in self._cache:
            self._cache.move_to_end(identity)
            return self._cache[identity]
        folder = self.path / self.arm / identity
        frames = np.load(folder / 'frames.npy', mmap_mode='r')
        with np.load(folder / 'index.npz') as z:
            arrays = {k: z[k] for k in z.files}
        self._cache[identity] = frames, arrays
        if len(self._cache) > 16:
            self._cache.popitem(last=False)
        return self._cache[identity]

    def __getitem__(self, index):
        identity, window, _, _, _ = self.rows[index]
        frames, arrays = self._load(identity)
        raw = encode_frames(frames[arrays['mapping'][window]], self.encoding)
        if raw.shape[-1] == 1:
            raw = np.repeat(raw, 3, axis=-1)
        x = torch.from_numpy(raw).permute(3, 0, 1, 2).float() / 255.
        x = ((x - torch.tensor(KINETICS_MEAN)[:, None, None, None])
             / torch.tensor(KINETICS_STD)[:, None, None, None])
        return x, torch.from_numpy(arrays['labels'][window]), torch.from_numpy(arrays['mask'][window])
