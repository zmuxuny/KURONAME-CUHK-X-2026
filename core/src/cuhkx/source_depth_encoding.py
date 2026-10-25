"""Fixed source pixel transforms; no target-fitted map or per-frame rescaling."""
from __future__ import annotations

import hashlib

import cv2
import numpy as np


def inverse_jet_table():
    # Existing VDA uint8 is normalized relative inverse depth. Reverse its rank
    # before JET. This is a sensor-color hypothesis, not recovered metric depth.
    ramp = np.arange(255, -1, -1, dtype=np.uint8)[:, None]
    return np.ascontiguousarray(cv2.applyColorMap(ramp, cv2.COLORMAP_JET)[:, 0, ::-1])


def encoding_contract(name, arm):
    if name == 'identity':
        return {'name': name, 'arm': arm, 'policy': 'Unchanged original source pixels'}
    if name != 'inverse_jet_v1' or arm != 'depth':
        raise ValueError('Only depth supports inverse_jet_v1')
    table = inverse_jet_table()
    return {'name': name, 'arm': arm, 'opencv': cv2.__version__,
            'rgb_table': table.tolist(), 'table_sha256': hashlib.sha256(table.tobytes()).hexdigest(),
            'policy': 'RGB=OpenCVJET(255-existing_uint8_relative_inverse_depth); '
                      'no new quantiles, target fitting, geometry or label changes; '
                      'not an assertion of official sensor mapping'}


def encode_frames(raw, name='identity'):
    if name == 'identity':
        return raw
    if name != 'inverse_jet_v1' or raw.dtype != np.uint8 or raw.shape[-1] != 1:
        raise ValueError('inverse_jet_v1 requires single-channel uint8 source depth')
    return inverse_jet_table()[raw[..., 0]]
