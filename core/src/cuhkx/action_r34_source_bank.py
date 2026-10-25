"""Source-only R34 pooled banks for person-excluded direct HAU supervision."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cuhkx.action_joint_harn_head import HarnWindowBank
from cuhkx.metadata import subject_id
from cuhkx.validation import sha256_file


def read_centers(directory, raw_records):
    directory, raw_records = Path(directory), Path(raw_records)
    invocation = json.loads((directory / 'invocation.json').read_text())
    if sha256_file(raw_records) != invocation['config']['source_sha256']:
        raise ValueError('Physical timeline source differs from feature production')
    manifest = json.loads((directory / 'manifest.json').read_text())
    units = {r['unit'] for r in manifest['units']}
    if any(not u.startswith('Training/HAU/') for u in units):
        raise ValueError('Only official Train HAU time axes are supported')
    result = {}
    for row in json.loads(raw_records.read_text())['records']:
        if row['unit'] not in units or not row['available']:
            continue
        indices = np.asarray(row['timing']['frame_indices'])
        count = row['metadata']['frame_count']
        if indices.shape != (8, 32) or count <= 1 or indices.min() < 0 or indices.max() >= count:
            raise ValueError('Physical source window indices differ')
        # Preserve the existing temporal decoder convention, before interpolation.
        times = indices.mean(axis=1) / (count - 1)
        if (np.diff(times) < 0).any():
            raise ValueError('Physical windows are not ordered')
        result[(row['unit'], row['modality'])] = times
    return result


def read_window_bank(directory, arm):
    directory = Path(directory)
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if (manifest['kind'] != 'cuhkx_r34_source_hau_pools_v1' or manifest['probe']
            or manifest['channels'] != 512 or manifest['windows'] != 8
            or manifest['target_labels_used'] or arm not in manifest['arms']):
        raise ValueError('Requires completed source-only R34 HAU bank')
    units, values, masks = [], [], []
    for entry in manifest['units']:
        path = directory / entry['file']
        if sha256_file(path) != entry['sha256']:
            raise ValueError('HAU pooled input changed')
        with np.load(path, allow_pickle=False) as z:
            unit, features, mask = str(z['unit']), z[arm].copy(), z['mask'].copy()
        if (unit != entry['unit'] or not unit.startswith('Training/HAU/')
                or features.shape != (8, 2, 512) or not np.isfinite(features).all()
                or mask.shape != (8, 2) or mask.dtype != bool or not mask.any()):
            raise ValueError('HAU feature scope or shape differs')
        units.append(unit)
        values.append(features)
        masks.append(mask)
    if len(units) != 809 or len(set(units)) != 809:
        raise ValueError('HAU bank must cover809 unique Train recordings')
    matrix = np.concatenate([np.stack(values).reshape(809, -1), np.stack(masks).reshape(809, -1)], axis=1)
    names = [f'r34_w{w}_{v}_{c}' for w in range(8) for v in ('depthcolor', 'ir') for c in range(512)]
    names += [f'available_w{w}_{v}' for w in range(8) for v in ('depthcolor', 'ir')]
    return {'X': matrix, 'units': np.asarray(units), 'feature_names': np.asarray(names),
            'index': {u: i for i, u in enumerate(units)}, 'source_manifest_sha256': sha256_file(manifest_path)}


class R34SourceHarnBank(HarnWindowBank):
    """Reuse existing canonical Train crop features, never target-adapted encoders."""

    def __init__(self, directory, expected_source_sha256):
        directory = Path(directory)
        self.manifest_path = str(directory / 'extraction_report.json')
        self.manifest_sha256 = sha256_file(Path(self.manifest_path))
        report = json.loads(Path(self.manifest_path).read_text())
        if report['fits'] != 0 or report['source'] != 'official Train HARn only':
            raise ValueError('HARn requires unfitted target features')
        for filename, field in [('features.npy', 'feature_sha256'), ('rows.json', 'rows_sha256')]:
            if sha256_file(directory / filename) != report[field]:
                raise ValueError('HARn source-only pooled bank changed')
        identity = directory / 'external_source.json'
        if expected_source_sha256 is not None:
            if json.loads(identity.read_text())['sha256'] != expected_source_sha256:
                raise ValueError('HAU and auxiliary HARn have different source encoders')
        elif identity.exists() or not report['precision'].startswith('generic '):
            raise ValueError('Generic control must use generic pooled features')
        rows = json.loads((directory / 'rows.json').read_text())
        raw = np.load(directory / 'features.npy', allow_pickle=False)
        keys = [(r['unit'], r['modality']) for r in rows]
        if raw.shape != (len(rows), 512) or len(set(keys)) != len(keys) or not np.isfinite(raw).all():
            raise ValueError('HARn pooled feature identity/shape differs')
        self.units = np.asarray(sorted({u for u, _ in keys}))
        if len(self.units) != 3098 or any(not u.startswith('Training/HARn/') for u in self.units):
            raise ValueError('HARn scope must be3098 officialTrain units')
        self.features = np.zeros((len(self.units), 1, 2, 512), np.float32)
        self.mask = np.zeros((len(self.units), 1, 2), bool)
        lookup = {u: i for i, u in enumerate(self.units)}
        for i, (unit, modality) in enumerate(keys):
            view = ('Depth_Color', 'IR').index(modality)
            self.features[lookup[unit], 0, view] = raw[i]
            self.mask[lookup[unit], 0, view] = True
        self.subjects = np.asarray([subject_id(u) for u in self.units])
        # Existing crop bank already selects canonical available/aligned views.
        if not self.mask.any(axis=(1, 2)).all():
            raise ValueError('HARn auxiliary recording has no canonical input')
