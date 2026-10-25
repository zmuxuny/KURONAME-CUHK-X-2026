"""Load keyed whole-clip HARn descriptors without interpreting action path labels."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cuhkx.harn_nested import FeatureBank
from cuhkx.harn_videomae import SPEC
from cuhkx.validation import sha256_file


class InferenceVideoMAEBank(FeatureBank):
    @classmethod
    def load(cls, skeleton_path, visual_path):
        directory = Path(visual_path).parent
        manifest = json.loads(Path(visual_path).read_text())
        if (manifest['kind'] not in {'cuhkx_videomae_ssv2_harn_train_whole_v1',
                                     'cuhkx_videomae_ssv2_harn_test_whole_v1'}
                or manifest['windows'] != 1 or manifest['labels_used']):
            raise ValueError('expected the declared label-free whole-clip HARn bank')
        for entry in manifest['files']:
            path = directory / entry['path']
            if path.stat().st_size != entry['bytes'] or sha256_file(path) != entry['sha256']:
                raise ValueError('HARn descriptor input changed')
        with np.load(skeleton_path, allow_pickle=False) as archive:
            units, skeleton, names = archive['units'].astype(str), archive['X'].astype(np.float32), archive['feature_names'].astype(str)
        if skeleton.shape != (len(units), len(names)) or len(set(units)) != len(units) or not np.isfinite(skeleton).all():
            raise ValueError('invalid Skeleton descriptor axes')
        with np.load(directory / 'metadata.npz', allow_pickle=False) as archive:
            meta = {key: archive[key] for key in archive.files}
        features = np.load(directory / 'features.npy', allow_pickle=False)
        complete = np.load(directory / 'completed.npy', allow_pickle=False)
        count = len(meta['units'])
        if (features.shape != (count, 1, 768) or complete.shape != (count,) or not complete.all()
                or not np.isfinite(features).all() or meta['available'].dtype != bool
                or meta['alignment_ok'].dtype != bool):
            raise ValueError('incomplete HARn whole-clip descriptors')
        keys = list(zip(meta['units'].astype(str), meta['modalities'].astype(str), strict=True))
        lookup = {key: row for row, key in enumerate(keys)}
        if len(lookup) != count or set(meta['modalities']) != {'Depth_Color', 'IR'}:
            raise ValueError('duplicate or unknown HARn view keys')
        visual_units = np.asarray(sorted({key[0] for key in keys}))
        visual = np.zeros((len(visual_units), 2, 768), dtype=np.float32)
        mask = np.zeros((len(visual_units), 2), dtype=bool)
        for index, unit in enumerate(visual_units):
            rows = [lookup[(unit, view)] for view in ('Depth_Color', 'IR')]
            visual[index] = features[rows, 0]
            mask[index] = meta['available'][rows]
            if mask[index].all() and not meta['alignment_ok'][rows].all():
                mask[index, 0] = False
        visible = mask.any(axis=1)
        matrix = np.concatenate((visual[visible].reshape(int(visible.sum()), 1536), mask[visible]), axis=1)
        return cls(units, skeleton, names, visual_units[visible], matrix, SPEC)
