#!/usr/bin/env python3
"""Infer a preselected external-source final R34 model from original HARn media."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
import pandas as pd
import torch

from cuhkx.external_candidate_contract import selected_external_recipe
from cuhkx.harn_nested import harn_rows
from cuhkx.harn_tail import configure_torch
from cuhkx.metadata import feature_unit
from cuhkx.r34_adaptation import restore
from cuhkx.r34_video import sample_preprocessed
from cuhkx.spatial_training import write_json
from cuhkx.validation import sha256_file
from scripts.evaluate_r34_adaptation import probabilities


def run(args, recipe_validator=selected_external_recipe):
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(args.config.read_text())
    protected = {}

    def read(path, expected=None):
        path = Path(path).resolve()
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError('Input identity changed: ' + str(path))
        protected[str(path)] = digest
        return path

    read(args.config)
    source, fusion = recipe_validator(cfg, read)
    read(args.qa_csv)
    model_path = read(cfg['r34_model']['path'], cfg['r34_model']['sha256'])
    receipt = json.loads(read(model_path.with_suffix('.json'), cfg['r34_model']['receipt_sha256']).read_text())
    assert receipt['sha256'] == cfg['r34_model']['sha256']
    assert receipt['external_source_arm'] == source
    assert receipt['external_source_sha256'] == cfg['external_source']['sha256']
    adaptation = json.loads(read(cfg['r34_config']['path'], cfg['r34_config']['sha256']).read_text())
    read(adaptation['weights'], adaptation['weights_sha256'])
    raw = json.loads(read(adaptation['source_manifest'], adaptation['source_sha256']).read_text())
    qa = harn_rows(pd.read_csv(args.qa_csv, dtype=str, keep_default_na=False))
    units = sorted({feature_unit(p, training=False) for p in qa.path})
    assert units and all(u.startswith('Testing/') for u in units)
    grouped = {u: [] for u in units}
    keys = set()
    for record in raw['records']:
        if record['unit'] not in grouped:
            continue
        assert record['source'] == 'HARn'
        key = (record['unit'], record['modality'])
        assert key not in keys
        keys.add(key)
        if record['available']:
            record = dict(record)
            path = Path(adaptation['data_root']) / record['unit'] / record['modality'] / (record['modality'] + '.mp4')
            record['path'] = str(read(path, record['raw']['sha256']))
            grouped[record['unit']].append(record)
    assert keys == {(u, m) for u in units for m in ['Depth_Color', 'IR']}
    configure_torch()
    model, payload = restore(adaptation, model_path)
    assert payload['arm'] == 'R' and payload['outer'] == 'final'
    assert payload['external_source_arm'] == source
    assert payload['external_source_sha256'] == cfg['external_source']['sha256']
    assert payload['training_subjects'] == sorted(adaptation['subjects'])
    assert all(u.startswith('Training/') for u in payload['training_units'])
    for field in ['training_subjects', 'training_units', 'supported_classes', 'steps', 'initial_state_sha256', 'sample_order_sha256']:
        assert payload[field] == receipt[field], field
    classes = payload['supported_classes']
    matrix = np.zeros((len(units), 44), dtype=np.float32)
    present = np.asarray([bool(grouped[u]) for u in units])
    view_values, view_units, view_modalities, crop_digests = [], [], [], []
    # Match the validation producer's 16-unit batches and two-view microbatches.
    for start in range(0, len(units), 16):
        chosen = units[start:start + 16]
        tensors, owners = [], []
        for owner, unit in enumerate(chosen):
            for record in grouped[unit]:
                tensor = sample_preprocessed(record)
                tensors.append(tensor)
                owners.append(owner)
                view_units.append(unit)
                view_modalities.append(record['modality'])
                import hashlib
                crop_digests.append(hashlib.sha256(tensor.numpy().tobytes()).hexdigest())
        if not tensors:
            continue
        values = probabilities(model, torch.cat(tensors), classes)
        view_values.extend(values)
        owners = np.asarray(owners)
        for i, unit in enumerate(chosen):
            if grouped[unit]:
                matrix[start + i] = values[owners == i].mean(0)
        print('raw R34 units', min(start + 16, len(units)), '/', len(units), flush=True)
    np.testing.assert_allclose(matrix[present].sum(1), 1., rtol=0, atol=4 * np.finfo(np.float32).eps)
    assert np.isfinite(matrix).all() and (matrix >= 0).all()
    np.savez_compressed(args.output / 'probabilities.npz', units=np.asarray(units), probabilities=matrix,
                        present=present, classes=np.asarray(classes))
    np.savez_compressed(args.output / 'views.npz', units=np.asarray(view_units), modalities=np.asarray(view_modalities),
                        probabilities=np.asarray(view_values), preprocessed_sha256=np.asarray(crop_digests))
    write_json(args.output / 'records.json', {'records': [r for rs in grouped.values() for r in rs]})
    for path in [Path(__file__), ROOT / 'scripts/evaluate_r34_adaptation.py', *[ROOT / 'src/cuhkx' / n for n in
                 ['r34_adaptation.py', 'r34_video.py', 'generic_video_windows.py', 'video.py', 'harn_tail.py', 'metadata.py', 'external_candidate_contract.py']]]:
        read(path)
    for path, digest in protected.items():
        assert sha256_file(Path(path)) == digest, path
    write_json(args.output / 'report.json', {'status': 'complete_raw_final_external_R34_HARn', 'units': len(units),
               'available_units': int(present.sum()), 'views': len(view_units), 'input_sha256': protected,
               'model_sha256': cfg['r34_model']['sha256'], 'external_source_arm': source,
               'external_source_sha256': cfg['external_source']['sha256'], 'selected_fusion': fusion, 'new_fits': 0, 'Test_truth_read': False,
               'scope': 'Actual original media, fixed manifest ROI/frame producer, same16-unit/two-view validation batching. Frozen final R34; missing views explicit. Existing manifest-based media contract, not a new arbitrary-input discovery implementation.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['config', 'qa-csv', 'output']:
        parser.add_argument('--' + name, required=True, type=Path)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
