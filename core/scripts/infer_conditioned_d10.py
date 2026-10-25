#!/usr/bin/env python3
"""Infer matched temporal forests and D10 from current questions and raw logits."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import joblib
import numpy as np
import pandas as pd

from cuhkx.acquisition_v2 import read_acquisition
from cuhkx.inference import question_metadata
from cuhkx.metadata import clip_key, feature_unit
from cuhkx.repeated_temporal import share_repeated_pair_probabilities
from cuhkx.server_temporal_d6_residual import ANSWERS, BITS, PAIRS, logit
from cuhkx.server_temporal_d10_backoff import predict
from cuhkx.validation import sha256_file
from scripts.run_hau_pairwise_temporal import make_pair_rows, question_classes
from scripts.temporal_checkpoint import load_logits


def infer(questions, maps, groups, bundle, prior_model, config):
    frame = question_metadata(questions)
    frame = frame.loc[frame.source.eq('HAU') & frame.category.eq('sequence')].reset_index(drop=True)
    if frame.empty:
        raise ValueError('D10 requires an active sequence question')
    frame['unit'] = frame.path.map(lambda path: feature_unit(path, training=False))
    frame['repeat_group'] = frame.path.map(lambda path: groups[clip_key(path)])
    if any(set(frame.unit) - set(source) for source in maps):
        raise ValueError('Current temporal features do not cover sequence questions')
    classes = question_classes(frame)
    pairs = make_pair_rows(frame, classes, maps, training=False)
    forward, reverse = np.asarray(pairs.forward, np.float32), np.asarray(pairs.reverse, np.float32)
    probabilities = []
    if bundle['config']['sibling_weight'] != config['sibling_weight']:
        raise ValueError('D10 parent pooling policy differs')
    for model in bundle['models']['control']:
        if list(model.classes_) != [0, 1] or model.n_features_in_ != forward.shape[1]:
            raise ValueError('Matched temporal forest schema differs')
        model.n_jobs = 1
        probabilities.append((.5 * (model.predict_proba(forward)[:, 1]
                                   + 1 - model.predict_proba(reverse)[:, 1])).reshape(-1, 6))
    raw = np.asarray(probabilities)
    pooled, shared = share_repeated_pair_probabilities(raw, classes, frame.repeat_group.to_numpy(),
        family_indices=[0, 1], sibling_weight=config['sibling_weight'])
    counts, smoothing = bundle['counts'], config['prior_smoothing']
    prior = np.asarray([[np.log((counts.get((int(c[a]), int(c[b])), 0) + smoothing)
                               / (counts.get((int(c[b]), int(c[a])), 0) + smoothing))
                         for a, b in PAIRS] for c in classes])
    prediction, scores, support, parent_scores = predict(prior_model, pooled, prior, classes)
    result = frame[['qa_id']].copy()
    result['matched_parent'] = ANSWERS[((logit(pooled) + prior) @ BITS.T).argmax(1)]
    result['prediction'] = prediction
    arrays = {'qa_ids': frame.qa_id.to_numpy(dtype=str), 'classes': classes, 'raw_families': raw,
              'pooled': pooled, 'prior_logits': prior, 'scores': scores, 'support': support,
              'parent_scores': parent_scores, 'shared_pairs': shared}
    return result, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--descriptor', type=Path, required=True)
    parser.add_argument('--qa-csv', type=Path, required=True)
    parser.add_argument('--logits', type=Path, action='append', required=True)
    parser.add_argument('--acquisition', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.descriptor.read_text())
    if config['kind'] != 'cuhkx_conditioned_D10_v1':
        raise ValueError('Unsupported D10 descriptor')
    paths = [args.descriptor, args.qa_csv, *args.logits, Path(__file__)]
    for name in ['parent_model', 'prior_model']:
        record = config[name]
        path = Path(record['path'])
        if not path.is_absolute():
            path = args.descriptor.parent / path
        if sha256_file(path) != record['sha256']:
            raise ValueError('D10 fitted model identity changed')
        record['path'] = str(path)
        paths.append(path)
    questions = question_metadata(pd.read_csv(args.qa_csv, dtype=str, keep_default_na=False))
    clips = set(questions.loc[questions.source.eq('HAU'), 'path'].map(clip_key))
    if args.acquisition:
        groups = read_acquisition(questions, args.acquisition).groups
        paths.append(args.acquisition)
    else:
        groups = {clip: clip for clip in clips}
    hashes = {str(path): sha256_file(path) for path in paths}
    maps, contracts = load_logits(args.logits, config['temperature'], config['feature_contracts'])
    bundle = joblib.load(config['parent_model']['path'])
    prior_model = json.loads(Path(config['prior_model']['path']).read_text())
    if prior_model['training_subjects'] != bundle['training_subjects']:
        raise ValueError('Matched forest/count-model training people differ')
    result, arrays = infer(questions, maps, groups, bundle, prior_model, config)
    args.output.mkdir(parents=True, exist_ok=False)
    result.to_csv(args.output / 'predictions.csv', index=False)
    np.savez_compressed(args.output / 'pair_scores.npz', **arrays)
    for path, digest in hashes.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError('D10 inference input changed')
    report = {'status': 'complete_conditioned_D10', 'rows': len(result), 'new_fits': 0,
        'condition': 'grouped' if args.acquisition else 'independent', 'feature_contracts': contracts,
        'inputs_sha256': hashes, 'changed_vs_matched_parent': int(result.prediction.ne(result.matched_parent).sum()),
        'scope': 'Actual matched forests and D10 on current metadata/logits. No D6 calibration, inferred groups, labels or historical answers. KUNO upstream roster unknown.'}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
