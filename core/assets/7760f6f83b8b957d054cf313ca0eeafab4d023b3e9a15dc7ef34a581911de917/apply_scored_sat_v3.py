"""Apply frozen SAT stages to computed parent predictions; explicit group columns use bracket lookup; partial-option recordings preserved."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor/server_sat_v1'))

import joblib
import numpy as np
import pandas as pd
from consistency_repair import repair
from frequency_model import features
from global_sat import CATEGORIES, audit, clip_key, prepare


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qa-csv', type=Path, required=True)
    parser.add_argument('--parent-csv', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--scene-manifest', type=Path)
    args = parser.parse_args()
    config = json.loads(args.profile.read_text())
    if config['kind'] != 'cuhkx_sat_scored_postprocess_v1':
        raise ValueError('Unexpected SAT profile')
    qa = pd.read_csv(args.qa_csv, keep_default_na=False, dtype=str)
    required = ['qa_id', 'source', 'path', 'category', 'A', 'B', 'C', 'D']
    qa = qa[required].copy()  # No labels, user predictions, or hidden targets reach SAT/features.
    parent = pd.read_csv(args.parent_csv, keep_default_na=False, dtype=str)
    if qa.qa_id.duplicated().any() or parent.qa_id.duplicated().any() or set(qa.qa_id) != set(parent.qa_id):
        raise ValueError('Parent must cover precisely the supplied unique question IDs')
    parent = parent.set_index('qa_id').loc[qa.qa_id].reset_index()[['qa_id', 'prediction']]
    for row in qa.merge(parent, on='qa_id', validate='one_to_one').itertuples():
        value = row.prediction
        if not value or set(value) - set('ABCD') or len(set(value)) != len(value) or any(not str(getattr(row, c)).strip() for c in value):
            raise ValueError('Invalid parent answer')
        if row.category == 'sequence' and sorted(value) != [c for c in 'ABCD' if str(getattr(row, c)).strip()]:
            raise ValueError('Sequence parent must permute exactly its available options')
        if row.category not in ('sequence', 'multi') and len(value) != 1:
            raise ValueError('Single-answer parent must select one option')
    if config['layout'] == 'frozen_official_testing_20260912':
        if digest(args.qa_csv) != config['qa_sha256'] or args.scene_manifest:
            raise ValueError('Frozen scored layout accepts only its exact official QA version; use general profile for new inputs')
        frame = prepare(qa)
    elif config['layout'] == 'explicit_or_independent':
        frame = qa.assign(clip=qa.path.map(clip_key), subject='opaque')
        frame['scene'] = frame['clip']
        if args.scene_manifest:
            groups = pd.read_csv(args.scene_manifest, dtype=str, keep_default_na=False)
            if groups['clip'].duplicated().any() or not set(frame.loc[frame.source.eq('HAU'), 'clip']) <= set(groups['clip']):
                raise ValueError('Explicit scene mapping must uniquely cover all HAU recordings')
            mapping = groups.set_index('clip').group.to_dict()
            frame['scene'] = [mapping.get(c, c) for c in frame['clip']]
    else:
        raise ValueError('Unknown layout protocol')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    parent_map = dict(zip(parent.qa_id, parent.prediction, strict=True))
    actionable = frame.loc[frame.source.eq('HAU') & frame.category.isin(CATEGORIES), 'clip']
    selected_frame = frame.loc[frame['clip'].isin(set(actionable))]
    skipped_partial_clips = []
    if config['layout'] == 'explicit_or_independent':
        incomplete = selected_frame.category.isin([*CATEGORIES, 'sequence']) & selected_frame[['A', 'B', 'C', 'D']].apply(lambda c: c.str.strip().eq('')).any(axis=1)
        skipped_partial_clips = sorted(set(selected_frame.loc[incomplete, 'clip']))
        selected_frame = selected_frame.loc[~selected_frame['clip'].isin(skipped_partial_clips)]
    model_hashes = []
    if selected_frame.empty:
        rows, groups = pd.DataFrame(columns=['qa_id', 'prediction']), pd.DataFrame()
    elif config['mode'] == 'sat_clip':
        rows, groups = audit(selected_frame, 'clip', parent_map)
    elif config['mode'] == 'sat_min_edit':
        matrix, meta = features(frame)
        probabilities = []
        for row in config['models']:
            path = ROOT / row['path']
            if digest(path) != row['sha256']:
                raise ValueError('SAT frequency model hash mismatch')
            model = joblib.load(path)
            probabilities.append(model.predict_proba(matrix)[:, 1].astype(np.float32))
            model_hashes.append(row)
        p = np.stack(probabilities, axis=1)
        np.savez_compressed(args.output_dir / 'frequency_probabilities.npz', probabilities=p)
        rows = repair(selected_frame, meta, p.mean(1), parent_map)
        _, groups = audit(selected_frame, 'clip', parent_map)
    else:
        raise ValueError('Unknown SAT mode')
    overlay = rows.set_index('qa_id').prediction.to_dict()
    result = parent.copy()
    result['prediction'] = [overlay.get(q, v) for q, v in zip(parent.qa_id, parent.prediction, strict=True)]
    result.to_csv(args.output_dir / 'submission.csv', index=False, lineterminator='\n')
    rows.to_csv(args.output_dir / 'stage_decisions.csv', index=False)
    groups.to_csv(args.output_dir / 'group_audit.csv', index=False)
    parent.to_csv(args.output_dir / 'computed_parent.csv', index=False)
    report = {'rows': len(result), 'mode': config['mode'], 'layout': config['layout'],
              'qa_sha256': digest(args.qa_csv), 'parent_sha256': digest(args.parent_csv),
              'profile_sha256': digest(args.profile), 'model_hashes': model_hashes,
              'source_sha256': digest(__file__), 'output_sha256': digest(args.output_dir / 'submission.csv'),
              'partial_option_recordings_preserved': skipped_partial_clips,
              'partial_option_policy': 'Preserve complete parent answers for recordings outside the frozen four-option SAT contract',
              'changed_from_computed_parent': int(result.prediction.ne(parent.prediction).sum()),
              'new_fits': 0, 'pseudo_dependency': False,
              'scope': 'SAT downstream applied to supplied computed parent; upstream provenance must accompany caller run'}
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
