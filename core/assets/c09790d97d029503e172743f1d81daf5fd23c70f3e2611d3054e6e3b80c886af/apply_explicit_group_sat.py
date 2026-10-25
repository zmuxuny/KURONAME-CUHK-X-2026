#!/usr/bin/env python3
"""Apply frozen frequency-weighted minimum-edit SAT with explicit acquisition groups."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor/server_sat_v1')]

import numpy as np
import pandas as pd
from global_sat import CATEGORIES, LETTERS, atoms

from scripts import apply_scored_sat_v3 as legacy


def complete_target_features(frame):
    """Keep all visible occurrence context, skipping unsupported target clips only."""
    hau = frame.loc[frame.source.eq('HAU')]
    unsupported = set()
    occurrences = defaultdict(list)
    for row in hau.loc[~hau.category.eq('emotion')].itertuples():
        for letter in LETTERS:
            values = atoms(getattr(row, letter))
            if row.category in (*CATEGORIES, 'sequence') and not values:
                unsupported.add(row.clip)
            for value in set(values):
                occurrences[(row.scene, value)].append((row.clip, row.qa_id, row.category))
    matrix, metadata = [], []
    for row in hau.loc[hau.category.isin(CATEGORIES) & ~hau['clip'].isin(unsupported)].itertuples():
        raw = []
        for letter in LETTERS:
            values, atom_features = atoms(getattr(row, letter)), []
            for value in values:
                items = occurrences[(row.scene, value)]
                sibling = [v for v in items if v[0] != row.clip]
                same = [v for v in items if v[0] == row.clip and v[1] != row.qa_id]
                counts = [len(sibling), len({v[1] for v in sibling}), len({v[0] for v in sibling}), len(items), len(same)]
                for category in (*CATEGORIES, 'sequence'):
                    counts += [sum(v[2] == category for v in sibling), len({v[1] for v in sibling if v[2] == category}), sum(v[2] == category for v in same)]
                atom_features.append(counts)
            af = np.asarray(atom_features, dtype=np.float32)
            raw.append(np.concatenate([af.min(0), af.mean(0), af.max(0), af.sum(0), [len(values)]]))
        raw = np.asarray(raw, dtype=np.float32)
        for i, letter in enumerate(LETTERS):
            matrix.append(np.concatenate([raw[i], (raw < raw[i]).sum(0), raw[i] - raw.mean(0), [row.category == c for c in CATEGORIES]]))
            metadata.append({'qa_id': row.qa_id, 'clip': row.clip, 'scene': row.scene, 'category': row.category, 'subject': row.subject, 'letter': letter})
    result = np.asarray(matrix, dtype=np.float32).reshape(-1, 210)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite frequency features')
    return result, pd.DataFrame(metadata, columns=['qa_id', 'clip', 'scene', 'category', 'subject', 'letter'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['qa-csv', 'parent-csv', 'profile', 'output-dir', 'scene-manifest']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.profile.read_text())
    if config['mode'] != 'sat_min_edit' or config['layout'] != 'explicit_or_independent':
        raise ValueError('Explicit-group minimum-edit profile required')
    qa = pd.read_csv(args.qa_csv, dtype=str, keep_default_na=False)
    groups = pd.read_csv(args.scene_manifest, dtype=str, keep_default_na=False)
    if set(groups.columns) != {'clip', 'group'} or not groups['clip'].is_unique or groups['group'].str.strip().eq('').any():
        raise ValueError('A unique explicit clip/group table is required')
    expected = set(qa.loc[qa.source.eq('HAU'), 'path'].map(legacy.clip_key))
    if set(groups['clip']) != expected:
        raise ValueError('Explicit groups must exactly cover the current HAU recordings')
    paths = [Path(__file__), ROOT / 'scripts/apply_scored_sat_v3.py',
             *[ROOT / 'vendor/server_sat_v1' / (name + '.py') for name in ['frequency_model', 'global_sat', 'consistency_repair', 'posterior_decode']],
             args.qa_csv, args.parent_csv, args.profile, args.scene_manifest]
    hashes = {str(p.resolve()): legacy.digest(p) for p in paths}
    legacy.features = complete_target_features
    legacy.main()
    for path, digest in hashes.items():
        if legacy.digest(path) != digest:
            raise ValueError('SAT source/input changed during execution')
    snapshot = args.output_dir / 'source_snapshot'
    snapshot.mkdir()
    for path in paths[:6]:
        (snapshot / path.name).write_bytes(path.read_bytes())
    (snapshot / 'pip_freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze', '--all'], text=True))
    (args.output_dir / 'explicit_group_receipt.json').write_text(json.dumps({'source_input_hashes': hashes,
        'command': sys.argv, 'python': sys.executable, 'groups': int(groups['group'].nunique()),
        'group_derivation': 'Caller supplied only; IDs and adjacent recordings never infer membership.',
        'features': 'Same210columns and fixed3forests. Partial-option clips retain parent answers; their supplied atoms remain occurrence context for complete targets.',
        'new_fits': 0, 'scope': 'Computed parent to minimum-edit SAT, not raw upstream inference or a new score.'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
