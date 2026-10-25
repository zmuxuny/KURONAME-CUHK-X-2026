#!/usr/bin/env python3
"""Freeze the already trained local A/B models for an explicit-scene component run."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from cuhkx.validation import sha256_file
from scripts.research_write_guard import install
from scripts.run_local_emotion_distillation import write


def entry(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha256_file(path), 'bytes': path.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    install(Path('/root/workspace/CUHK-X'))
    root = ROOT / 'runs/20260912_local_emotion_distillation_v1'
    final = root / 'scopes/final'
    parent = ROOT / 'runs/20260905_emotion_recipe_nested_v6/inference_policy.json'
    if json.loads((root / 'report.json').read_text())['status'] != 'completed_fixed_QA_vs_crossfit_distillation':
        raise ValueError('student experiment incomplete')
    if json.loads((ROOT / 'runs/20260912_local_emotion_full_fusion_replay_v1/report.json').read_text())['status'] != 'passed_fixed_full_scene_state_fusion':
        raise ValueError('independent fixed fusion replay incomplete')
    sources = [ROOT / name for name in ('configs/local_emotion_distillation_v1.json', 'configs/local_emotion_full_fusion_v1.json',
        'scripts/infer_local_emotion_full_fusion_v2.py', 'scripts/prepare_local_emotion_final_inference_v2.py',
        'scripts/assess_local_emotion_full_fusion_v2.py', 'scripts/run_local_emotion_distillation.py',
        'src/cuhkx/local_emotion_student.py', 'src/cuhkx/local_emotion_teacher.py', 'src/cuhkx/acquisition_v2.py',
        'src/cuhkx/acquisition.py', 'scripts/research_write_guard.py')]
    sources += [root / 'report.json', root / 'completion_freeze.json',
                ROOT / 'runs/20260912_local_emotion_replay_v1/report.json',
                ROOT / 'runs/20260912_local_emotion_full_fusion_v1/report.json',
                ROOT / 'runs/20260912_local_emotion_full_fusion_replay_v1/report.json']
    # Freeze every source imported by the historical parent; no Test prediction table.
    policy = json.loads(parent.read_text())
    sources += [ROOT / path for path in policy['recipe_source_hashes']]
    descriptor = {'kind': 'cuhkx_local_emotion_full_state_fusion_final_v1', 'parent_policy': entry(parent),
        'students': {arm: entry(final / f'{arm}.pt') for arm in ('A', 'B')},
        'normalizer': entry(final / 'normalizer.joblib'), 'student_scope': entry(final / 'scope.json'),
        'parent_weight': .5, 'student_weight': .5, 'sources': [entry(path) for path in sorted(set(sources))],
        'default_candidate': 'A', 'B': 'predeclared control retained, no weight search',
        'selection_evidence': {'local_A': '339/809', 'local_B': '341/809', 'full_parent': '749/809',
                              'full_A': '752/809', 'full_B': '750/809', 'all_CIs_cross_zero': True},
        'decision_before_Test_component': datetime.now(UTC).isoformat(),
        'pseudo_dependency': {'training': False, 'tuning': False, 'selection': False}}
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'descriptor.json', descriptor)
    best = json.loads((ROOT / 'configs/checkpoint_inference_three_family_harn_v1.json').read_text())
    paths = {'qa_csv': ROOT / 'data/raw/kaggle/test_qa.csv',
        'scene_manifest': ROOT / 'runs/20260909_conditioned_four_expert_execution_v3/inputs/historical_release_scene_fixture.csv',
        **{name: ROOT / best['features'][name if name != 'visual' else 'emotion_visual'] for name in ('imu', 'skeleton', 'radar', 'visual')}}
    write(out / 'inputs.json', {name: entry(path) for name, path in paths.items()})
    write(out / 'execution_freeze.json', {'files': [entry(out / 'descriptor.json'), entry(out / 'inputs.json')],
        'protection': 'all old models and caches read-only; only this new root receives outputs; research_write_guard',
        'recovery': 'retain source assets; isolate new run; verify every descriptor/input sha256', 'fits': 0})
    print(json.dumps({'descriptor': str(out / 'descriptor.json'), 'sha256': sha256_file(out / 'descriptor.json')}))


if __name__ == '__main__':
    main()
