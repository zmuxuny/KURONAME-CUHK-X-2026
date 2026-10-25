#!/usr/bin/env python3
"""Run all four experts with four-model, three-family equal HARn inference."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from scripts import infer_experts as engine
from scripts import infer_seed_ensemble_experts as parent
from scripts.infer_harn_three_family_v2 import KIND, resolve, verify_entries
from scripts.research_write_guard import install


def resolve_config(config):
    if config.get('profile') != 'three_family_harn_equal':
        raise ValueError('expected three-family HARn profile')
    harn = resolve(config['checkpoints']['harn'])
    descriptor = json.loads(harn.read_text())
    if descriptor['kind'] != KIND:
        raise ValueError('unexpected three-family HARn descriptor')
    verify_entries([descriptor['parent_checkpoint']])
    translated = {**config, 'profile': 'independent_equal_seed_harn',
                  'checkpoints': {**config['checkpoints'], 'harn': descriptor['parent_checkpoint']['path']},
                  'features': {name: path for name, path in config['features'].items() if name not in {'harn_r34', 'harn_videomae'}}}
    checkpoints, features = parent.resolve_config(translated)
    for name in ['harn_r34', 'harn_videomae']:
        path = resolve(config['features'][name])
        if not path.is_file() or path.suffix != '.json':
            raise ValueError('extra HARn family requires an existing feature manifest')
        features[name] = path
    checkpoints['harn'] = harn
    return checkpoints, features


def stage_commands(qa, checkpoints, features, work, workers):
    descriptor = json.loads(checkpoints['harn'].read_text())
    original = {**checkpoints, 'harn': resolve(descriptor['parent_checkpoint']['path'])}
    commands = parent.stage_commands(qa, original, features, work, workers)
    commands['harn'] = [sys.executable, str(ROOT / 'scripts/infer_harn_three_family_v2.py'),
                        '--checkpoint', str(checkpoints['harn']), '--qa-csv', str(qa),
                        '--skeleton', str(features['skeleton']), '--visual', str(features['harn_middle']),
                        '--r34', str(features['harn_r34']), '--videomae', str(features['harn_videomae']),
                        '--workers', str(min(workers, 2)), '--output-dir', str(work / 'harn')]
    return {name: [cmd[0], str(ROOT / 'scripts/research_write_guard.py'), *cmd[1:]] for name, cmd in commands.items()}


def main():
    install(Path('/root/workspace/CUHK-X'))
    engine.resolve_config = resolve_config
    engine.stage_commands = stage_commands
    engine.SOURCES = [*engine.SOURCES, 'scripts/infer_adapted_experts.py', 'scripts/infer_seed_ensemble_experts.py',
                      'scripts/infer_three_family_harn_experts_v2.py', 'scripts/infer_harn_three_family_v2.py',
                      'src/cuhkx/harn_three_family.py', 'src/cuhkx/harn_seed_ensemble.py',
                      'src/cuhkx/harn_videomae_inference.py', 'src/cuhkx/harn_videomae.py',
                      'src/cuhkx/r34_tail.py', 'src/cuhkx/r34_video.py', 'scripts/research_write_guard.py']
    return engine.main()


if __name__ == '__main__':
    raise SystemExit(main())
