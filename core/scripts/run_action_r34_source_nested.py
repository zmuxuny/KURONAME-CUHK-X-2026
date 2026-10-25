#!/usr/bin/env python3
"""Matched source-only R34 representation with the existing nonlinear joint recipe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_action_window_nested as runner

from cuhkx.action_r34_source_bank import R34SourceHarnBank, read_window_bank
from cuhkx.action_variable_window_head import VariableWindowJointClassifier
from cuhkx.validation import sha256_file


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', required=True, type=Path)
    args, _ = parser.parse_known_args()
    config = json.loads(args.config.read_text())
    if config['presence']['channels'] != 512 or config['windows'] != 8:
        raise ValueError('Requires explicit R34 dimensions')
    bank_path = runner.ROOT / config['window_bank']
    manifest = json.loads((bank_path / 'manifest.json').read_text())
    arm = config['arm']
    identity = manifest['source_models'][arm]['sha256'] if arm != 'generic' else None
    bank = R34SourceHarnBank(runner.ROOT / config['auxiliary_bank'], identity)
    for entry in config['inputs']:
        if sha256_file(runner.ROOT / entry['path']) != entry['sha256']:
            raise ValueError('Declared experiment input changed')
    original_fit = runner.fit_scope

    def matched_scope(qa, prepared, poisoned, compiled, visual, train, valid, settings, seed, folder, sources):
        auxiliary = bank.scope(train, valid)
        runner.WindowPartialActionClassifier = lambda head_config, head_seed: VariableWindowJointClassifier(
            head_config, head_seed, auxiliary)
        result = original_fit(qa, prepared, poisoned, compiled, visual, train, valid, settings, seed, folder, sources)
        runner.write_json(folder / 'auxiliary_scope.json', auxiliary[3])
        return result

    runner.fit_scope = matched_scope
    runner.read_window_bank = lambda directory: read_window_bank(directory, arm)
    runner.KIND = 'cuhkx_action_r34_source_nested_v1'
    runner.COMPONENTS = [*runner.COMPONENTS, 'HARn_primary_action_joint_supervision']
    runner.main()


if __name__ == '__main__':
    main()
