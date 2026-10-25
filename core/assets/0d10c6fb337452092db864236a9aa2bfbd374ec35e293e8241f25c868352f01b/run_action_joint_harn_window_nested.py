#!/usr/bin/env python3
"""Reuse frozen nested selection with explicitly matched HARn auxiliary training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_action_window_nested as runner

from cuhkx.action_joint_harn_head import HarnWindowBank, JointHarnWindowClassifier
from cuhkx.validation import sha256_file


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, required=True)
    args, _ = parser.parse_known_args()
    config = json.loads(args.config.read_text())
    path = runner.ROOT / config["auxiliary_bank"] / "manifest.json"
    expected = next(entry["sha256"] for entry in config["inputs"] if entry["path"] == str(path.relative_to(runner.ROOT)))
    assert sha256_file(path) == expected
    bank = HarnWindowBank(path.parent)
    original_fit = runner.fit_scope

    def matched_scope(qa, prepared, poisoned, compiled, visual, train, valid, settings, seed, folder, sources):
        auxiliary = bank.scope(train, valid)
        runner.WindowPartialActionClassifier = lambda head_config, head_seed: JointHarnWindowClassifier(head_config, head_seed, auxiliary)
        result = original_fit(qa, prepared, poisoned, compiled, visual, train, valid, settings, seed, folder, sources)
        assert result[0]["presence"].auxiliary_scope == auxiliary[3]
        runner.write_json(folder / "auxiliary_scope.json", auxiliary[3])
        return result

    runner.fit_scope = matched_scope
    runner.KIND = "cuhkx_videomae_action_joint_harn_window_nested_v1"
    runner.COMPONENTS = [*runner.COMPONENTS, "HARn_primary_action_joint_supervision"]
    runner.main()


if __name__ == "__main__":
    main()
