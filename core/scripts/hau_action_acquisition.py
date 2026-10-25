#!/usr/bin/env python3
"""Run the frozen action expert with explicit unlabelled acquisition groups."""

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from cuhkx import clip_constraints, option_frequency, repeated_actions
from cuhkx.acquisition import Acquisition
from cuhkx.inference import QUESTION_COLUMNS
from scripts import hau_action_checkpoint as frozen
from scripts.temporal_checkpoint import resolve_groups


@contextmanager
def acquisition_scope(acquisition):
    modules = (clip_constraints, option_frequency, repeated_actions, frozen)
    previous = [module.repeated_recording_groups for module in modules]
    previous_resolve = frozen.resolve_groups
    try:
        for module in modules:
            module.repeated_recording_groups = acquisition.groups_for
        frozen.resolve_groups = lambda qa, layout: resolve_groups(qa, layout, acquisition.temporal_manifest())
        yield
    finally:
        for module, function in zip(modules, previous, strict=True):
            module.repeated_recording_groups = function
        frozen.resolve_groups = previous_resolve


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    options, remaining = parser.parse_known_args()
    if not remaining or remaining[0] != "infer":
        raise ValueError("acquisition adapter supports load-only inference")
    qa_path = Path(remaining[remaining.index("--qa-csv") + 1])
    output = Path(remaining[remaining.index("--output-dir") + 1])
    qa = pd.read_csv(qa_path, usecols=QUESTION_COLUMNS, dtype=str, keep_default_na=False)
    acquisition = Acquisition.parse(qa, pd.read_csv(options.scene_manifest, dtype=str, keep_default_na=False))
    original_args = sys.argv
    try:
        sys.argv = [str(ROOT / "scripts/hau_action_checkpoint.py"), *remaining]
        with acquisition_scope(acquisition):
            result = frozen.main()
    finally:
        sys.argv = original_args
    receipt = {"status": "explicit_acquisition_inference_complete",
        "scene_manifest": {"path": str(options.scene_manifest.resolve()),
                           "sha256": frozen.sha256(options.scene_manifest)},
        "clips": len(acquisition.scenes), "groups": acquisition.scenes.group.nunique(),
        "runtime_adapter_sources": {name: frozen.sha256(ROOT / name) for name in
            ("scripts/hau_action_acquisition.py", "src/cuhkx/acquisition.py")},
        "scope": "Fitted weights and frozen source verification retained; grouping functions use supplied acquisition metadata.",
        "accuracy_claim": False}
    (output / "acquisition_adapter.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
