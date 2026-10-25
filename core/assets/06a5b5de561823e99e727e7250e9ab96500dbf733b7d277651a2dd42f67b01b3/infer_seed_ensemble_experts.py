#!/usr/bin/env python3
"""Load four independent experts with the fixed equal-seed HARn checkpoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import infer_adapted_experts as adapted
from scripts import infer_experts as engine
from scripts.run_harn_seed_ensemble import KIND


def resolve_config(config):
    if (set(config) != {"schema_version", "profile", "checkpoints", "features"}
            or config["schema_version"] != 1 or config["profile"] != "independent_equal_seed_harn"
            or set(config["checkpoints"]) != set(engine.SCOPES)):
        raise ValueError("expected complete independent equal-seed configuration")
    checkpoints = {}
    for name, value in config["checkpoints"].items():
        if not isinstance(value, str) or not value:
            raise ValueError("checkpoint path must be nonempty")
        path = (ROOT / value).resolve()
        if not path.is_file() or path.suffix != ".json":
            raise ValueError("checkpoint must be an existing model descriptor")
        checkpoints[name] = path
    descriptor = json.loads(checkpoints["harn"].read_text())
    if (descriptor.get("kind") != KIND or descriptor.get("visual_weights") != [.5, .5]
            or set(descriptor.get("classifiers", {})) != {"original", "repeated"}):
        raise ValueError("HARn checkpoint is not the frozen equal two-seed model")
    emotion = json.loads(checkpoints["emotion"].read_text()).get("model_kind")
    if emotion not in (adapted.EMOTION_BASE, adapted.EMOTION_RECIPE):
        raise ValueError("unsupported emotion checkpoint")
    expected = adapted.BASE_FEATURES | {"harn_middle"}
    if emotion == adapted.EMOTION_RECIPE:
        expected |= {"emotion_visual"}
    if set(config["features"]) != expected:
        raise ValueError("feature coverage differs from the exact selected experts")
    features = {}
    for name, value in config["features"].items():
        if not isinstance(value, str) or not value:
            raise ValueError("feature path must be nonempty")
        path = (ROOT / value).resolve()
        if not path.is_file() or path.suffix != (".json" if name == "harn_middle" else ".npz"):
            raise ValueError("feature input is not the declared archive or middle manifest")
        features[name] = path
    return checkpoints, features


def stage_commands(qa, checkpoints, features, work, workers):
    descriptor = json.loads(checkpoints["harn"].read_text())
    # Reuse the three unchanged experts; the temporary single-tail command is never run.
    original = Path(descriptor["classifiers"]["original"]["path"])
    commands = adapted.stage_commands(qa, {**checkpoints, "harn": original}, features, work, workers)
    commands["harn"] = [sys.executable, str(ROOT / "scripts/run_harn_seed_ensemble.py"), "infer",
                        "--checkpoint", str(checkpoints["harn"]), "--qa-csv", str(qa),
                        "--skeleton", str(features["skeleton"]), "--visual", str(features["harn_middle"]),
                        "--output-dir", str(work / "harn")]
    return commands


def main():
    if not any(value == "--config" or value.startswith("--config=") for value in sys.argv[1:]):
        raise ValueError("equal-seed inference requires an explicit frozen config")
    engine.resolve_config = resolve_config
    engine.stage_commands = stage_commands
    engine.SOURCES = [*engine.SOURCES, "scripts/infer_adapted_experts.py", "scripts/infer_seed_ensemble_experts.py"]
    return engine.main()


if __name__ == "__main__":
    raise SystemExit(main())
