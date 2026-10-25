#!/usr/bin/env python3
"""Load four independent experts with a fitted HARn visual adaptation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import infer_experts as engine

BASE_FEATURES = engine.FEATURES - {"harn_visual"}
EMOTION_BASE = "cuhkx_emotion_context_checkpoint_v1"
EMOTION_RECIPE = "cuhkx_emotion_recipe_nested_v6"
HARN_MODULES = {"cuhkx_harn_adapted_mixture_v1": "run_harn_adapted_fusion.py",
                "cuhkx_harn_category_adapted_mixture_v1": "run_harn_category_fusion.py"}


def formats(checkpoints):
    emotion = json.loads(checkpoints["emotion"].read_text()).get("model_kind")
    if emotion not in (EMOTION_BASE, EMOTION_RECIPE):
        raise ValueError("emotion checkpoint is outside the independently verified profiles")
    harn = "tail" if checkpoints["harn"].suffix == ".joblib" else json.loads(checkpoints["harn"].read_text()).get("kind")
    if harn != "tail" and harn not in HARN_MODULES:
        raise ValueError("HARn checkpoint is outside the fitted adapted profiles")
    return emotion, harn


def resolve_config(config):
    if (set(config) != {"schema_version", "profile", "checkpoints", "features"}
            or config["schema_version"] != 1 or config["profile"] != "independent_adapted_harn"
            or set(config["checkpoints"]) != set(engine.SCOPES)):
        raise ValueError("expected complete independent adapted checkpoint configuration")
    checkpoints, features = {}, {}
    for name, value in config["checkpoints"].items():
        if not isinstance(value, str) or not value:
            raise ValueError("checkpoint path must be nonempty")
        path = (ROOT / value).resolve()
        valid = {".json", ".joblib"} if name == "harn" else {".json"}
        if not path.is_file() or path.suffix not in valid:
            raise ValueError("checkpoint is missing or is not a model descriptor/state")
        checkpoints[name] = path
    emotion, harn = formats(checkpoints)
    expected = BASE_FEATURES | {"harn_middle"}
    if emotion == EMOTION_RECIPE:
        expected |= {"emotion_visual"}
    if harn != "tail":
        expected |= {"harn_multivideo"}
    if set(config["features"]) != expected:
        raise ValueError("feature archive coverage differs from the exact selected experts")
    for name, value in config["features"].items():
        if not isinstance(value, str) or not value:
            raise ValueError("feature path must be nonempty")
        path = (ROOT / value).resolve()
        if not path.is_file() or path.suffix != (".json" if name == "harn_middle" else ".npz"):
            raise ValueError("feature input is not the declared archive or middle manifest")
        features[name] = path
    return checkpoints, features


def stage_commands(qa, checkpoints, features, work, workers):
    def command(script, *args):
        return [sys.executable, str(ROOT / "scripts" / script), "infer", *map(str, args)]

    emotion, harn = formats(checkpoints)
    commands = {
        "action": command("hau_action_checkpoint.py", "--checkpoint", checkpoints["action"], "--qa-csv", qa,
                           "--logits", features["logits8"], features["logits16"], "--window-features", features["windows16"],
                           features["windows32"], features["windows48"], "--output-dir", work / "action", "--workers", workers),
        "emotion": command("emotion_context_checkpoint.py", "--checkpoint", checkpoints["emotion"], "--test-csv", qa,
                            "--imu", features["imu"], "--skeleton", features["skeleton"], "--radar", features["radar"],
                            "--output-dir", work / "emotion"),
        "sequence": command("temporal_checkpoint.py", "--checkpoint", checkpoints["sequence"], "--qa-csv", qa,
                             "--logits", features["logits8"], "--logits", features["logits16"],
                             "--output-dir", work / "sequence", "--workers", workers),
    }
    if emotion == EMOTION_RECIPE:
        commands["emotion"] = command("run_emotion_recipe_nested.py", "--policy", checkpoints["emotion"], "--test-csv", qa,
                                       "--imu", features["imu"], "--skeleton", features["skeleton"], "--radar", features["radar"],
                                       "--visual", features["emotion_visual"], "--output-dir", work / "emotion")
    if harn == "tail":
        commands["harn"] = command("run_harn_tail_nested.py", "--checkpoint", checkpoints["harn"], "--qa-csv", qa,
                                    "--skeleton", features["skeleton"], "--visual", features["harn_middle"],
                                    "--output-dir", work / "harn", "--workers", workers)
    else:
        commands["harn"] = command(HARN_MODULES[harn], "--checkpoint", checkpoints["harn"], "--qa-csv", qa,
                                    "--skeleton", features["skeleton"], "--generic-visual", features["harn_multivideo"],
                                    "--adapted-visual", features["harn_middle"], "--output-dir", work / "harn")
    return commands


def main():
    if not any(value == "--config" or value.startswith("--config=") for value in sys.argv[1:]):
        raise ValueError("adapted expert inference requires an explicit frozen config")
    engine.resolve_config = resolve_config
    engine.stage_commands = stage_commands
    engine.SOURCES = [*engine.SOURCES, "scripts/infer_adapted_experts.py"]
    return engine.main()


if __name__ == "__main__":
    raise SystemExit(main())
