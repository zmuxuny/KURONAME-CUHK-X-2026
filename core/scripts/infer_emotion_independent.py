#!/usr/bin/env python3
"""Export or run the standalone emotion expert without acquisition metadata."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import joblib
import numpy as np
import pandas as pd

from cuhkx import emotion_independent as mechanism
from cuhkx.emotion_nested import read_sensor
from cuhkx.inference import QUESTION_COLUMNS
from cuhkx.validation import sha256_file
from scripts.run_emotion_nested import validate_output, write_json


def export(run, destination):
    """Export feature names once so inference never needs the Train feature files."""
    report = json.loads((run / "report.json").read_text())
    if report["status"] != "completed_nested_study_not_promoted" or report["models"] != 76:
        raise ValueError("export requires a completed76-model nested study")
    frozen = json.loads((run / "frozen_execution.json").read_text())
    config = frozen["config"]
    record = next(item for item in json.loads((run / "models.json").read_text())
                  if Path(item["path"]).name == "all_train")
    checkpoint = Path(record["path"]) / "models.joblib"
    if sha256_file(checkpoint) != record["checkpoint_sha256"]:
        raise ValueError("final checkpoint hash differs")
    schemas = {}
    for name, feature in config["features"].items():
        path = ROOT / feature["path"]
        if sha256_file(path) != feature["sha256"]:
            raise ValueError("training feature schema source changed")
        with np.load(path, allow_pickle=False) as archive:
            names = archive["feature_names"].astype(str)
        schemas[name] = [value for value in names if re.search(config["fit"]["views"][name], value)]
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(checkpoint, destination / "models.joblib")
    write_json(destination / "manifest.json", {"kind": "cuhkx_independent_emotion_checkpoint_v1",
        "model": {"path": "models.joblib", "sha256": record["checkpoint_sha256"]},
        "fit": config["fit"], "feature_names": schemas,
        "decoder": json.loads((run / "inference_policy.json").read_text()),
        "study": {"path": str(run.resolve()), "report_sha256": sha256_file(run / "report.json"),
                  "execution_sha256": sha256_file(run / "frozen_execution.json")},
        "code": {str(path.relative_to(ROOT)): sha256_file(path) for path in
                 [Path(__file__), ROOT / "src/cuhkx/emotion_independent.py", ROOT / "src/cuhkx/emotion_nested.py",
                  ROOT / "src/cuhkx/emotion_scene_robustness.py", ROOT / "src/cuhkx/inference.py",
                  ROOT / "src/cuhkx/metadata.py", ROOT / "src/cuhkx/actions.py", ROOT / "src/cuhkx/submission.py"]}})


def infer(bundle_path, qa_path, features, output):
    manifest = json.loads((bundle_path / "manifest.json").read_text())
    if manifest["kind"] != "cuhkx_independent_emotion_checkpoint_v1":
        raise ValueError("unsupported standalone emotion bundle")
    if output.exists() or output.with_suffix(".report.json").exists():
        raise FileExistsError("new standalone inference output required")
    hashes = {}
    for path, digest in manifest["code"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError("standalone inference source changed")
    checkpoint = bundle_path / manifest["model"]["path"]
    if sha256_file(checkpoint) != manifest["model"]["sha256"]:
        raise ValueError("standalone model hash changed")
    bundle = joblib.load(checkpoint)
    if bundle["config"] != manifest["fit"]:
        raise ValueError("checkpoint fit configuration differs")
    qa = pd.read_csv(qa_path, usecols=QUESTION_COLUMNS, dtype=str, keep_default_na=False)
    frame, contexts = mechanism.prepare_independent(qa, training=False)
    matrices, masks = {}, {}
    for name, path in features.items():
        data = read_sensor(path, manifest["fit"]["views"][name])
        if data["names"].tolist() != manifest["feature_names"][name]:
            raise ValueError(f"{name} feature names/order differ from training")
        matrices[name], masks[name] = mechanism.absolute_matrix(frame, data)
        hashes[str(path.resolve())] = sha256_file(path)
    scores = mechanism.evidence(bundle, frame, contexts, matrices, masks, np.arange(len(frame)))
    prediction = mechanism.predict(frame, scores, manifest["decoder"])
    validate_output(frame, prediction)
    output.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(output, index=False)
    write_json(output.with_suffix(".report.json"), {"rows": len(prediction), "groups": "one recording each",
        "acquisition_metadata": False, "qa_sha256": sha256_file(qa_path),
        "checkpoint_sha256": manifest["model"]["sha256"], "feature_hashes": hashes,
        "active_rows": {name: int(mask.sum()) for name, mask in masks.items()},
        "prediction_sha256": sha256_file(output),
        "scope": "Expert prediction from supplied descriptors; not raw extraction or a full competition submission"})
    return prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    exporter = commands.add_parser("export")
    exporter.add_argument("--run", type=Path, required=True)
    exporter.add_argument("--output-dir", type=Path, required=True)
    predictor = commands.add_parser("infer")
    predictor.add_argument("--bundle", type=Path, required=True)
    predictor.add_argument("--qa-csv", type=Path, required=True)
    predictor.add_argument("--output", type=Path, required=True)
    for name in ("imu", "skeleton", "radar", "visual"):
        predictor.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        export(args.run, args.output_dir)
    else:
        infer(args.bundle, args.qa_csv, {name: getattr(args, name) for name in ("imu", "skeleton", "radar", "visual")}, args.output)


if __name__ == "__main__":
    main()
