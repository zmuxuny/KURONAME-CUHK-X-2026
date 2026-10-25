#!/usr/bin/env python3
"""Export/load the existing option-frequency expert without replaying its OOF search.

Only an all-training-subject refit is needed to recover this missing checkpoint.
Export must reproduce the frozen test option probabilities and decoded answers.
Prediction requires the previously audited repeat layout and produces evidence,
not a full competition submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from cuhkx.inference import question_metadata
from cuhkx.metadata import subject_id
from cuhkx.option_frequency import build_option_frequency_features, decode_option_probabilities
from cuhkx.submission import read_csv_str, validate_submission_frames
from scripts.run_hau_repeated_option_frequency import model

KIND = "cuhkx_option_frequency_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    paths = (
        "src/cuhkx/option_frequency.py", "src/cuhkx/clip_constraints.py",
        "src/cuhkx/metadata.py", "src/cuhkx/inference.py", "src/cuhkx/submission.py",
        "scripts/option_frequency_checkpoint.py", "scripts/run_hau_repeated_option_frequency.py",
    )
    return {path: sha256(ROOT / path) for path in paths}


def predict(bundle: dict, questions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    matrix, metadata = build_option_frequency_features(question_metadata(questions))
    scores, decisions = metadata.copy(), {}
    for name, estimator in bundle["models"].items():
        if list(estimator.classes_) != [0, 1]:
            raise ValueError("binary classifier classes differ from the frozen contract")
        probabilities = estimator.predict_proba(matrix)[:, 1]
        scores[f"probability_{name}"] = probabilities
        decisions[name] = decode_option_probabilities(
            metadata, probabilities, multi_threshold=bundle["config"]["multi_threshold"]
        )
    return scores, decisions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("export")
    fit.add_argument("--train-csv", type=Path, required=True)
    fit.add_argument("--test-csv", type=Path, required=True)
    fit.add_argument("--reference-run", type=Path, required=True)
    fit.add_argument("--output-dir", type=Path, required=True)
    inference = commands.add_parser("predict")
    inference.add_argument("--test-csv", type=Path, required=True)
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("output directory must be new")
    inputs = [args.test_csv]
    if args.command == "export":
        reference_report = args.reference_run / "report.json"
        config = json.loads(reference_report.read_text())["config"]
        train = read_csv_str(args.train_csv)
        training = question_metadata(train).assign(answer=train["answer"])
        labels = training[["qa_id", "answer"]].rename(columns={"answer": "prediction"})
        validate_submission_frames(training, labels, labels)
        allowed = {f"user{i}" for i in (*range(1, 10), *range(16, 25))}
        if not set(training.path.map(subject_id)) <= allowed:
            raise ValueError("training input includes non-training subjects")
        matrix, metadata = build_option_frequency_features(training)
        models = {}
        for index, (name, family) in enumerate(config["families"].items()):
            estimator = model(family, config["trees"], config["seed"] + 1000 * index)
            estimator.fit(matrix, metadata["label"].to_numpy(dtype=int))
            models[name] = estimator
        bundle = {
            "model_kind": KIND, "config": config, "models": models,
            "source_hashes": source_hashes(), "training_rows": len(training),
            "train_sha256": sha256(args.train_csv),
        }
        inputs.extend([args.train_csv, reference_report])
    else:
        expected = args.checkpoint.with_suffix(".sha256").read_text().strip()
        if sha256(args.checkpoint) != expected:
            raise ValueError("checkpoint SHA256 mismatch")
        bundle = joblib.load(args.checkpoint)
        if bundle["model_kind"] != KIND or bundle["source_hashes"] != source_hashes():
            raise ValueError("checkpoint model kind or source hashes differ")
        inputs.append(args.checkpoint)
    scores, decisions = predict(bundle, read_csv_str(args.test_csv))
    comparisons = {}
    if args.command == "export":
        reference_path = args.reference_run / "test_option_scores.csv"
        columns = ["qa_id", "letter", *[f"probability_{name}" for name in bundle["models"]]]
        reference = pd.read_csv(reference_path, usecols=columns, keep_default_na=False)
        if reference[["qa_id", "letter"]].to_dict("list") != scores[
            ["qa_id", "letter"]
        ].to_dict("list"):
            raise ValueError("reference option IDs/order differ from regenerated features")
        for name in bundle["models"]:
            column = f"probability_{name}"
            before = reference[column].to_numpy(dtype=float)
            after = scores[column].to_numpy(dtype=float)
            delta = float(np.max(np.abs(before - after)))
            old_decisions = decode_option_probabilities(
                scores, before, multi_threshold=bundle["config"]["multi_threshold"]
            )
            identical = decisions[name].prediction.equals(old_decisions.prediction)
            if delta > 1e-12 or not identical:
                raise ValueError(f"refit differs from frozen {name}: max absolute error {delta}")
            comparisons[name] = {"max_abs_probability_error": delta, "same_predictions": identical}
        inputs.append(reference_path)
    args.output_dir.mkdir(parents=True)
    if args.command == "export":
        checkpoint = args.output_dir / "models.joblib"
        joblib.dump(bundle, checkpoint, compress=3)
        checkpoint.with_suffix(".sha256").write_text(sha256(checkpoint) + "\n")
    scores.to_csv(args.output_dir / "option_scores.csv", index=False, lineterminator="\n")
    for name, frame in decisions.items():
        frame.to_csv(args.output_dir / f"predictions_{name}.csv", index=False, lineterminator="\n")
    report = {
        "model_kind": KIND, "operation": args.command, "config": bundle["config"],
        "scope": "recovered production expert; no new OOF selection or Public submission",
        "layout": "requires independently audited three-recording scene boundaries",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_hashes": source_hashes(), "frozen_score_comparison": comparisons,
        "environment": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib")},
        "inputs": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in inputs],
        "outputs": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                    for p in args.output_dir.iterdir()},
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"operation": args.command, "comparison": comparisons, "outputs": report["outputs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
