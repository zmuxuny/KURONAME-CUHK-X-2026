#!/usr/bin/env python3
"""Recover/load the frozen HAU visual presence calibrators from temporal logits.

Only final fits are recovered; the old OOF search is not repeated. Runtime uses
new question metadata and compatible K-KUNO logits, without a training CSV,
historical answer vector, or oracle multi-answer cardinality. The output is
action/option evidence for the joint decoder, not a complete submission.
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
sys.path.insert(0, str(ROOT / "scripts"))

from run_hau_kuno_presence import (
    ACTION_CATEGORIES,
    MODEL_LEAVES,
    build_model,
    feature_cube,
    option_scores,
    training_options,
)
from run_hau_kuno_temporal import load_logit_archives

from cuhkx.inference import question_metadata
from cuhkx.metadata import feature_unit, subject_id
from cuhkx.submission import read_csv_str, validate_submission_frames

KIND = "cuhkx_kuno_presence_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    paths = (
        "scripts/kuno_presence_checkpoint.py", "scripts/run_hau_kuno_presence.py",
        "scripts/run_hau_kuno_temporal.py", "src/cuhkx/actions.py",
        "src/cuhkx/inference.py", "src/cuhkx/metadata.py", "src/cuhkx/submission.py",
        "src/cuhkx/temporal.py",
    )
    return {path: sha256(ROOT / path) for path in paths}


def load_features(paths: list[Path], config: dict):
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            if not np.array_equal(archive["classes"], np.arange(40)):
                raise ValueError("unexpected action class order")
    logits, units, weights, metadata = load_logit_archives(paths)
    if len(set(units)) != len(units) or any(not value for value in units):
        raise ValueError("logit unit IDs must be nonempty and unique")
    if not np.isfinite(logits).all() or not np.isfinite(weights).all():
        raise ValueError("logits and weights must be finite")
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("ensemble weights must be nonnegative with a positive sum")
    features, direct = feature_cube(
        logits, weights, config["temperature"], config["top_windows"]
    )
    contract = {"logit_tail": [logits.shape[i] for i in (0, 2, 3)],
                "weights": weights.tolist(), "feature_width": features.shape[-1]}
    return features, direct, units, contract, metadata


def predict(bundle: dict, qa: pd.DataFrame, features, direct, units):
    metadata = question_metadata(qa)
    target = metadata.loc[
        metadata.source.eq("HAU") & metadata.category.isin(ACTION_CATEGORIES)
    ]
    unit_rows = {unit: index for index, unit in enumerate(units)}
    required = list(dict.fromkeys(feature_unit(path, training=False) for path in target.path))
    if not required or not set(required) <= set(unit_rows):
        raise ValueError("logits must cover every requested HAU action clip")
    indices = np.asarray([unit_rows[unit] for unit in required])
    matrix = features[indices].reshape(-1, features.shape[-1])
    scores = {}
    for name, model in bundle["models"].items():
        if list(model.classes_) != [0, 1]:
            raise ValueError("unexpected binary classifier classes")
        scores[name] = model.predict_proba(matrix)[:, 1].reshape(len(required), 40).astype(
            np.float32
        )
    learned = np.mean(np.stack(list(scores.values())), axis=0)
    scores["direct"] = direct[indices]
    weight = bundle["config"]["base_blend"]
    scores["selected"] = (1 - weight) * learned + weight * scores["direct"]
    lookup = {unit: index for index, unit in enumerate(required)}
    records = []
    for _, row in target.iterrows():
        unit = feature_unit(row.path, training=False)
        record = {"qa_id": row.qa_id, "source": row.source, "category": row.category,
                  "unit": unit}
        for name, values in scores.items():
            prefix = "option_score_" if name == "selected" else f"option_score_{name}_"
            for letter, value in option_scores(row, values[lookup[unit]]).items():
                record[prefix + letter] = value
        records.append(record)
    return pd.DataFrame(records), required, scores


def compare_frozen(scores: pd.DataFrame, reference_path: Path) -> dict:
    columns = [name for name in scores if name.startswith("option_score_")]
    reference = pd.read_csv(reference_path, usecols=["qa_id", *columns], keep_default_na=False)
    if reference.qa_id.tolist() != scores.qa_id.tolist():
        raise ValueError("frozen question IDs or order differ")
    before = reference[columns].apply(pd.to_numeric, errors="coerce").to_numpy()
    after = scores[columns].to_numpy(dtype=float)
    if not np.array_equal(np.isnan(before), np.isnan(after)):
        raise ValueError("frozen valid-option coverage differs")
    error = float(np.nanmax(np.abs(before - after)))
    if error > 5e-8:
        raise ValueError(f"refit differs from frozen presence scores: {error}")
    return {"rows": len(scores), "option_score_values": int(np.isfinite(after).sum()),
            "max_abs_score_error": error, "tolerance": 5e-8,
            "no_parent_or_cardinality_input": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    fit = commands.add_parser("export")
    fit.add_argument("--train-csv", type=Path, required=True)
    fit.add_argument("--reference-run", type=Path, required=True)
    inference = commands.add_parser("predict")
    inference.add_argument("--checkpoint", type=Path, required=True)
    for command in (fit, inference):
        command.add_argument("--test-csv", type=Path, required=True)
        command.add_argument("--logits", type=Path, nargs="+", required=True)
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("output directory must be new")
    inputs = [args.test_csv, *args.logits]
    if args.operation == "export":
        report_path = args.reference_run / "report.json"
        reference = json.loads(report_path.read_text())
        config = reference["config"]
        expected = [item["sha256"] for item in reference["logit_archives"]]
        if [sha256(path) for path in args.logits] != expected:
            raise ValueError("recovery logits must match frozen input hashes and order")
        features, direct, units, contract, _ = load_features(args.logits, config)
        train = read_csv_str(args.train_csv)
        training = question_metadata(train).assign(answer=train["answer"])
        labels = training[["qa_id", "answer"]].rename(columns={"answer": "prediction"})
        validate_submission_frames(training, labels, labels)
        allowed = {f"user{i}" for i in (*range(1, 10), *range(16, 25))}
        if not set(training.path.map(subject_id)) <= allowed:
            raise ValueError("training includes non-training subjects")
        options = training_options(
            training, {unit: i for i, unit in enumerate(units)}, config["supervision"]
        )
        if len(options) != reference["training_option_rows"]:
            raise ValueError("training option coverage differs from frozen calibration")
        matrix = features[options.unit_row.to_numpy(), options.class_id.to_numpy()]
        models = {}
        parameters = argparse.Namespace(**(config | {"workers": 3}))
        for index, (name, leaf) in enumerate(MODEL_LEAVES.items()):
            estimator = build_model(parameters, leaf, config["seed"] + 1000 * index)
            estimator.fit(matrix, options.label.to_numpy())
            models[name] = estimator
            print(f"recovered={name} trees={config['trees']}", flush=True)
        bundle = {"model_kind": KIND, "config": config, "models": models,
                  "feature_contract": contract, "source_hashes": source_hashes(),
                  "train_sha256": sha256(args.train_csv), "training_options": len(options)}
        inputs.extend([args.train_csv, report_path])
    else:
        if sha256(args.checkpoint) != args.checkpoint.with_suffix(".sha256").read_text().strip():
            raise ValueError("checkpoint SHA256 mismatch")
        bundle = joblib.load(args.checkpoint)
        if bundle["model_kind"] != KIND or bundle["source_hashes"] != source_hashes():
            raise ValueError("checkpoint kind or source hashes differ")
        features, direct, units, contract, _ = load_features(args.logits, bundle["config"])
        if contract != bundle["feature_contract"]:
            raise ValueError("logit feature contract differs from training")
        inputs.append(args.checkpoint)
    scores, test_units, action_scores = predict(
        bundle, read_csv_str(args.test_csv), features, direct, units
    )
    comparison = {}
    if args.operation == "export":
        frozen = args.reference_run / "test_predictions.csv"
        comparison = compare_frozen(scores, frozen)
        inputs.append(frozen)
    args.output_dir.mkdir(parents=True)
    if args.operation == "export":
        checkpoint = args.output_dir / "models.joblib"
        joblib.dump(bundle, checkpoint, compress=3)
        checkpoint.with_suffix(".sha256").write_text(sha256(checkpoint) + "\n")
    scores.to_csv(args.output_dir / "option_scores.csv", index=False, lineterminator="\n")
    np.savez_compressed(args.output_dir / "action_scores.npz", units=np.asarray(test_units),
                        classes=np.arange(40), view_names=np.asarray(list(action_scores)),
                        scores=np.stack(list(action_scores.values())))
    report = {
        "model_kind": KIND, "operation": args.operation, "config": bundle["config"],
        "feature_contract": bundle["feature_contract"], "runtime_workers": 3,
        "scope": "final calibrator recovery only; no new validation selection or submission",
        "limitations": ["Input logits still require the K-KUNO raw media extractor.",
                        "Backbone subject-fold membership remains unverified.",
                        "Scores require the joint decoder; no multi-answer oracle is used."],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_hashes": source_hashes(), "frozen_score_comparison": comparison,
        "environment": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib")},
        "inputs": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in inputs],
        "outputs": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                    for p in args.output_dir.iterdir()},
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"operation": args.operation, "comparison": comparison,
                      "outputs": report["outputs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
