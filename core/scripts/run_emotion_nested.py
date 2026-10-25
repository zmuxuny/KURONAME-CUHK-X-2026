#!/usr/bin/env python3
"""Nested subject evaluation and checkpoint inference for independent emotion QA."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuhkx.emotion_nested import (
    base_evidence,
    choose_decoder,
    fit_bundle,
    predict_rows,
    prepare,
    read_sensor,
    sensor_matrix,
    validation_views,
)
from cuhkx.submission import read_csv_str, validate_prediction

KIND = "cuhkx_emotion_nested_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def sources():
    paths = [Path(__file__), ROOT / "src/cuhkx/emotion_nested.py",
             ROOT / "src/cuhkx/clip_constraints.py", ROOT / "src/cuhkx/inference.py",
             ROOT / "src/cuhkx/metadata.py", ROOT / "src/cuhkx/actions.py",
             ROOT / "src/cuhkx/submission.py"]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def feature_data(frame, source_data):
    matrices, masks = {}, {}
    for name, data in source_data.items():
        matrices[name], masks[name] = sensor_matrix(frame, data)
    return matrices, masks


def scope_record(path, bundle, frame, fit_subjects, valid_subjects, input_hashes):
    fit_subjects, valid_subjects = set(fit_subjects), set(valid_subjects)
    if fit_subjects & valid_subjects:
        raise ValueError("training and validation subjects overlap")
    selected = frame.subject.isin(fit_subjects)
    record = {
        "train_subjects": sorted(fit_subjects), "validation_subjects": sorted(valid_subjects),
        "training_qa_ids": frame.loc[selected, "qa_id"].tolist(),
        "training_units_by_sensor": bundle["fitted_units"],
        "fitted_estimator_digest_joblib": joblib.hash(bundle["models"]),
        "source_hashes": sources(), "input_hashes": input_hashes,
    }
    write_json(path, record)
    return {"path": str(path.resolve()), "sha256": sha(path)}


def select_inner(frame, contexts, matrices, masks, fit_subjects, config, out, seed, inputs):
    selected = np.flatnonzero(frame.subject.isin(fit_subjects))
    subjects = frame.iloc[selected].subject.to_numpy()
    scores, folds = [], []
    for index, (fit_indices, valid_indices) in enumerate(
        GroupKFold(n_splits=config["inner_folds"]).split(selected, groups=subjects)
    ):
        fit_users = set(subjects[fit_indices])
        valid_users = set(subjects[valid_indices])
        if fit_users & valid_users or not (fit_users | valid_users) <= set(fit_subjects):
            raise ValueError("invalid inner subject partition")
        bundle = fit_bundle(frame, contexts, matrices, masks, fit_users, config, seed + index * 10)
        rows = selected[valid_indices]
        evidence = base_evidence(bundle, frame, contexts, matrices, masks, rows)
        scores.append(validation_views(frame.iloc[rows], evidence, config["decoders"]))
        provenance = scope_record(out / f"inner_{index}_scope.json", bundle, frame,
                                  fit_users, valid_users, inputs)
        folds.append({"fold_id": f"inner_{index}", "train_subjects": sorted(fit_users),
                      "validation_subjects": sorted(valid_users), "provenance": provenance})
        del bundle
    combined = pd.concat(scores, ignore_index=True)
    decoder, ranking = choose_decoder(combined, config)
    combined.to_csv(out / "inner_validation_views.csv", index=False)
    write_json(out / "inner_selection.json", {"selected": decoder, "ranking": ranking})
    return decoder, ranking, folds


def validate_output(qa, output):
    if not output.qa_id.is_unique or set(output.qa_id) != set(qa.qa_id):
        raise ValueError("emotion expert IDs do not cover the target scope")
    by_id = qa.set_index("qa_id")
    for row in output.itertuples():
        error = validate_prediction(row.prediction, "emotion", by_id.loc[row.qa_id, "letters"])
        if error:
            raise ValueError(error)


def fit(args):
    if args.output_dir.exists():
        raise ValueError("output directory must be new")
    started = time.perf_counter()
    config = json.loads(args.config.read_text())
    if not 1 <= config["workers"] <= 3:
        raise ValueError("workers must be one through three")
    args.output_dir.mkdir(parents=True)
    paths = {"imu": args.imu, "skeleton": args.skeleton, "radar": args.radar}
    inputs = {str(path.resolve()): sha(path) for path in [args.train_csv, args.test_csv, args.config, *paths.values()]}
    write_json(args.output_dir / "frozen_config.json", {"config": config, "inputs": inputs,
               "source_hashes": sources(), "frozen_before_fitting": True})
    train_qa, test_qa = read_csv_str(args.train_csv), read_csv_str(args.test_csv)
    frame, contexts = prepare(train_qa, training=True)
    test, test_contexts = prepare(test_qa, training=False)
    sensor_data = {name: read_sensor(path, config["views"][name]) for name, path in paths.items()}
    matrices, masks = feature_data(frame, sensor_data)
    test_matrices, test_masks = feature_data(test, sensor_data)
    subjects = sorted(set(frame.subject))
    oof, protocols, selection_counts, outer_views = [], [], Counter(), []
    for outer_index, subject in enumerate(subjects):
        out = args.output_dir / "folds" / subject
        out.mkdir(parents=True)
        fit_users = set(subjects) - {subject}
        seed = config["seed"] + outer_index * 10000
        decoder, _ranking, inner_folds = select_inner(
            frame, contexts, matrices, masks, fit_users, config, out, seed, inputs
        )
        bundle = fit_bundle(frame, contexts, matrices, masks, fit_users, config, seed + 99)
        valid_rows = np.flatnonzero(frame.subject.eq(subject))
        target = frame.iloc[valid_rows]
        evidence = base_evidence(bundle, frame, contexts, matrices, masks, valid_rows)
        output = predict_rows(target, evidence, decoder)
        validate_output(target, output)
        output["subject"] = subject
        output["selected_decoder"] = decoder["name"]
        output.to_csv(out / "oof_predictions.csv", index=False)
        oof.append(output)
        selection_counts[decoder["name"]] += 1
        views = validation_views(target, evidence, [decoder])
        views["outer_subject"] = subject
        outer_views.append(views)
        checkpoint = out / "models.joblib"
        bundle["decoder"] = decoder
        joblib.dump(bundle, checkpoint, compress=3)
        provenance = scope_record(out / "training_scope.json", bundle, frame, fit_users, {subject}, inputs)
        protocols.append({
            "fold_id": subject, "train_subjects": sorted(fit_users),
            "validation_subjects": [subject], "prediction_qa_ids": output.qa_id.tolist(),
            "inner_folds": inner_folds,
            "upstream_training": [{"name": "sensor_models_and_word_context_priors",
                                   "train_subjects": sorted(fit_users), "provenance": provenance}],
            "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha(checkpoint)},
        })
        accuracy = float(np.mean(output.prediction.to_numpy() == target.answer.to_numpy()))
        print(f"outer={subject} decoder={decoder['name']} rows={len(output)} accuracy={accuracy:.5f}", flush=True)
        del bundle
    oof = pd.concat(oof, ignore_index=True).set_index("qa_id").loc[frame.qa_id].reset_index()
    oof.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    pd.concat(outer_views, ignore_index=True).to_csv(args.output_dir / "outer_validation_views.csv", index=False)
    protocol = {"schema_version": 1, "validation_kind": "nested_subject", "folds": protocols,
                "limitations": [config["validation_caveat"], "Previously exposed subjects; not a new blind holdout.",
                                "Template-masked views share subjects/scenes; they are not independent samples."],
                "upstream_feature_note": "All sensor transforms are deterministic within each unlabeled scene; no pretrained CUHK classifier is used."}
    write_json(args.output_dir / "protocol.json", protocol)
    final_dir = args.output_dir / "final_selection"
    final_dir.mkdir()
    final_decoder, final_ranking, _ = select_inner(
        frame, contexts, matrices, masks, subjects, config, final_dir, config["seed"] + 900000, inputs
    )
    bundle = fit_bundle(frame, contexts, matrices, masks, subjects, config, config["seed"] + 999999)
    bundle.update({"model_kind": KIND, "decoder": final_decoder, "source_hashes": sources(),
                   "feature_names": {name: data["names"] for name, data in sensor_data.items()},
                   "input_hashes": inputs})
    checkpoint = args.output_dir / "models.joblib"
    joblib.dump(bundle, checkpoint, compress=3)
    checkpoint.with_suffix(".sha256").write_text(sha(checkpoint) + "\n")
    evidence = base_evidence(bundle, test, test_contexts, test_matrices, test_masks, np.arange(len(test)))
    test_output = predict_rows(test, evidence, final_decoder)
    validate_output(test, test_output)
    test_output.to_csv(args.output_dir / "test_predictions.csv", index=False)
    by_subject = pd.DataFrame({"subject": frame.subject, "correct": oof.prediction.to_numpy() == frame.answer.to_numpy()}).groupby("subject").correct.mean()
    report = {
        "status": "nested_component_evaluated", "scope": "HAU emotion only", "rows": len(oof),
        "accuracy": float(np.mean(oof.prediction.to_numpy() == frame.answer.to_numpy())),
        "worst_subject_accuracy": float(by_subject.min()), "by_subject": by_subject.to_dict(),
        "outer_decoder_counts": dict(selection_counts), "final_decoder": final_decoder,
        "final_inner_ranking": final_ranking, "test_rows": len(test_output),
        "input_hashes": inputs, "source_hashes": sources(), "runtime_seconds": time.perf_counter() - started,
        "environment": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy", "joblib")},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "outputs": {p.name: sha(p) for p in args.output_dir.iterdir() if p.is_file()},
        "limitations": protocol["limitations"],
    }
    for path, digest in inputs.items():
        if sha(path) != digest:
            raise ValueError("input changed during nested evaluation")
    write_json(args.output_dir / "report.json", report)
    print(json.dumps({k: report[k] for k in ("accuracy", "worst_subject_accuracy", "outer_decoder_counts", "final_decoder", "runtime_seconds")}, indent=2))


def infer(args):
    if args.output_dir.exists():
        raise ValueError("output directory must be new")
    if sha(args.checkpoint) != args.checkpoint.with_suffix(".sha256").read_text().strip():
        raise ValueError("checkpoint SHA256 mismatch")
    bundle = joblib.load(args.checkpoint)
    if bundle["model_kind"] != KIND or bundle["source_hashes"] != sources():
        raise ValueError("checkpoint kind or source hashes differ")
    manifest = read_csv_str(args.scene_manifest) if args.scene_manifest else None
    frame, contexts = prepare(read_csv_str(args.test_csv), training=False, manifest=manifest)
    paths = {"imu": args.imu, "skeleton": args.skeleton, "radar": args.radar}
    data = {name: read_sensor(path, bundle["config"]["views"][name]) for name, path in paths.items()}
    if any(not np.array_equal(data[name]["names"], bundle["feature_names"][name]) for name in data):
        raise ValueError("sensor feature names/order differ from checkpoint")
    matrices, masks = feature_data(frame, data)
    evidence = base_evidence(bundle, frame, contexts, matrices, masks, np.arange(len(frame)))
    output = predict_rows(frame, evidence, bundle["decoder"])
    validate_output(frame, output)
    args.output_dir.mkdir(parents=True)
    output.to_csv(args.output_dir / "test_predictions.csv", index=False)
    write_json(args.output_dir / "report.json", {"status": "load_only", "rows": len(output),
               "checkpoint_sha256": sha(args.checkpoint), "prediction_sha256": sha(args.output_dir / "test_predictions.csv"),
               "source_hashes": sources(), "decoder": bundle["decoder"],
               "inputs": {str(p.resolve()): sha(p) for p in [args.test_csv, *paths.values(), *([args.scene_manifest] if args.scene_manifest else [])]}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("fit")
    training.add_argument("--train-csv", type=Path, required=True)
    training.add_argument("--config", type=Path, default=ROOT / "configs/emotion_nested_v1.json")
    prediction = sub.add_parser("infer")
    prediction.add_argument("--checkpoint", type=Path, required=True)
    prediction.add_argument("--scene-manifest", type=Path)
    for command in (training, prediction):
        command.add_argument("--test-csv", type=Path, required=True)
        for name in ("imu", "skeleton", "radar"):
            command.add_argument(f"--{name}", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    fit(args) if args.command == "fit" else infer(args)


if __name__ == "__main__":
    main()
