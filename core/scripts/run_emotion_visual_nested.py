#!/usr/bin/env python3
"""Frozen v3 nested emotion pipeline with one generic visual modality."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from cuhkx import emotion_fusion_nested, emotion_template_nested
from cuhkx.emotion_nested import (
    base_evidence,
    predict_rows,
    prepare,
    read_sensor,
    sensor_matrix,
    validation_views,
)
from cuhkx.emotion_visual_nested import fit_bundle
from cuhkx.submission import read_csv_str, validate_prediction
from scripts import run_emotion_template_nested as template_runner

KIND = "cuhkx_emotion_visual_nested_v5"
SENSORS = ("imu", "skeleton", "radar", "visual")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def sources():
    extra = ["scripts/run_emotion_visual_nested.py", "src/cuhkx/emotion_visual_nested.py",
             "scripts/pack_emotion_generic_visual.py", "scripts/audit_emotion_visual_nested.py",
             "scripts/audit_emotion_nested_provenance.py", "src/cuhkx/validation.py",
             "scripts/evaluate_local_predictions.py",
             "scripts/extract_generic_hau_windows.py", "src/cuhkx/generic_video_windows.py",
             "src/cuhkx/video.py", "scripts/extract_video_embeddings.py"]
    return {**template_runner.sources(), **{name: sha(ROOT / name) for name in extra}}


def validate_config(config):
    parent = json.loads((ROOT / "configs/emotion_template_nested_v3.json").read_text())
    expected = dict(parent)
    expected["views"] = {**parent["views"], "visual": "^visual_"}
    expected["sensor_weights"] = {**parent["sensor_weights"], "visual": 1.0}
    descriptive = {"version", "objective", "validation_caveat", "visual_specification"}
    if set(config) != set(expected) | {"visual_specification"}:
        raise ValueError("visual config changed its declared field set")
    if any(config[key] != value for key, value in expected.items() if key not in descriptive):
        raise ValueError("visual experiment may only add the fixed 1.0-weight modality to v3")
    if config["visual_specification"] != {
        "dimensions": 2048, "aggregation": "Depth mean,std then IR mean,std over 8 windows; population std",
        "relative_normalization": "unchanged sensor_matrix within complete three-recording scene",
        "invalid_unit_policy": "omit missing or physically misaligned paired video; disable visual for its whole scene",
        "duplicate_windows": "retain fixed eight-window sample; zero std is allowed"}:
        raise ValueError("visual descriptor or missing-scene policy differs")


def install_hooks():
    global base_evidence, predict_rows, validation_views, choose_decoder, select_inner
    template_runner.install_hooks()
    template_runner.runner.sources = sources
    template_runner.runner.write_json = write_json
    template_runner.runner.scope_record = scope_record
    template_runner.runner.fit_bundle = fit_bundle
    template_runner.runner.KIND = KIND
    base_evidence = emotion_fusion_nested.base_evidence
    predict_rows = emotion_template_nested.predict_rows
    validation_views = emotion_template_nested.validation_views
    choose_decoder = emotion_template_nested.choose_decoder
    select_inner = template_runner.select_inner


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
        "training_content": bundle["training_content"],
        "fitted_estimator_digest_joblib": joblib.hash(bundle["models"]),
        "source_hashes": sources(), "input_hashes": input_hashes,
    }
    write_json(path, record)
    return {"path": str(path.resolve()), "sha256": sha(path)}


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
    validate_config(config)
    if not 1 <= config["workers"] <= 3:
        raise ValueError("workers must be one through three")
    args.output_dir.mkdir(parents=True)
    paths = {name: getattr(args, name) for name in SENSORS}
    inputs = {str(path.resolve()): sha(path) for path in [args.train_csv, args.test_csv, args.config, *paths.values(),
                args.visual.with_suffix(".json"), ROOT / "configs/emotion_template_nested_v3.json"]}
    write_json(args.output_dir / "frozen_config.json", {"config": config, "inputs": inputs,
               "source_hashes": sources(), "frozen_before_fitting": True})
    for name, digest in sources().items():
        target = args.output_dir / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        if sha(target) != digest:
            raise ValueError("source changed while freezing")
    train_qa, test_qa = read_csv_str(args.train_csv), read_csv_str(args.test_csv)
    frame, contexts = prepare(train_qa, training=True)
    test, test_contexts = prepare(test_qa, training=False)
    sensor_data = {name: read_sensor(path, config["views"][name]) for name, path in paths.items()}
    matrices, masks = feature_data(frame, sensor_data)
    test_matrices, test_masks = feature_data(test, sensor_data)
    subjects = sorted(set(frame.subject))
    if len(subjects) != 18:
        raise ValueError("complete nested emotion evaluation requires all 18 official subjects")
    write_json(args.output_dir / "modality_coverage.json", {
        split: {name: {"rows": int(mask.sum()), "scenes": int(values.loc[mask, "group"].nunique()),
                       "masked_units": values.loc[~mask, "unit"].tolist()}
                for name, mask in split_masks.items()}
        for split, values, split_masks in (("train", frame, masks), ("test", test, test_masks))})
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
                                "Relaxed-core views retain all questions and the same vocabulary; their repeated observations are dependent."],
                "upstream_feature_note": "All sensor transforms are deterministic within each unlabeled scene; no pretrained CUHK classifier is used."}
    protocol.update({"fitted_components": ["sensor_models_and_word_context_priors"],
                     "selection_history": "Historically observed subjects and prior v3/v4 results; no new blind design holdout.",
                     "mechanism": "Add fixed generic-video mean/std visual weight 1.0; unchanged v3 inner-only template decoder selection",
                     "visual_missing_policy": "Invalid paired visual unit disables this modality for its whole scene; other sensor masks remain independent"})
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
    scope_record(args.output_dir / "final_training_scope.json", bundle, frame, subjects, set(), inputs)
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
        "retained_checkpoints": {"outer": 18, "inner": 54, "final_inner": 3, "final": 1},
    }
    for path, digest in inputs.items():
        if sha(path) != digest:
            raise ValueError("input changed during nested evaluation")
    frozen = json.loads((args.output_dir / "frozen_config.json").read_text())
    if sources() != frozen["source_hashes"]:
        raise ValueError("source changed during nested evaluation")
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
    paths = {name: getattr(args, name) for name in SENSORS}
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
    training.add_argument("--config", type=Path, default=ROOT / "configs/emotion_visual_nested_v5.json")
    prediction = sub.add_parser("infer")
    prediction.add_argument("--checkpoint", type=Path, required=True)
    prediction.add_argument("--scene-manifest", type=Path)
    for command in (training, prediction):
        command.add_argument("--test-csv", type=Path, required=True)
        for name in SENSORS:
            command.add_argument(f"--{name}", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    install_hooks()
    fit(args) if args.command == "fit" else infer(args)


if __name__ == "__main__":
    main()
