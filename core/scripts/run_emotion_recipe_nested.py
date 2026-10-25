#!/usr/bin/env python3
"""Reselect, audit, or load two fixed emotion recipes over 76 frozen v5 models."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from cuhkx import emotion_recipe_nested as recipe
from cuhkx.emotion_nested import prepare, read_sensor
from cuhkx.emotion_visual_nested import verify_training_content
from cuhkx.submission import read_csv_str
from cuhkx.validation import validate_protocol
from scripts import run_emotion_visual_nested as parent_driver
from scripts.audit_emotion_nested_provenance import VerifiedFiles, audit_checkpoint

SENSORS = ("imu", "skeleton", "radar", "visual")
KIND = "cuhkx_emotion_recipe_nested_v6"
sha = parent_driver.sha
write_json = parent_driver.write_json


def sources():
    return {**parent_driver.sources(), **{name: sha(ROOT / name) for name in
            ("scripts/run_emotion_recipe_nested.py", "src/cuhkx/emotion_recipe_nested.py")}}


def reference(path):
    return {"path": str(path.resolve()), "sha256": sha(path)}


def source_manifest(parent):
    frozen = json.loads((parent / "frozen_config.json").read_text())
    protocol = json.loads((parent / "protocol.json").read_text())
    original_report = json.loads((parent / "report.json").read_text())
    for name in ("models.joblib", "final_training_scope.json", "protocol.json"):
        if sha(parent / name) != original_report["outputs"][name]:
            raise ValueError("parent final model/scope/protocol differs from its completed report")
    if sha(parent / "models.joblib") != (parent / "models.sha256").read_text().strip():
        raise ValueError("parent final model SHA sidecar differs")
    entries, folds = {}, []

    def add(key, checkpoint, scope, train, valid):
        entries[key] = {"checkpoint": checkpoint, "scope": scope,
                        "train_subjects": sorted(train), "validation_subjects": sorted(valid)}

    for fold in protocol["folds"]:
        subject = fold["fold_id"]
        key = f"outer/{subject}"
        add(key, fold["checkpoint"], fold["upstream_training"][0]["provenance"],
            fold["train_subjects"], fold["validation_subjects"])
        inner = json.loads((parent / "folds" / subject / "inner_fold_manifest.json").read_text())
        if inner != fold["inner_folds"]:
            raise ValueError("parent inner manifest differs from its original protocol")
        inner_keys = []
        for row in inner:
            inner_key = key + "/" + row["fold_id"]
            add(inner_key, row["checkpoint"], row["provenance"], row["train_subjects"], row["validation_subjects"])
            inner_keys.append(inner_key)
        folds.append({"subject": subject, "model_key": key, "inner_keys": inner_keys})
    final_inner = json.loads((parent / "final_selection" / "inner_fold_manifest.json").read_text())
    final_keys = []
    for row in final_inner:
        key = "final/" + row["fold_id"]
        add(key, row["checkpoint"], row["provenance"], row["train_subjects"], row["validation_subjects"])
        final_keys.append(key)
    final_scope = json.loads((parent / "final_training_scope.json").read_text())
    add("final", reference(parent / "models.joblib"), reference(parent / "final_training_scope.json"),
        final_scope["train_subjects"], final_scope["validation_subjects"])
    if len(entries) != 76 or len(folds) != 18 or len(final_keys) != 3:
        raise ValueError("exactly 76 original v5 models are required")
    return {"kind": "reused_v5_fitted_models", "parent_run": str(parent.resolve()),
            "parent_frozen_config": reference(parent / "frozen_config.json"),
            "parent_protocol": reference(parent / "protocol.json"),
            "parent_report": reference(parent / "report.json"),
            "original_fit_config": frozen["config"], "original_source_hashes": frozen["source_hashes"],
            "original_input_hashes": frozen["inputs"], "models": entries, "outer_folds": folds,
            "final_inner_keys": final_keys, "final_model_key": "final", "new_forest_fits": 0}


def check_sources(manifest):
    parent = Path(manifest["parent_run"])
    if parent_driver.sources() != manifest["original_source_hashes"]:
        raise ValueError("original v5 fitted-model sources changed")
    for name, digest in manifest["original_source_hashes"].items():
        if sha(parent / "source_snapshot" / name) != digest:
            raise ValueError("original v5 source snapshot changed")
    for name in ("parent_frozen_config", "parent_protocol", "parent_report"):
        item = manifest[name]
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("original model provenance changed")


def read_data(args, manifest):
    paths = {name: getattr(args, name) for name in SENSORS}
    for path in (args.train_csv, args.test_csv, *paths.values()):
        if manifest["original_input_hashes"].get(str(path.resolve())) != sha(path):
            raise ValueError("re-selection must use the original v5 fitting/evaluation inputs")
    frame, contexts = prepare(read_csv_str(args.train_csv), training=True)
    test, test_contexts = prepare(read_csv_str(args.test_csv), training=False)
    data = {name: read_sensor(path, manifest["original_fit_config"]["views"][name]) for name, path in paths.items()}
    matrices, masks = parent_driver.feature_data(frame, data)
    test_matrices, test_masks = parent_driver.feature_data(test, data)
    if len(set(frame.subject)) != 18 or len(frame) != 809 or len(test) != 144:
        raise ValueError("fixed recipe evaluation requires the complete official emotion scopes")
    return frame, contexts, matrices, masks, test, test_contexts, test_matrices, test_masks, data


def execute(args, manifest, config, *, audit_run=None):
    started = time.perf_counter()
    check_sources(manifest)
    recipe.validate_config(config, manifest["original_fit_config"])
    frame, contexts, matrices, masks, test, test_contexts, test_matrices, test_masks, data = read_data(args, manifest)
    subjects = set(frame.subject)
    files, loaded, inner_scores_count = VerifiedFiles(), [], 0

    def load(key):
        if key in loaded:
            raise ValueError("an original model was unexpectedly loaded twice in one replay")
        item = manifest["models"][key]
        scope = files.json(item["scope"]["path"], "original fitting scope", item["scope"]["sha256"])
        train, valid = set(item["train_subjects"]), set(item["validation_subjects"])
        if train & valid or not train or (train | valid) - subjects:
            raise ValueError("invalid original model subject exclusion")
        if scope["train_subjects"] != sorted(train) or scope["validation_subjects"] != sorted(valid):
            raise ValueError("model source manifest changed its actual original fitting subjects")
        if scope["source_hashes"] != manifest["original_source_hashes"] or scope["input_hashes"] != manifest["original_input_hashes"]:
            raise ValueError("original model source/input provenance differs")
        path = files.verify(item["checkpoint"]["path"], item["checkpoint"]["sha256"], "original retained model")
        bundle = joblib.load(path)
        if bundle["config"] != manifest["original_fit_config"] or bundle["training_subjects"] != sorted(train):
            raise ValueError("source model configuration or subjects differ")
        verify_training_content(bundle, scope, frame, matrices, masks)
        if audit_run is not None:
            audit_checkpoint(path, item["checkpoint"]["sha256"], scope, frame, contexts,
                             {name: len(data[name]["names"]) for name in SENSORS},
                             manifest["original_fit_config"], files)
        loaded.append(key)
        return bundle

    def select(keys, universe, directory):
        nonlocal inner_scores_count
        if len(keys) != 3:
            raise ValueError("exactly three original inner models are required")
        seen, outputs, inner_folds = set(), [], []
        for key in keys:
            item = manifest["models"][key]
            train, valid = set(item["train_subjects"]), set(item["validation_subjects"])
            if train & valid or train | valid != universe or seen & valid:
                raise ValueError("inner models do not partition only the allowed outer fitting subjects")
            seen |= valid
            bundle = load(key)
            rows = np.flatnonzero(frame.subject.isin(valid))
            evidence = recipe.recipe_evidence(bundle, frame, contexts, matrices, masks, rows, config)
            outputs.append(recipe.validation_views(frame.iloc[rows], evidence, config))
            inner_folds.append({"fold_id": key.rsplit("/", 1)[-1], "train_subjects": sorted(train),
                                "validation_subjects": sorted(valid), "prediction_qa_ids": frame.iloc[rows].qa_id.tolist(),
                                "provenance": item["scope"], "checkpoint": item["checkpoint"],
                                "upstream_training": [{"name": "sensor_models_and_word_context_priors",
                                                       "train_subjects": sorted(train), "provenance": item["scope"]}]})
        if seen != universe:
            raise ValueError("original inner folds omit allowed validation subjects")
        scores = pd.concat(outputs, ignore_index=True)
        policy, ranking = recipe.choose_decoder(scores, config)
        recipe.check_policy(policy, config)
        inner_scores_count += len(scores)
        record = {"selected": policy, "ranking": ranking, "fit_subjects": sorted(universe),
                  "source_model_keys": keys, "new_forest_fits": 0}
        if audit_run is None:
            directory.mkdir(parents=True)
            scores.to_csv(directory / "inner_validation_views.csv", index=False)
            write_json(directory / "inner_selection.json", record)
            write_json(directory / "inner_fold_manifest.json", inner_folds)
        else:
            stored = pd.read_csv(directory / "inner_validation_views.csv")
            pd.testing.assert_frame_equal(scores, stored, check_dtype=False)
            if json.loads((directory / "inner_selection.json").read_text()) != record:
                raise ValueError("saved policy/ranking differs from actual complete inner replay")
            if json.loads((directory / "inner_fold_manifest.json").read_text()) != inner_folds:
                raise ValueError("saved inner model source mapping differs")
        return policy, inner_folds

    output_root = args.output_dir if audit_run is None else audit_run
    outputs, folds, outer_policies = [], [], {}
    for fold in manifest["outer_folds"]:
        subject, key = fold["subject"], fold["model_key"]
        item = manifest["models"][key]
        train, valid = set(item["train_subjects"]), set(item["validation_subjects"])
        if valid != {subject} or train != subjects - valid:
            raise ValueError("outer source model is not complete leave-one-subject-out")
        directory = output_root / "folds" / subject
        policy, inner = select(fold["inner_keys"], train, directory)
        bundle = load(key)
        rows = np.flatnonzero(frame.subject.eq(subject))
        evidence = recipe.recipe_evidence(bundle, frame, contexts, matrices, masks, rows, config)
        output = recipe.predict_rows(frame.iloc[rows], evidence, policy)
        parent_driver.validate_output(frame.iloc[rows], output)
        output["subject"] = subject
        output["selected_decoder"] = policy["name"]
        if audit_run is None:
            output.to_csv(directory / "oof_predictions.csv", index=False)
        else:
            recipe.assert_outputs_equal(pd.read_csv(directory / "oof_predictions.csv"), output)
        outputs.append(output)
        outer_policies[subject] = policy
        folds.append({"fold_id": subject, "train_subjects": sorted(train), "validation_subjects": [subject],
                      "prediction_qa_ids": output.qa_id.tolist(), "inner_folds": inner,
                      "checkpoint": item["checkpoint"], "provenance": item["scope"],
                      "upstream_training": [{"name": "sensor_models_and_word_context_priors",
                                             "train_subjects": sorted(train), "provenance": item["scope"]}]})
        print(f"{'audit' if audit_run else 'reselect'} outer={subject} rows={len(output)} complete={policy['complete']['name']} partial={policy['partial']['name']}", flush=True)
    combined = pd.concat(outputs, ignore_index=True).set_index("qa_id").loc[frame.qa_id].reset_index()
    final_policy, _ = select(manifest["final_inner_keys"], subjects, output_root / "final_selection")
    final = load(manifest["final_model_key"])
    if set(final["training_subjects"]) != subjects:
        raise ValueError("original final model does not use exactly the eighteen official subjects")
    test_evidence = recipe.recipe_evidence(final, test, test_contexts, test_matrices, test_masks, np.arange(len(test)), config)
    test_output = recipe.predict_rows(test, test_evidence, final_policy)
    parent_driver.validate_output(test, test_output)
    if len(loaded) != 76 or set(loaded) != set(manifest["models"]):
        raise ValueError("not all and only the 76 original models were reused")
    correctness = combined.prediction.to_numpy() == frame.answer.to_numpy()
    by_subject = pd.Series(correctness).groupby(frame.subject).mean()
    protocol = {"schema_version": 1, "validation_kind": "nested_subject", "folds": folds,
                "fitted_components": ["sensor_models_and_word_context_priors"],
                "selection_history": config["validation_caveat"], "new_forest_fits": 0,
                "model_reuse": "Exactly76 previously fitted v5 models; only recipe/family selection is new"}
    protocol_check = validate_protocol(protocol, frame, subjects, base_dir=output_root)
    report = {"status": "nested_recipe_reselection_evaluated" if audit_run is None else "all_reused_models_inner_scores_and_outputs_replayed",
              "rows": len(frame), "correct_rows": int(correctness.sum()), "accuracy": float(correctness.mean()),
              "subject_macro_accuracy": float(by_subject.mean()), "worst_subject_accuracy": float(by_subject.min()),
              "by_subject": by_subject.to_dict(), "test_rows": len(test_output), "new_forest_fits": 0,
              "reused_models": len(loaded), "inner_score_rows": inner_scores_count,
              "outer_policy_counts": {scope: dict(Counter(policy[scope]["name"] for policy in outer_policies.values()))
                                      for scope in ("complete", "partial")},
              "final_policy": final_policy, "runtime_seconds": time.perf_counter() - started,
              "verified_original_files": files.finish(), "protocol_audit": protocol_check,
              "limitations": [config["validation_caveat"], "No forest was refitted; original fit_config/source remain distinct from these inference recipes."]}
    expected_policy = {"model_kind": KIND, "recipe_config": config, "selected_policy": final_policy,
                       "recipe_source_hashes": sources(), "original_fit_config": manifest["original_fit_config"],
                       "original_source_hashes": manifest["original_source_hashes"],
                       "source_model": manifest["models"][manifest["final_model_key"]],
                       "source_manifest": reference(output_root / "model_source_manifest.json"), "new_forest_fits": 0}
    if audit_run is None:
        combined.to_csv(output_root / "oof_predictions.csv", index=False)
        test_output.to_csv(output_root / "test_predictions.csv", index=False)
        write_json(output_root / "protocol.json", protocol)
        write_json(output_root / "inference_policy.json", expected_policy)
        (output_root / "inference_policy.sha256").write_text(sha(output_root / "inference_policy.json") + "\n")
    else:
        recipe.assert_outputs_equal(pd.read_csv(output_root / "oof_predictions.csv"), combined)
        recipe.assert_outputs_equal(pd.read_csv(output_root / "test_predictions.csv"), test_output)
        if json.loads((output_root / "protocol.json").read_text()) != protocol:
            raise ValueError("saved source/fold protocol differs from actual replay")
        policy = json.loads((output_root / "inference_policy.json").read_text())
        if policy != expected_policy:
            raise ValueError("final recipe policy differs from original three final-inner model selections")
    return report


def reselect(args):
    if args.output_dir.exists():
        raise FileExistsError("recipe selection run must be new")
    manifest = source_manifest(args.parent_run)
    config = json.loads(args.config.read_text())
    recipe.validate_config(config, manifest["original_fit_config"])
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "model_source_manifest.json", manifest)
    paths = [args.train_csv, args.test_csv, args.config, *[getattr(args, name) for name in SENSORS],
             args.output_dir / "model_source_manifest.json"]
    frozen = {"recipe_config": config, "inputs": {str(path.resolve()): sha(path) for path in paths},
              "source_hashes": sources(), "frozen_before_selection": True, "created_utc": datetime.now(UTC).isoformat()}
    write_json(args.output_dir / "frozen_config.json", frozen)
    for name, digest in frozen["source_hashes"].items():
        target = args.output_dir / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        if sha(target) != digest:
            raise ValueError("source changed while freezing recipe selection")
    report = execute(args, manifest, config)
    verify_frozen(args.output_dir)
    report["file_hashes"] = {path.name: sha(path) for path in args.output_dir.iterdir() if path.is_file()}
    write_json(args.output_dir / "report.json", report)
    print(json.dumps({key: report[key] for key in ("status", "correct_rows", "accuracy", "worst_subject_accuracy", "reused_models", "new_forest_fits", "final_policy", "runtime_seconds")}, indent=2))


def verify_frozen(run):
    frozen = json.loads((run / "frozen_config.json").read_text())
    if not frozen["frozen_before_selection"] or frozen["source_hashes"] != sources():
        raise ValueError("recipe code differs from its frozen selection source")
    for name, digest in frozen["source_hashes"].items():
        if sha(run / "source_snapshot" / name) != digest:
            raise ValueError("recipe source snapshot differs")
    for name, digest in frozen["inputs"].items():
        if sha(name) != digest:
            raise ValueError("recipe selection input changed")
    return frozen


def audit(args):
    if args.output_dir.exists():
        raise FileExistsError("recipe audit directory must be new")
    frozen = verify_frozen(args.run_dir)
    for path in [args.train_csv, args.test_csv, *[getattr(args, name) for name in SENSORS]]:
        if str(path.resolve()) not in frozen["inputs"]:
            raise ValueError("audit inputs were not frozen by re-selection")
    manifest = json.loads((args.run_dir / "model_source_manifest.json").read_text())
    report = execute(args, manifest, frozen["recipe_config"], audit_run=args.run_dir)
    verify_frozen(args.run_dir)
    policy = args.run_dir / "inference_policy.json"
    if sha(policy) != policy.with_suffix(".sha256").read_text().strip():
        raise ValueError("saved inference policy SHA differs")
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "report.json", report)
    print(json.dumps({key: report[key] for key in ("status", "reused_models", "new_forest_fits", "inner_score_rows", "runtime_seconds")}, indent=2))


def infer(args):
    if args.output_dir.exists():
        raise FileExistsError("recipe inference directory must be new")
    if sha(args.policy) != args.policy.with_suffix(".sha256").read_text().strip():
        raise ValueError("inference policy SHA differs")
    policy = json.loads(args.policy.read_text())
    if policy["model_kind"] != KIND or policy["recipe_source_hashes"] != sources():
        raise ValueError("inference recipe kind/source differs")
    if policy["original_source_hashes"] != parent_driver.sources():
        raise ValueError("original fitted-model sources differ")
    manifest_ref = policy["source_manifest"]
    if sha(manifest_ref["path"]) != manifest_ref["sha256"]:
        raise ValueError("original model source manifest differs")
    recipe.validate_config(policy["recipe_config"], policy["original_fit_config"])
    recipe.check_policy(policy["selected_policy"], policy["recipe_config"])
    model = policy["source_model"]
    if sha(model["checkpoint"]["path"]) != model["checkpoint"]["sha256"]:
        raise ValueError("original final model SHA differs")
    if sha(model["scope"]["path"]) != model["scope"]["sha256"]:
        raise ValueError("original final scope SHA differs")
    bundle = joblib.load(model["checkpoint"]["path"])
    if bundle["config"] != policy["original_fit_config"] or bundle["source_hashes"] != policy["original_source_hashes"]:
        raise ValueError("original final fitting configuration/source differs")
    if bundle["training_subjects"] != model["train_subjects"] or model["validation_subjects"]:
        raise ValueError("inference source is not the declared all-training-subject final model")
    scene = read_csv_str(args.scene_manifest) if args.scene_manifest else None
    frame, contexts = prepare(read_csv_str(args.test_csv), training=False, manifest=scene)
    paths = {name: getattr(args, name) for name in SENSORS}
    data = {name: read_sensor(path, bundle["config"]["views"][name]) for name, path in paths.items()}
    if any(not np.array_equal(data[name]["names"], bundle["feature_names"][name]) for name in SENSORS):
        raise ValueError("inference feature schema differs from original fitted models")
    matrices, masks = parent_driver.feature_data(frame, data)
    evidence = recipe.recipe_evidence(bundle, frame, contexts, matrices, masks, np.arange(len(frame)), policy["recipe_config"])
    output = recipe.predict_rows(frame, evidence, policy["selected_policy"])
    parent_driver.validate_output(frame, output)
    args.output_dir.mkdir(parents=True)
    output.to_csv(args.output_dir / "test_predictions.csv", index=False)
    write_json(args.output_dir / "report.json", {"status": "load_only_reused_model", "rows": len(output),
               "policy_sha256": sha(args.policy), "original_checkpoint": model["checkpoint"],
               "original_fit_config": policy["original_fit_config"], "selected_policy": policy["selected_policy"],
               "prediction_sha256": sha(args.output_dir / "test_predictions.csv"),
               "inputs": {str(path.resolve()): sha(path) for path in [args.test_csv, *paths.values()]},
               "new_forest_fits": 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    selection = commands.add_parser("reselect")
    selection.add_argument("--parent-run", type=Path, required=True)
    selection.add_argument("--config", type=Path, required=True)
    auditing = commands.add_parser("audit")
    auditing.add_argument("--run-dir", type=Path, required=True)
    for command in (selection, auditing):
        command.add_argument("--train-csv", type=Path, required=True)
    prediction = commands.add_parser("infer")
    prediction.add_argument("--policy", type=Path, required=True)
    prediction.add_argument("--scene-manifest", type=Path)
    for command in (selection, auditing, prediction):
        command.add_argument("--test-csv", type=Path, required=True)
        for name in SENSORS:
            command.add_argument(f"--{name}", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    {"reselect": reselect, "audit": audit, "infer": infer}[args.command](args)


if __name__ == "__main__":
    main()
