#!/usr/bin/env python3
"""Audit saved nested emotion scopes and complete their evaluation protocol schema.

The original protocol, checkpoints, predictions, and decoder choices remain unchanged.
Only locally generated training OOF predictions may be evaluated. This verifies saved
scope records against official QA and checkpoint contents; it does not create a fresh
blind holdout or prove the historical execution of every estimator fit.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from cuhkx.emotion_nested import prepare
from cuhkx.metadata import subject_id
from cuhkx.validation import (
    attach_predictions,
    prediction_path,
    prepare_qa,
    sha256_file,
    validate_protocol,
)
from scripts.evaluate_local_predictions import evaluate


def string_set(values, name):
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a nonempty list")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{name} contains an invalid value")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate values")
    return set(values)


class VerifiedFiles:
    def __init__(self):
        self.files = {}

    def verify(self, path, digest, role):
        path = Path(path).resolve()
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"invalid SHA256 for {path}")
        if path not in self.files:
            actual = sha256_file(path)
            self.files[path] = {"path": str(path), "sha256": actual, "roles": []}
        record = self.files[path]
        if record["sha256"] != digest:
            raise ValueError(f"SHA256 mismatch for {path}")
        if role not in record["roles"]:
            record["roles"].append(role)
        return path

    def current(self, path, role):
        return self.verify(path, sha256_file(Path(path)), role)

    def json(self, path, role, digest=None):
        path = self.verify(path, digest, role) if digest else self.current(path, role)
        return json.loads(path.read_text())

    def finish(self):
        for path, record in self.files.items():
            if sha256_file(path) != record["sha256"]:
                raise ValueError(f"audited file changed during execution: {path}")
        return list(self.files.values())


def sensor_contract(frame, archives, config, frozen_inputs, files):
    eligible, counts = {}, {}
    for name, path in archives.items():
        key = str(path.resolve())
        if key not in frozen_inputs:
            raise ValueError(f"sensor input was not frozen before training: {key}")
        path = files.verify(path, frozen_inputs[key], f"{name} sensor archive")
        with np.load(path, allow_pickle=False) as data:
            units = data["units"].astype(str).tolist()
            names = data["feature_names"].astype(str)
        if len(set(units)) != len(units):
            raise ValueError("sensor archive has duplicate units")
        counts[name] = sum(re.search(config["views"][name], value) is not None for value in names)
        if not counts[name]:
            raise ValueError("sensor feature pattern selected no columns")
        units = set(units)
        eligible[name] = set()
        for _, group in frame.groupby("group", sort=False):
            if len(group) == 3 and set(group.unit) <= units:
                eligible[name].update(group.unit)
    return eligible, counts


def audit_scope(record, frame, train_subjects, validation_subjects, eligible, frozen):
    train = string_set(record.get("train_subjects"), "record train_subjects")
    valid = string_set(record.get("validation_subjects"), "record validation_subjects")
    all_subjects = set(frame.subject)
    if train != set(train_subjects) or valid != set(validation_subjects):
        raise ValueError("scope subject sets differ from their protocol fold")
    if train & valid or (train | valid) - all_subjects:
        raise ValueError("scope has overlapping or unknown subjects")
    ids = string_set(record.get("training_qa_ids"), "training_qa_ids")
    selected = frame.loc[frame.subject.isin(train)]
    if ids != set(selected.qa_id):
        raise ValueError("training QA IDs do not match the actual official training subjects")
    if record.get("source_hashes") != frozen["source_hashes"]:
        raise ValueError("scope source hashes differ from the frozen source snapshot")
    if record.get("input_hashes") != frozen["inputs"]:
        raise ValueError("scope input hashes differ from the frozen input snapshot")
    recorded_units = record.get("training_units_by_sensor", {})
    if set(recorded_units) != set(eligible):
        raise ValueError("scope does not declare every fitted sensor")
    units_by_sensor = {}
    for name, units in recorded_units.items():
        units = string_set(units, name + " training units")
        if any(subject_id(unit) not in train for unit in units):
            raise ValueError("sensor training unit belongs to an excluded or unknown subject")
        expected = set(selected.unit) & eligible[name]
        if units != expected:
            raise ValueError("sensor training units differ from the eligible official QA units")
        units_by_sensor[name] = len(units)
    digest = record.get("fitted_estimator_digest_joblib")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{32}", digest):
        raise ValueError("scope is missing its estimator object digest record")
    return {"train_subjects": sorted(train), "validation_subjects": sorted(valid),
            "training_qa_rows": len(ids), "training_units_by_sensor": units_by_sensor}


def audit_word_priors(bundle, frame, contexts, train_subjects):
    train = frame.loc[frame.subject.isin(train_subjects)]
    words = sorted({word for options in train.options for word in options})
    index = {word: i for i, word in enumerate(words)}
    if bundle["words"] != words or bundle["word_index"] != index:
        raise ValueError("checkpoint word vocabulary differs from its training subjects")
    counts = np.ones((len(words), 2, 3))
    association = np.ones((len(words), 3))
    for row in train.itertuples():
        for word in row.options:
            counts[index[word], int(word != row.truth_word), row.position] += 1
        association[index[row.truth_word], row.position] += 1
    association /= association.sum(axis=1, keepdims=True)
    if not np.array_equal(bundle["lexical"], np.log(counts[:, 0]) - np.log(counts[:, 1])):
        raise ValueError("checkpoint lexical prior includes different training labels")
    if not np.array_equal(bundle["association"], association):
        raise ValueError("checkpoint word association includes different training labels")
    groups = list(train.groupby("group", sort=False))
    if len(bundle["references"]) != len(groups):
        raise ValueError("checkpoint context reference count differs from training scenes")
    for reference, (group_id, group) in zip(bundle["references"], groups, strict=True):
        expected = [(row.truth_word, row.position) for row in group.itertuples()]
        if reference["word_states"] != expected:
            raise ValueError("checkpoint context references include different training labels")
        if not np.array_equal(reference["context"], contexts[group_id]):
            raise ValueError("checkpoint action context differs from official unlabeled options")


def audit_checkpoint(path, digest, scope, frame, contexts, feature_counts, config, files):
    path = files.verify(path, digest, "outer checkpoint; verified before trusted-local loading")
    bundle = joblib.load(path)
    if set(bundle["training_subjects"]) != set(scope["train_subjects"]):
        raise ValueError("checkpoint training subjects differ from its scope record")
    if bundle["fitted_units"] != scope["training_units_by_sensor"]:
        raise ValueError("checkpoint fitted units differ from its scope record")
    if bundle["config"] != config or set(bundle["models"]) != set(feature_counts):
        raise ValueError("checkpoint model configuration differs from the frozen configuration")
    for name, model in bundle["models"].items():
        if type(model).__name__ != "ExtraTreesClassifier":
            raise ValueError("unexpected saved estimator type")
        parameters = model.get_params()
        expected = {"n_estimators": config["trees"], "max_depth": config["max_depth"],
                    "min_samples_leaf": config["minimum_leaf"], "max_features": "sqrt",
                    "class_weight": "balanced", "n_jobs": config["workers"]}
        if any(parameters[key] != value for key, value in expected.items()):
            raise ValueError("saved estimator parameters differ from the frozen configuration")
        if model.n_features_in_ != feature_counts[name] or set(model.classes_) != {0, 1, 2}:
            raise ValueError("saved estimator feature count or target classes differ")
    audit_word_priors(bundle, frame, contexts, set(scope["train_subjects"]))
    object_digest = joblib.hash(bundle["models"])
    return {
        "path": str(path), "sha256": digest, "selected_decoder": bundle["decoder"],
        "word_priors_exact": True, "saved_model_parameters_verified": True,
        "recorded_joblib_object_digest": scope["fitted_estimator_digest_joblib"],
        "reloaded_joblib_object_digest": object_digest,
        "object_digest_matches": object_digest == scope["fitted_estimator_digest_joblib"],
        "object_digest_note": (
            "joblib.hash of sklearn estimator objects is not stable across joblib dump/load; "
            "the checkpoint byte SHA256 and explicit parameter/prior checks are authoritative. "
            "Scope records remain declarations of sensor fitting rows."
        ),
    }


def audit_run(args):
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite audit output: {args.output_dir}")
    run = args.run_dir.resolve()
    files = VerifiedFiles()
    original_path = run / "protocol.json"
    original = files.json(original_path, "original protocol; preserved byte for byte")
    frozen = files.json(run / "frozen_config.json", "pretraining frozen configuration")
    if not frozen.get("frozen_before_fitting"):
        raise ValueError("run lacks a frozen-before-fitting configuration declaration")
    config = frozen["config"]
    for path, digest in frozen["source_hashes"].items():
        files.verify(ROOT / path, digest, "frozen training source")
    for path, digest in frozen["inputs"].items():
        files.verify(path, digest, "frozen training input; only official QA labels are parsed")
    training_key = str(args.train_csv.resolve())
    if training_key not in frozen["inputs"]:
        raise ValueError("official training QA is absent from the frozen input contract")
    official = pd.read_csv(args.train_csv, dtype=str, keep_default_na=False)
    frame, contexts = prepare(official, training=True)
    qa = prepare_qa(official, ["HAU:emotion"])
    subjects = set(qa.subject)
    archives = {name: getattr(args, name) for name in ("imu", "skeleton", "radar")}
    eligible, feature_counts = sensor_contract(frame, archives, config, frozen["inputs"], files)
    amended = copy.deepcopy(original)
    components = {item["name"] for fold in original["folds"] for item in fold["upstream_training"]}
    if components != {"sensor_models_and_word_context_priors"}:
        raise ValueError("unexpected or incomplete emotion fitted-component bundle")
    if original.get("fitted_components", sorted(components)) != sorted(components):
        raise ValueError("original fitted_components disagrees with the saved fold declarations")
    amended["fitted_components"] = sorted(components)
    amended["selection_history"] = "Previously exposed subjects; no fresh blind holdout."
    audits, outer_frames = [], []
    for fold in amended["folds"]:
        valid = string_set(fold["validation_subjects"], "outer validation subjects")
        train = string_set(fold["train_subjects"], "outer training subjects")
        if len(valid) != 1 or train != subjects - valid or fold["fold_id"] != next(iter(valid)):
            raise ValueError("outer fold is not leave-one-official-subject-out")
        folder = run / "folds" / fold["fold_id"]
        upstream = fold["upstream_training"][0]
        if upstream["train_subjects"] != fold["train_subjects"]:
            raise ValueError("outer upstream subject declaration differs from its fold")
        provenance = upstream["provenance"]
        scope = files.json(provenance["path"], "outer fitting scope", provenance["sha256"])
        audit = audit_scope(scope, frame, train, valid, eligible, frozen)
        checkpoint = fold["checkpoint"]
        audit["checkpoint"] = audit_checkpoint(
            checkpoint["path"], checkpoint["sha256"], scope, frame, contexts,
            feature_counts, config, files,
        )
        selection = files.json(folder / "inner_selection.json", "frozen inner decoder selection")
        if selection["selected"] != audit["checkpoint"]["selected_decoder"]:
            raise ValueError("checkpoint decoder differs from the saved inner selection")
        if selection["selected"] not in config["decoders"]:
            raise ValueError("selected decoder was not in the frozen candidate list")
        inner_audits = []
        if len(fold["inner_folds"]) != config["inner_folds"]:
            raise ValueError("inner fold count differs from the frozen configuration")
        for inner in fold["inner_folds"]:
            inner_train = string_set(inner["train_subjects"], "inner training subjects")
            inner_valid = string_set(inner["validation_subjects"], "inner validation subjects")
            if inner_train & inner_valid or inner_train | inner_valid != train:
                raise ValueError("inner fold does not partition only the outer training subjects")
            provenance = inner["provenance"]
            inner_scope = files.json(provenance["path"], "inner fitting scope", provenance["sha256"])
            inner_audits.append(audit_scope(
                inner_scope, frame, inner_train, inner_valid, eligible, frozen
            ))
            inner["prediction_qa_ids"] = qa.loc[qa.subject.isin(inner_valid), "qa_id"].tolist()
            inner["upstream_training"] = [{"name": next(iter(components)),
                                           "train_subjects": sorted(inner_train),
                                           "provenance": provenance}]
        views_path = files.current(folder / "inner_validation_views.csv", "recorded inner validation rows")
        views = pd.read_csv(views_path, usecols=["qa_id", "subject", "decoder", "view"],
                            dtype=str, keep_default_na=False)
        true_subjects = views.qa_id.map(qa.set_index("qa_id").subject)
        if true_subjects.isna().any() or not true_subjects.equals(views.subject):
            raise ValueError("inner validation row subject disagrees with official QA")
        expected_ids = set(qa.loc[qa.subject.isin(train), "qa_id"])
        if set(views.qa_id) != expected_ids or not set(views.subject) <= train:
            raise ValueError("inner validation views include the outer subject or omit training QA")
        declared_decoders = {item["name"] for item in config["decoders"]}
        if set(views.decoder) != declared_decoders:
            raise ValueError("inner validation views omit a frozen decoder")
        for _, group in views.groupby(["decoder", "view"]):
            if set(group.qa_id) != expected_ids:
                raise ValueError("a decoder/view does not cover the complete inner validation scope")
        output_path = files.current(prediction_path(folder / "oof_predictions.csv"), "outer OOF predictions")
        output = pd.read_csv(output_path, dtype=str, keep_default_na=False)
        expected_qa = qa.loc[qa.subject.isin(valid)].reset_index(drop=True)
        checked, _ = attach_predictions(expected_qa, output, require_order=True)
        if output.qa_id.tolist() != fold["prediction_qa_ids"]:
            raise ValueError("outer OOF IDs disagree with the original protocol")
        if not output.subject.eq(next(iter(valid))).all():
            raise ValueError("outer OOF subject column differs from official QA")
        if not output.selected_decoder.eq(selection["selected"]["name"]).all():
            raise ValueError("outer OOF decoder differs from its frozen inner selection")
        outer_frames.append(checked[["qa_id", "prediction"]])
        audit["inner_folds"] = inner_audits
        audit["selected_decoder"] = selection["selected"]
        audits.append(audit)
    protocol_check = validate_protocol(amended, qa, subjects, base_dir=run)
    combined_path = files.current(prediction_path(run / "oof_predictions.csv"), "combined emotion OOF")
    combined = pd.read_csv(combined_path, usecols=["qa_id", "prediction"], dtype=str, keep_default_na=False)
    checked, _ = attach_predictions(qa, combined, require_order=True)
    gathered = pd.concat(outer_frames).set_index("qa_id").loc[qa.qa_id, "prediction"]
    if not np.array_equal(gathered.to_numpy(), checked.prediction.to_numpy()):
        raise ValueError("combined OOF differs from the individual outer-fold predictions")
    baseline = None
    if args.baseline_oof is not None:
        path = files.current(prediction_path(args.baseline_oof), "independent full local baseline OOF")
        full = pd.read_csv(path, usecols=["qa_id", "prediction"], dtype=str, keep_default_na=False)
        baseline, _ = attach_predictions(prepare_qa(official), full)
        baseline = baseline.loc[baseline.source.eq("HAU") & baseline.category.eq("emotion"),
                                ["qa_id", "prediction"]]
    amendment = {
        "operation": "schema completion and saved-artifact audit; no fitting or decoder reselection",
        "original_protocol": {"path": str(original_path), "sha256": sha256_file(original_path)},
        "added_fields": ["fitted_components", "selection_history", "inner prediction_qa_ids", "inner upstream_training"],
        "limitations": [
            "Inner estimators were discarded by the original runner; their scope records are audited, not reloaded.",
            "Historical subjects and structural rules were previously observed; this is not a new blind test.",
            "Masked question views approximate partial shared-option cores, not missing sensor recordings.",
            "Object digests of sklearn estimators are not stable across serialization; checkpoint byte hashes are checked.",
        ],
    }
    amended["schema_amendment"] = amendment
    report = {"status": "saved_scopes_and_checkpoints_audited", "created_utc": datetime.now(UTC).isoformat(),
              "amendment": amendment, "folds": audits, "protocol_audit": protocol_check,
              "feature_counts": feature_counts, "files": files.finish(),
              "source_code": {str(Path(__file__).resolve()): sha256_file(Path(__file__))}}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    protocol_path = args.output_dir / "protocol_audited.json"
    protocol_path.write_text(json.dumps(amended, indent=2) + "\n")
    report["audited_protocol"] = {"path": str(protocol_path.resolve()), "sha256": sha256_file(protocol_path)}
    if baseline is not None:
        baseline_path = args.output_dir / "baseline_emotion_oof.csv"
        baseline.to_csv(baseline_path, index=False, lineterminator="\n")
        evaluation = evaluate(argparse.Namespace(
            train_csv=args.train_csv, predictions=combined_path, baseline_predictions=baseline_path,
            protocol=protocol_path, scope=["HAU:emotion"], require_input_order=True,
            bootstrap_samples=10000, bootstrap_seed=20260905, confidence=0.95,
            output_dir=args.output_dir / "evaluation",
        ))
        report["evaluation"] = evaluation["validation"]
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    for name in ("imu", "skeleton", "radar"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--baseline-oof", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    report = audit_run(parser.parse_args())
    print(json.dumps({"status": report["status"], "outer_folds": len(report["folds"]),
                      "protocol_audit": report["protocol_audit"],
                      "evaluation": report.get("evaluation")}, indent=2))


if __name__ == "__main__":
    main()
