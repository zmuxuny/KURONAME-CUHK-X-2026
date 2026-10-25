#!/usr/bin/env python3
"""Audit four-modality visual emotion models and replay every inner selection."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from cuhkx import emotion_fusion_nested as fusion
from cuhkx import emotion_nested as legacy
from cuhkx import emotion_template_nested as template
from cuhkx.emotion_visual_nested import verify_training_content
from cuhkx.validation import attach_predictions, prepare_qa, validate_protocol
from scripts.audit_emotion_nested_provenance import (
    VerifiedFiles,
    audit_checkpoint,
    audit_scope,
    sensor_contract,
)
from scripts.run_emotion_nested import feature_data


def check_policy(policy, config):
    if set(policy) != {"name", "complete", "partial"} or policy["name"] != template.POLICY_NAME:
        raise ValueError("invalid template policy")
    if any(policy[scope] not in config["decoders"] for scope in ("complete", "partial")):
        raise ValueError("template policy contains an unfrozen decoder")


def audit(args):
    if args.output_dir.exists():
        raise FileExistsError("audit output must be new")
    run = args.run_dir.resolve()
    files = VerifiedFiles()
    frozen = files.json(run / "frozen_config.json", "pretraining contract")
    for path, digest in frozen["inputs"].items():
        files.verify(path, digest, "frozen input")
    for path, digest in frozen["source_hashes"].items():
        files.verify(ROOT / path, digest, "frozen source")
        files.verify(run / "source_snapshot" / path, digest, "immutable source copy")
    config = frozen["config"]
    official = pd.read_csv(args.train_csv, dtype=str, keep_default_na=False)
    frame, contexts = legacy.prepare(official, training=True)
    qa = prepare_qa(official, ["HAU:emotion"])
    subjects = set(frame.subject)
    archives = {name: getattr(args, name) for name in ("imu", "skeleton", "radar", "visual")}
    eligible, feature_counts = sensor_contract(frame, archives, config, frozen["inputs"], files)
    sensor_data = {name: legacy.read_sensor(path, config["views"][name]) for name, path in archives.items()}
    matrices, masks = feature_data(frame, sensor_data)
    protocol = files.json(run / "protocol.json", "outer protocol")
    folds, outer_outputs, scope_count, model_count = [], [], 0, 0

    def audit_inner(folder, universe):
        nonlocal scope_count, model_count
        manifest = files.json(folder / "inner_fold_manifest.json", "retained inner models")
        if len(manifest) != config["inner_folds"]:
            raise ValueError("unexpected inner fold count")
        records, validation_users = [], set()
        for inner in manifest:
            train, valid = set(inner["train_subjects"]), set(inner["validation_subjects"])
            if train & valid or train | valid != universe or valid & validation_users:
                raise ValueError("inner model trained outside its allowed subject partition")
            validation_users |= valid
            provenance = inner["provenance"]
            scope = files.json(provenance["path"], "inner scope", provenance["sha256"])
            audit_scope(scope, frame, train, valid, eligible, frozen)
            checkpoint = inner["checkpoint"]
            audit_checkpoint(checkpoint["path"], checkpoint["sha256"], scope, frame,
                             contexts, feature_counts, config, files)
            scope_count += 1
            model_count += 1
            bundle = joblib.load(checkpoint["path"])
            verify_training_content(bundle, scope, frame, matrices, masks)
            rows = np.flatnonzero(frame.subject.isin(valid))
            if inner["prediction_qa_ids"] != frame.iloc[rows].qa_id.tolist():
                raise ValueError("inner prediction IDs differ from actual held-out QA")
            evidence = fusion.base_evidence(bundle, frame, contexts, matrices, masks, rows)
            records.append(template.validation_views(frame.iloc[rows], evidence, config["decoders"]))
        if validation_users != universe:
            raise ValueError("inner models do not cover every subject in their training universe")
        replayed = pd.concat(records, ignore_index=True)
        stored_path = files.current(folder / "inner_validation_views.csv", "saved inner validation scores")
        stored = pd.read_csv(stored_path)
        pd.testing.assert_frame_equal(replayed, stored, check_dtype=False)
        expected_policy, _ = template.choose_decoder(replayed, config)
        selection = files.json(folder / "inner_selection.json", "selected template policy")
        check_policy(selection["selected"], config)
        if selection["selected"] != expected_policy:
            raise ValueError("saved policy differs from full inner score replay")
        return {"inner_folds": len(manifest), "score_rows_replayed": len(replayed),
                "selected_policy": expected_policy}

    for fold in protocol["folds"]:
        train, valid = set(fold["train_subjects"]), set(fold["validation_subjects"])
        if len(valid) != 1 or train != subjects - valid:
            raise ValueError("outer fold is not leave-one-official-subject-out")
        folder = run / "folds" / fold["fold_id"]
        selection = audit_inner(folder, train)
        provenance = fold["upstream_training"][0]["provenance"]
        scope = files.json(provenance["path"], "outer scope", provenance["sha256"])
        audit_scope(scope, frame, train, valid, eligible, frozen)
        checkpoint = fold["checkpoint"]
        checked = audit_checkpoint(checkpoint["path"], checkpoint["sha256"], scope,
                                   frame, contexts, feature_counts, config, files)
        scope_count += 1
        model_count += 1
        if checked["selected_decoder"] != selection["selected_policy"]:
            raise ValueError("outer model uses a different decoder than its inner selection")
        bundle = joblib.load(checkpoint["path"])
        verify_training_content(bundle, scope, frame, matrices, masks)
        rows = np.flatnonzero(frame.subject.isin(valid))
        evidence = fusion.base_evidence(bundle, frame, contexts, matrices, masks, rows)
        replayed = template.predict_rows(frame.iloc[rows], evidence, bundle["decoder"])
        output_path = files.current(folder / "oof_predictions.csv", "stored outer predictions")
        output = pd.read_csv(output_path, usecols=["qa_id", "prediction", "margin"])
        pd.testing.assert_frame_equal(replayed, output, check_exact=False, rtol=1e-12, atol=1e-12)
        checked_qa, _ = attach_predictions(qa.loc[qa.subject.isin(valid)], output)
        outer_outputs.append(checked_qa[["qa_id", "prediction"]])
        folds.append({"subject": next(iter(valid)), "selection": selection, "rows_replayed": len(output)})
    final_selection = audit_inner(run / "final_selection", subjects)
    checkpoint = files.current(run / "models.joblib", "final model")
    recorded_digest = (run / "models.sha256").read_text().strip()
    files.verify(checkpoint, recorded_digest, "final model SHA sidecar")
    bundle = joblib.load(checkpoint)
    final_scope = files.json(run / "final_training_scope.json", "final refit scope")
    verify_training_content(bundle, final_scope, frame, matrices, masks)
    if set(final_scope["train_subjects"]) != subjects or final_scope["validation_subjects"]:
        raise ValueError("final model was not fitted on the declared official subject universe")
    if final_scope["training_qa_ids"] != frame.qa_id.tolist():
        raise ValueError("final model scope does not match official training QA")
    if bundle["decoder"] != final_selection["selected_policy"] or bundle["config"] != config:
        raise ValueError("final model violates its frozen configuration or decoder selection")
    if bundle["source_hashes"] != frozen["source_hashes"] or bundle["input_hashes"] != frozen["inputs"]:
        raise ValueError("final model source/input hashes differ from frozen training inputs")
    for name in archives:
        expected_units = set(frame.unit) & eligible[name]
        if set(bundle["fitted_units"][name]) != expected_units:
            raise ValueError("final model sensor scope differs from official eligible units")
    audit_checkpoint(checkpoint, recorded_digest, final_scope, frame, contexts,
                     feature_counts, config, files)
    if final_scope["source_hashes"] != frozen["source_hashes"] or final_scope["input_hashes"] != frozen["inputs"]:
        raise ValueError("final scope source/input hashes differ from the frozen contract")
    scope_count += 1
    model_count += 1
    stored_oof = files.current(run / "oof_predictions.csv", "combined outer predictions")
    output = pd.read_csv(stored_oof, usecols=["qa_id", "prediction"])
    gathered = pd.concat(outer_outputs).set_index("qa_id").loc[qa.qa_id, "prediction"]
    checked, _ = attach_predictions(qa, output)
    if not np.array_equal(checked.prediction, gathered):
        raise ValueError("combined OOF differs from the separately replayed outer models")
    protocol_check = validate_protocol(protocol, qa, subjects, base_dir=run)
    files.current(Path(__file__), "audit source")
    report = {"status": "all_models_scopes_and_inner_selections_replayed",
              "created_utc": datetime.now(UTC).isoformat(), "folds": folds,
              "final_selection": final_selection, "model_checkpoints_checked": model_count,
              "training_scopes_checked": scope_count, "protocol_audit": protocol_check,
              "files": files.finish(), "audit_source_sha256": files.files[Path(__file__).resolve()]["sha256"],
              "limitations": ["Scope content and retained trained models are corroborated; this is not an unseen-subject design holdout.",
                              "Synthetic partial views alter constraints, not observed word vocabularies; they share scenes."]}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    for name in ("imu", "skeleton", "radar", "visual"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    report = audit(parser.parse_args())
    print(json.dumps({key: report[key] for key in (
        "status", "model_checkpoints_checked", "training_scopes_checked")}, indent=2))


if __name__ == "__main__":
    main()
