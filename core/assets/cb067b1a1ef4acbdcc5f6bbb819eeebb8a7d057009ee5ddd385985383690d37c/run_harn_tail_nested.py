#!/usr/bin/env python3
"""Nested HARn evaluation with a subject-fitted final video residual stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from cuhkx import harn_tail as tail
from cuhkx.harn_nested import QA_COLUMNS, predict_evidence, training_manifest
from cuhkx.video_midfeatures import MidFeatureBank
from scripts import run_harn_nested_fusion as engine

SOURCES = [*engine.SOURCE_PATHS, "scripts/run_harn_tail_nested.py", "src/cuhkx/harn_tail.py",
           "src/cuhkx/video_midfeatures.py", "scripts/extract_video_midfeatures.py"]
ORIGINAL_FIT_LOGGED = engine.fit_logged
ORIGINAL_WRITE = engine.write_json


def validate_config(config):
    adaptation = config["visual"]["adaptation"]
    fixed = {"epochs": 4, "batch_units": 16, "layer4_lr": 0.0001, "classifier_lr": 0.001,
             "weight_decay": 0.01, "gradient_clip": 5.0, "class_balance": "inverse training clip frequency",
             "stored_weights": "float16 before validation", "optimizer": "AdamW", "schedule": "cosine per optimizer step"}
    if (any(adaptation.get(key) != value for key, value in fixed.items())
            or config["seed"] != 20260905 or config["workers"] != 3
            or config["skeleton"]["trees"] != 200
            or config["validation"]["inner_splits"] != 3
            or config["validation"]["skeleton_weights"] != [0.0, 0.5, 0.75, 1.0]):
        raise ValueError("configuration differs from the single frozen residual-stage mechanism")


def row_digest(matrix, rows, labels):
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": [len(rows), matrix.shape[1]], "dtype": matrix.dtype.str,
                              "labels_dtype": labels.dtype.str}, sort_keys=True).encode())
    for start in range(0, len(rows), 64):
        digest.update(np.ascontiguousarray(matrix[rows[start:start + 64]]).tobytes())
    digest.update(np.ascontiguousarray(labels).tobytes())
    return digest.hexdigest()


def compare_prediction_frames(actual, path):
    expected = pd.read_csv(path, keep_default_na=False)
    keys = [key for key in ("qa_id", "skeleton_weight", "inner_fold") if key in actual]
    if set(actual.columns) != set(expected.columns):
        raise ValueError("replayed decoded prediction columns differ")
    pd.testing.assert_frame_equal(
        actual.sort_values(keys).reset_index(drop=True),
        expected[actual.columns].sort_values(keys).reset_index(drop=True),
        check_dtype=False, check_exact=False, rtol=0, atol=1e-6,
    )


def fit_logged(bank, qa, subjects, config, output, fit_id, inputs):
    bundle, _ = ORIGINAL_FIT_LOGGED(bank, qa, subjects, config, output, fit_id, inputs)
    path = output / "fits" / f"{fit_id}.json"
    scope = json.loads(path.read_text())
    rows, labels = tail.training_rows(bank, subjects)
    bundle["fit_id"] = fit_id
    bundle["fit_evidence_path"] = str(output / "fits" / f"{fit_id}.saved_evidence.npz")
    bundle["source_manifest"] = [engine.manifest_entry(ROOT / name) for name in SOURCES]
    checkpoint = output / "fits" / f"{fit_id}.models.joblib"
    engine.checkpoint_save(checkpoint, bundle)
    scope["adapted_visual"] = {"rows": rows.tolist(), "labels": labels.tolist(),
                               "actual_matrix_and_labels_sha256": row_digest(bank.visual, rows, labels),
                               "history": bundle["visual_model"].history,
                               "checkpoint_sha256": engine.sha256_file(checkpoint),
                               "middle_spec": bank.visual_spec,
                               "retained_before_validation": True}
    ORIGINAL_WRITE(path, scope)
    return bundle, {"path": str(path.resolve()), "sha256": engine.sha256_file(path)}


def predict_and_save(bundle, bank, qa, *, training_paths):
    evidence = predict_evidence(bundle, bank, qa, training_paths=training_paths)
    if training_paths:
        engine.save_evidence(Path(bundle["fit_evidence_path"]), evidence)
    return evidence


def write_json(path, payload):
    if path.name == "protocol.json":
        payload["selection_history"] = payload["historical_subject_exposure"]
        payload["feature_pretraining"]["visual"] = (
            "Kinetics400 R2Plus1D stem/layer1..3 frozen; layer4+44-class head trained only on each inner/outer HARn training subject scope")
    if path.name == "report.json" and "limitations" in payload:
        payload["limitations"][1:3] = [
            "Four fixed adaptation epochs; all visual fitting, Skeleton models and answer dictionaries exclude validation subjects.",
            "Checkpoint expects fp16 generic layer3 cache plus Skeleton statistics; CUDA replay is fixed for adapted video predictions.",
        ]
    ORIGINAL_WRITE(path, payload)
    if path.name == "source_manifest.json":
        for entry in payload:
            source = Path(entry["path"])
            dest = path.parent / "source_snapshot" / source.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            if engine.sha256_file(dest) != entry["sha256"]:
                raise ValueError("source changed while freezing the residual-stage experiment")


def audit(args):
    run = args.run_dir
    config = json.loads((run / "config.json").read_text())
    validate_config(config)
    for entry in json.loads((run / "source_manifest.json").read_text()):
        if engine.sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("audit source differs from fitted source")
    qa = engine.harn_rows(engine.read_qa(args.train_csv, labeled=True))
    bank = MidFeatureBank.load(args.skeleton, args.visual)
    checks, replayed = [], 0
    inner_predictions, outer_predictions = {}, []
    final_fit = None
    for checkpoint in sorted((run / "fits").glob("*.models.joblib")):
        fit_id = checkpoint.name.removesuffix(".models.joblib")
        scope = json.loads((run / "fits" / f"{fit_id}.json").read_text())
        if (engine.sha256_file(checkpoint) != scope["adapted_visual"]["checkpoint_sha256"]
                or engine.sha256_file(checkpoint) != checkpoint.with_suffix(".joblib.sha256").read_text().strip()):
            raise ValueError("actual retained model SHA differs")
        bundle = joblib.load(checkpoint)
        train_qa = qa.loc[qa.qa_id.isin(scope["training_qa_ids"])].copy()
        original = training_manifest(bank, train_qa, set(scope["train_subjects"]), config)
        if any(scope[key] != value for key, value in original.items()):
            raise ValueError("actual fitted HARn units, labels or subjects differ")
        rows, labels = tail.training_rows(bank, set(scope["train_subjects"]))
        learned = bundle["visual_model"]
        if learned.settings != config["visual"]["adaptation"]:
            raise ValueError("retained adaptation settings differ from frozen configuration")
        np.testing.assert_array_equal(rows, learned.training_rows)
        np.testing.assert_array_equal(labels, learned.training_labels)
        if (rows.tolist() != scope["adapted_visual"]["rows"] or labels.tolist() != scope["adapted_visual"]["labels"]
                or learned.history != scope["adapted_visual"]["history"]
                or row_digest(bank.visual, rows, labels) != scope["adapted_visual"]["actual_matrix_and_labels_sha256"]):
            raise ValueError("actual adapted fitting rows or matrices differ")
        evidence_path = run / "fits" / f"{fit_id}.saved_evidence.npz"
        if evidence_path.is_file():
            with np.load(evidence_path, allow_pickle=False) as archive:
                expected = {key: archive[key].copy() for key in archive.files}
            units = set(expected["units"].astype(str))
            from cuhkx.metadata import feature_unit, subject_id

            target = qa.loc[qa.path.map(lambda value: feature_unit(value, training=True)).isin(units)]
            if set(target.path.map(subject_id)) & set(scope["train_subjects"]):
                raise ValueError("saved held-out evidence includes a fitted subject")
            actual = predict_evidence(bundle, bank, target[QA_COLUMNS], training_paths=True)
            for key, values in actual.items():
                if np.issubdtype(np.asarray(values).dtype, np.floating):
                    np.testing.assert_allclose(values, expected[key], rtol=0, atol=1e-6)
                else:
                    np.testing.assert_array_equal(values, expected[key])
            if "_inner" in fit_id:
                parent, fold = fit_id.rsplit("_inner", 1)
                collection = inner_predictions.setdefault(parent, {})
                for weight in config["validation"]["skeleton_weights"]:
                    prediction = engine.decode_questions(target[QA_COLUMNS], actual, bundle, weight,
                                                         training_paths=True)
                    prediction["inner_fold"] = int(fold)
                    collection.setdefault(weight, []).append(prediction)
            else:
                parent = fit_id.removesuffix("_final")
                selection = json.loads((run / "fits" / f"{parent}_selection.json").read_text())
                prediction = engine.decode_questions(target[QA_COLUMNS], actual, bundle,
                                                     selection["selected_skeleton_weight"], training_paths=True)
                prediction["outer_fold"] = parent
                outer_predictions.append(prediction)
            replayed += len(target)
        else:
            if fit_id != "final_all_subjects_fit":
                raise ValueError("a validation model is missing its saved evidence")
            final_fit = bundle
        checks.append({"fit_id": fit_id, "subjects": scope["train_subjects"],
                       "actual_matrix_and_labels_sha256": scope["adapted_visual"]["actual_matrix_and_labels_sha256"]})
        print(f"tail_audit_scopes={len(checks)} held_out_QA_rows={replayed}", flush=True)
    if len(checks) != 76 or replayed != len(qa) * 19:
        raise ValueError("audit must cover all 76 fitted models and 19 repeats of held-out QA evidence")
    selections, decoded_rows = [], 0
    for parent, weights in inner_predictions.items():
        frames, reports = [], []
        for weight, parts in weights.items():
            if len(parts) != 3:
                raise ValueError("each inner recipe must replay all three folds")
            frame = pd.concat(parts, ignore_index=True)
            target = qa.loc[qa.qa_id.isin(frame.qa_id)]
            reports.append({"skeleton_weight": weight, "metrics": engine.evaluate_predictions(target, frame)})
            frames.append(frame)
        combined = pd.concat(frames, ignore_index=True)
        compare_prediction_frames(combined, run / "fits" / f"{parent}_inner_predictions.csv")
        selected = engine.select_weight(reports, config["validation"]["weight_tie_order"])
        frozen_selection = json.loads((run / "fits" / f"{parent}_selection.json").read_text())
        if (selected != frozen_selection["selected_skeleton_weight"]
                or reports != frozen_selection["candidate_reports"]):
            raise ValueError("actual inner metrics or selected fusion weight differ")
        selections.append({"scope": parent, "selected_skeleton_weight": selected})
        decoded_rows += len(combined)
    if len(selections) != 19 or len(outer_predictions) != 18 or final_fit is None:
        raise ValueError("decoded replay did not cover every selection and outer fold")
    oof = pd.concat(outer_predictions, ignore_index=True)
    compare_prediction_frames(oof, run / "oof_predictions.csv")
    final = engine.checkpoint_load(run / "models.joblib", workers=3)
    for key, values in final["visual_model"].state.items():
        np.testing.assert_array_equal(values, final_fit["visual_model"].state[key])
    if final["selected_skeleton_weight"] != next(item["selected_skeleton_weight"] for item in selections
                                                if item["scope"] == "final_all_subjects"):
        raise ValueError("final checkpoint fusion differs from replayed nested selection")
    report = json.loads((run / "report.json").read_text())
    if engine.evaluate_predictions(qa, oof) != report["metrics"]:
        raise ValueError("outer decoded metrics differ from the saved report")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    ORIGINAL_WRITE(args.output_dir / "report.json", {"status": "actual_scope_models_features_probabilities_replayed",
                                                   "checks": checks, "held_out_QA_rows_replayed": replayed,
                                                   "probability_tolerance": 1e-6,
                                                   "inner_selections_replayed": selections,
                                                   "decoded_inner_rows_replayed": decoded_rows,
                                                   "decoded_outer_rows_replayed": len(oof)})


def configure():
    engine.FeatureBank = MidFeatureBank
    engine.CHECKPOINT_KIND = tail.KIND
    engine.fit_bundle = tail.fit_bundle
    engine.fit_logged = fit_logged
    engine.predict_evidence = predict_and_save
    engine.write_json = write_json
    engine.SOURCE_PATHS = SOURCES
    engine.COMPONENTS = ["skeleton_extra_full", "skeleton_extra_aggregate", "subject_scoped_r2plus1d_layer4_and_head",
                         "action_dictionary", "object_dictionary", "option_prior"]


def main():
    configure()
    if len(sys.argv) > 1 and sys.argv[1] == "audit":
        parser = argparse.ArgumentParser(description="Replay every fitted residual-stage scope")
        for name in ("run-dir", "train-csv", "skeleton", "visual", "output-dir"):
            parser.add_argument(f"--{name}", type=Path, required=True)
        args = parser.parse_args(sys.argv[2:])
        with threadpool_limits(limits=3):
            audit(args)
        return 0
    args = engine.parse_args()
    if args.command == "train":
        validate_config(json.loads(args.config.read_text()))
    else:
        bundle = engine.checkpoint_load(args.checkpoint, workers=args.workers)
        for entry in bundle["source_manifest"]:
            if engine.sha256_file(Path(entry["path"])) != entry["sha256"]:
                raise ValueError("inference source differs from frozen fitting code")
    with threadpool_limits(limits=3):
        return engine.train_main(args) if args.command == "train" else engine.infer_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
