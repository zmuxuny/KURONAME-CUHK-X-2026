#!/usr/bin/env python3
"""Fit, replay, and audit nested subject-disjoint generic HAU action experts."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from cuhkx import action_generic_nested as model
from cuhkx.submission import read_csv_str
from cuhkx.validation import (
    attach_predictions,
    prepare_qa,
    score_rows,
    sha256_file,
    validate_protocol,
)

KIND = "cuhkx_action_generic_nested_v1"
SCOPES = ["HAU:single", "HAU:multi", "HAU:combination"]
COMPONENTS = ["generic_visual_presence", "option_frequency"]
SOURCES = ["scripts/run_action_generic_nested.py", "src/cuhkx/action_generic_nested.py",
           "src/cuhkx/actions.py", "src/cuhkx/clip_constraints.py", "src/cuhkx/joint_actions.py",
           "src/cuhkx/metadata.py", "src/cuhkx/option_frequency.py", "src/cuhkx/sequence_skeleton_nested.py",
           "src/cuhkx/validation.py", "src/cuhkx/submission.py"]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def utc():
    return datetime.now(UTC).isoformat()


def manifest(paths):
    return [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in paths]


def verify_manifest(entries):
    for entry in entries:
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"frozen input changed: {entry['path']}")


def validate_config(config, features, *, synthetic=False):
    fixed = {"schema_version": 1, "frozen_before_experiments": True, "seed": 20260905,
             "workers": 3, "inner_folds": 3,
             "presence": {"algorithm": "per_action_logistic_regression", "C": 1.0, "class_weight": "balanced",
                          "solver": "liblinear", "max_iter": 1000, "tol": 0.0001,
                          "standardize": "training_subjects_only"},
             "frequency": {"n_estimators": 200, "min_samples_leaf": 5, "max_depth": 12,
                           "max_features": 0.3, "class_weight": "balanced"}}
    if any(config.get(key) != value for key, value in fixed.items()):
        raise ValueError("configuration differs from the predeclared action mechanism")
    expected = [("visual_only", 1.0, 0.0), ("visual_option005", 1.0, 0.05), ("option_only", 0.0, 1.0)]
    actual = [(row["name"], row["visual_weight"], row["option_weight"]) for row in config["decoders"]]
    if actual != expected:
        raise ValueError("exactly three predeclared action decoders are required")
    if not synthetic and (config.get("feature_dimension") != 2048
                          or sha256_file(features) != config["feature_archive_sha256"]):
        raise ValueError("official generic visual archive differs from the frozen configuration")


def saved_bundle(path, bundle):
    joblib.dump(bundle, path, compress=3)
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n")
    return digest


def load_bundle(path):
    expected = path.with_suffix(path.suffix + ".sha256").read_text().strip()
    if sha256_file(path) != expected:
        raise ValueError("checkpoint digest differs from its frozen sidecar")
    bundle = joblib.load(path)
    if bundle["kind"] != KIND:
        raise ValueError("unexpected checkpoint kind")
    verify_manifest(bundle["source_manifest"])
    return bundle


def fit_scope(qa, prepared, visual, train, valid, config, seed, folder, sources):
    folder.mkdir(parents=True, exist_ok=False)
    bundle, scope = model.fit_bundle(qa, prepared, visual, train, valid, config, seed)
    bundle.update(kind=KIND, source_manifest=sources)
    digest = saved_bundle(folder / "retained_models.joblib", bundle)
    scope.update(checkpoint_sha256=digest, fitted_components=COMPONENTS)
    write_json(folder / "training_scope.json", scope)
    return bundle, scope


def splits(frame, subjects):
    rows = frame.loc[frame.subject.isin(subjects)]
    groups = rows.subject.to_numpy()
    for index, (train, valid) in enumerate(GroupKFold(n_splits=3).split(rows, groups=groups)):
        yield index, sorted(set(groups[train])), sorted(set(groups[valid]))


def accuracy(qa, predictions):
    scored, _ = attach_predictions(qa, predictions, require_order=True)
    overall = score_rows(scored)
    overall["worst_subject_accuracy"] = min(score_rows(group)["accuracy"] for _, group in scored.groupby("subject"))
    return overall


def select_inner(qa, prepared, visual, train, config, seed, folder, sources):
    predictions = {decoder["name"]: [] for decoder in config["decoders"]}
    records = []
    for index, inner_train, inner_valid in splits(prepared.frame, train):
        inner_folder = folder / f"inner_{index}"
        bundle, scope = fit_scope(qa, prepared, visual, inner_train, inner_valid, config,
                                  seed + index, inner_folder, sources)
        values = model.evidence(bundle, prepared, visual, scope["prediction_qa_ids"])
        for decoder in config["decoders"]:
            output = model.predict(prepared, values, decoder)
            output.to_csv(inner_folder / f"predictions_{decoder['name']}.csv", index=False)
            predictions[decoder["name"]].append(output)
        records.append({"train_subjects": inner_train, "validation_subjects": inner_valid,
                        "training_scope": str((inner_folder / "training_scope.json").resolve())})
    ids = prepared.frame.loc[prepared.frame.subject.isin(train)].qa_id.tolist()
    target = prepare_qa(qa, SCOPES).set_index("qa_id").loc[ids].reset_index()
    rankings = []
    for index, decoder in enumerate(config["decoders"]):
        output = pd.concat(predictions[decoder["name"]]).set_index("qa_id").loc[ids].reset_index()
        rankings.append({"decoder": decoder, "order": index, **accuracy(target, output)})
    ranked = sorted(rankings, key=lambda row: (-row["accuracy"], -row["worst_subject_accuracy"], row["order"]))
    write_json(folder / "inner_selection.json", {"rankings": rankings, "selected_decoder": ranked[0]["decoder"],
                                                "selection_qa_ids": ids, "inner_folds": records})
    return ranked[0]["decoder"], records


def fit(args):
    started = time.monotonic()
    config = json.loads(args.config.read_text())
    validate_config(config, args.features, synthetic=args.synthetic_preflight)
    qa, test = read_csv_str(args.train_csv), read_csv_str(args.test_csv)
    prepared, test_prepared = model.prepare(qa, training=True), model.prepare(test, training=False)
    target = prepare_qa(qa, SCOPES)
    visual = model.read_visual(args.features)
    if not args.synthetic_preflight and visual["X"].shape[1] != config["feature_dimension"]:
        raise ValueError("generic visual descriptor has the wrong dimension")
    subjects = sorted(set(prepared.frame.subject))
    if len(subjects) != 18:
        raise ValueError("this frozen protocol requires exactly eighteen outer subjects")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    sources = manifest([ROOT / path for path in SOURCES])
    inputs = manifest([args.config, args.train_csv, args.test_csv, args.features])
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    frozen = {"kind": KIND, "config": config, "source_manifest": sources, "input_manifest": inputs,
              "started_at_utc": utc(), "git_commit": commit, "synthetic_preflight": args.synthetic_preflight}
    write_json(out / "frozen_config.json", frozen)
    for source in sources:
        path = Path(source["path"])
        dest = out / "source_snapshot" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        if sha256_file(dest) != source["sha256"]:
            raise ValueError("source changed while freezing the experiment")
    folds, outputs = [], []
    for index, valid_subject in enumerate(subjects):
        train, valid = sorted(set(subjects) - {valid_subject}), [valid_subject]
        folder = out / f"outer_{index:02d}_{valid_subject}"
        decoder, inner = select_inner(qa, prepared, visual, train, config,
                                      config["seed"] + index * 100, folder, sources)
        bundle, scope = fit_scope(qa, prepared, visual, train, valid, config,
                                  config["seed"] + index * 100 + 50, folder / "outer_fit", sources)
        output = model.predict(prepared, model.evidence(bundle, prepared, visual, scope["prediction_qa_ids"]), decoder)
        output.to_csv(folder / "oof_predictions.csv", index=False)
        outputs.append(output)
        provenance = manifest([folder / "outer_fit" / "training_scope.json"])[0]
        folds.append({"fold_id": f"outer_{index:02d}", "train_subjects": train, "validation_subjects": valid,
                      "prediction_qa_ids": scope["prediction_qa_ids"], "inner_folds": inner,
                      "upstream_training": [{"name": name, "train_subjects": train, "provenance": provenance} for name in COMPONENTS]})
        scored = accuracy(target.loc[target.subject.isin(valid)].reset_index(drop=True), output)
        print(json.dumps({"outer": index, "subject": valid_subject, "selected": decoder["name"], **scored}), flush=True)
    oof = pd.concat(outputs).set_index("qa_id").loc[target.qa_id].reset_index()
    oof.to_csv(out / "oof_predictions.csv", index=False)
    decoder, final_inner = select_inner(qa, prepared, visual, subjects, config, config["seed"] + 9000,
                                        out / "final_selection", sources)
    bundle, _ = fit_scope(qa, prepared, visual, subjects, [], config, config["seed"] + 9050,
                          out / "final_fit", sources)
    bundle.update(selected_decoder=decoder, input_manifest=inputs, final_inner_folds=final_inner)
    checkpoint = out / "model.joblib"
    digest = saved_bundle(checkpoint, bundle)
    test_output = model.predict(test_prepared, model.evidence(bundle, test_prepared, visual, test_prepared.frame.qa_id), decoder)
    test_output.to_csv(out / "test_predictions.csv", index=False)
    protocol = {"schema_version": 1, "validation_kind": "nested_subject", "fitted_components": COMPONENTS,
                "folds": folds, "selection_history": config["selection_history"],
                "limitations": ["All subjects have historical exposure; this is not a fresh blind holdout.",
                                "Generic pretraining is fixed; only downstream fitting and decoder selection are nested.",
                                "Official repeated-recording acquisition layout is required; unseen layouts are not yet tested."]}
    write_json(out / "protocol.json", protocol)
    validation = validate_protocol(protocol, target, set(subjects), base_dir=out)
    if validation["status"] != "declared_provenance_verified":
        raise ValueError("nested subject protocol did not verify")
    verify_manifest(sources)
    verify_manifest(inputs)
    report = {"kind": KIND, "completed_at_utc": utc(), "duration_seconds": time.monotonic() - started,
              "overall": accuracy(target, oof), "validation": validation, "test_rows": len(test_output),
              "selected_decoder": decoder, "checkpoint_sha256": digest,
              "models_retained": 76, "synthetic_preflight": args.synthetic_preflight}
    write_json(out / "report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def infer(args):
    bundle = load_bundle(args.checkpoint)
    # Only the current unlabeled questions and descriptors are loaded. Historical
    # train paths in provenance are not opened or hashed by this inference path.
    qa = read_csv_str(args.test_csv)
    prepared = model.prepare(qa, training=False)
    visual = model.read_visual(args.features)
    output = model.predict(prepared, model.evidence(bundle, prepared, visual, prepared.frame.qa_id), bundle["selected_decoder"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output.to_csv(args.output_dir / "test_predictions.csv", index=False)
    write_json(args.output_dir / "report.json", {"rows": len(output), "selected_decoder": bundle["selected_decoder"],
                                               "checkpoint_sha256": sha256_file(args.checkpoint),
                                               "inputs": manifest([args.test_csv, args.features])})
    print(f"checkpoint replay action_rows={len(output)}", flush=True)


def compare_predictions(actual, expected):
    wanted = pd.read_csv(expected, keep_default_na=False)
    pd.testing.assert_frame_equal(actual.fillna("").reset_index(drop=True), wanted,
                                  check_dtype=False, check_exact=False, rtol=0, atol=1e-12)


def audit_selections(run, qa, prepared, config):
    checks = []
    target = prepare_qa(qa, SCOPES)
    for path in sorted(run.rglob("inner_selection.json")):
        selection = json.loads(path.read_text())
        expected_subjects = set(prepared.frame.subject)
        if path.parent.name != "final_selection":
            outer_scope = json.loads((path.parent / "outer_fit" / "training_scope.json").read_text())
            expected_subjects = set(outer_scope["train_subjects"])
        ids = prepared.frame.loc[prepared.frame.subject.isin(expected_subjects)].qa_id.tolist()
        if ids != selection["selection_qa_ids"]:
            raise ValueError("inner decoder selection used the wrong question/person scope")
        seen = set()
        for split in selection["inner_folds"]:
            fitted = json.loads(Path(split["training_scope"]).read_text())
            train, valid = set(split["train_subjects"]), set(split["validation_subjects"])
            if (train | valid != expected_subjects or train & valid or seen & valid
                    or fitted["train_subjects"] != split["train_subjects"]
                    or fitted["validation_subjects"] != split["validation_subjects"]):
                raise ValueError("inner fitted models differ from their outer allowed training scope")
            seen.update(valid)
        if seen != expected_subjects:
            raise ValueError("inner validation does not cover its allowed training people")
        rankings = []
        local_target = target.set_index("qa_id").loc[ids].reset_index()
        for index, decoder in enumerate(config["decoders"]):
            outputs = [pd.read_csv(Path(split["training_scope"]).parent / f"predictions_{decoder['name']}.csv", keep_default_na=False)
                       for split in selection["inner_folds"]]
            output = pd.concat(outputs).set_index("qa_id").loc[ids].reset_index()
            rankings.append({"decoder": decoder, "order": index, **accuracy(local_target, output)})
        best = min(rankings, key=lambda row: (-row["accuracy"], -row["worst_subject_accuracy"], row["order"]))
        if rankings != selection["rankings"] or best["decoder"] != selection["selected_decoder"]:
            raise ValueError("actual inner prediction scores do not reconstruct the saved selection")
        checks.append({"selection": str(path.resolve()), "selected_decoder": best["decoder"]})
    if len(checks) != 19:
        raise ValueError("eighteen outer and one final inner selections must be audited")
    return checks


def audit(args):
    frozen = json.loads((args.run_dir / "frozen_config.json").read_text())
    verify_manifest(frozen["source_manifest"])
    verify_manifest(frozen["input_manifest"])
    qa = read_csv_str(args.train_csv)
    prepared = model.prepare(qa, training=True)
    visual = model.read_visual(args.features)
    checks, scoring_rows = [], 0
    for path in sorted(args.run_dir.rglob("training_scope.json")):
        scope = json.loads(path.read_text())
        bundle = load_bundle(path.parent / "retained_models.joblib")
        if sha256_file(path.parent / "retained_models.joblib") != scope["checkpoint_sha256"]:
            raise ValueError("retained model differs from the fitted scope")
        arrays, recomputed = model.fitting_arrays(qa, prepared, visual, scope["train_subjects"], scope["validation_subjects"])
        if any(scope[key] != value for key, value in recomputed.items()):
            raise ValueError("actual fitted subjects, rows, matrices or labels differ")
        if (bundle["training_subjects"] != scope["train_subjects"]
                or bundle["known_action_classes"] != scope["known_action_classes"]
                or bundle["presence"].get_params() != scope["model_parameters"]["presence"]
                or bundle["frequency"].get_params() != scope["model_parameters"]["frequency"]
                or bundle["presence"].n_features_in_ != arrays[0].shape[1]
                or bundle["presence"].n_outputs_ != 40
                or bundle["frequency"].n_features_in_ != arrays[2].shape[1]):
            raise ValueError("actual retained model schema or training parameters differ")
        scaler = model.StandardScaler().fit(arrays[0])
        for key in ("mean_", "var_", "scale_"):
            np.testing.assert_array_equal(getattr(bundle["presence"].scaler, key), getattr(scaler, key))
        for action, estimator in enumerate(bundle["presence"].models):
            rows = np.flatnonzero(arrays[1][:, action] != -1)
            np.testing.assert_array_equal(bundle["presence"].fitting_rows[action], rows)
            known = action in scope["known_action_classes"]
            if (estimator is not None) != known:
                raise ValueError("partial-label classifier exists for an unfittable action")
            if known and (estimator.n_features_in_ != arrays[0].shape[1]
                          or list(estimator.classes_) != [0, 1]
                          or estimator.random_state != scope["seed"] + action):
                raise ValueError("actual per-action fitting schema differs")
        paths = sorted(path.parent.glob("predictions_*.csv"))
        selection = None
        if path.parent.name == "outer_fit":
            selection = json.loads((path.parent.parent / "inner_selection.json").read_text())["selected_decoder"]
            paths = [path.parent.parent / "oof_predictions.csv"]
        if paths:
            values = model.evidence(bundle, prepared, visual, scope["prediction_qa_ids"])
            decoders = [selection] if selection else frozen["config"]["decoders"]
            for decoder in decoders:
                output = model.predict(prepared, values, decoder)
                destination = paths[0] if selection else path.parent / f"predictions_{decoder['name']}.csv"
                compare_predictions(output, destination)
                scoring_rows += len(output)
        checks.append({"scope": str(path.resolve()), "checkpoint_sha256": scope["checkpoint_sha256"],
                       "visual_matrix_and_targets_sha256": recomputed["visual_matrix_and_targets_sha256"],
                       "option_matrix_and_targets_sha256": recomputed["option_matrix_and_targets_sha256"]})
        print(f"audited_scopes={len(checks)} replayed_scoring_rows={scoring_rows}", flush=True)
    expected_rows = len(prepared.frame) * 55
    if len(checks) != 76 or scoring_rows != expected_rows:
        raise ValueError("all 76 models and exact inner/outer prediction rows must be replayed")
    selections = audit_selections(args.run_dir, qa, prepared, frozen["config"])
    final = load_bundle(args.run_dir / "model.joblib")
    retained = load_bundle(args.run_dir / "final_fit" / "retained_models.joblib")
    selected = json.loads((args.run_dir / "final_selection" / "inner_selection.json").read_text())["selected_decoder"]
    if final["selected_decoder"] != selected or final["training_subjects"] != sorted(set(prepared.frame.subject)):
        raise ValueError("inference checkpoint changed final selection or training scope")
    for left, right in zip(final["presence"].models, retained["presence"].models, strict=True):
        if (left is None) != (right is None):
            raise ValueError("final action models differ from retained fitting models")
        if left is not None:
            np.testing.assert_array_equal(left.coef_, right.coef_)
            np.testing.assert_array_equal(left.intercept_, right.intercept_)
    for left, right in zip(final["frequency"].estimators_, retained["frequency"].estimators_, strict=True):
        for key in ("nodes", "values"):
            np.testing.assert_array_equal(left.tree_.__getstate__()[key], right.tree_.__getstate__()[key])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "report.json", {"status": "76_actual_fitting_scopes_and_models_predictions_verified",
                                               "scoring_rows_replayed": scoring_rows, "checks": checks,
                                               "inner_selections_recomputed": selections,
                                               "final_models_match_retained_fit": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit_parser = commands.add_parser("fit")
    fit_parser.add_argument("--train-csv", type=Path, required=True)
    fit_parser.add_argument("--test-csv", type=Path, required=True)
    fit_parser.add_argument("--config", type=Path, required=True)
    fit_parser.add_argument("--synthetic-preflight", action="store_true")
    infer_parser = commands.add_parser("infer")
    infer_parser.add_argument("--checkpoint", type=Path, required=True)
    infer_parser.add_argument("--test-csv", type=Path, required=True)
    audit_parser = commands.add_parser("audit")
    audit_parser.add_argument("--run-dir", type=Path, required=True)
    audit_parser.add_argument("--train-csv", type=Path, required=True)
    for item in (fit_parser, infer_parser, audit_parser):
        item.add_argument("--features", type=Path, required=True)
        item.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=3):
        {"fit": fit, "infer": infer, "audit": audit}[args.command](args)


if __name__ == "__main__":
    main()
