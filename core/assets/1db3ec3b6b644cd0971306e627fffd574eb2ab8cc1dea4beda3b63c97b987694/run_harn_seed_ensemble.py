#!/usr/bin/env python3
"""Replay a fixed equal two-seed HARn visual ensemble with nested decoding."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from cuhkx.harn_nested import (
    OFFICIAL_SUBJECTS,
    QA_COLUMNS,
    decode_questions,
    evaluate_predictions,
    harn_rows,
    predict_evidence,
    select_weight,
    sha256_file,
    write_json,
)
from cuhkx.harn_seed_ensemble import (
    NAMES,
    SEEDS,
    check_final_models,
    check_model,
    check_scope,
    equal_evidence,
)
from cuhkx.metadata import feature_unit, subject_id
from cuhkx.validation import prepare_qa, validate_protocol
from cuhkx.video_midfeatures import MidFeatureBank
from scripts.run_harn_nested_fusion import manifest_entry, read_qa, save_evidence
from scripts.run_harn_tail_seed_repeat import SOURCES as PARENT_SOURCES

KIND = "cuhkx_harn_equal_seed_ensemble_v1"
COMPONENTS = [f"{name}_seed_adapted_video_skeleton_and_dictionary" for name in NAMES]
SOURCES = list(dict.fromkeys([*PARENT_SOURCES, "src/cuhkx/harn_seed_ensemble.py",
                             "scripts/run_harn_seed_ensemble.py", "src/cuhkx/validation.py",
                             "scripts/check_harn_seed_ensemble_inference.py"]))


def validate_config(config):
    if (config.get("schema_version") != 1 or not config.get("frozen_before_experiment")
            or set(config["subjects"]) != OFFICIAL_SUBJECTS or config["workers"] != 3
            or config["visual_weights"] != [.5, .5] or config["visual_seeds"] != SEEDS
            or config["skeleton_weights"] != [0., .5, .75, 1.]
            or config["weight_tie_order"] != [.5, .75, 0., 1.]
            or config["inner_splits"] != 3 or set(config["parents"]) != set(NAMES)):
        raise ValueError("configuration differs from the fixed equal two-seed mechanism")


def load_evidence(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def compare_frame(actual, expected):
    if isinstance(expected, Path):
        expected = pd.read_csv(expected, keep_default_na=False)
    if set(actual.columns) != set(expected.columns):
        raise ValueError("decoded output columns differ")
    keys = [key for key in ("qa_id", "skeleton_weight", "inner_fold") if key in actual]
    pd.testing.assert_frame_equal(
        actual.fillna("").sort_values(keys).reset_index(drop=True),
        expected[actual.columns].fillna("").sort_values(keys).reset_index(drop=True),
        check_dtype=False, check_exact=False, rtol=0, atol=1e-6,
    )


def split_signature(fold):
    return {key: fold[key] for key in ("fold_id", "train_subjects", "validation_subjects", "prediction_qa_ids")}


class ParentPair:
    """Verify both real retained model collections before using their evidence."""

    def __init__(self, config, qa):
        self.config, self.qa = config, qa
        self.paths = {name: (ROOT / config["parents"][name]["run"]).resolve() for name in NAMES}
        self.inputs, self.audits, self.parent_configs = {}, {}, {}
        self.scopes, self.evidence, self.dictionaries, self.model_checks = {}, {}, {}, []
        protocols, selections = {}, {}
        for name, path in self.paths.items():
            declaration = config["parents"][name]
            for file in ("config.json", "source_manifest.json"):
                if self.register(path / file)["sha256"] != declaration[file + "_sha256"]:
                    raise ValueError("parent fitting configuration or source manifest changed")
            parent_config = json.loads((path / "config.json").read_text())
            self.parent_configs[name] = parent_config
            for entry in json.loads((path / "source_manifest.json").read_text()):
                if self.register(Path(entry["path"]))["sha256"] != entry["sha256"]:
                    raise ValueError("parent fitted source changed")
            generated = json.loads((path / "generation_frozen.json").read_text())
            self.register(path / "generation_frozen.json")
            for entry in generated["outputs"].values():
                if self.register(Path(entry["path"]))["sha256"] != entry["sha256"]:
                    raise ValueError("original generated parent output changed")
            for file in ("input_manifest.json", "protocol.json", "report.json"):
                self.register(path / file)
            original_inputs = json.loads((path / "input_manifest.json").read_text())
            for key in ("train_csv", "test_csv"):
                if self.register(ROOT / config[key])["sha256"] != original_inputs[key]["sha256"]:
                    raise ValueError("official QA differs from actual parent fitting input")
            audit_path = (ROOT / declaration["actual_scope_audit"]).resolve()
            self.register(audit_path)
            audit = json.loads(audit_path.read_text())
            if (audit["status"] != "actual_scope_models_features_probabilities_replayed"
                    or len(audit["checks"]) != 76 or audit["held_out_QA_rows_replayed"] != len(qa) * 19
                    or len(audit["inner_selections_replayed"]) != 19):
                raise ValueError("each adapted parent needs its completed76-model actual replay audit")
            self.audits[name] = {check["fit_id"]: check for check in audit["checks"]}
            if len(self.audits[name]) != 76:
                raise ValueError("duplicate scope in parent actual model audit")
            protocols[name] = json.loads((path / "protocol.json").read_text())
            selections[name] = {}
            for selection in sorted((path / "fits").glob("*_selection.json")):
                self.register(selection)
                selections[name][selection.name.removesuffix("_selection.json")] = json.loads(selection.read_text())
            if len(selections[name]) != 19 or len(list((path / "fits").glob("*.saved_evidence.npz"))) != 75:
                raise ValueError("both parents must retain19 selections and75 saved evidence archives")
            if len(list((path / "fits").glob("*.models.joblib"))) != 76:
                raise ValueError("both adapted parents must retain all76 actual models")
        for key in ("subjects", "seed", "skeleton", "visual"):
            if self.parent_configs["original"][key] != self.parent_configs["repeated"][key]:
                raise ValueError("seed repeat changed the original fitting recipe")
        if self.parent_configs["repeated"]["repeat_protocol"]["repeated_visual_seed"] != SEEDS["repeated"]:
            raise ValueError("repeat fitting configuration has an unexpected visual seed")
        self.protocol = protocols["original"]
        self.selections = selections
        if len(self.protocol["folds"]) != 18:
            raise ValueError("expected18 outer subjects")
        for left, right in zip(self.protocol["folds"], protocols["repeated"]["folds"], strict=True):
            if split_signature(left) != split_signature(right):
                raise ValueError("outer subject/QA assignments differ between seeds")
            for name, fold in (("original", left), ("repeated", right)):
                selected_folds = selections[name][fold["fold_id"]]["inner_folds"]
                if [split_signature(x) for x in fold["inner_folds"]] != [split_signature(x) for x in selected_folds]:
                    raise ValueError("actual inner selections differ from the parent outer protocol")
        prepared = prepare_qa(qa)
        for protocol in protocols.values():
            if validate_protocol(protocol, prepared, OFFICIAL_SUBJECTS, base_dir=ROOT)["status"] != "declared_provenance_verified":
                raise ValueError("parent subject protocol is not fully declared")
        planned = {}
        for outer in self.protocol["folds"]:
            planned[outer["fold_id"] + "_final"] = (set(outer["train_subjects"]), set(outer["validation_subjects"]))
        for scope in selections["original"]:
            left, right = (selections[name][scope]["inner_folds"] for name in NAMES)
            if len(left) != 3 or len(right) != 3:
                raise ValueError("each scope needs three exact inner folds")
            for a, b in zip(left, right, strict=True):
                if split_signature(a) != split_signature(b):
                    raise ValueError("inner subject/QA assignments differ between seeds")
                train = set(a["train_subjects"])
                planned[a["fold_id"]] = (train, OFFICIAL_SUBJECTS - train)
        planned["final_all_subjects_fit"] = (OFFICIAL_SUBJECTS, set())
        if len(planned) != 76:
            raise ValueError("expected76 distinct paired fitted scopes")
        for index, (fit_id, (train, valid)) in enumerate(sorted(planned.items())):
            self.load_scope(fit_id, train, valid)
            if (index + 1) % 8 == 0 or index == 75:
                print(f"paired_retained_scopes_checked={index+1}/76", flush=True)

    def register(self, path):
        path = Path(path).resolve()
        key = str(path)
        if key not in self.inputs:
            self.inputs[key] = manifest_entry(path)
        return self.inputs[key]

    def load_scope(self, fit_id, train, valid):
        fitting = self.qa.loc[self.qa.path.map(subject_id).isin(train)].copy()
        scopes, probabilities, summaries = {}, {}, {}
        for name, parent in self.paths.items():
            path = parent / "fits" / f"{fit_id}.json"
            self.register(path)
            scope = json.loads(path.read_text())
            check_scope(scope, fitting, train, valid)
            audit = self.audits[name][fit_id]
            if (audit["subjects"] != scope["train_subjects"]
                    or audit["actual_matrix_and_labels_sha256"] != scope["adapted_visual"]["actual_matrix_and_labels_sha256"]):
                raise ValueError("parent actual model replay audited another training matrix/scope")
            checkpoint = parent / "fits" / f"{fit_id}.models.joblib"
            digest = self.register(checkpoint)["sha256"]
            sidecar = checkpoint.with_suffix(".joblib.sha256")
            self.register(sidecar)
            if digest != sidecar.read_text().strip() or digest != scope["adapted_visual"]["checkpoint_sha256"]:
                raise ValueError("actual retained fitted model SHA differs")
            bundle = joblib.load(checkpoint)
            check_model(bundle, scope, fitting, SEEDS[name], self.parent_configs[name])
            summaries[name] = {key: bundle[key] for key in ("action_mapping", "object_mapping", "prior")}
            self.model_checks.append({"parent": name, "fit_id": fit_id, "checkpoint": self.inputs[str(checkpoint)],
                                      "train_subjects": sorted(train), "visual_seed": bundle["visual_model"].seed,
                                      "actual_matrix_and_labels_sha256": audit["actual_matrix_and_labels_sha256"]})
            evidence_path = parent / "fits" / f"{fit_id}.saved_evidence.npz"
            if fit_id != "final_all_subjects_fit":
                self.register(evidence_path)
                evidence = load_evidence(evidence_path)
                for family in ("skeleton", "visual"):
                    np.testing.assert_array_equal(evidence[f"{family}_classes"], bundle[f"{family}_classes"])
                target_subjects = {subject_id(unit) for unit in evidence["units"]}
                if target_subjects & train or not target_subjects <= valid:
                    raise ValueError("held-out saved evidence includes an excluded-scope mismatch")
                target = self.qa.loc[self.qa.path.map(subject_id).isin(target_subjects)]
                np.testing.assert_array_equal(evidence["units"], sorted({feature_unit(p, training=True) for p in target.path}))
                probabilities[name] = evidence
            scopes[name] = scope
            del bundle
            gc.collect()
        if scopes["original"]["features"] != scopes["repeated"]["features"]:
            raise ValueError("the two adapted models fitted different feature units/labels")
        for key in ("rows", "labels", "actual_matrix_and_labels_sha256", "middle_spec"):
            if scopes["original"]["adapted_visual"][key] != scopes["repeated"]["adapted_visual"][key]:
                raise ValueError("the repeated seed changed its actual adapted input matrix")
        if summaries["original"] != summaries["repeated"]:
            raise ValueError("actual paired model dictionaries differ")
        if probabilities:
            equal_evidence(probabilities["original"], probabilities["repeated"])
        self.scopes[fit_id], self.evidence[fit_id] = scopes, probabilities
        self.dictionaries[fit_id] = summaries["original"]

    def upstream(self, fit_id):
        return [{"name": component, "train_subjects": self.scopes[fit_id][name]["train_subjects"],
                 "provenance": self.register(self.paths[name] / "fits" / f"{fit_id}.json")}
                for name, component in zip(NAMES, COMPONENTS, strict=True)]


def decode(qa, evidence, dictionary, weight, *, training):
    return decode_questions(qa[QA_COLUMNS], evidence, dictionary, weight, training_paths=training)


def choose(pair, qa, scope, output):
    config = pair.config
    folds = copy.deepcopy(pair.selections["original"][scope]["inner_folds"])
    parts = {name: {weight: [] for weight in config["skeleton_weights"]}
             for name in (*NAMES, "ensemble")}
    used = []
    for index, fold in enumerate(folds):
        fitting_subjects = set(qa.path.map(subject_id))
        if (set(fold["train_subjects"]) & set(fold["validation_subjects"])
                or set(fold["train_subjects"]) | set(fold["validation_subjects"]) != fitting_subjects):
            raise ValueError("inner fitting/validation scope differs from outer-training subjects")
        target = qa.loc[qa.qa_id.isin(fold["prediction_qa_ids"])].copy()
        if target.qa_id.tolist() != fold["prediction_qa_ids"]:
            raise ValueError("exact inner QA rows/order differ")
        fit_id = fold["fold_id"]
        evidence = pair.evidence[fit_id]
        values = {**evidence, "ensemble": equal_evidence(evidence["original"], evidence["repeated"])}
        dictionary = pair.dictionaries[fit_id]
        for name in parts:
            for weight in config["skeleton_weights"]:
                frame = decode(target, values[name], dictionary, weight, training=True)
                frame["inner_fold"] = index
                parts[name][weight].append(frame)
        fold["upstream_training"] = pair.upstream(fit_id)
        used.extend(target.qa_id)
    if len(used) != len(set(used)) or set(used) != set(qa.qa_id):
        raise ValueError("inner held-out QA must cover the fitting scope exactly once")
    reports = {}
    for name, collections in parts.items():
        candidates, frames = [], []
        for weight, collection in collections.items():
            frame = pd.concat(collection, ignore_index=True)
            frames.append(frame)
            candidates.append({"skeleton_weight": weight, "metrics": evaluate_predictions(qa, frame)})
        combined = pd.concat(frames, ignore_index=True)
        reports[name] = candidates
        directory = "selections" if name == "ensemble" else "controls"
        destination = output / directory / f"{name}_{scope}_inner_predictions.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(destination, index=False)
        if name in NAMES:
            expected_path = pair.paths[name] / "fits" / f"{scope}_inner_predictions.csv"
            pair.register(expected_path)
            compare_frame(combined, expected_path)
            prior = pair.selections[name][scope]
            if (candidates != prior["candidate_reports"]
                    or select_weight(candidates, config["weight_tie_order"]) != prior["selected_skeleton_weight"]):
                raise ValueError("original endpoint inner metrics/selection failed control replay")
    selected = select_weight(reports["ensemble"], config["weight_tie_order"])
    record = {"selected_skeleton_weight": selected, "visual_weights": [.5, .5],
              "candidate_reports": reports["ensemble"], "endpoint_control_reports": {name: reports[name] for name in NAMES},
              "endpoint_controls_participate_in_selection": False, "inner_folds": folds}
    write_json(output / "selections" / f"{scope}.json", record)
    return selected, folds, record


def execute(config, output):
    validate_config(config)
    qa = harn_rows(read_qa(ROOT / config["train_csv"], labeled=True))
    test = harn_rows(read_qa(ROOT / config["test_csv"], labeled=False))
    pair = ParentPair(config, qa)
    oof_parts, folds, choices = [], [], []
    subjects = qa.path.map(subject_id)
    for original in pair.protocol["folds"]:
        scope = original["fold_id"]
        fitting = qa.loc[subjects.isin(original["train_subjects"])].copy()
        target = qa.loc[qa.qa_id.isin(original["prediction_qa_ids"])].copy()
        selected, inner, _ = choose(pair, fitting, scope, output)
        fit_id = scope + "_final"
        evidence = pair.evidence[fit_id]
        combined = equal_evidence(evidence["original"], evidence["repeated"])
        prediction = decode(target, combined, pair.dictionaries[fit_id], selected, training=True)
        prediction["outer_fold"] = scope
        oof_parts.append(prediction)
        fold = {**split_signature(original), "inner_folds": inner, "upstream_training": pair.upstream(fit_id)}
        folds.append(fold)
        choices.append({"outer_subject": original["validation_subjects"][0], "selected_skeleton_weight": selected})
        save_evidence(output / "selections" / f"{scope}.ensemble_evidence.npz", combined)
        print(f"equal_seed_outer_complete={len(oof_parts)}/18", flush=True)
    oof = pd.concat(oof_parts, ignore_index=True).set_index("qa_id", drop=False).loc[qa.qa_id].reset_index(drop=True)
    oof.to_csv(output / "oof_predictions.csv", index=False)
    final_weight, final_inner, final_selection = choose(pair, qa, "final_all_subjects", output)
    final_models = {name: joblib.load(pair.paths[name] / "models.joblib") for name in NAMES}
    check_final_models(final_models)
    for name, bundle in final_models.items():
        check_model(bundle, pair.scopes["final_all_subjects_fit"][name], qa, SEEDS[name], pair.parent_configs[name])
        retained = joblib.load(pair.paths[name] / "fits/final_all_subjects_fit.models.joblib")
        for key, value in retained["visual_model"].state.items():
            np.testing.assert_array_equal(bundle["visual_model"].state[key], value)
        del retained
    final_evidence = {name: load_evidence(pair.paths[name] / "test_probabilities.npz") for name in NAMES}
    combined = equal_evidence(final_evidence["original"], final_evidence["repeated"])
    np.testing.assert_array_equal(combined["units"], sorted({feature_unit(path, training=False) for path in test.path}))
    predictions = decode(test, combined, final_models["original"], final_weight, training=False)
    predictions.to_csv(output / "test_predictions.csv", index=False)
    save_evidence(output / "test_probabilities.npz", combined)
    sources = [manifest_entry(ROOT / source) for source in SOURCES]
    checkpoint = {"schema_version": 1, "kind": KIND, "config": config, "sources": sources,
                  "visual_weights": [.5, .5], "selected_skeleton_weight": final_weight,
                  "classifiers": {name: pair.register(parent / "models.joblib") for name, parent in pair.paths.items()}}
    write_json(output / "checkpoint.json", checkpoint)
    protocol = {"schema_version": 1, "validation_kind": "nested_subject", "fitted_components": COMPONENTS,
                "folds": folds, "final_inner_folds": final_inner, "selection_history": config["selection_history"],
                "visual_weights": [.5, .5], "endpoint_controls_participate_in_selection": False,
                "retention": "Both76 actual adapted model collections retained,152 in total; no generic SVC source.",
                "source_protocols": {name: pair.register(path / "protocol.json") for name, path in pair.paths.items()}}
    audit = validate_protocol(protocol, prepare_qa(qa), OFFICIAL_SUBJECTS, base_dir=output)
    if audit["status"] != "declared_provenance_verified":
        raise ValueError("ensemble shared-evaluator protocol is incomplete")
    write_json(output / "protocol.json", protocol)
    write_json(output / "retained_model_checks.json", pair.model_checks)
    write_json(output / "input_manifest.json", list(pair.inputs.values()))
    generated = {name: manifest_entry(output / name) for name in (
        "oof_predictions.csv", "test_predictions.csv", "test_probabilities.npz", "checkpoint.json", "protocol.json")}
    write_json(output / "generation_frozen.json", {"frozen_utc": datetime.now(UTC).isoformat(), "outputs": generated,
                                                   "historical_parent_read_yet": False})
    report = {"schema_version": 1, "status": "fixed_equal_seed_nested_decoding_complete", "metrics": evaluate_predictions(qa, oof),
              "outer_selections": choices, "final_skeleton_weight": final_weight, "final_inner_selection": final_selection,
              "test_rows": len(predictions), "new_neural_fits": 0, "retained_scope_models_loaded": len(pair.model_checks),
              "inner_selections": 19, "decoded_inner_rows": len(qa) * 18 * 4,
              "decoded_endpoint_control_rows": len(qa) * 18 * 4 * 2, "decoded_outer_rows": len(qa),
              "parent_actual_gpu_audits": {name: pair.register(ROOT / config["parents"][name]["actual_scope_audit"]) for name in NAMES},
              "neural_predictions_recomputed_in_this_decode_run": False, "protocol_audit": audit,
              "limitations": [config["selection_history"], "The two original full76-model probability replay audits are required and bound to these retained scopes; this stage reloads152 retained models and replays their saved probabilities/decoding.",
                              "No missing visual modality is imputed or seed selected; unequal unit/class/mask axes are rejected."]}
    write_json(output / "report.json", report)
    for entry in [*pair.inputs.values(), *sources]:
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("source or retained parent input changed during ensemble decoding")
    return report


def freeze(config_path, output):
    config = json.loads(config_path.read_text())
    validate_config(config)
    output.mkdir(parents=True, exist_ok=False)
    sources = {name: sha256_file(ROOT / name) for name in SOURCES}
    write_json(output / "frozen_config.json", {"config": config, "config_sha256": sha256_file(config_path),
                                               "source_hashes": sources, "created_utc": datetime.now(UTC).isoformat()})
    for name in sources:
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    return config


def verify_frozen(run):
    frozen = json.loads((run / "frozen_config.json").read_text())
    for source, digest in frozen["source_hashes"].items():
        if sha256_file(ROOT / source) != digest:
            raise ValueError("ensemble source differs from the predeclared implementation")
    return frozen


def fit(args):
    if args.frozen_dir:
        frozen = verify_frozen(args.frozen_dir)
        if sha256_file(args.config) != frozen["config_sha256"]:
            raise ValueError("configuration differs from the predeclared freeze")
    config = freeze(args.config, args.output_dir)
    started = time.monotonic()
    report = execute(config, args.output_dir)
    verify_frozen(args.output_dir)
    write_json(args.output_dir / "execution.json", {"exit_code": 0, "seconds": time.monotonic() - started,
                                                   "completed_utc": datetime.now(UTC).isoformat()})
    print(json.dumps({"correct": report["metrics"]["correct"], "rows": report["metrics"]["rows"],
                      "final_skeleton_weight": report["final_skeleton_weight"]}), flush=True)


def audit(args):
    frozen = verify_frozen(args.run_dir)
    for entry in json.loads((args.run_dir / "generation_frozen.json").read_text())["outputs"].values():
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("original ensemble output changed")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = execute(frozen["config"], args.output_dir)
    for directory in ("controls", "selections"):
        for path in sorted((args.output_dir / directory).glob("*.csv")):
            compare_frame(pd.read_csv(path, keep_default_na=False), args.run_dir / directory / path.name)
        for path in sorted((args.output_dir / directory).glob("*.json")):
            if json.loads(path.read_text()) != json.loads((args.run_dir / directory / path.name).read_text()):
                raise ValueError("replayed inner selection differs")
    for name in ("oof_predictions.csv", "test_predictions.csv"):
        compare_frame(pd.read_csv(args.output_dir / name, keep_default_na=False), args.run_dir / name)
    for name in ("checkpoint.json", "protocol.json", "retained_model_checks.json"):
        if json.loads((args.output_dir / name).read_text()) != json.loads((args.run_dir / name).read_text()):
            raise ValueError("replayed checkpoint/protocol/retained model scope differs")
    write_json(args.output_dir / "completion_verification.json", {
        "status": "both152_retained_models_scopes_endpoint_controls_and_equal_ensemble_choices_replayed",
        "inner_selections": 19, "decoded_inner_rows": report["decoded_inner_rows"],
        "decoded_endpoint_control_rows": report["decoded_endpoint_control_rows"], "decoded_outer_rows": report["decoded_outer_rows"],
        "test_rows": report["test_rows"], "new_neural_fits": 0, "parent_actual_gpu_audits_required": True})


def infer(args):
    checkpoint = json.loads(args.checkpoint.read_text())
    if checkpoint.get("schema_version") != 1 or checkpoint["kind"] != KIND or checkpoint["visual_weights"] != [.5, .5]:
        raise ValueError("unexpected fixed equal-seed checkpoint")
    validate_config(checkpoint["config"])
    if checkpoint["selected_skeleton_weight"] not in checkpoint["config"]["skeleton_weights"]:
        raise ValueError("final Skeleton weight was not predeclared")
    for entry in [*checkpoint["sources"], *checkpoint["classifiers"].values()]:
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("final model or frozen source changed")
    qa = harn_rows(read_qa(args.qa_csv, labeled=False))
    models = {name: joblib.load(checkpoint["classifiers"][name]["path"]) for name in NAMES}
    check_final_models(models)
    bank = MidFeatureBank.load(args.skeleton, args.visual)
    if any(not np.char.startswith(axis.astype(str), "Testing/").all()
           for axis in (bank.skeleton_units, bank.visual_units)):
        raise ValueError("inference requires Testing-only feature archives")
    for bundle in models.values():
        for forest in bundle["models"].values():
            forest.n_jobs = 3
    evidence = {name: predict_evidence(bundle, bank, qa[QA_COLUMNS], training_paths=False) for name, bundle in models.items()}
    combined = equal_evidence(evidence["original"], evidence["repeated"])
    output = decode(qa, combined, models["original"], checkpoint["selected_skeleton_weight"], training=False)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output.to_csv(args.output_dir / "test_predictions.csv", index=False)
    save_evidence(args.output_dir / "test_probabilities.npz", combined)
    write_json(args.output_dir / "report.json", {"status": "actual_two_adapted_final_classifiers_loaded", "rows": len(output),
                                                "new_neural_fits": 0, "visual_weights": [.5, .5],
                                                "prediction_sha256": sha256_file(args.output_dir / "test_predictions.csv")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    predeclare = sub.add_parser("freeze")
    fitting = sub.add_parser("fit")
    for command in (predeclare, fitting):
        command.add_argument("--config", type=Path, required=True)
    fitting.add_argument("--frozen-dir", type=Path)
    auditing = sub.add_parser("audit")
    auditing.add_argument("--run-dir", type=Path, required=True)
    prediction = sub.add_parser("infer")
    for name in ("checkpoint", "qa-csv", "skeleton", "visual"):
        prediction.add_argument(f"--{name}", type=Path, required=True)
    for command in (predeclare, fitting, auditing, prediction):
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=3):
        if args.command == "freeze":
            freeze(args.config, args.output_dir)
        else:
            {"fit": fit, "audit": audit, "infer": infer}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
