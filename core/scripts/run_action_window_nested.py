#!/usr/bin/env python3
"""Train and replay all76 VideoMAE HAU window heads with nested subject selection."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier
from threadpoolctl import threadpool_limits

from cuhkx import action_generic_nested as base
from cuhkx.action_videomae_bank import read_window_bank
from cuhkx.action_window_decoding import compile_questions, evidence_from_probabilities
from cuhkx.action_window_head import WindowPartialActionClassifier
from cuhkx.emotion_nested import OFFICIAL_SUBJECTS
from cuhkx.validation import (
    attach_predictions,
    prepare_qa,
    score_rows,
    sha256_file,
    validate_protocol,
)
from scripts.run_action_generic_nested import (
    accuracy,
    manifest,
    splits,
    verify_manifest,
    write_json,
)

KIND = "cuhkx_videomae_action_window_nested_v1"
SCOPES = ["HAU:single", "HAU:multi", "HAU:combination"]
COMPONENTS = ["window_normalization_and_mil", "training_label_prior", "option_frequency"]


def fit_scope(qa, prepared, poisoned, compiled, visual, train, valid, config, seed, folder, sources):
    folder.mkdir(parents=True, exist_ok=False)
    (matrix, targets, options, labels), scope = base.fitting_arrays(qa, prepared, visual, train, valid)
    head = WindowPartialActionClassifier(config["presence"], seed).fit(matrix, targets)
    assert head.fitting_content_sha256 == scope["visual_matrix_and_targets_sha256"]
    frequency = ExtraTreesClassifier(**config["frequency"], random_state=seed + 1000000, n_jobs=3).fit(options, labels)
    bundle = {"kind": KIND, "presence": head, "frequency": frequency,
              "training_subjects": sorted(train), "validation_subjects": sorted(valid),
              "known_action_classes": scope["known_action_classes"], "feature_names": visual["feature_names"],
              "source_manifest": sources, "seed": seed, "config": config}
    path = folder / "model.joblib"
    joblib.dump(bundle, path, compress=3)
    digest = sha256_file(path)
    (folder / "model.sha256").write_text(digest + "\n")
    reloaded = joblib.load(path)
    scope.update(checkpoint_sha256=digest, fitted_components=COMPONENTS)
    write_json(folder / "training_scope.json", scope)
    write_json(folder / "training_history.json", {"history": head.history, "fit_seconds": head.fit_seconds})
    if not valid:
        # Final fitting still checks the stored model on real fitting inputs, without accuracy scoring.
        np.testing.assert_array_equal(head.predict(matrix[:2]), reloaded["presence"].predict(matrix[:2]))
        write_json(folder / "replay_audit.json", {"final_training_input_replay_rows": 2, "max_difference": 0.})
        return reloaded, scope, {}
    target_frame = prepared.frame.loc[prepared.frame.subject.isin(valid)]
    units = target_frame.drop_duplicates("unit").unit.tolist()
    indices = [visual["index"][unit] for unit in units]
    values = visual["X"][indices]
    probabilities = reloaded["presence"].predict(values)
    np.testing.assert_array_equal(probabilities, head.predict(values))
    prior = reloaded["presence"].predict_prior(values)
    np.savez_compressed(folder / "validation_probabilities.npz", units=np.asarray(units),
                        probabilities=probabilities, prior=prior)
    # Every prediction is formed from the reloaded model before any held-out label is scored.
    predictions = {}
    checks = 0
    for branch, matrix_probability in (("visual", probabilities), ("prior", prior)):
        evidence = evidence_from_probabilities(reloaded, prepared, compiled, units, matrix_probability,
                                                scope["prediction_qa_ids"])
        for recipe in config["decoders"]:
            if recipe["branch"] != branch:
                continue
            prediction = base.predict(prepared, evidence[recipe["likelihood"]], recipe)
            pd.testing.assert_frame_equal(prediction, base.predict(poisoned, evidence[recipe["likelihood"]], recipe))
            prediction.to_csv(folder / (recipe["name"] + ".csv"), index=False)
            predictions[recipe["name"]] = prediction
            checks += 1
    assert set(predictions) == {recipe["name"] for recipe in config["decoders"]}
    assert sha256_file(path) == digest
    write_json(folder / "replay_audit.json", {"validation_units": units, "reload_max_difference": 0.,
                                              "target_poison_predictions": checks,
                                              "actual_fitting_content_sha256": head.fitting_content_sha256})
    return reloaded, scope, predictions


def select_inner(qa, prepared, poisoned, compiled, visual, train, config, seed, folder, sources, update):
    outputs = {recipe["name"]: [] for recipe in config["decoders"]}
    folds = []
    for index, inner_train, inner_valid in splits(prepared.frame, train):
        current = folder / f"inner_{index}"
        update(str(current))
        _, _, predictions = fit_scope(qa, prepared, poisoned, compiled, visual, inner_train, inner_valid,
                                          config, seed + index, current, sources)
        for name, prediction in predictions.items():
            outputs[name].append(prediction)
        folds.append({"train_subjects": inner_train, "validation_subjects": inner_valid,
                      "training_scope": str((current / "training_scope.json").resolve())})
    ids = prepared.frame.loc[prepared.frame.subject.isin(train)].qa_id.tolist()
    truth = prepare_qa(qa, SCOPES).set_index("qa_id").loc[ids].reset_index()
    rankings = []
    for index, recipe in enumerate(config["decoders"]):
        prediction = pd.concat(outputs[recipe["name"]]).set_index("qa_id").loc[ids].reset_index()
        rankings.append({"decoder": recipe, "order": index, **accuracy(truth, prediction)})
    ranked = sorted(rankings, key=lambda item: (-item["accuracy"], -item["worst_subject_accuracy"], item["order"]))
    selected = ranked[0]["decoder"]
    controls = {name: next(item["decoder"] for item in ranked
                           if item["decoder"]["name"] == "option_only" or item["decoder"]["branch"] in branches)
                for name, branches in config["control_policies"].items()}
    write_json(folder / "selection.json", {"rankings": rankings, "selected_decoder": selected,
                                            "control_selections": controls, "selection_qa_ids": ids, "inner_folds": folds})
    return selected, folds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    threadpool_limits(limits=2)
    for entry in config["inputs"] + config["implementation"]:
        assert sha256_file(ROOT / entry["path"]) == entry["sha256"]
    qa = pd.read_csv(ROOT / config["train_csv"], dtype=str, keep_default_na=False)
    prepared = base.prepare(qa, training=True)
    poisoned = base.prepare(qa.assign(answer="POISON", target="POISON", prediction="POISON"), training=True)
    np.testing.assert_array_equal(prepared.option_matrix, poisoned.option_matrix)
    compiled = compile_questions(qa, prepared)
    assert compiled == compile_questions(qa.assign(answer="POISON", target="POISON", prediction="POISON"), poisoned)
    visual = read_window_bank(ROOT / config["window_bank"])
    assert set(visual["units"]) == set(prepared.frame.unit)
    assert set(prepared.frame.subject) == OFFICIAL_SUBJECTS
    sources = manifest([ROOT / entry["path"] for entry in config["implementation"]])
    inputs = manifest([args.config] + [ROOT / entry["path"] for entry in config["inputs"]])
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    write_json(output / "frozen_config.json", {"kind": KIND, "config": config, "source_manifest": sources,
                                               "input_manifest": inputs, "git_commit": commit,
                                               "started_utc": datetime.now(UTC).isoformat()})
    for entry in sources:
        path = Path(entry["path"])
        destination = output / "source_snapshot" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        assert sha256_file(destination) == entry["sha256"]
    state = {"status": "running", "pid": os.getpid(), "completed_outer_folds": 0,
             "started_utc": datetime.now(UTC).isoformat()}

    def update(active):
        state.update(active_fit=active, updated_utc=datetime.now(UTC).isoformat())
        write_json(output / "state.json", state)
        print(json.dumps(state), flush=True)

    start = time.perf_counter()
    subjects = sorted(OFFICIAL_SUBJECTS)
    target = prepare_qa(qa, SCOPES)
    baseline = pd.read_csv(ROOT / config["option_control"], dtype={"qa_id": str, "prediction": str})
    folds, oof = [], []
    fixed = {recipe["name"]: [] for recipe in config["decoders"]}
    policies = {name: [] for name in config["control_policies"]}
    try:
        for index, subject in enumerate(subjects):
            train, valid = sorted(set(subjects) - {subject}), [subject]
            folder = output / ("outer_" + subject)
            recipe, inner = select_inner(qa, prepared, poisoned, compiled, visual, train, config,
                                          config["seed"] + index * 100, folder, sources, update)
            update(str(folder / "outer_fit"))
            _, scope, predictions = fit_scope(qa, prepared, poisoned, compiled, visual, train, valid, config,
                                               config["seed"] + index * 100 + 50, folder / "outer_fit", sources)
            chosen = predictions[recipe["name"]]
            chosen.to_csv(folder / "oof_predictions.csv", index=False)
            oof.append(chosen)
            control = baseline.set_index("qa_id").loc[chosen.qa_id, "prediction"]
            assert np.array_equal(predictions["option_only"].prediction.to_numpy(), control.to_numpy())
            for name, prediction in predictions.items():
                fixed[name].append(prediction)
            controls = json.loads((folder / "selection.json").read_text())["control_selections"]
            for name, control_recipe in controls.items():
                policies[name].append(predictions[control_recipe["name"]])
            provenance = manifest([folder / "outer_fit" / "training_scope.json"])[0]
            folds.append({"fold_id": f"outer_{index:02d}", "train_subjects": train, "validation_subjects": valid,
                          "prediction_qa_ids": scope["prediction_qa_ids"], "inner_folds": inner,
                          "upstream_training": [{"name": name, "train_subjects": train, "provenance": provenance}
                                                for name in COMPONENTS]})
            state["completed_outer_folds"] += 1
            print(json.dumps({"subject": subject, "selected": recipe["name"],
                              **accuracy(target.loc[target.subject.eq(subject)].reset_index(drop=True), chosen)}), flush=True)
        selected = pd.concat(oof).set_index("qa_id").loc[target.qa_id].reset_index()
        selected.to_csv(output / "oof_predictions.csv", index=False)
        for name, parts in fixed.items():
            pd.concat(parts).set_index("qa_id").loc[target.qa_id].reset_index().to_csv(output / ("oof_" + name + ".csv"), index=False)
        policy_predictions = {}
        for name, parts in policies.items():
            prediction = pd.concat(parts).set_index("qa_id").loc[target.qa_id].reset_index()
            prediction.to_csv(output / ("oof_policy_" + name + ".csv"), index=False)
            policy_predictions[name] = prediction
        recipe, final_inner = select_inner(qa, prepared, poisoned, compiled, visual, subjects, config,
                                           config["seed"] + 9000, output / "final_selection", sources, update)
        update(str(output / "final_fit"))
        bundle, _, _ = fit_scope(qa, prepared, poisoned, compiled, visual, subjects, [], config,
                                  config["seed"] + 9050, output / "final_fit", sources)
        bundle.update(selected_decoder=recipe, final_inner_folds=final_inner, input_manifest=inputs)
        joblib.dump(bundle, output / "model.joblib", compress=3)
        (output / "model.sha256").write_text(sha256_file(output / "model.joblib") + "\n")
        protocol = {"schema_version": 1, "validation_kind": "nested_subject", "fitted_components": COMPONENTS,
                    "folds": folds, "selection_history": config["selection_history"],
                    "limitations": ["All18people historically exposed;not a fresh blind holdout.",
                                    "Fixed generic SSv2 upstream;only normalization/head/prior/option fitting and decoding selection are nested.",
                                    "Repeated-recording metadata required;independent-recording accuracy not established."]}
        write_json(output / "protocol.json", protocol)
        validation = validate_protocol(protocol, target, set(subjects), base_dir=output)
        assert validation["status"] == "declared_provenance_verified"
        verify_manifest(sources)
        verify_manifest(inputs)
        scored, _ = attach_predictions(target, selected, require_order=True)
        scored, _ = attach_predictions(scored, baseline, column="baseline_prediction", require_order=False)
        without_visual, _ = attach_predictions(target, selected, require_order=True)
        without_visual, _ = attach_predictions(without_visual, policy_predictions["without_visual"],
                                               column="baseline_prediction", require_order=True)
        report = {"kind": KIND, "status": "completed_nested_training_no_test_candidate", "overall": score_rows(scored),
                  "by_subject": {name: score_rows(part) for name, part in scored.groupby("subject")},
                  "by_category": {name: score_rows(part) for name, part in scored.groupby("category")},
                  "versus_nested_without_visual": score_rows(without_visual),
                  "control_policy_metrics": {name: accuracy(target, prediction) for name, prediction in policy_predictions.items()},
                  "selected_final_decoder": recipe, "validation": validation, "models_retained": 76,
                  "actual_reload_checks": 76, "target_poison_prediction_checks": 75 * len(config["decoders"]),
                  "duration_seconds": time.perf_counter() - start, "test_qa_read": False, "uploads": 0}
        write_json(output / "report.json", report)
        state.update(status="complete", active_fit=None)
        write_json(output / "state.json", state)
        print(json.dumps(report), flush=True)
    except Exception as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        write_json(output / "state.json", state)
        raise


if __name__ == "__main__":
    main()
