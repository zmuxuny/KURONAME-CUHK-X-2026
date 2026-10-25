#!/usr/bin/env python3
"""Refit independent-recording option models, reusing exactly matched visual heads."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
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
from cuhkx.action_explicit_layout import prepare
from cuhkx.action_videomae_bank import read_window_bank
from cuhkx.action_window_decoding import compile_questions, evidence_from_probabilities
from cuhkx.emotion_nested import OFFICIAL_SUBJECTS
from cuhkx.validation import prepare_qa, sha256_file, validate_protocol
from scripts import run_action_window_nested as nested
from scripts.assess_action_window_nested import compare
from scripts.run_action_generic_nested import manifest, verify_manifest, write_json

KIND = "cuhkx_action_independent_window_nested_v1"
COMPONENTS = ["matched_reused_joint_window_head", "training_label_prior", "independent_option_frequency"]


def fit_scope(qa, prepared, poisoned, compiled, visual, train, valid, config, seed, folder, sources):
    folder.mkdir(parents=True, exist_ok=False)
    parent = ROOT / config["parent_run"] / folder.relative_to(Path(config["runtime_output"]))
    parent_scope = json.loads((parent / "training_scope.json").read_text())
    digest = sha256_file(parent / "model.joblib")
    assert digest == parent_scope["checkpoint_sha256"]
    bundle = joblib.load(parent / "model.joblib")
    assert bundle["seed"] == seed
    assert bundle["training_subjects"] == sorted(train)
    assert bundle["validation_subjects"] == sorted(valid)
    (matrix, targets, options, labels), scope = base.fitting_arrays(qa, prepared, visual, train, valid)
    # Group sharing changes option features; visual training and labels must be identical.
    for name, value in scope.items():
        if name != "option_matrix_and_targets_sha256":
            assert parent_scope[name] == value, name
    head = bundle["presence"]
    assert head.fitting_content_sha256 == scope["visual_matrix_and_targets_sha256"]
    np.testing.assert_array_equal(head.training_targets, targets)
    if valid:
        poisoned_qa = qa.copy()
        poisoned_qa.loc[poisoned_qa.qa_id.isin(scope["prediction_qa_ids"]), "answer"] = "FORBIDDEN_LABEL"
        _, poison_scope = base.fitting_arrays(poisoned_qa, prepared, visual, train, valid)
        assert poison_scope == scope
    frequency = ExtraTreesClassifier(**config["frequency"], random_state=seed + 1000000, n_jobs=3).fit(options, labels)
    bundle.update(kind=KIND, frequency=frequency, source_manifest=sources, config=config,
                  parent_checkpoint={"path": str(parent / "model.joblib"), "sha256": digest},
                  layout="independent_recordings")
    checkpoint = folder / "model.joblib"
    joblib.dump(bundle, checkpoint, compress=3)
    saved_digest = sha256_file(checkpoint)
    (folder / "model.sha256").write_text(saved_digest + "\n")
    reloaded = joblib.load(checkpoint)
    scope.update(checkpoint_sha256=saved_digest, fitted_components=COMPONENTS,
                 reused_visual_checkpoint=bundle["parent_checkpoint"])
    write_json(folder / "training_scope.json", scope)
    checks, predictions = 0, {}
    np.testing.assert_allclose(frequency.predict_proba(options[:8]),
                               reloaded["frequency"].predict_proba(options[:8]), rtol=0, atol=1e-12)
    if valid:
        with np.load(parent / "validation_probabilities.npz", allow_pickle=False) as archive:
            units, probability, prior = archive["units"].astype(str), archive["probabilities"], archive["prior"]
        target_units = prepared.frame.loc[prepared.frame.subject.isin(valid)].drop_duplicates("unit").unit.to_numpy()
        np.testing.assert_array_equal(units, target_units)
        values = visual["X"][[visual["index"][unit] for unit in units]]
        np.testing.assert_array_equal(reloaded["presence"].predict(values), probability)
        np.testing.assert_array_equal(reloaded["presence"].predict_prior(values), prior)
        np.savez_compressed(folder / "validation_probabilities.npz", units=units, probabilities=probability, prior=prior)
        for branch, probabilities in (("visual", probability), ("prior", prior)):
            evidence = evidence_from_probabilities(reloaded, prepared, compiled, units, probabilities, scope["prediction_qa_ids"])
            poison_evidence = evidence_from_probabilities(reloaded, poisoned, compiled, units, probabilities, scope["prediction_qa_ids"])
            for recipe in config["decoders"]:
                if recipe["branch"] != branch:
                    continue
                prediction = base.predict(prepared, evidence[recipe["likelihood"]], recipe)
                pd.testing.assert_frame_equal(prediction, base.predict(poisoned, poison_evidence[recipe["likelihood"]], recipe),
                                              rtol=0, atol=1e-12)
                prediction.to_csv(folder / (recipe["name"] + ".csv"), index=False)
                predictions[recipe["name"]] = prediction
                checks += 1
    else:
        np.testing.assert_array_equal(head.predict(matrix[:2]), reloaded["presence"].predict(matrix[:2]))
    assert sha256_file(parent / "model.joblib") == digest
    assert sha256_file(checkpoint) == saved_digest
    write_json(folder / "replay_audit.json", {"reused_head_actual_inputs_matched": True,
               "new_option_fits": 1, "new_visual_fits": 0, "target_poison_predictions": checks,
               "heldout_label_poison_checked": bool(valid), "checkpoint_reload_checked": True})
    return reloaded, scope, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for item in config["implementation"] + config["inputs"]:
        assert sha256_file(ROOT / item["path"]) == item["sha256"]
    parent = ROOT / config["parent_run"]
    parent_frozen = json.loads((parent / "frozen_config.json").read_text())
    verify_manifest(parent_frozen["source_manifest"])
    verify_manifest(parent_frozen["input_manifest"])
    sources = manifest([ROOT / item["path"] for item in config["implementation"]])
    inputs = manifest([args.config] + [ROOT / item["path"] for item in config["inputs"]])
    commit = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True).stdout.strip()
    write_json(output / "frozen_config.json", {"config": config, "source_manifest": sources,
               "input_manifest": inputs, "git_commit": commit, "started_utc": datetime.now(UTC).isoformat()})
    for item in sources:
        path = Path(item["path"])
        destination = output / "source_snapshot" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    config = {**config, "runtime_output": str(output)}
    torch.set_num_threads(2)
    threadpool_limits(limits=2)
    qa = pd.read_csv(ROOT / config["train_csv"], dtype=str, keep_default_na=False)
    prepared = prepare(qa, training=True)
    poisoned = prepare(qa.assign(answer="POISON", target="POISON", prediction="POISON"), training=True)
    np.testing.assert_array_equal(prepared.option_matrix, poisoned.option_matrix)
    assert len({clip["repeat_group"] for clip in prepared.clips}) == len(prepared.clips)
    compiled = compile_questions(qa, prepared)
    assert compiled == compile_questions(qa.assign(answer="POISON"), poisoned)
    visual = read_window_bank(ROOT / config["window_bank"])
    target = prepare_qa(qa, nested.SCOPES)
    subjects = sorted(OFFICIAL_SUBJECTS)
    assert set(prepared.frame.subject) == set(subjects) and len(target) == 2408
    # Reuse the frozen inner-selection algorithm, replacing only its fit implementation.
    nested.fit_scope = fit_scope
    state = {"status": "running", "pid": os.getpid(), "completed_outer_folds": 0}

    def update(active):
        state.update(active_fit=active, updated_utc=datetime.now(UTC).isoformat())
        write_json(output / "state.json", state)
        print(json.dumps(state), flush=True)

    parts = {"selected": [], **{recipe["name"]: [] for recipe in config["decoders"]},
             **{"policy_" + name: [] for name in config["control_policies"]}}
    folds = []
    try:
        for index, subject in enumerate(subjects):
            train, valid = sorted(set(subjects) - {subject}), [subject]
            folder = output / ("outer_" + subject)
            recipe, inner = nested.select_inner(qa, prepared, poisoned, compiled, visual, train, config,
                                                config["seed"] + index * 100, folder, sources, update)
            update(str(folder / "outer_fit"))
            _, scope, predictions = fit_scope(qa, prepared, poisoned, compiled, visual, train, valid, config,
                                               config["seed"] + index * 100 + 50, folder / "outer_fit", sources)
            parts["selected"].append(predictions[recipe["name"]])
            predictions[recipe["name"]].to_csv(folder / "oof_predictions.csv", index=False)
            for name, prediction in predictions.items():
                parts[name].append(prediction)
            controls = json.loads((folder / "selection.json").read_text())["control_selections"]
            for name, chosen in controls.items():
                parts["policy_" + name].append(predictions[chosen["name"]])
            provenance = manifest([folder / "outer_fit" / "training_scope.json"])[0]
            folds.append({"fold_id": f"outer_{index:02d}", "train_subjects": train, "validation_subjects": valid,
                          "prediction_qa_ids": scope["prediction_qa_ids"], "inner_folds": inner,
                          "upstream_training": [{"name": name, "train_subjects": train, "provenance": provenance}
                                                for name in COMPONENTS]})
            state["completed_outer_folds"] += 1
        assembled = {}
        for name, chunks in parts.items():
            prediction = pd.concat(chunks).set_index("qa_id").loc[target.qa_id].reset_index()
            prediction.to_csv(output / ("oof_predictions.csv" if name == "selected" else "oof_" + name + ".csv"), index=False)
            assembled[name] = prediction
        recipe, inner = nested.select_inner(qa, prepared, poisoned, compiled, visual, subjects, config,
                                             config["seed"] + 9000, output / "final_selection", sources, update)
        bundle, _, _ = fit_scope(qa, prepared, poisoned, compiled, visual, subjects, [], config,
                                  config["seed"] + 9050, output / "final_fit", sources)
        bundle.update(selected_decoder=recipe, final_inner_folds=inner)
        joblib.dump(bundle, output / "model.joblib", compress=3)
        (output / "model.sha256").write_text(sha256_file(output / "model.joblib") + "\n")
        protocol = {"schema_version": 1, "validation_kind": "nested_subject", "fitted_components": COMPONENTS,
                    "folds": folds, "selection_history": config["selection_history"],
                    "limitations": ["All18people historically exposed; not a fresh blind holdout.",
                                    "Independent recordings retain all same-clip questions and modalities.",
                                    "Heads reused only after exact fitting arrays/labels/scope checks; option forests refitted."]}
        write_json(output / "protocol.json", protocol)
        validation = validate_protocol(protocol, target, set(subjects), base_dir=output)
        assert validation["status"] == "declared_provenance_verified"
        parent_prediction = pd.read_csv(parent / "oof_predictions.csv", dtype={"qa_id": str, "prediction": str})
        parent_prediction = parent_prediction.set_index("qa_id").loc[target.qa_id].reset_index()
        metrics = {"independent_vs_same_condition_option": compare(target, assembled["selected"], assembled["option_only"]),
                   "independent_vs_same_condition_without_visual": compare(target, assembled["selected"], assembled["policy_without_visual"]),
                   "independent_vs_grouped_input_ablation": compare(target, assembled["selected"], parent_prediction)}
        verify_manifest(sources)
        verify_manifest(inputs)
        report = {"status": "completed_independent_nested_option_refit", "metrics": metrics,
                  "selected_final_decoder": recipe, "new_option_models": 76, "reused_visual_models": 76,
                  "new_visual_fits": 0, "target_poison_prediction_checks": 975, "validation": validation,
                  "test_qa_read": False, "uploads": 0,
                  "limits": "HAUaction2408QA only; grouped comparison is an input ablation, not a superiority claim. No independent whole-system accuracy or new-acquisition guarantee."}
        write_json(output / "report.json", report)
        state.update(status="complete", active_fit=None)
        write_json(output / "state.json", state)
        print(json.dumps({name: result["overall"] for name, result in metrics.items()}), flush=True)
    except Exception as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        write_json(output / "state.json", state)
        raise


if __name__ == "__main__":
    main()
