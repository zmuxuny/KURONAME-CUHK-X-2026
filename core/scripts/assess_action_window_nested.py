#!/usr/bin/env python3
"""Independently replay nested window models and compare predeclared policies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from cuhkx import action_generic_nested as base
from cuhkx.action_videomae_bank import read_window_bank
from cuhkx.action_window_decoding import compile_questions, evidence_from_probabilities
from cuhkx.action_window_head import unpack
from cuhkx.validation import attach_predictions, prepare_qa, score_rows, sha256_file
from scripts.run_action_generic_nested import accuracy, verify_manifest, write_json
from scripts.run_action_window_nested import SCOPES


def compare(truth, prediction, baseline):
    scored, _ = attach_predictions(truth.copy(), prediction, require_order=True)
    scored, _ = attach_predictions(scored, baseline, column="baseline_prediction", require_order=True)
    people = {name: score_rows(part) for name, part in scored.groupby("subject")}
    counts = np.asarray([[part["net_correct"], part["rows"]] for part in people.values()])
    sample = np.random.default_rng(20260908).integers(0, len(counts), size=(10000, len(counts)))
    sampled = counts[sample].sum(axis=1)
    delta = sampled[:, 0] / sampled[:, 1]
    return {"overall": score_rows(scored), "by_subject": people,
            "by_category": {name: score_rows(part) for name, part in scored.groupby("category")},
            "worst_subject": min(part["accuracy"] for part in people.values()),
            "macro_subject": float(np.mean([part["accuracy"] for part in people.values()])),
            "person_bootstrap_delta95": np.quantile(delta, [.025, .975]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    assert json.loads((run / "state.json").read_text())["status"] == "complete"
    frozen = json.loads((run / "frozen_config.json").read_text())
    verify_manifest(frozen["source_manifest"])
    verify_manifest(frozen["input_manifest"])
    config = frozen["config"]
    torch.set_num_threads(2)
    threadpool_limits(limits=2)
    qa = pd.read_csv(ROOT / config["train_csv"], dtype=str, keep_default_na=False)
    prepared = base.prepare(qa, training=True)
    compiled = compile_questions(qa, prepared)
    visual = read_window_bank(ROOT / config["window_bank"])
    audits = []
    scope_files = sorted(run.rglob("training_scope.json"))
    assert len(scope_files) == 76
    for path in scope_files:
        folder = path.parent
        scope = json.loads(path.read_text())
        checkpoint = folder / "model.joblib"
        assert sha256_file(checkpoint) == scope["checkpoint_sha256"] == (folder / "model.sha256").read_text().strip()
        bundle = joblib.load(checkpoint)
        train, valid = scope["train_subjects"], scope["validation_subjects"]
        assert bundle["training_subjects"] == train and bundle["validation_subjects"] == valid
        arrays, expected = base.fitting_arrays(qa, prepared, visual, train, valid)
        for name, value in expected.items():
            assert scope[name] == value, (str(path), name)
        matrix, targets = arrays[:2]
        head = bundle["presence"]
        np.testing.assert_array_equal(head.training_targets, targets)
        assert head.fitting_content_sha256 == expected["visual_matrix_and_targets_sha256"]
        features, mask = unpack(matrix)
        for view in range(2):
            values = features[:, :, view][mask[:, :, view]]
            if len(values):
                np.testing.assert_array_equal(head.mean[view], values.mean(axis=0))
                np.testing.assert_array_equal(head.scale[view], np.maximum(values.std(axis=0), config["presence"]["std_floor"]))
        checks = 0
        if valid:
            poison = qa.copy()
            poison.loc[poison.qa_id.isin(scope["prediction_qa_ids"]), "answer"] = "FORBIDDEN_VALIDATION_LABEL"
            _, poisoned_scope = base.fitting_arrays(poison, prepared, visual, train, valid)
            assert poisoned_scope == expected
            with np.load(folder / "validation_probabilities.npz", allow_pickle=False) as archive:
                units, probabilities, prior = archive["units"].astype(str), archive["probabilities"], archive["prior"]
            target_units = prepared.frame.loc[prepared.frame.subject.isin(valid)].drop_duplicates("unit").unit.to_numpy()
            np.testing.assert_array_equal(units, target_units)
            values = visual["X"][[visual["index"][unit] for unit in units]]
            np.testing.assert_array_equal(head.predict(values), probabilities)
            np.testing.assert_array_equal(head.predict_prior(values), prior)
            for branch, probability in (("visual", probabilities), ("prior", prior)):
                evidence = evidence_from_probabilities(bundle, prepared, compiled, units, probability, scope["prediction_qa_ids"])
                for recipe in config["decoders"]:
                    if recipe["branch"] != branch:
                        continue
                    prediction = base.predict(prepared, evidence[recipe["likelihood"]], recipe)
                    saved = pd.read_csv(folder / (recipe["name"] + ".csv"), dtype={"qa_id": str, "prediction": str})
                    pd.testing.assert_frame_equal(prediction, saved, check_dtype=False, rtol=0, atol=1e-12)
                    checks += 1
        else:
            final = joblib.load(run / "model.joblib")
            assert sha256_file(run / "model.joblib") == (run / "model.sha256").read_text().strip()
            assert final["training_subjects"] == train
            np.testing.assert_array_equal(final["presence"].predict(matrix[:2]), head.predict(matrix[:2]))
        audit = {"scope": str(path.relative_to(run)), "actual_fitting_arrays_recomputed": True,
                 "training_normalization_recomputed": True, "heldout_label_poison_checked": bool(valid),
                 "checkpoint_sha256": scope["checkpoint_sha256"], "prediction_replays": checks}
        audits.append(audit)
        write_json(output / "progress.json", {"scopes_completed": len(audits), "total_scopes": 76})
        print(json.dumps({"scopes_completed": len(audits), **audit}), flush=True)
    selections = []
    selected_parts = []
    policy_parts = {name: [] for name in config["control_policies"]}
    truth = prepare_qa(qa, SCOPES)
    for path in sorted(run.rglob("selection.json")):
        saved = json.loads(path.read_text())
        ids = saved["selection_qa_ids"]
        target = truth.set_index("qa_id").loc[ids].reset_index()
        ranking = []
        for index, recipe in enumerate(config["decoders"]):
            parts = [pd.read_csv(folder / (recipe["name"] + ".csv"), dtype={"qa_id": str, "prediction": str})
                     for folder in sorted(path.parent.glob("inner_*"))]
            prediction = pd.concat(parts).set_index("qa_id").loc[ids].reset_index()
            ranking.append({"decoder": recipe, "order": index, **accuracy(target, prediction)})
        assert ranking == saved["rankings"]
        ordered = sorted(ranking, key=lambda row: (-row["accuracy"], -row["worst_subject_accuracy"], row["order"]))
        assert ordered[0]["decoder"] == saved["selected_decoder"]
        for name, branches in config["control_policies"].items():
            chosen = next(row["decoder"] for row in ordered if row["decoder"]["name"] == "option_only" or row["decoder"]["branch"] in branches)
            assert chosen == saved["control_selections"][name]
        if path.parent.name.startswith("outer_"):
            read_fold = lambda recipe, current=path.parent: pd.read_csv(current / "outer_fit" / (recipe["name"] + ".csv"),
                                                   dtype={"qa_id": str, "prediction": str})
            selected_parts.append(read_fold(saved["selected_decoder"]))
            for name, parts in policy_parts.items():
                parts.append(read_fold(saved["control_selections"][name]))
        else:
            assert joblib.load(run / "model.joblib")["selected_decoder"] == saved["selected_decoder"]
        selections.append(str(path.relative_to(run)))
    assert len(selections) == 19
    read = lambda path: pd.read_csv(path, dtype={"qa_id": str, "prediction": str}).set_index("qa_id").loc[truth.qa_id].reset_index()
    prediction = read(run / "oof_predictions.csv")
    controls = {name: read(run / ("oof_policy_" + name + ".csv")) for name in config["control_policies"]}
    assembled = pd.concat(selected_parts).set_index("qa_id").loc[truth.qa_id].reset_index()
    pd.testing.assert_frame_equal(prediction, assembled, check_dtype=False, rtol=0, atol=1e-12)
    for name, parts in policy_parts.items():
        assembled = pd.concat(parts).set_index("qa_id").loc[truth.qa_id].reset_index()
        pd.testing.assert_frame_equal(controls[name], assembled, check_dtype=False, rtol=0, atol=1e-12)
    option = read(ROOT / config["option_control"])
    metrics = {"nested_all_vs_option": compare(truth, prediction, option),
               "nested_all_vs_without_visual": compare(truth, prediction, controls["without_visual"]),
               "nested_visual_vs_without_visual": compare(truth, controls["visual_and_option"], controls["without_visual"])}
    fixed = {recipe["name"]: compare(truth, read(run / ("oof_" + recipe["name"] + ".csv")), option)
             for recipe in config["decoders"]}
    verify_manifest(frozen["source_manifest"])
    verify_manifest(frozen["input_manifest"])
    report = {"status": "completed_actual_nested_model_and_selection_replay", "audits": audits,
              "selection_replays": selections, "metrics": metrics, "fixed_recipe_metrics": fixed,
              "new_model_fits": 0, "test_qa_read": False, "uploads": 0,
              "implementation_sha256": sha256_file(Path(__file__)),
              "limits": "2408HAUactionQA only;all18people historicallyexposed. Personbootstrap does not correct historical selection bias. Fixedrecipe metrics are diagnostics;never select a finalrecipe onouterQA. CurrentstrongKUNOchain has unknown upstreamfolds andhas not been replaced."}
    write_json(output / "report.json", report)
    print(json.dumps({name: item["overall"] for name, item in metrics.items()}), flush=True)


if __name__ == "__main__":
    main()
