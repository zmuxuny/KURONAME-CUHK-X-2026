#!/usr/bin/env python3
"""Nested selection of support-preserving fusion on retained two-seed evidence."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from cuhkx.harn_nested import (
    OFFICIAL_SUBJECTS,
    QA_COLUMNS,
    evaluate_predictions,
    harn_rows,
    write_json,
)
from cuhkx.harn_seed_ensemble import check_scope, equal_evidence, mappings
from cuhkx.harn_support_fusion import decode_supported
from cuhkx.metadata import feature_unit, subject_id
from scripts.run_harn_nested_fusion import manifest_entry, read_qa
from scripts.run_harn_seed_ensemble import compare_frame, load_evidence
from scripts.run_submission_window import sha

SOURCES = ["scripts/run_harn_support_fusion.py", "src/cuhkx/harn_support_fusion.py"]


def freeze(config_path, output):
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(config_path.read_text())
    parent = ROOT / config["parent"]
    old = json.loads((parent / "frozen_config.json").read_text())
    sources = {str(ROOT / path): digest for path, digest in old["source_hashes"].items()}
    sources.update({str(ROOT / path): sha(ROOT / path) for path in SOURCES})
    inputs = [parent / name for name in ("frozen_config.json", "input_manifest.json", "protocol.json",
                                         "generation_frozen.json", "checkpoint.json")]
    inputs += [ROOT / config["train_csv"], ROOT / config["test_csv"], ROOT / config["upstream_audit"]]
    if sha(inputs[-1]) != config["upstream_audit_sha256"]:
        raise ValueError("upstream completed audit differs")
    inputs += [parent / name for name in ("test_probabilities.npz", "test_predictions.csv", "oof_predictions.csv")]
    inputs += sorted((parent / "selections").glob("*.json"))
    inputs += sorted((parent / "selections").glob("*.csv"))
    for name, digest in sources.items():
        if sha(name) != digest:
            raise ValueError("fitted parent source changed")
        destination = output / "source_snapshot" / Path(name).relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, destination)
    write_json(output / "config.json", config)
    write_json(output / "frozen.json", {"created_utc": datetime.now(UTC).isoformat(),
               "config_sha256": sha(output / "config.json"), "sources": sources,
               "inputs": [manifest_entry(path) for path in inputs]})
    print("support fusion recipe and sources frozen", flush=True)


class CachedScopes:
    def __init__(self, parent, qa):
        self.qa, self.parent = qa, parent
        self.inputs = {str(Path(x["path"]).resolve()): x for x in json.loads((parent / "input_manifest.json").read_text())}
        config = json.loads((parent / "frozen_config.json").read_text())["config"]
        self.parents = {key: ROOT / value["run"] for key, value in config["parents"].items()}
        self.used, self.scopes = {}, {}

    def read(self, path):
        path = path.resolve()
        if str(path) not in self.used:
            expected = self.inputs[str(path)]
            if sha(path) != expected["sha256"]:
                raise ValueError("retained scope input changed")
            self.used[str(path)] = expected
        return path

    def load(self, fit_id, subjects, target):
        train = self.qa.loc[self.qa.path.map(subject_id).isin(subjects)].copy()
        values = {}
        for name, root in self.parents.items():
            scope = json.loads(self.read(root / "fits" / f"{fit_id}.json").read_text())
            check_scope(scope, train, subjects, OFFICIAL_SUBJECTS - set(subjects))
            evidence = load_evidence(self.read(root / "fits" / f"{fit_id}.saved_evidence.npz"))
            np.testing.assert_array_equal(evidence["units"], sorted({feature_unit(p, training=True) for p in target.path}))
            for family in ("skeleton", "visual"):
                np.testing.assert_array_equal(evidence[f"{family}_classes"], np.unique(scope["features"][family]["labels"]))
            values[name] = evidence
        self.scopes[fit_id] = {"train_subjects": sorted(subjects), "dictionary_qa_ids": train.qa_id.tolist()}
        return equal_evidence(values["original"], values["repeated"]), mappings(train)


def choose(cache, fitting, scope, config, output):
    declaration = json.loads((cache.parent / "selections" / f"{scope}.json").read_text())
    parts = {recipe["id"]: [] for recipe in config["recipes"]}
    for index, fold in enumerate(declaration["inner_folds"]):
        train, valid = set(fold["train_subjects"]), set(fold["validation_subjects"])
        if train & valid or train | valid != set(fitting.path.map(subject_id)):
            raise ValueError("inner people overlap or differ from outer training scope")
        target = fitting.loc[fitting.qa_id.isin(fold["prediction_qa_ids"])].copy()
        if target.qa_id.tolist() != fold["prediction_qa_ids"] or set(target.path.map(subject_id)) != valid:
            raise ValueError("inner validation coverage differs")
        evidence, dictionary = cache.load(fold["fold_id"], train, target)
        for recipe in config["recipes"]:
            frame = decode_supported(target[QA_COLUMNS], evidence, dictionary, recipe["mode"], recipe["weight"], training_paths=True)
            frame["inner_fold"] = index
            parts[recipe["id"]].append(frame)
    frames = {name: pd.concat(values, ignore_index=True) for name, values in parts.items()}
    reports = {name: evaluate_predictions(fitting, frame) for name, frame in frames.items()}
    reference = pd.read_csv(cache.parent / "selections" / f"ensemble_{scope}_inner_predictions.csv", keep_default_na=False)
    controls = pd.concat([frames[r["id"]] for r in config["recipes"] if r["mode"] == "baseline"], ignore_index=True)
    compare_frame(controls[reference.columns], reference)
    selected = max(config["recipes"], key=lambda r: (reports[r["id"]]["accuracy"], reports[r["id"]]["worst_subject_accuracy"], -config["tie_order"].index(r["id"])))
    for recipe in config["recipes"]:
        frames[recipe["id"]]["recipe"] = recipe["id"]
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(frames.values(), ignore_index=True).to_csv(output / f"{scope}.csv", index=False)
    write_json(output / f"{scope}.json", {"selected": selected, "metrics": reports, "inner_folds": declaration["inner_folds"], "baseline_replay_exact": True})
    return selected


def fit(run):
    frozen = json.loads((run / "frozen.json").read_text())
    if sha(run / "config.json") != frozen["config_sha256"]:
        raise ValueError("configuration changed after freeze")
    for path, digest in frozen["sources"].items():
        if sha(path) != digest:
            raise ValueError("source changed after freeze")
    for item in frozen["inputs"]:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("input changed after freeze")
    config = json.loads((run / "config.json").read_text())
    output = run / "fit"
    output.mkdir(exist_ok=False)
    start = time.monotonic()
    qa = harn_rows(read_qa(ROOT / config["train_csv"], labeled=True))
    parent = ROOT / config["parent"]
    cache = CachedScopes(parent, qa)
    protocol = copy.deepcopy(json.loads((parent / "protocol.json").read_text()))
    parts, selections = [], []
    for fold in protocol["folds"]:
        fitting = qa.loc[qa.path.map(subject_id).isin(fold["train_subjects"])].copy()
        target = qa.loc[qa.qa_id.isin(fold["prediction_qa_ids"])].copy()
        selected = choose(cache, fitting, fold["fold_id"], config, output / "selections")
        evidence, dictionary = cache.load(fold["fold_id"] + "_final", set(fold["train_subjects"]), target)
        frame = decode_supported(target[QA_COLUMNS], evidence, dictionary, selected["mode"], selected["weight"], training_paths=True)
        frame["outer_fold"] = fold["fold_id"]
        parts.append(frame)
        selections.append({"fold": fold["fold_id"], "selected": selected})
        print(f"support outer={len(parts)}/18 recipe={selected['id']}", flush=True)
    oof = pd.concat(parts, ignore_index=True).set_index("qa_id", drop=False).loc[qa.qa_id].reset_index(drop=True)
    oof.to_csv(output / "oof_predictions.csv", index=False)
    final = choose(cache, qa, "final_all_subjects", config, output / "selections")
    test = harn_rows(read_qa(ROOT / config["test_csv"], labeled=False))
    evidence = load_evidence(parent / "test_probabilities.npz")
    predictions = decode_supported(test[QA_COLUMNS], evidence, mappings(qa), final["mode"], final["weight"], training_paths=False)
    predictions.to_csv(output / "test_predictions.csv", index=False)
    parent_test = pd.read_csv(parent / "test_predictions.csv", keep_default_na=False)
    parent_oof = pd.read_csv(parent / "oof_predictions.csv", keep_default_na=False)
    comparison = qa[["qa_id", "answer"]].merge(oof[["qa_id", "prediction"]], on="qa_id").merge(parent_oof[["qa_id", "prediction"]].rename(columns={"prediction": "baseline"}), on="qa_id")
    old_ok, new_ok = comparison.baseline.eq(comparison.answer), comparison.prediction.eq(comparison.answer)
    protocol["selection_history"] = config["history"]
    protocol["support_fusion_sources"] = frozen["sources"]
    protocol["support_fusion_config"] = config
    write_json(output / "protocol.json", protocol)
    write_json(output / "used_inputs.json", list(cache.used.values()))
    write_json(output / "scope_dictionaries.json", cache.scopes)
    write_json(output / "checkpoint.json", {"kind": "cuhkx_harn_support_fusion_v1", "parent_checkpoint": manifest_entry(parent / "checkpoint.json"), "selected_recipe": final, "sources": frozen["sources"]})
    report = {"completed_utc": datetime.now(UTC).isoformat(), "metrics": evaluate_predictions(qa, oof), "selections": selections, "final_recipe": final,
              "corrected": int((~old_ok & new_ok).sum()), "harmed": int((old_ok & ~new_ok).sum()), "changed": int(comparison.prediction.ne(comparison.baseline).sum()),
              "test_changed": int(predictions.prediction.ne(parent_test.prediction).sum()), "new_neural_fits": 0, "reused_prior152_model_audit": True,
              "seconds": time.monotonic() - start, "history": config["history"]}
    write_json(output / "report.json", report)
    print(json.dumps({k: report[k] for k in ("corrected", "harmed", "changed", "test_changed", "final_recipe")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    predeclare = sub.add_parser("freeze")
    predeclare.add_argument("--config", type=Path, required=True)
    predeclare.add_argument("--output-dir", type=Path, required=True)
    fitting = sub.add_parser("fit")
    fitting.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=3):
        if args.phase == "freeze":
            freeze(args.config, args.output_dir)
        else:
            fit(args.run_dir)


if __name__ == "__main__":
    main()
