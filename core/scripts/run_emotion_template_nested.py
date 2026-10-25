#!/usr/bin/env python3
"""Run frozen template-matched emotion validation without changing prior sources."""

from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from cuhkx import emotion_fusion_nested, emotion_template_nested
from scripts import run_emotion_nested as runner

LEGACY_SOURCES = runner.sources
LEGACY_WRITE_JSON = runner.write_json
EXTRA_SOURCES = ["scripts/run_emotion_template_nested.py", "src/cuhkx/emotion_template_nested.py",
                 "src/cuhkx/emotion_fusion_nested.py", "src/cuhkx/emotion_relative.py"]


def sources():
    return {**LEGACY_SOURCES(), **{name: runner.sha(ROOT / name) for name in EXTRA_SOURCES}}


def select_inner(frame, contexts, matrices, masks, fit_subjects, config, out, seed, inputs):
    selected = np.flatnonzero(frame.subject.isin(fit_subjects))
    subjects = frame.iloc[selected].subject.to_numpy()
    scores, folds = [], []
    for index, (fit_indices, valid_indices) in enumerate(
        GroupKFold(n_splits=config["inner_folds"]).split(selected, groups=subjects)
    ):
        fit_users, valid_users = set(subjects[fit_indices]), set(subjects[valid_indices])
        if fit_users & valid_users or (fit_users | valid_users) != set(fit_subjects):
            raise ValueError("invalid inner subject partition")
        bundle = runner.fit_bundle(frame, contexts, matrices, masks, fit_users, config, seed + index * 10)
        rows = selected[valid_indices]
        evidence = runner.base_evidence(bundle, frame, contexts, matrices, masks, rows)
        scores.append(runner.validation_views(frame.iloc[rows], evidence, config["decoders"]))
        provenance = runner.scope_record(out / f"inner_{index}_scope.json", bundle, frame,
                                         fit_users, valid_users, inputs)
        bundle["decoder"] = {"name": "all_predeclared_inner_candidates"}
        checkpoint = out / f"inner_{index}_models.joblib"
        joblib.dump(bundle, checkpoint, compress=3)
        folds.append({"fold_id": f"inner_{index}", "train_subjects": sorted(fit_users),
                      "validation_subjects": sorted(valid_users), "provenance": provenance,
                      "prediction_qa_ids": frame.iloc[rows].qa_id.tolist(),
                      "checkpoint": {"path": str(checkpoint.resolve()), "sha256": runner.sha(checkpoint)},
                      "upstream_training": [{"name": "sensor_models_and_word_context_priors",
                                             "train_subjects": sorted(fit_users), "provenance": provenance}]})
        del bundle
    combined = pd.concat(scores, ignore_index=True)
    policy, ranking = runner.choose_decoder(combined, config)
    combined.to_csv(out / "inner_validation_views.csv", index=False)
    runner.write_json(out / "inner_selection.json", {"selected": policy, "ranking": ranking})
    runner.write_json(out / "inner_fold_manifest.json", folds)
    return policy, ranking, folds


def write_json(path, payload):
    if path.name == "protocol.json":
        payload = copy.deepcopy(payload)
        payload["fitted_components"] = ["sensor_models_and_word_context_priors"]
        payload["selection_history"] = "Previously observed subjects and v2b error slices; no new blind holdout."
        payload["mechanism"] = "three-question partial-core proxies and inner-only template-specific decoder choice"
        payload["limitations"][-1] = "Relaxed-core views retain all questions; proxy vocabulary remains training vocabulary and views are dependent."
    if path.name == "report.json" and payload.get("status") == "nested_component_evaluated":
        bundle = joblib.load(path.parent / "models.joblib")
        frame, _ = runner.prepare(runner.read_csv_str(Path(next(
            name for name in payload["input_hashes"] if name.endswith("training_qa.csv")))), training=True)
        runner.scope_record(path.parent / "final_training_scope.json", bundle, frame,
                            set(frame.subject), set(), payload["input_hashes"])
        payload = copy.deepcopy(payload)
        payload["outputs"]["final_training_scope.json"] = runner.sha(path.parent / "final_training_scope.json")
        payload["retained_checkpoints"] = {"outer": 18, "inner": 54, "final_inner": 3, "final": 1}
    LEGACY_WRITE_JSON(path, payload)
    if path.name == "frozen_config.json":
        snapshot = path.parent / "source_snapshot"
        snapshot.mkdir()
        records = []
        for name, digest in payload["source_hashes"].items():
            source = ROOT / name
            if runner.sha(source) != digest:
                raise ValueError("source changed while freezing nested run")
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            records.append({"path": str(target.resolve()), "sha256": digest})
        LEGACY_WRITE_JSON(path.parent / "source_snapshot_manifest.json", records)


def install_hooks():
    runner.base_evidence = emotion_fusion_nested.base_evidence
    runner.predict_rows = emotion_template_nested.predict_rows
    runner.validation_views = emotion_template_nested.validation_views
    runner.choose_decoder = emotion_template_nested.choose_decoder
    runner.select_inner = select_inner
    runner.sources = sources
    runner.write_json = write_json
    runner.KIND = "cuhkx_emotion_template_nested_v3"


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "fit":
        if "--config" not in sys.argv:
            sys.argv.extend(["--config", str(ROOT / "configs/emotion_template_nested_v3.json")])
        config = json.loads(Path(sys.argv[sys.argv.index("--config") + 1]).read_text())
        if not config.get("frozen_before_experiments") or len(config["decoders"]) != 5:
            raise ValueError("template experiment requires exactly five frozen v2 decoder families")
    install_hooks()
    runner.main()


if __name__ == "__main__":
    main()
