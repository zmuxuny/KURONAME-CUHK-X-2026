#!/usr/bin/env python3
"""Infer the independent HAU action chain from fitted models and test features.

This recreates the complete-set blend, confidence gates, option-frequency
models, and repeat constraints. No historical prediction CSV is an input.
The original release's independently audited acquisition layout is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cuhkx.actions import hau_action_class
from cuhkx.clip_constraints import apply_repeated_sequence_constraints, repeated_recording_groups
from cuhkx.inference import QUESTION_COLUMNS, question_metadata
from cuhkx.metadata import feature_unit
from cuhkx.repeated_actions import (
    SOURCE_CATEGORIES,
    apply_repeated_action_decisions,
    repeated_action_decisions,
)
from cuhkx.submission import available_letters, validate_submission_frames
from scripts import hau_skeleton_transfer_checkpoint as skeleton
from scripts import kuno_presence_checkpoint as presence
from scripts import option_frequency_checkpoint as frequency
from scripts import run_hau_complete_set_calibration as complete
from scripts import run_hau_joint_actions as joint
from scripts import run_hau_joint_complete_blend as broad
from scripts.apply_local_evidence_constraints import (
    applicable_action_updates,
    frequency_updates,
    source_maps,
)
from scripts.apply_repeated_action_evidence import GATE_COLUMNS
from scripts.build_complete_set_joint_consensus import apply_gates
from scripts.run_hau_repeated_option_frequency import gate_predictions
from scripts.temporal_checkpoint import SOURCE_PATHS, load_logits, resolve_groups

KIND = "cuhkx_full_hau_action_checkpoint_v1"
ACTION_CATEGORIES = {"single", "multi", "combination"}
BLEND_WEIGHTS = {"calibrated": 0.5, "visual16": 0.5, "complete_set": 1.5, "skeleton": 0.2}
SOURCES = sorted({
    "scripts/hau_action_checkpoint.py", "scripts/run_hau_complete_set_calibration.py",
    "scripts/run_hau_joint_actions.py", "scripts/run_hau_joint_complete_blend.py",
    "scripts/build_complete_set_joint_consensus.py", "scripts/apply_local_evidence_constraints.py",
    "scripts/apply_repeated_action_evidence.py", "scripts/apply_harn_paired_constraints.py",
    "scripts/build_local_evidence_baseline.py", "scripts/run_harn_object_interaction.py",
    "scripts/run_hau_repeated_option_frequency.py", "src/cuhkx/joint_actions.py",
    "src/cuhkx/repeated_actions.py", "src/cuhkx/validation.py", "src/cuhkx/option_frequency.py",
    *presence.source_hashes(), *frequency.source_hashes(), *skeleton.SOURCES, *SOURCE_PATHS,
})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sources() -> dict:
    return {name: sha256(ROOT / name) for name in SOURCES}


def load_pickle(path: Path, *, workers: int):
    bundle = joblib.load(path)
    models = bundle.get("models", {})
    for model in models.values() if isinstance(models, dict) else models:
        model.n_jobs = workers
    return bundle


def calibration_map(qa, scores, fallback):
    offered = qa.loc[qa.source.eq("HAU") & qa.category.isin({"single", "multi"})]
    scores = scores.set_index("qa_id")
    values = defaultdict(list)
    for _, row in offered.iterrows():
        unit = feature_unit(row.path, training=False)
        for letter in available_letters(row):
            values[(unit, hau_action_class(row[letter]))].append(float(scores.at[row.qa_id, f"option_score_{letter}"]))
    output = {unit: vector.copy() for unit, vector in fallback.items()}
    for (unit, cls), entries in values.items():
        if max(entries) - min(entries) > 1e-5:
            raise ValueError("inconsistent per-clip calibrated class score")
        output[unit][cls] = np.mean(entries)
    return output


def frequency_option_map(scores):
    columns = [name for name in scores if name.startswith("probability_")]
    values = scores[columns].to_numpy(dtype=float).clip(1e-5, 1 - 1e-5)
    logits = np.log(values / (1 - values)).mean(axis=1)
    frame = scores[["qa_id", "letter"]].assign(probability=1 / (1 + np.exp(-logits)))
    return {qa_id: dict(zip(group.letter, group.probability, strict=True))
            for qa_id, group in frame.groupby("qa_id", sort=False)}


def constrain_actions(qa, initial, joint_gates, frequency_gates):
    # Non-action placeholders only satisfy the shared helper's complete-ID API.
    # Those helpers use sequence option sets, never sequence/emotion predictions.
    answers = {row.qa_id: (available_letters(row) if row.category == "sequence"
                           else available_letters(row)[0]) for _, row in qa.iterrows()}
    answers.update(initial.set_index("qa_id").prediction.to_dict())
    traces = []

    def update(stage, values):
        for qa_id, prediction in values.items():
            traces.append({"qa_id": qa_id, "stage": stage, "previous": answers[qa_id],
                           "prediction": prediction, "changed": answers[qa_id] != prediction})
            answers[qa_id] = prediction

    update("frequency", frequency_updates(frequency_gates))
    _, decisions = apply_repeated_sequence_constraints(qa, answers)
    update("repeated_sequence_1", applicable_action_updates(decisions))
    joint_gates = joint_gates.copy()
    frequency_gates = frequency_gates.copy()
    joint_gates["source_eligible"] = joint_gates[list(GATE_COLUMNS)].all(axis=1) & joint_gates.category.isin(SOURCE_CATEGORIES)
    frequency_gates["source_eligible"] = frequency_gates.direct_frequency_gate & frequency_gates.category.isin(SOURCE_CATEGORIES)
    evidence, eligible, conflicts = source_maps(joint_gates, frequency_gates)
    decisions = repeated_action_decisions(qa, evidence, eligible)
    _, applied = apply_repeated_action_decisions(answers, decisions)
    update("repeated_actions", applicable_action_updates(applied))
    _, decisions = apply_repeated_sequence_constraints(qa, answers)
    update("repeated_sequence_2", applicable_action_updates(decisions))
    target = qa.loc[qa.source.eq("HAU") & qa.category.isin(ACTION_CATEGORIES)].copy()
    result = target[["qa_id", "source", "category"]].assign(prediction=target.qa_id.map(answers))
    validation = result[["qa_id", "prediction"]]
    validate_submission_frames(target, validation, validation)
    return result, pd.DataFrame(traces), conflicts


def infer(qa, paths, logits, window_features, manifest, output, workers):
    resolve_groups(qa, manifest["layout"])
    load_logits(logits, 4.0, manifest["logit_contracts"])
    presence_bundle = load_pickle(paths["presence"], workers=workers)
    if presence_bundle["source_hashes"] != presence.source_hashes():
        raise ValueError("presence checkpoint source mismatch")
    features, direct, units, contract, _ = presence.load_features([logits[0]], presence_bundle["config"])
    if contract != presence_bundle["feature_contract"]:
        raise ValueError("presence feature contract mismatch")
    option_scores, _, _ = presence.predict(presence_bundle, qa, features, direct, units)
    v8 = joint.temporal_presence_map(logits[0], temperature=4.0, top_windows=2)
    v16 = joint.temporal_presence_map(logits[1], temperature=4.0, top_windows=4)
    calibrated = calibration_map(qa, option_scores, v8)
    option_scores.to_csv(output / "presence_option_scores.csv", index=False)
    del presence_bundle, features

    skeleton_manifest = json.loads(paths["skeleton"].read_text())
    if skeleton_manifest["kind"] != skeleton.KIND or skeleton_manifest["source_hashes"] != skeleton.source_hashes():
        raise ValueError("Skeleton transfer checkpoint source mismatch")
    banks = [skeleton.read_windows(path, inference=True) for path in window_features]
    seed_scores, skeleton_units = [], None
    for entry in skeleton_manifest["models"]:
        model_path = (paths["skeleton"].parent / entry["path"]).resolve()
        if sha256(model_path) != entry["sha256"]:
            raise ValueError("Skeleton transfer model SHA mismatch")
        bundle = load_pickle(model_path, workers=workers)
        axis, scores, _, _ = skeleton.predict_seed(bundle, qa, banks, workers)
        if skeleton_units is not None and not np.array_equal(skeleton_units, axis):
            raise ValueError("Skeleton seed unit coverage differs")
        skeleton_units = axis
        seed_scores.append(scores.mean(axis=0))
        del bundle
    skeleton_values = np.mean(np.stack(seed_scores), axis=0).astype(np.float32)
    skeleton_map = dict(zip(skeleton_units, skeleton_values, strict=True))
    del banks

    complete_features, complete_units, _ = complete.load_features(argparse.Namespace(
        logit_view=[[logits[0]], [logits[1]]], top_windows=[2, 4], temperature=4.0,
    ))
    complete_frames, complete_scores = [], []
    for index in range(3):
        bundle = load_pickle(paths[f"complete_{index}"], workers=workers)
        if (bundle["config"]["temperature"] != 4.0 or bundle["config"]["top_windows"] != [2, 4]
                or set(bundle["models"]) != {"extra_leaf2"}):
            raise ValueError("unexpected complete-set classifier configuration")
        values = complete.positive_probabilities(
            bundle["models"]["extra_leaf2"], complete_features.reshape(-1, complete_features.shape[-1]),
        ).reshape(len(complete_units), 40)
        decisions = complete.decode_frame(qa, complete_units, {"extra_leaf2": values, "mean": values}, training=False)
        complete_frames.append(decisions)
        complete_scores.append(values)
        decisions.to_csv(output / f"complete_predictions_{index}.csv", index=False)
        del bundle
    complete_map = dict(zip(complete_units, np.mean(np.stack(complete_scores), axis=0).astype(np.float32), strict=True))
    del complete_features

    joint_sources = {"calibrated": calibrated, "v8": v8, "v16": v16,
                     "v16_peak": joint.temporal_presence_map(logits[1], temperature=5.0, top_windows=1),
                     "skeleton": skeleton_map}
    joint_frame, _ = joint.decode_frame(qa, joint_sources, training=False)
    joined = joint_frame.copy()
    for index, frame in enumerate(complete_frames):
        joined = joined.merge(frame[["qa_id", "prediction", "margin"]].rename(columns={
            "prediction": f"complete_prediction_{index}", "margin": f"complete_margin_{index}",
        }), on="qa_id", validate="one_to_one")
    joint_gates = apply_gates(joined, argparse.Namespace(complete_set_run=[0, 1, 2],
        strict_joint_agreement=7, strict_joint_margin=0.05, complete_margin=0.20,
        skeleton_complete_margin=0.15))
    frequency_bundle = load_pickle(paths["frequency"], workers=workers)
    if frequency_bundle["source_hashes"] != frequency.source_hashes():
        raise ValueError("option frequency checkpoint source mismatch")
    frequency_scores, frequency_frames = frequency.predict(frequency_bundle, qa)
    frequency_gates = gate_predictions(frequency_frames, joint_gates, minimum_margin=0.5,
                                      minimum_direct_margin=0.8, required_joint_agreement=7)
    broad_frame, _ = broad.decode_frame(qa, {"calibrated": calibrated, "visual16": v16,
        "complete_set": complete_map, "skeleton": skeleton_map}, BLEND_WEIGHTS,
        frequency_option_map(frequency_scores), 0.05, training=False)
    result, trace, conflicts = constrain_actions(qa, broad_frame, joint_gates, frequency_gates)
    for name, frame in {"joint_predictions": joint_frame, "joint_complete_gates": joint_gates,
                        "frequency_gates": frequency_gates, "option_frequency_scores": frequency_scores,
                        "broad_predictions": broad_frame, "constraint_trace": trace,
                        "test_predictions": result}.items():
        frame.to_csv(output / f"{name}.csv", index=False, lineterminator="\n")
    return {"rows": len(result), "skeleton_units": len(skeleton_units), "source_conflicts": conflicts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["export", "infer"])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--presence-checkpoint", type=Path)
    parser.add_argument("--frequency-checkpoint", type=Path)
    parser.add_argument("--skeleton-checkpoint", type=Path)
    parser.add_argument("--complete-checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--qa-csv", type=Path, required=True)
    parser.add_argument("--logits", type=Path, nargs=2, required=True)
    parser.add_argument("--window-features", type=Path, nargs=3, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.output_dir.exists() or not 1 <= args.workers <= 3:
        raise ValueError("output must be new and workers within1..3")
    qa = question_metadata(pd.read_csv(args.qa_csv, usecols=QUESTION_COLUMNS, dtype=str, keep_default_na=False))
    inputs = [args.qa_csv, *args.logits, *args.window_features]
    if args.operation == "export":
        if (args.checkpoint or not all((args.presence_checkpoint, args.frequency_checkpoint,
                                       args.skeleton_checkpoint)) or len(args.complete_checkpoint) != 3):
            raise ValueError("export requires presence/frequency/Skeleton and three complete-set checkpoints")
        paths = {"presence": args.presence_checkpoint, "frequency": args.frequency_checkpoint,
                 "skeleton": args.skeleton_checkpoint,
                 **{f"complete_{i}": p for i, p in enumerate(args.complete_checkpoint)}}
        _, contracts = load_logits(args.logits, 4.0)
        manifest = {"kind": KIND, "source_hashes": sources(),
            "models": {name: {"path": os.path.relpath(p.resolve(), args.output_dir.resolve()),
                              "sha256": sha256(p)} for name, p in paths.items()},
            "layout": {"groups": repeated_recording_groups(qa)}, "logit_contracts": contracts,
            "profile": {"log_odds_weights": BLEND_WEIGHTS, "option_weight": 0.05,
                        "constraints": "unchanged independent local-v2 HAU recipe"},
            "scope": "load-only historical HAU action chain; no new OOF or backbone exclusion claim"}
    else:
        if args.checkpoint is None:
            raise ValueError("infer requires checkpoint.json")
        if sha256(args.checkpoint) != args.checkpoint.with_suffix(".json.sha256").read_text().strip():
            raise ValueError("manifest SHA mismatch")
        manifest = json.loads(args.checkpoint.read_text())
        if manifest["kind"] != KIND or manifest["source_hashes"] != sources():
            raise ValueError("HAU checkpoint identity or sources changed")
        paths = {name: (args.checkpoint.parent / entry["path"]).resolve()
                 for name, entry in manifest["models"].items()}
        inputs.append(args.checkpoint)
    for name, path in paths.items():
        if sha256(path) != manifest["models"][name]["sha256"]:
            raise ValueError(f"checkpoint SHA mismatch: {name}")
    inputs.extend(paths.values())
    args.output_dir.mkdir(parents=True)
    details = infer(qa, paths, args.logits, args.window_features, manifest, args.output_dir, args.workers)
    if args.operation == "export":
        path = args.output_dir / "checkpoint.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        path.with_suffix(".json.sha256").write_text(sha256(path) + "\n")
        for name in SOURCES:
            target = args.output_dir / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
    report = {"operation": args.operation, "kind": KIND, **details,
              "source_hashes": sources(), "inputs": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in inputs],
              "outputs": {p.name: sha256(p) for p in args.output_dir.iterdir() if p.is_file()},
              "limits": ["requires original audited acquisition layout",
                         "features supplied upstream; no raw-media or complete finalist claim",
                         "K-KUNO backbone subject memberships remain unverified"]}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(details, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
