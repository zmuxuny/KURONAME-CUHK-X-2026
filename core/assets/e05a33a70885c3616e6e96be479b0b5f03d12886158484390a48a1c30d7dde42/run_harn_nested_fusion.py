#!/usr/bin/env python3
"""Train nested-subject HARn fusion, or infer by loading its saved checkpoint.

No public submission is accepted. The optional independent parent OOF file is
read only after predictions and the final model are frozen, for diagnostics.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuhkx.harn_nested import (
    CHECKPOINT_KIND,
    CLASSES,
    OFFICIAL_SUBJECTS,
    QA_COLUMNS,
    FeatureBank,
    decode_questions,
    evaluate_predictions,
    fit_bundle,
    harn_rows,
    predict_evidence,
    select_weight,
    sha256_file,
    training_manifest,
    write_json,
)
from cuhkx.metadata import subject_id

COMPONENTS = [
    "skeleton_extra_full", "skeleton_extra_aggregate", "kinetics_r3d_svc",
    "action_dictionary", "object_dictionary", "option_prior",
]
SOURCE_PATHS = [
    "scripts/run_harn_nested_fusion.py", "src/cuhkx/harn_nested.py", "src/cuhkx/harn.py",
    "src/cuhkx/metadata.py", "src/cuhkx/submission.py", "src/cuhkx/skeleton.py",
    "scripts/extract_skeleton_features.py", "scripts/extract_video_embeddings.py",
    "scripts/extract_kuno_transfer_logits.py", "src/cuhkx/video.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--config", type=Path, default=ROOT / "configs/harn_nested_fusion_v1.json")
    train.add_argument("--output-dir", type=Path, required=True)
    for name in ("train-csv", "test-csv", "sample-csv", "skeleton", "visual", "parent-oof"):
        train.add_argument(f"--{name}", type=Path)
    infer = subparsers.add_parser("infer")
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--skeleton", type=Path, required=True)
    infer.add_argument("--visual", type=Path, required=True)
    infer.add_argument("--qa-csv", type=Path, required=True)
    infer.add_argument("--output-dir", type=Path, required=True)
    infer.add_argument("--workers", type=int, default=3)
    return parser.parse_args()


def read_qa(path: Path, *, labeled: bool) -> pd.DataFrame:
    return pd.read_csv(path, usecols=QA_COLUMNS + (["answer"] if labeled else []),
                       dtype=str, keep_default_na=False)


def manifest_entry(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def fresh_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def save_evidence(path: Path, evidence: dict[str, Any]) -> None:
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in evidence.items()})


def fit_logged(
    bank: FeatureBank, qa: pd.DataFrame, subjects: set[str], config: dict[str, Any],
    output: Path, fit_id: str, inputs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = training_manifest(bank, qa, subjects, config)
    manifest["fit_id"] = fit_id
    manifest["inputs"] = inputs
    path = output / "fits" / f"{fit_id}.json"
    write_json(path, manifest)
    provenance = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    started = time.perf_counter()
    bundle = fit_bundle(bank, qa, subjects, config)
    elapsed = time.perf_counter() - started
    write_json(output / "fits" / f"{fit_id}.runtime.json", {"elapsed_seconds": elapsed})
    print(f"fit={fit_id} subjects={len(subjects)} seconds={elapsed:.2f}", flush=True)
    return bundle, provenance


def upstream(subjects: set[str], provenance: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"name": name, "train_subjects": sorted(subjects), "provenance": provenance}
            for name in COMPONENTS]


def inner_selection(
    bank: FeatureBank, qa: pd.DataFrame, config: dict[str, Any],
    output: Path, outer_id: str, inputs: dict[str, Any],
) -> tuple[float, list[dict[str, Any]], list[dict[str, Any]]]:
    groups = qa["path"].map(subject_id).to_numpy()
    splitter = GroupKFold(n_splits=config["validation"]["inner_splits"])
    all_predictions: dict[float, list[pd.DataFrame]] = {
        float(weight): [] for weight in config["validation"]["skeleton_weights"]
    }
    folds = []
    for index, (fit_rows, val_rows) in enumerate(splitter.split(qa, groups=groups)):
        fitting, validation = qa.iloc[fit_rows].copy(), qa.iloc[val_rows].copy()
        fit_subjects = set(groups[fit_rows])
        validation_subjects = set(groups[val_rows])
        if fit_subjects & validation_subjects:
            raise ValueError("inner subject overlap")
        fit_id = f"{outer_id}_inner{index}"
        bundle, provenance = fit_logged(bank, fitting, fit_subjects, config, output, fit_id, inputs)
        evidence = predict_evidence(bundle, bank, validation[QA_COLUMNS], training_paths=True)
        for weight, parts in all_predictions.items():
            predictions = decode_questions(validation[QA_COLUMNS], evidence, bundle, weight,
                                           training_paths=True)
            predictions["inner_fold"] = index
            parts.append(predictions)
        folds.append({
            "fold_id": fit_id, "train_subjects": sorted(fit_subjects),
            "validation_subjects": sorted(validation_subjects),
            "prediction_qa_ids": validation["qa_id"].tolist(),
            "upstream_training": upstream(fit_subjects, provenance),
        })
        del bundle, evidence
        gc.collect()
    reports = []
    all_frames = []
    for weight, parts in all_predictions.items():
        frame = pd.concat(parts, ignore_index=True)
        reports.append({"skeleton_weight": weight, "metrics": evaluate_predictions(qa, frame)})
        all_frames.append(frame)
    selected = select_weight(reports, config["validation"]["weight_tie_order"])
    pd.concat(all_frames, ignore_index=True).to_csv(
        output / "fits" / f"{outer_id}_inner_predictions.csv", index=False
    )
    write_json(output / "fits" / f"{outer_id}_selection.json", {
        "selected_skeleton_weight": selected, "candidate_reports": reports, "inner_folds": folds,
    })
    print(f"selection={outer_id} skeleton_weight={selected}", flush=True)
    return selected, folds, reports


def diagnostic_comparison(
    qa: pd.DataFrame, predictions: pd.DataFrame, parent_path: Path, output: Path,
) -> dict[str, Any]:
    if parent_path.name == "submission.csv" or "submissions" in parent_path.resolve().parts:
        raise ValueError("a submission file cannot be a diagnostic parent")
    parent = pd.read_csv(parent_path, usecols=["qa_id", "prediction"], dtype=str,
                         keep_default_na=False)
    parent = parent.loc[parent["qa_id"].isin(qa["qa_id"])].copy()
    parent_metrics = evaluate_predictions(qa, parent)
    rows = qa[["qa_id", "path", "category", "answer"]].merge(
        parent.rename(columns={"prediction": "parent_prediction"}), on="qa_id", validate="one_to_one"
    ).merge(predictions[["qa_id", "prediction"]], on="qa_id", validate="one_to_one")
    rows["parent_correct"] = rows["parent_prediction"].eq(rows["answer"])
    rows["correct"] = rows["prediction"].eq(rows["answer"])
    changed = rows["prediction"].ne(rows["parent_prediction"])
    rows.loc[changed].to_csv(output / "parent_changed_rows.csv", index=False)
    corrected = int((~rows["parent_correct"] & rows["correct"]).sum())
    harmed = int((rows["parent_correct"] & ~rows["correct"]).sum())
    return {
        "role": "post-freeze diagnostic only; parent is not nested and has previously exposed subjects",
        "input": manifest_entry(parent_path), "parent_metrics": parent_metrics,
        "changed_rows": int(changed.sum()), "corrected": corrected, "harmed": harmed,
        "net_correct": corrected - harmed,
        "changed_accuracy": float(rows.loc[changed, "correct"].mean()) if changed.any() else None,
        "changed_parent_accuracy": float(rows.loc[changed, "parent_correct"].mean())
        if changed.any() else None,
    }


def checkpoint_save(path: Path, bundle: dict[str, Any]) -> None:
    joblib.dump(bundle, path, compress=3)
    path.with_suffix(path.suffix + ".sha256").write_text(sha256_file(path) + "\n", encoding="ascii")


def checkpoint_load(path: Path, *, workers: int) -> dict[str, Any]:
    if not 1 <= workers <= 3:
        raise ValueError("workers must be within 1..3")
    expected = path.with_suffix(path.suffix + ".sha256").read_text(encoding="ascii").strip()
    if len(expected) != 64 or sha256_file(path) != expected:
        raise ValueError("checkpoint SHA256 mismatch")
    bundle = joblib.load(path)
    if bundle.get("kind") != CHECKPOINT_KIND or bundle.get("status") != "diagnostic_only":
        raise ValueError("unexpected checkpoint schema/status")
    if "selected_skeleton_weight" not in bundle:
        raise ValueError("checkpoint lacks frozen final selection")
    for model in bundle["models"].values():
        model.n_jobs = workers
    return bundle


def train_main(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not config["frozen_before_experiments"] or set(config["subjects"]) != OFFICIAL_SUBJECTS:
        raise ValueError("configuration must freeze the official 18-subject protocol")
    if not 1 <= config["workers"] <= 3:
        raise ValueError("workers must be within 1..3")
    paths = {key: ROOT / value for key, value in config["input_defaults"].items()}
    for name in ("train_csv", "test_csv", "sample_csv", "skeleton", "visual", "parent_oof"):
        if getattr(args, name) is not None:
            paths[name] = getattr(args, name).resolve()
    output = args.output_dir.resolve()
    fresh_output(output)
    write_json(output / "config.json", config)
    constraints_path = ROOT / config["global_constraints"]
    constraints = json.loads(constraints_path.read_text(encoding="utf-8"))
    # The parent is deliberately not opened or hashed until generated outputs are frozen.
    inputs = {name: manifest_entry(path) for name, path in paths.items() if name != "parent_oof"}
    inputs["config"] = manifest_entry(args.config)
    inputs["global_constraints"] = manifest_entry(constraints_path)
    source_manifest = [manifest_entry(ROOT / path) for path in SOURCE_PATHS]
    write_json(output / "input_manifest.json", inputs)
    write_json(output / "source_manifest.json", source_manifest)
    by_path = {item["path"]: item["sha256"] for item in inputs.values()}
    for expected in constraints["inputs"]:
        expected_path = str((ROOT / expected["path"]).resolve())
        if expected_path in by_path and by_path[expected_path] != expected["sha256"]:
            raise ValueError(f"global input SHA mismatch: {expected_path}")
    metadata = json.loads(paths["visual_metadata"].read_text(encoding="utf-8"))
    if metadata["sha256"] != inputs["visual"]["sha256"]:
        raise ValueError("R3D embedding SHA differs from extractor report")
    train = harn_rows(read_qa(paths["train_csv"], labeled=True))
    full_test = read_qa(paths["test_csv"], labeled=False)
    test = harn_rows(full_test)
    sample_ids = pd.read_csv(paths["sample_csv"], usecols=["qa_id"], dtype=str)["qa_id"].tolist()
    if sample_ids != full_test["qa_id"].tolist():
        raise ValueError("official sample and test QA ID order mismatch")
    subjects = train["path"].map(subject_id)
    if set(subjects) != OFFICIAL_SUBJECTS:
        raise ValueError("training QA subject set mismatch")
    bank = FeatureBank.load(paths["skeleton"], paths["visual"])
    oof_parts, outer_folds, selections, probability_parts = [], [], [], []
    for outer_subject in config["subjects"]:
        fit_qa = train.loc[subjects.ne(outer_subject)].copy()
        validation = train.loc[subjects.eq(outer_subject)].copy()
        fit_subjects = set(config["subjects"]) - {outer_subject}
        outer_id = f"outer_{outer_subject}"
        weight, inner_folds, candidate_reports = inner_selection(
            bank, fit_qa, config, output, outer_id, inputs
        )
        bundle, provenance = fit_logged(bank, fit_qa, fit_subjects, config, output,
                                        f"{outer_id}_final", inputs)
        evidence = predict_evidence(bundle, bank, validation[QA_COLUMNS], training_paths=True)
        prediction = decode_questions(validation[QA_COLUMNS], evidence, bundle, weight,
                                      training_paths=True)
        prediction["outer_fold"] = outer_id
        oof_parts.append(prediction)
        save_evidence(output / "fits" / f"{outer_id}_evidence.npz", evidence)
        probability_parts.append({
            key: value for key, value in evidence.items() if not key.endswith("_classes")
        })
        outer_folds.append({
            "fold_id": outer_id, "train_subjects": sorted(fit_subjects),
            "validation_subjects": [outer_subject], "prediction_qa_ids": validation["qa_id"].tolist(),
            "upstream_training": upstream(fit_subjects, provenance), "inner_folds": inner_folds,
        })
        selections.append({
            "outer_subject": outer_subject, "selected_skeleton_weight": weight,
            "inner_candidate_reports": candidate_reports,
            "outer_metrics": evaluate_predictions(validation, prediction),
        })
        pd.concat(oof_parts, ignore_index=True).to_csv(output / "oof_partial.csv", index=False)
        write_json(output / "outer_progress.json", selections)
        print(f"outer_complete={outer_subject} accuracy={selections[-1]['outer_metrics']['accuracy']:.5f}"
              f" elapsed_seconds={time.perf_counter() - started:.1f}", flush=True)
        del bundle, evidence
        gc.collect()
    oof = pd.concat(oof_parts, ignore_index=True).set_index("qa_id", drop=False)
    oof = oof.loc[train["qa_id"]].reset_index(drop=True)
    oof.to_csv(output / "oof_predictions.csv", index=False)
    save_evidence(output / "oof_probabilities.npz", {
        **{key: np.concatenate([part[key] for part in probability_parts])
           for key in probability_parts[0]}, "classes": CLASSES,
    })
    final_weight, final_inner, final_reports = inner_selection(
        bank, train, config, output, "final_all_subjects", inputs
    )
    bundle, final_provenance = fit_logged(bank, train, OFFICIAL_SUBJECTS, config, output,
                                          "final_all_subjects_fit", inputs)
    bundle["selected_skeleton_weight"] = final_weight
    bundle["training_provenance"] = final_provenance
    bundle["input_manifest"] = inputs
    bundle["source_manifest"] = source_manifest
    evidence = predict_evidence(bundle, bank, test[QA_COLUMNS], training_paths=False)
    predictions = decode_questions(test[QA_COLUMNS], evidence, bundle, final_weight,
                                   training_paths=False)
    predictions.to_csv(output / "test_predictions.csv", index=False)
    save_evidence(output / "test_probabilities.npz", evidence)
    checkpoint_save(output / "models.joblib", bundle)
    protocol = {
        "schema_version": 1, "validation_kind": "nested_subject",
        "fitted_components": COMPONENTS, "folds": outer_folds,
        "historical_subject_exposure": config["validation"]["historical_subject_exposure"],
        "feature_pretraining": {
            "visual": "torchvision Kinetics400 generic R3D, no CUHK-X fitting",
            "crop": config["visual"]["crop_provenance"],
            "skeleton": "deterministic statistics of official Skeleton JSON; no learned extractor",
        },
    }
    write_json(output / "protocol.json", protocol)
    generated = {name: manifest_entry(output / name) for name in (
        "oof_predictions.csv", "test_predictions.csv", "oof_probabilities.npz",
        "test_probabilities.npz", "models.joblib", "protocol.json",
    )}
    write_json(output / "generation_frozen.json", {
        "frozen_utc": datetime.now(UTC).isoformat(), "outputs": generated,
        "parent_read_yet": False,
    })
    parent_comparison = diagnostic_comparison(train, oof, paths["parent_oof"], output)
    for item in [*inputs.values(), *source_manifest]:
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"input/source changed during run: {item['path']}")
    report = {
        "status": "diagnostic_only", "validation_kind": "nested_subject",
        "public_prediction_inputs": [], "metrics": evaluate_predictions(train, oof),
        "outer_selections": selections,
        "selection_counts": {str(weight): sum(item["selected_skeleton_weight"] == weight
                                              for item in selections)
                             for weight in config["validation"]["skeleton_weights"]},
        "final_skeleton_weight": final_weight, "final_inner_reports": final_reports,
        "final_inner_folds": final_inner, "parent_comparison": parent_comparison,
        "oof_evidence_source_counts": oof["evidence_source"].value_counts().to_dict(),
        "test_evidence_source_counts": predictions["evidence_source"].value_counts().to_dict(),
        "test_rows": len(predictions), "test_raw_missing_skeleton_rows": int((~predictions["skeleton_present"]).sum()),
        "test_raw_missing_visual_rows": int((~predictions["visual_present"]).sum()),
        "inputs": inputs, "sources": source_manifest, "generated": generated,
        "runtime_seconds": time.perf_counter() - started,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "sklearn": sklearn.__version__,
                        "joblib": joblib.__version__},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                               text=True).strip(),
        "limitations": [
            config["validation"]["historical_subject_exposure"],
            "SVC uses fixed internal Platt calibration within fitting subjects; inner selection and outer evaluation remain subject-disjoint",
            "Final checkpoint expects deterministic Skeleton feature names and generic Kinetics R3D Depth+IR embeddings; raw extraction is upstream",
            "Missing Depth or IR makes the visual family unavailable; missing all usable features uses training-only option priors",
            "Only HARn questions are emitted; this is not a complete competition submission",
        ],
    }
    write_json(output / "report.json", report)
    print(json.dumps({key: report[key] for key in (
        "status", "metrics", "selection_counts", "final_skeleton_weight", "parent_comparison",
        "runtime_seconds",
    )}, indent=2), flush=True)
    return 0


def infer_main(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    fresh_output(args.output_dir)
    bundle = checkpoint_load(args.checkpoint, workers=args.workers)
    bank = FeatureBank.load(args.skeleton, args.visual)
    qa = harn_rows(read_qa(args.qa_csv, labeled=False))
    evidence = predict_evidence(bundle, bank, qa, training_paths=False)
    predictions = decode_questions(qa, evidence, bundle, bundle["selected_skeleton_weight"],
                                   training_paths=False)
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    save_evidence(args.output_dir / "test_probabilities.npz", evidence)
    write_json(args.output_dir / "report.json", {
        "status": "diagnostic_only", "mode": "load_only", "rows": len(predictions),
        "selected_skeleton_weight": bundle["selected_skeleton_weight"],
        "inputs": [manifest_entry(path) for path in (
            args.checkpoint, args.checkpoint.with_suffix(args.checkpoint.suffix + ".sha256"),
            args.skeleton, args.visual, args.qa_csv,
        )],
        "outputs": [manifest_entry(args.output_dir / name) for name in (
            "test_predictions.csv", "test_probabilities.npz",
        )],
        "runtime_seconds": time.perf_counter() - started,
    })
    print(f"load_only_rows={len(predictions)} seconds={time.perf_counter() - started:.2f}", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    workers = getattr(args, "workers", 3)
    with threadpool_limits(limits=min(workers, 3)):
        return train_main(args) if args.command == "train" else infer_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
