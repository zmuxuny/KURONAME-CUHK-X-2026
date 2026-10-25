#!/usr/bin/env python3
"""Recover final HARn-window classifiers used by HAU action-set inference.

Export repeats only the missing full-training fits with their original seeds.
Inference uses question metadata and Testing Skeleton windows; it requires no
training QA, class targets, cached score archive, or submission vector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from cuhkx.harn import action_class
from cuhkx.harn_nested import OFFICIAL_SUBJECTS
from cuhkx.inference import QUESTION_COLUMNS, question_metadata
from cuhkx.metadata import feature_unit, subject_id
from scripts.extract_hau_skeleton_action_scores import (
    AGGREGATIONS,
    aggregate_unit_scores,
    aligned_probabilities,
    max_features_value,
    unit_weights,
)

KIND = "cuhkx_hau_skeleton_transfer_v1"
SOURCES = (
    "scripts/hau_skeleton_transfer_checkpoint.py", "scripts/extract_hau_skeleton_action_scores.py",
    "src/cuhkx/harn.py", "src/cuhkx/harn_nested.py", "src/cuhkx/inference.py",
    "src/cuhkx/metadata.py", "src/cuhkx/submission.py", "src/cuhkx/actions.py",
    "src/cuhkx/structural.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict:
    return {name: sha256(ROOT / name) for name in SOURCES}


def read_windows(path: Path, expected_names=None, *, inference=False):
    with np.load(path, allow_pickle=False) as archive:
        units = archive["units"].astype(str)
        names = archive["feature_names"].astype(str)
        matrix = archive["X"].astype(np.float32)
        starts, stops = archive["starts"], archive["stops"]
        window_size = int(archive["window_size"]) if "window_size" in archive else int(
            np.max(stops - starts, initial=0)
        )
    if (matrix.shape != (len(units), len(names)) or not np.isfinite(matrix).all()
            or len(starts) != len(units) or len(stops) != len(units)):
        raise ValueError("invalid window feature axes or values")
    if (len(set(zip(units, starts.tolist(), stops.tolist(), strict=True))) != len(units)
            or np.any(starts >= stops) or np.any(stops - starts > window_size)):
        raise ValueError("duplicate or invalid window interval")
    if expected_names is not None and not np.array_equal(names, expected_names):
        raise ValueError("Skeleton window feature order differs from checkpoint")
    if inference:
        selected = np.char.startswith(units, "Testing/")
        units, matrix = units[selected], matrix[selected]
    return matrix, units, names, window_size


def training_indices(units: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = np.flatnonzero(np.char.startswith(units, "Training/HARn/"))
    subjects = {subject_id(unit) for unit in units[rows]}
    if subjects != OFFICIAL_SUBJECTS:
        raise ValueError("final refit requires exactly the 18 official HARn training subjects")
    labels = np.asarray([action_class(unit) for unit in units[rows]])
    if not np.array_equal(np.unique(labels), np.arange(40)):
        raise ValueError("HARn Skeleton window labels must cover classes 0..39")
    return rows, labels


def predict_seed(bundle: dict, questions: pd.DataFrame, banks: list, workers: int):
    questions = question_metadata(questions)
    target = questions.loc[questions.source.eq("HAU")]
    units = np.asarray(sorted({feature_unit(p, training=False) for p in target.path}))
    if not len(units) or len(banks) != len(bundle["models"]):
        raise ValueError("HAU questions and all temporal scales are required")
    shared = set(units)
    for _, axis, _, _ in banks:
        shared &= set(axis)
    available = np.asarray([unit for unit in units if unit in shared], dtype=str)
    missing = sorted(set(units) - shared)
    views, names = [], []
    for index, ((matrix, axis, features, window_size), model) in enumerate(
        zip(banks, bundle["models"], strict=True)
    ):
        if not np.array_equal(features, bundle["feature_names"][index]):
            raise ValueError("Skeleton window feature order differs from checkpoint")
        if window_size != bundle["window_sizes"][index]:
            raise ValueError("Skeleton window scale differs from checkpoint; short-only inputs need window_size metadata")
        if not np.array_equal(model.classes_, np.arange(40)):
            raise ValueError("unexpected action classifier classes")
        model.n_jobs = workers
        rows = np.flatnonzero(np.isin(axis, available))
        if len(rows):
            probabilities = aligned_probabilities(model, matrix[rows], np.arange(40))
            aggregated = aggregate_unit_scores(axis[rows], probabilities, available)
        else:
            aggregated = {name: np.zeros((0, 40), dtype=np.float32) for name in AGGREGATIONS}
        for aggregation in AGGREGATIONS:
            views.append(aggregated[aggregation])
            names.append(f"{bundle['scale_names'][index]}_{aggregation}")
    return available, np.stack(views), np.asarray(names), missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["export", "infer"])
    parser.add_argument("--reference-run", type=Path, action="append", default=[])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--features", type=Path, nargs="+", required=True)
    parser.add_argument("--qa-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.output_dir.exists() or not 1 <= args.workers <= 3 or len(args.features) != 3:
        raise ValueError("new output, one to three workers, and three ordered scales required")
    questions = question_metadata(pd.read_csv(args.qa_csv, usecols=QUESTION_COLUMNS, dtype=str,
                                              keep_default_na=False))
    inputs = [args.qa_csv, *args.features]
    banks = [read_windows(path, inference=args.operation == "infer") for path in args.features]
    args.output_dir.mkdir(parents=True)
    model_entries, comparisons, cubes, common_units = [], {}, [], None
    if args.operation == "export":
        if len(args.reference_run) != 3 or args.checkpoint is not None:
            raise ValueError("export requires exactly three seed reference runs")
        jobs = [(path, None) for path in args.reference_run]
    else:
        if args.checkpoint is None or args.reference_run:
            raise ValueError("infer requires only an exported checkpoint manifest")
        if sha256(args.checkpoint) != args.checkpoint.with_suffix(".json.sha256").read_text().strip():
            raise ValueError("checkpoint manifest SHA mismatch")
        manifest = json.loads(args.checkpoint.read_text())
        if manifest["kind"] != KIND or manifest["source_hashes"] != source_hashes():
            raise ValueError("checkpoint identity or sources changed")
        jobs = [(None, entry) for entry in manifest["models"]]
        inputs.append(args.checkpoint)
    for reference, entry in jobs:
        if args.operation == "export":
            report_path = reference / "report.json"
            source = json.loads(report_path.read_text())
            config = source["config"]
            expected = source["feature_archives"]
            if [sha256(path) for path in args.features] != [item["sha256"] for item in expected]:
                raise ValueError("refit feature archives differ from original run")
            models, training = [], []
            for index, (matrix, axis, _, _) in enumerate(banks):
                rows, labels = training_indices(axis)
                model = ExtraTreesClassifier(
                    n_estimators=config["trees"], max_features=max_features_value(config["max_features"]),
                    min_samples_leaf=config["min_samples_leaf"], class_weight="balanced",
                    n_jobs=args.workers, random_state=config["seed"] + 1000 * index,
                )
                weights = unit_weights(axis[rows]) if config["balance_units"] else None
                with threadpool_limits(limits=args.workers):
                    model.fit(matrix[rows], labels, sample_weight=weights)
                models.append(model)
                training.append({"scale": args.features[index].stem, "window_rows": rows.tolist(),
                                 "subjects": sorted(OFFICIAL_SUBJECTS),
                                 "unit_labels_sha256": hashlib.sha256(json.dumps(
                                     list(zip(axis[rows].tolist(), labels.tolist(), strict=True)),
                                     separators=(",", ":"),
                                 ).encode()).hexdigest()})
                print(f"refit_seed={config['seed']} scale={index + 1}/3", flush=True)
            bundle = {"kind": KIND, "config": config, "models": models,
                      "feature_names": [item[2] for item in banks],
                      "window_sizes": [item[3] for item in banks],
                      "scale_names": [path.stem for path in args.features], "training": training,
                      "source_hashes": source_hashes()}
            inputs.append(report_path)
        else:
            weights_path = (args.checkpoint.parent / entry["path"]).resolve()
            if sha256(weights_path) != entry["sha256"]:
                raise ValueError("Skeleton transfer model SHA mismatch")
            bundle = joblib.load(weights_path)
            if bundle["kind"] != KIND or bundle["source_hashes"] != source_hashes():
                raise ValueError("Skeleton transfer model identity or sources changed")
            inputs.append(weights_path)
        units, scores, view_names, missing = predict_seed(bundle, questions, banks, args.workers)
        if common_units is not None and not np.array_equal(units, common_units):
            raise ValueError("seed experts have different unit coverage")
        common_units = units
        seed = bundle["config"]["seed"]
        if args.operation == "export":
            score_path = reference / "action_scores.npz"
            with np.load(score_path, allow_pickle=False) as archive:
                if (not np.array_equal(archive["test_units"].astype(str), units)
                        or not np.array_equal(archive["view_names"].astype(str), view_names)):
                    raise ValueError("refit unit/view axes differ from frozen evidence")
                delta = float(np.max(np.abs(archive["test_scores"].astype(np.float32) - scores)))
            if delta > 1e-7:
                raise ValueError(f"refit does not reproduce frozen action scores: {delta}")
            comparisons[str(seed)] = {"score_values": int(scores.size), "max_abs_error": delta}
            model_path = args.output_dir / f"models_seed{seed}.joblib"
            joblib.dump(bundle, model_path, compress=3)
            model_entries.append({"seed": seed, "path": os.path.relpath(model_path, args.output_dir),
                                  "sha256": sha256(model_path), "bytes": model_path.stat().st_size})
            inputs.append(score_path)
        np.savez_compressed(args.output_dir / f"scores_seed{seed}.npz", units=units, scores=scores,
                            view_names=view_names, classes=np.arange(40))
        cubes.append(scores)
        del bundle
    combined = np.mean(np.stack([cube.mean(axis=0) for cube in cubes]), axis=0).astype(np.float32)
    np.savez_compressed(args.output_dir / "action_scores.npz", units=common_units, scores=combined,
                        classes=np.arange(40))
    if args.operation == "export":
        manifest = {"kind": KIND, "source_hashes": source_hashes(), "models": model_entries,
                    "verification": comparisons, "scope": "recovered final weights, no new OOF search",
                    "score_combination": "float32 mean over12 views per seed, then float32 mean over3 seeds"}
        path = args.output_dir / "checkpoint.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        path.with_suffix(".json.sha256").write_text(sha256(path) + "\n")
        for name in SOURCES:
            target = args.output_dir / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
    report = {"operation": args.operation, "kind": KIND, "test_units": len(common_units),
              "missing_units": missing, "verification": comparisons, "source_hashes": source_hashes(),
              "inputs": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in inputs],
              "outputs": {p.name: sha256(p) for p in args.output_dir.iterdir() if p.is_file()},
              "environment": {name: version(name) for name in ("numpy", "scikit-learn", "joblib")}}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"test_units": len(common_units), "verification": comparisons}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
