#!/usr/bin/env python3
"""Cross-fit HARn Skeleton action models and score HAU clips.

The action models see HARn labels only.  For every training HAU subject, the
corresponding HARn subject is removed before scoring that subject's HAU clips.
This produces leakage-safe action-presence features for a downstream HAU
calibrator while retaining a final model trained on every HARn subject for
anonymous test clips.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuhkx.actions import hau_action_class
from cuhkx.harn import action_class
from cuhkx.metadata import feature_unit, subject_id
from cuhkx.submission import available_letters, read_csv_str

AGGREGATIONS = ("max", "mean", "top3", "q90")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--features", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--max-features", default="sqrt")
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--balance-units", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def max_features_value(value: str) -> str | float:
    if value in {"sqrt", "log2"}:
        return value
    result = float(value)
    if not 0 < result <= 1:
        raise ValueError("numeric max-features must be in (0, 1]")
    return result


def unit_weights(units: np.ndarray) -> np.ndarray:
    counts = Counter(units)
    return np.asarray([1.0 / counts[unit] for unit in units], dtype=np.float64)


def aligned_probabilities(
    model: ExtraTreesClassifier, matrix: np.ndarray, classes: np.ndarray
) -> np.ndarray:
    raw = model.predict_proba(matrix)
    result = np.zeros((len(matrix), len(classes)), dtype=np.float32)
    target_columns = {int(value): index for index, value in enumerate(classes)}
    for source_column, value in enumerate(model.classes_):
        result[:, target_columns[int(value)]] = raw[:, source_column]
    return result


def aggregate_unit_scores(
    row_units: np.ndarray,
    probabilities: np.ndarray,
    target_units: np.ndarray,
) -> dict[str, np.ndarray]:
    """Aggregate window probabilities into fixed per-unit action scores."""

    result = {
        name: np.zeros((len(target_units), probabilities.shape[1]), dtype=np.float32)
        for name in AGGREGATIONS
    }
    row_lookup: dict[str, list[int]] = {}
    for row_index, unit in enumerate(row_units):
        row_lookup.setdefault(str(unit), []).append(row_index)
    for target_index, unit in enumerate(target_units):
        rows = row_lookup.get(str(unit))
        if not rows:
            raise ValueError(f"no window rows for target unit: {unit}")
        values = probabilities[np.asarray(rows)]
        result["max"][target_index] = values.max(axis=0)
        result["mean"][target_index] = values.mean(axis=0)
        result["top3"][target_index] = np.sort(values, axis=0)[-min(3, len(values)) :].mean(
            axis=0
        )
        result["q90"][target_index] = np.quantile(values, 0.90, axis=0)
    return result


def single_action_report(
    qa: pd.DataFrame,
    units: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
) -> dict[str, object]:
    unit_to_row = {unit: index for index, unit in enumerate(units)}
    class_to_column = {int(value): index for index, value in enumerate(classes)}
    records: list[dict[str, object]] = []
    for _, row in qa.loc[
        qa["source"].eq("HAU") & qa["category"].eq("single")
    ].iterrows():
        unit = feature_unit(row["path"], training=True)
        if unit not in unit_to_row:
            continue
        option_scores = {
            letter: float(
                scores[
                    unit_to_row[unit],
                    class_to_column[hau_action_class(row[letter])],
                ]
            )
            for letter in available_letters(row)
        }
        ranked = sorted(option_scores, key=lambda letter: (-option_scores[letter], letter))
        records.append(
            {
                "subject": subject_id(row["path"]),
                "correct": ranked[0] == row["answer"],
                "margin": option_scores[ranked[0]] - option_scores[ranked[1]],
            }
        )
    frame = pd.DataFrame(records)
    confidence = []
    for threshold in (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30):
        selected = frame["margin"].ge(threshold)
        confidence.append(
            {
                "minimum_margin": threshold,
                "rows": int(selected.sum()),
                "accuracy": float(frame.loc[selected, "correct"].mean())
                if selected.any()
                else None,
                "subjects": int(frame.loc[selected, "subject"].nunique()),
            }
        )
    subject_scores = frame.groupby("subject")["correct"].mean()
    return {
        "rows": len(frame),
        "accuracy": float(frame["correct"].mean()),
        "worst_subject": float(subject_scores.min()),
        "confidence": confidence,
    }


def target_units(frame: pd.DataFrame, *, training: bool) -> np.ndarray:
    rows = frame.loc[frame["source"].eq("HAU")]
    return np.asarray(
        sorted(
            {
                feature_unit(value, training=training)
                for value in rows["path"]
            }
        )
    )


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    if args.folds < 2:
        raise ValueError("fold count must be at least two")
    train = read_csv_str(args.train_csv)
    test = read_csv_str(args.test_csv)
    requested_train_units = target_units(train, training=True)
    requested_test_units = target_units(test, training=False)

    archive_units: list[set[str]] = []
    for path in args.features:
        with np.load(path, allow_pickle=False) as archive:
            archive_units.append(set(archive["units"].astype(str)))
    common = set.intersection(*archive_units)
    train_units = np.asarray([unit for unit in requested_train_units if unit in common])
    test_units = np.asarray([unit for unit in requested_test_units if unit in common])
    train_subjects = np.asarray([subject_id(unit) for unit in train_units])
    if any(subject is None for subject in train_subjects):
        raise ValueError("a HAU training unit has no subject")
    splits = list(
        GroupKFold(n_splits=args.folds).split(train_units, groups=train_subjects)
    )

    train_views: list[np.ndarray] = []
    test_views: list[np.ndarray] = []
    view_names: list[str] = []
    archive_reports: list[dict[str, object]] = []
    classes: np.ndarray | None = None
    for archive_index, path in enumerate(args.features):
        with np.load(path, allow_pickle=False) as archive:
            matrix = archive["X"]
            row_units = archive["units"].astype(str)
            harn_rows = np.flatnonzero(np.char.startswith(row_units, "Training/HARn/"))
            harn_labels = np.asarray([action_class(row_units[index]) for index in harn_rows])
            harn_groups = np.asarray([subject_id(row_units[index]) for index in harn_rows])
            archive_classes = np.unique(harn_labels)
            if classes is None:
                classes = archive_classes
            elif not np.array_equal(classes, archive_classes):
                raise ValueError(f"HARn classes differ across archives: {path}")

            train_score_cube = np.zeros(
                (len(train_units), len(archive_classes)), dtype=np.float32
            )
            for fold, (_, valid_index) in enumerate(splits, start=1):
                held_out = set(train_subjects[valid_index])
                fit_mask = np.asarray([group not in held_out for group in harn_groups])
                model = ExtraTreesClassifier(
                    n_estimators=args.trees,
                    max_features=max_features_value(args.max_features),
                    min_samples_leaf=args.min_samples_leaf,
                    class_weight="balanced",
                    n_jobs=args.workers,
                    random_state=args.seed + 1000 * archive_index + fold,
                )
                weights = unit_weights(row_units[harn_rows][fit_mask]) if args.balance_units else None
                model.fit(
                    matrix[harn_rows][fit_mask],
                    harn_labels[fit_mask],
                    sample_weight=weights,
                )
                held_units = train_units[valid_index]
                held_set = set(held_units)
                held_rows = np.asarray(
                    [index for index, unit in enumerate(row_units) if unit in held_set]
                )
                probabilities = aligned_probabilities(
                    model, matrix[held_rows], archive_classes
                )
                aggregated = aggregate_unit_scores(
                    row_units[held_rows], probabilities, held_units
                )
                # Store each aggregation temporarily in a cube allocated on its first fold.
                if fold == 1:
                    fold_aggregates = {
                        name: np.zeros_like(train_score_cube) for name in AGGREGATIONS
                    }
                for name in AGGREGATIONS:
                    fold_aggregates[name][valid_index] = aggregated[name]
                print(
                    f"archive={path.stem} fold={fold}/{len(splits)} "
                    f"held_out={','.join(sorted(held_out))}",
                    flush=True,
                )

            final_model = ExtraTreesClassifier(
                n_estimators=args.trees,
                max_features=max_features_value(args.max_features),
                min_samples_leaf=args.min_samples_leaf,
                class_weight="balanced",
                n_jobs=args.workers,
                random_state=args.seed + 1000 * archive_index,
            )
            final_weights = unit_weights(row_units[harn_rows]) if args.balance_units else None
            final_model.fit(
                matrix[harn_rows], harn_labels, sample_weight=final_weights
            )
            test_set = set(test_units)
            test_rows = np.asarray(
                [index for index, unit in enumerate(row_units) if unit in test_set]
            )
            test_probabilities = aligned_probabilities(
                final_model, matrix[test_rows], archive_classes
            )
            test_aggregates = aggregate_unit_scores(
                row_units[test_rows], test_probabilities, test_units
            )
            for name in AGGREGATIONS:
                train_views.append(fold_aggregates[name])
                test_views.append(test_aggregates[name])
                view_names.append(f"{path.stem}_{name}")
            archive_reports.append(
                {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "shape": list(matrix.shape),
                    "harn_windows": len(harn_rows),
                }
            )

    if classes is None:
        raise ValueError("no feature archives were provided")
    train_scores = np.stack(train_views)
    test_scores = np.stack(test_views)
    individual = {
        name: single_action_report(train, train_units, score, classes)
        for name, score in zip(view_names, train_scores, strict=True)
    }
    mean_report = single_action_report(
        train, train_units, train_scores.mean(axis=0), classes
    )
    report = {
        "method": "HARn-only Skeleton window action transfer to HAU",
        "validation_protocol": (
            "six-fold HAU subject-group cross-fit; each fold removes held-out subjects from "
            "HARn before scoring their HAU clips; no HAU label is used by the action models"
        ),
        "config": {
            "folds": args.folds,
            "trees": args.trees,
            "workers": args.workers,
            "seed": args.seed,
            "max_features": args.max_features,
            "min_samples_leaf": args.min_samples_leaf,
            "balance_units": args.balance_units,
            "aggregations": list(AGGREGATIONS),
        },
        "train_units": len(train_units),
        "missing_train_units": sorted(set(requested_train_units) - set(train_units)),
        "test_units": len(test_units),
        "missing_test_units": sorted(set(requested_test_units) - set(test_units)),
        "classes": classes.tolist(),
        "views": view_names,
        "individual_single_validation": individual,
        "mean_view_single_validation": mean_report,
        "feature_archives": archive_reports,
        "runtime_seconds": time.perf_counter() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.output_dir / "action_scores.npz"
    np.savez_compressed(
        score_path,
        view_names=np.asarray(view_names),
        classes=classes,
        train_units=train_units,
        train_subjects=train_subjects,
        train_scores=train_scores,
        test_units=test_units,
        test_scores=test_scores,
    )
    report["score_archive"] = {
        "path": str(score_path.resolve()),
        "sha256": sha256_file(score_path),
        "train_shape": list(train_scores.shape),
        "test_shape": list(test_scores.shape),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
