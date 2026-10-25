#!/usr/bin/env python3
"""Learn HAU action presence from complete combination-answer action sets.

The combination question names the complete action set performed in a clip.
This script turns that answer into 40 binary targets and cross-fits a
multi-label calibrator by subject.  Frozen K-KUNO temporal logits are the only
input features; anonymous test labels are never used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_hau_kuno_presence import feature_cube
from run_hau_kuno_temporal import load_logit_archives

from cuhkx.actions import HAU_CLASS_TEXT, HAU_TEXT_CLASS
from cuhkx.joint_actions import JOINT_CATEGORIES, decode_joint_state
from cuhkx.metadata import clip_key, feature_unit, option_atoms, subject_id
from cuhkx.submission import read_csv_str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument(
        "--logit-view",
        type=Path,
        nargs="+",
        action="append",
        required=True,
        help=(
            "One compatible train/test archive group. Repeat the flag to concatenate "
            "feature views from multiple temporal resolutions."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--top-windows", type=int, action="append")
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--minimum-leaf", type=int, action="append", default=[])
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_features(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    top_windows = args.top_windows or []
    if top_windows and len(top_windows) != len(args.logit_view):
        raise ValueError("provide one --top-windows value per --logit-view")
    if not top_windows:
        top_windows = [2] * len(args.logit_view)

    unit_reference: np.ndarray | None = None
    views: list[np.ndarray] = []
    reports: list[dict[str, object]] = []
    for paths, top_count in zip(args.logit_view, top_windows, strict=True):
        logits, units, weights, metadata = load_logit_archives(paths)
        features, _ = feature_cube(logits, weights, args.temperature, top_count)
        if unit_reference is None:
            unit_reference = units
        elif not np.array_equal(units, unit_reference):
            raise ValueError("logit views have different unit order")
        views.append(features)
        reports.append(
            {
                "paths": metadata,
                "top_windows": top_count,
                "feature_shape": list(features.shape),
            }
        )
    assert unit_reference is not None
    return np.concatenate(views, axis=2), unit_reference, reports


def complete_targets(
    train: pd.DataFrame,
    unit_to_row: dict[str, int],
) -> tuple[pd.DataFrame, np.ndarray]:
    rows = train.loc[
        train["source"].eq("HAU") & train["category"].eq("combination")
    ].copy()
    rows["unit"] = rows["path"].map(lambda value: feature_unit(value, training=True))
    rows = rows.loc[rows["unit"].isin(unit_to_row)].reset_index(drop=True)
    targets = np.zeros((len(rows), len(HAU_CLASS_TEXT)), dtype=np.uint8)
    for row_index, row in rows.iterrows():
        for atom in option_atoms(row[row["answer"]]):
            targets[row_index, HAU_TEXT_CLASS[atom]] = 1
    rows["subject"] = rows["path"].map(subject_id)
    return rows, targets


def positive_probabilities(
    model: ExtraTreesClassifier,
    matrix: np.ndarray,
) -> np.ndarray:
    result = np.zeros(len(matrix), dtype=np.float32)
    positive = np.flatnonzero(model.classes_ == 1)
    if len(positive):
        result = model.predict_proba(matrix)[:, positive[0]].astype(np.float32)
    return result


def decode_frame(
    qa: pd.DataFrame,
    units: np.ndarray,
    family_scores: dict[str, np.ndarray],
    *,
    training: bool,
) -> pd.DataFrame:
    unit_to_row = {unit: index for index, unit in enumerate(units)}
    work = qa.loc[qa["source"].eq("HAU")].copy()
    work["unit"] = work["path"].map(
        lambda value: feature_unit(value, training=training)
    )
    work = work.loc[work["unit"].isin(unit_to_row)]
    work["_clip"] = work["path"].map(clip_key)
    records: list[dict[str, object]] = []
    for _, group in work.groupby("_clip", sort=False):
        unit = str(group.iloc[0]["unit"])
        source_row = unit_to_row[unit]
        decisions = {}
        for family, scores in family_scores.items():
            probabilities = {
                HAU_CLASS_TEXT[class_id]: float(scores[source_row, class_id])
                for class_id in HAU_CLASS_TEXT
            }
            decisions[family] = decode_joint_state(group, probabilities)
        answer_maps = {
            family: decision.state.answer_map() for family, decision in decisions.items()
        }
        for _, row in group.loc[group["category"].isin(JOINT_CATEGORIES)].iterrows():
            category = str(row["category"])
            predictions = {
                family: answers[category] for family, answers in answer_maps.items()
            }
            record: dict[str, object] = {
                "qa_id": row["qa_id"],
                "unit": unit,
                "category": category,
                "prediction": predictions["mean"],
                "margin": decisions["mean"].margin,
                "agreement_count": max(Counter(predictions.values()).values()),
                **{f"prediction_{name}": value for name, value in predictions.items()},
                **{
                    f"margin_{name}": decision.margin
                    for name, decision in decisions.items()
                },
            }
            if training:
                record["answer"] = row["answer"]
                record["correct"] = record["prediction"] == row["answer"]
                record["subject"] = subject_id(row["path"])
            records.append(record)
    return pd.DataFrame(records)


def accuracy_report(frame: pd.DataFrame, prediction_column: str) -> dict[str, object]:
    correct = frame[prediction_column].eq(frame["answer"])
    by_category = (
        pd.DataFrame({"category": frame["category"], "correct": correct})
        .groupby("category")["correct"]
        .agg(["size", "mean"])
    )
    return {
        "accuracy": float(correct.mean()),
        "worst_subject": float(correct.groupby(frame["subject"]).mean().min()),
        "category": {
            name: {"rows": int(row["size"]), "accuracy": float(row["mean"])}
            for name, row in by_category.iterrows()
        },
    }


def confidence_report(frame: pd.DataFrame, family_count: int) -> list[dict[str, object]]:
    result = []
    for margin in (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50):
        selected = frame["margin"].ge(margin) & frame["agreement_count"].ge(family_count)
        result.append(
            {
                "minimum_margin": margin,
                "rows": int(selected.sum()),
                "accuracy": float(frame.loc[selected, "correct"].mean())
                if selected.any()
                else None,
                "worst_subject": float(
                    frame.loc[selected, "correct"]
                    .groupby(frame.loc[selected, "subject"])
                    .mean()
                    .min()
                )
                if selected.any()
                else None,
            }
        )
    return result


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    leaves = args.minimum_leaf or [2]
    if args.folds < 2 or any(value < 1 for value in leaves):
        raise ValueError("folds must be >=2 and minimum-leaf values must be positive")
    train = read_csv_str(args.train_csv)
    test = read_csv_str(args.test_csv)
    features, archive_units, feature_reports = load_features(args)
    archive_index = {unit: index for index, unit in enumerate(archive_units)}
    labels, targets = complete_targets(train, archive_index)
    label_rows = np.asarray([archive_index[unit] for unit in labels["unit"]])
    groups = labels["subject"].to_numpy()
    splits = list(GroupKFold(args.folds).split(label_rows, groups=groups))
    training_matrix = features[label_rows].reshape(-1, features.shape[-1])
    training_targets = targets.reshape(-1)

    train_units = np.asarray(
        sorted(
            {
                feature_unit(path, training=True)
                for path in train.loc[train["source"].eq("HAU"), "path"]
            }
            & set(archive_units)
        )
    )
    test_units = np.asarray(
        sorted(
            {
                feature_unit(path, training=False)
                for path in test.loc[test["source"].eq("HAU"), "path"]
            }
            & set(archive_units)
        )
    )
    train_rows = np.asarray([archive_index[unit] for unit in train_units])
    test_rows = np.asarray([archive_index[unit] for unit in test_units])
    train_subjects = np.asarray([subject_id(unit) for unit in train_units])

    family_oof: dict[str, np.ndarray] = {}
    family_test: dict[str, np.ndarray] = {}
    fitted: dict[str, ExtraTreesClassifier] = {}
    for family_index, leaf in enumerate(leaves):
        name = f"extra_leaf{leaf}"
        if name in family_oof:
            raise ValueError(f"duplicate minimum-leaf value: {leaf}")
        oof = np.zeros((len(train_units), len(HAU_CLASS_TEXT)), dtype=np.float32)
        for fold, (fit_index, valid_index) in enumerate(splits, start=1):
            held_out = set(groups[valid_index])
            valid_units = np.asarray(
                [index for index, subject in enumerate(train_subjects) if subject in held_out]
            )
            model = ExtraTreesClassifier(
                n_estimators=args.trees,
                max_features="sqrt",
                min_samples_leaf=leaf,
                class_weight="balanced",
                n_jobs=args.workers,
                random_state=args.seed + 1000 * family_index + fold,
            )
            fit_rows = (
                fit_index[:, np.newaxis] * len(HAU_CLASS_TEXT)
                + np.arange(len(HAU_CLASS_TEXT))
            ).ravel()
            model.fit(training_matrix[fit_rows], training_targets[fit_rows])
            oof[valid_units] = positive_probabilities(
                model, features[train_rows[valid_units]].reshape(-1, features.shape[-1])
            ).reshape(len(valid_units), len(HAU_CLASS_TEXT))
            print(
                f"family={name} fold={fold}/{len(splits)} "
                f"held_out={','.join(sorted(held_out))}",
                flush=True,
            )
        family_oof[name] = oof

        model = ExtraTreesClassifier(
            n_estimators=args.trees,
            max_features="sqrt",
            min_samples_leaf=leaf,
            class_weight="balanced",
            n_jobs=args.workers,
            random_state=args.seed + 1000 * family_index,
        )
        model.fit(training_matrix, training_targets)
        fitted[name] = model
        family_test[name] = positive_probabilities(
            model, features[test_rows].reshape(-1, features.shape[-1])
        ).reshape(len(test_rows), len(HAU_CLASS_TEXT))

    family_oof["mean"] = np.mean(np.stack(list(family_oof.values())), axis=0)
    family_test["mean"] = np.mean(np.stack(list(family_test.values())), axis=0)
    oof_frame = decode_frame(train, train_units, family_oof, training=True)
    test_frame = decode_frame(test, test_units, family_test, training=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.output_dir / "action_scores.npz"
    np.savez_compressed(
        score_path,
        view_names=np.asarray(list(family_oof)),
        classes=np.arange(len(HAU_CLASS_TEXT)),
        train_units=train_units,
        train_subjects=train_subjects,
        train_scores=np.stack(list(family_oof.values())),
        test_units=test_units,
        test_scores=np.stack(list(family_test.values())),
    )
    oof_frame.to_csv(
        args.output_dir / "oof_predictions.csv", index=False, lineterminator="\n"
    )
    test_frame.to_csv(
        args.output_dir / "test_predictions.csv", index=False, lineterminator="\n"
    )
    joblib.dump(
        {"models": fitted, "config": vars(args)},
        args.output_dir / "models.joblib",
        compress=3,
    )

    family_names = list(family_oof)
    report = {
        "method": "complete-set K-KUNO action presence calibration",
        "validation_protocol": (
            "six-fold subject-group OOF; targets are the complete action atoms in the labeled "
            "HAU combination answer; every held-out subject is excluded from fitting"
        ),
        "config": {
            "folds": args.folds,
            "trees": args.trees,
            "workers": args.workers,
            "seed": args.seed,
            "minimum_leaf": leaves,
            "temperature": args.temperature,
            "top_windows": args.top_windows or [2] * len(args.logit_view),
        },
        "labeled_clips": len(labels),
        "mean_actions_per_clip": float(targets.sum(axis=1).mean()),
        "train_units": len(train_units),
        "test_units": len(test_units),
        "feature_shape": list(features.shape),
        "feature_views": feature_reports,
        "validation": {
            name: accuracy_report(oof_frame, f"prediction_{name}")
            for name in family_names
        },
        "confidence": confidence_report(oof_frame, len(family_names)),
        "score_archive": {
            "path": str(score_path.resolve()),
            "sha256": sha256_file(score_path),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
