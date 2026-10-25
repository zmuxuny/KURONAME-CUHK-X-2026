#!/usr/bin/env python3
"""Decode HAU action questions from time-resolved K-KUNO transfer logits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuhkx.actions import hau_action_class
from cuhkx.metadata import feature_unit, option_atoms, subject_id
from cuhkx.submission import available_letters, read_csv_str, validate_submission_files
from cuhkx.temporal import (
    normalized_score_margin,
    softmax,
    temporal_order_scores,
    temporal_presence,
)

ACTION_CATEGORIES = {"single", "multi", "combination", "sequence"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--sample-csv", type=Path, required=True)
    parser.add_argument(
        "--logits",
        type=Path,
        nargs="+",
        required=True,
        help="one or more compatible temporal-logit archives to concatenate by unit",
    )
    parser.add_argument("--parent-submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--top-windows", type=int, default=2)
    parser.add_argument("--multi-cardinality", type=int, default=1)
    parser.add_argument("--sequence-raw-time", action="store_true")
    parser.add_argument("--override-categories", default="sequence")
    parser.add_argument("--override-min-margin", type=float, default=0.20)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_logit_archives(
    paths: list[Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Load compatible archives and concatenate their unit axis."""

    logit_parts = []
    unit_parts = []
    metadata = []
    reference_weights: np.ndarray | None = None
    reference_tail: tuple[int, int, int] | None = None
    seen_units: set[str] = set()
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            logits = archive["logits"].astype(np.float32)
            units = archive["units"].astype(str)
            weights = archive["weights"].astype(np.float32)
        if logits.ndim != 4 or logits.shape[0] != len(weights) or logits.shape[-1] != 40:
            raise ValueError(f"{path}: expected [fold, unit, segment, 40] temporal logits")
        if logits.shape[1] != len(units):
            raise ValueError(f"{path}: unit count does not match the logit unit axis")
        tail = (logits.shape[0], logits.shape[2], logits.shape[3])
        if reference_tail is None:
            reference_tail = tail
            reference_weights = weights
        elif tail != reference_tail or not np.allclose(weights, reference_weights):
            raise ValueError(f"{path}: incompatible folds, segments, classes, or weights")
        duplicates = sorted(seen_units.intersection(units.tolist()))
        if duplicates:
            raise ValueError(f"{path}: duplicate units across archives: {duplicates[:3]}")
        seen_units.update(units.tolist())
        logit_parts.append(logits)
        unit_parts.append(units)
        metadata.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "shape": list(logits.shape),
            }
        )
    assert reference_weights is not None
    return (
        np.concatenate(logit_parts, axis=1),
        np.concatenate(unit_parts),
        reference_weights,
        metadata,
    )


def action_option_scores(row: pd.Series, presence: np.ndarray) -> dict[str, float]:
    scores = {}
    for letter in available_letters(row):
        atoms = option_atoms(row[letter]) if row["category"] == "combination" else (row[letter],)
        classes = [hau_action_class(atom) for atom in atoms]
        values = np.asarray([presence[class_id] for class_id in classes]).clip(1e-12)
        scores[letter] = float(np.log(values).mean())
    return scores


def decode_multi(scores: dict[str, float], cardinality: int) -> tuple[str, float, float]:
    if not 1 <= cardinality <= len(scores):
        raise ValueError("multi-answer cardinality is outside the available options")
    labels = sorted(scores)
    probabilities = softmax(np.asarray([scores[label] for label in labels]))
    ranked = sorted(zip(labels, probabilities, strict=True), key=lambda item: (-item[1], item[0]))
    selected = sorted(label for label, _ in ranked[:cardinality])
    if cardinality < len(ranked):
        margin = float(ranked[cardinality - 1][1] - ranked[cardinality][1])
    else:
        margin = float(min(value for _, value in ranked))
    return "".join(selected), float(sum(value for _, value in ranked[:cardinality])), margin


def decode_row(
    row: pd.Series,
    probabilities: np.ndarray,
    top_windows: int,
    multi_cardinality: int,
    *,
    normalize_sequence_time: bool,
) -> tuple[str, float, float]:
    if row["category"] == "sequence":
        classes = {letter: hau_action_class(row[letter]) for letter in available_letters(row)}
        scores = temporal_order_scores(
            probabilities, classes, normalize_time=normalize_sequence_time
        )
        return normalized_score_margin(scores)
    presence = temporal_presence(probabilities, top_windows)
    scores = action_option_scores(row, presence)
    if row["category"] == "multi":
        return decode_multi(scores, multi_cardinality)
    return normalized_score_margin(scores)


def prediction_frame(
    qa: pd.DataFrame,
    unit_to_row: dict[str, int],
    fold_probabilities: np.ndarray,
    weights: np.ndarray,
    args: argparse.Namespace,
    *,
    training: bool,
    parent_map: pd.Series | None = None,
    oracle_multi_cardinality: bool = False,
) -> pd.DataFrame:
    records = []
    normalized_weights = weights / weights.sum()
    ensemble = np.tensordot(normalized_weights, fold_probabilities, axes=(0, 0))
    for _, row in qa.iterrows():
        if row["source"] != "HAU" or row["category"] not in ACTION_CATEGORIES:
            continue
        unit = feature_unit(row["path"], training=training)
        if unit not in unit_to_row:
            continue
        source_row = unit_to_row[unit]
        parent = None if parent_map is None else parent_map.get(row["qa_id"])
        if row["category"] == "multi":
            if oracle_multi_cardinality and training:
                cardinality = len(row["answer"])
            elif parent is not None:
                cardinality = len(str(parent))
            else:
                cardinality = args.multi_cardinality
        else:
            cardinality = 1
        record: dict[str, object] = {
            "qa_id": row["qa_id"],
            "unit": unit,
            "category": row["category"],
        }
        fold_predictions = []
        for fold_index in range(fold_probabilities.shape[0]):
            prediction, score, margin = decode_row(
                row,
                fold_probabilities[fold_index, source_row],
                args.top_windows,
                cardinality,
                normalize_sequence_time=not args.sequence_raw_time,
            )
            fold_predictions.append(prediction)
            record[f"prediction_fold_{fold_index}"] = prediction
            record[f"score_fold_{fold_index}"] = score
            record[f"margin_fold_{fold_index}"] = margin
        prediction, score, margin = decode_row(
            row,
            ensemble[source_row],
            args.top_windows,
            cardinality,
            normalize_sequence_time=not args.sequence_raw_time,
        )
        record.update(
            {
                "prediction": prediction,
                "score": score,
                "margin": margin,
                "fold_consensus": len(set(fold_predictions)) == 1,
                "multi_cardinality": cardinality if row["category"] == "multi" else None,
            }
        )
        if parent is not None:
            record["parent"] = parent
            record["disagrees_parent"] = prediction != parent
        if training:
            record["answer"] = row["answer"]
            record["correct"] = prediction == row["answer"]
            record["subject"] = subject_id(row["path"])
        records.append(record)
    return pd.DataFrame(records)


def validation_report(frame: pd.DataFrame) -> dict[str, object] | None:
    if frame.empty:
        return None
    category = frame.groupby("category")["correct"].agg(["size", "mean"])
    subject = frame.groupby("subject")["correct"].mean().sort_values()
    confidence = []
    for category_name in sorted(frame["category"].unique()):
        category_rows = frame["category"].eq(category_name) & frame["fold_consensus"]
        for threshold in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
            selected = category_rows & frame["margin"].ge(threshold)
            confidence.append(
                {
                    "category": category_name,
                    "margin": threshold,
                    "rows": int(selected.sum()),
                    "accuracy": float(frame.loc[selected, "correct"].mean())
                    if selected.any()
                    else None,
                }
            )
    return {
        "protocol": (
            "direct external-transfer evaluation on HAU; the frozen checkpoint was trained "
            "on separate HAR single-action clips and receives no HAU labels"
        ),
        "rows": len(frame),
        "accuracy": float(frame["correct"].mean()),
        "worst_subject_accuracy": float(subject.iloc[0]),
        "category": {
            name: {"rows": int(row["size"]), "accuracy": float(row["mean"])}
            for name, row in category.iterrows()
        },
        "confidence": confidence,
    }


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    train = read_csv_str(args.train_csv)
    test = read_csv_str(args.test_csv)
    parent = read_csv_str(args.parent_submission)
    prediction_column = str(parent.columns[1])
    parent_map = parent.set_index("qa_id")[prediction_column]

    logits, units, weights, logit_archives = load_logit_archives(args.logits)
    probabilities = softmax(logits, args.temperature)
    unit_to_row = {unit: index for index, unit in enumerate(units)}

    train_result = prediction_frame(
        train, unit_to_row, probabilities, weights, args, training=True
    )
    oracle_result = prediction_frame(
        train,
        unit_to_row,
        probabilities,
        weights,
        args,
        training=True,
        oracle_multi_cardinality=True,
    )
    report: dict[str, object] = {
        "method": "time-resolved public K-KUNO action transfer to HAU",
        "config": {
            "temperature": args.temperature,
            "top_windows": args.top_windows,
            "multi_cardinality": args.multi_cardinality,
            "normalize_sequence_time": not args.sequence_raw_time,
            "override_categories": sorted(
                value.strip() for value in args.override_categories.split(",") if value.strip()
            ),
            "override_min_margin": args.override_min_margin,
        },
        "validation": validation_report(train_result),
        "oracle_multi_cardinality_validation": validation_report(oracle_result),
        "logit_archives": logit_archives,
        "logit_shape": list(logits.shape),
    }

    test_result = prediction_frame(
        test,
        unit_to_row,
        probabilities,
        weights,
        args,
        training=False,
        parent_map=parent_map,
    )
    override_categories = {
        value.strip() for value in args.override_categories.split(",") if value.strip()
    }
    if not override_categories <= ACTION_CATEGORIES:
        raise ValueError(f"invalid override categories: {sorted(override_categories)}")
    test_result["override"] = (
        test_result["category"].isin(override_categories)
        & test_result["fold_consensus"]
        & test_result["margin"].ge(args.override_min_margin)
        & test_result["disagrees_parent"]
    )
    submission = parent.copy()
    overlay = test_result.loc[test_result["override"]].set_index("qa_id")["prediction"]
    submission[prediction_column] = submission.apply(
        lambda row: overlay.get(row["qa_id"], row[prediction_column]), axis=1
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = args.output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    train_result.to_csv(args.output_dir / "train_predictions.csv", index=False)
    test_result.to_csv(args.output_dir / "test_predictions.csv", index=False)
    validation = validate_submission_files(args.test_csv, args.sample_csv, submission_path)
    report["test_rows"] = len(test_result)
    report["parent_disagreements"] = int(test_result["disagrees_parent"].sum())
    report["overrides"] = int(test_result["override"].sum())
    report["override_rows"] = test_result.loc[
        test_result["override"],
        ["qa_id", "category", "parent", "prediction", "margin", "fold_consensus"],
    ].to_dict("records")
    report["submission"] = validation.to_dict()
    report["runtime_seconds"] = time.perf_counter() - started
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
