#!/usr/bin/env python3
"""Decode HARn object interactions through cross-subject action evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuhkx.harn import action_class
from cuhkx.metadata import feature_unit, normalize_text, subject_id
from cuhkx.submission import available_letters, read_csv_str, validate_submission_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--sample-csv", type=Path, required=True)
    parser.add_argument("--skeleton-probabilities", type=Path, required=True)
    parser.add_argument("--kuno-logits", type=Path, required=True)
    parser.add_argument("--parent-submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-skeleton-margin", type=float, default=0.15)
    parser.add_argument("--minimum-kuno-margin", type=float, default=0.15)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def canonical_mapping(frame: pd.DataFrame) -> dict[str, tuple[int, ...]]:
    """Map canonical correct object words to the HARn action classes that use them."""

    class_words: dict[int, set[str]] = defaultdict(set)
    for _, row in frame.iterrows():
        class_words[action_class(row["path"])].add(normalize_text(row[str(row["answer"])]))
    ambiguous = {key: values for key, values in class_words.items() if len(values) != 1}
    if ambiguous:
        raise ValueError(f"an action class maps to multiple canonical objects: {ambiguous}")
    word_classes: dict[str, list[int]] = defaultdict(list)
    for class_id, words in class_words.items():
        word_classes[next(iter(words))].append(class_id)
    return {word: tuple(sorted(values)) for word, values in word_classes.items()}


def option_scores(
    row: pd.Series,
    probabilities: np.ndarray,
    classes: np.ndarray,
    mapping: dict[str, tuple[int, ...]],
) -> tuple[dict[str, float], int]:
    class_columns = {int(class_id): index for index, class_id in enumerate(classes)}
    scores = {}
    known = 0
    for letter in available_letters(row):
        object_word = normalize_text(row[letter])
        mapped = [class_columns[value] for value in mapping.get(object_word, ()) if value in class_columns]
        scores[letter] = float(probabilities[mapped].sum()) if mapped else 0.0
        known += bool(mapped)
    return scores, known


def decode(scores: dict[str, float]) -> tuple[str, float, float]:
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if not ranked:
        raise ValueError("object question has no valid options")
    margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else float("inf")
    return ranked[0][0], ranked[0][1], margin


def skeleton_predictions(
    frame: pd.DataFrame,
    *,
    probabilities: np.ndarray,
    units: np.ndarray,
    model_names: np.ndarray,
    classes: np.ndarray,
    mapping_frame: pd.DataFrame,
    training: bool,
) -> pd.DataFrame:
    unit_index = {unit: index for index, unit in enumerate(units)}
    subject_mappings = {
        subject: canonical_mapping(
            mapping_frame.loc[mapping_frame["path"].map(subject_id).ne(subject)]
        )
        for subject in mapping_frame["path"].map(subject_id).dropna().unique()
    }
    full_mapping = canonical_mapping(mapping_frame)
    records = []
    for _, row in frame.iterrows():
        unit = feature_unit(row["path"], training=training)
        if unit not in unit_index:
            continue
        mapping = subject_mappings[subject_id(row["path"])] if training else full_mapping
        predictions = []
        model_scores = []
        record: dict[str, object] = {"qa_id": row["qa_id"], "unit": unit}
        for model_number, model_name in enumerate(model_names.astype(str)):
            scores, known = option_scores(
                row, probabilities[model_number, unit_index[unit]], classes, mapping
            )
            prediction, score, margin = decode(scores)
            predictions.append(prediction)
            model_scores.append(np.asarray([scores[letter] for letter in available_letters(row)]))
            record[f"prediction_{model_name}"] = prediction
            record[f"score_{model_name}"] = score
            record[f"margin_{model_name}"] = margin
            record[f"known_options_{model_name}"] = known
        mean_scores = np.mean(np.stack(model_scores), axis=0)
        ensemble_scores = dict(zip(available_letters(row), mean_scores, strict=True))
        prediction, score, margin = decode(ensemble_scores)
        record.update(
            {
                "prediction": prediction,
                "score": score,
                "margin": margin,
                "consensus": len(set(predictions)) == 1,
                "known_options": int(
                    sum(normalize_text(row[letter]) in mapping for letter in available_letters(row))
                ),
            }
        )
        if training:
            record["answer"] = row["answer"]
            record["correct"] = prediction == row["answer"]
            record["subject"] = subject_id(row["path"])
            record["action_class"] = action_class(row["path"])
        records.append(record)
    return pd.DataFrame(records)


def kuno_predictions(
    frame: pd.DataFrame,
    *,
    logits: np.ndarray,
    units: np.ndarray,
    classes: np.ndarray,
    weights: np.ndarray,
    mapping: dict[str, tuple[int, ...]],
    training: bool,
) -> pd.DataFrame:
    probabilities = softmax(logits)
    unit_index = {unit: index for index, unit in enumerate(units)}
    weights = weights / weights.sum()
    ensemble = np.tensordot(weights, probabilities, axes=(0, 0))
    records = []
    for _, row in frame.iterrows():
        unit = feature_unit(row["path"], training=training)
        if unit not in unit_index:
            continue
        predictions = []
        record: dict[str, object] = {"qa_id": row["qa_id"]}
        for fold in range(len(probabilities)):
            scores, _ = option_scores(row, probabilities[fold, unit_index[unit]], classes, mapping)
            prediction, _, margin = decode(scores)
            predictions.append(prediction)
            record[f"prediction_fold_{fold}"] = prediction
            record[f"margin_fold_{fold}"] = margin
        scores, known = option_scores(row, ensemble[unit_index[unit]], classes, mapping)
        prediction, score, margin = decode(scores)
        record.update(
            {
                "prediction": prediction,
                "score": score,
                "margin": margin,
                "fold_consensus": len(set(predictions)) == 1,
                "known_options": known,
            }
        )
        if training:
            record["answer"] = row["answer"]
            record["correct"] = prediction == row["answer"]
            record["subject"] = subject_id(row["path"])
        records.append(record)
    return pd.DataFrame(records)


def confidence_report(frame: pd.DataFrame) -> list[dict[str, object]]:
    records = []
    for threshold in (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
        selected = frame["consensus"] & frame["margin"].ge(threshold)
        records.append(
            {
                "minimum_margin": threshold,
                "rows": int(selected.sum()),
                "accuracy": float(frame.loc[selected, "correct"].mean())
                if selected.any()
                else None,
                "subjects": int(frame.loc[selected, "subject"].nunique())
                if selected.any()
                else 0,
            }
        )
    return records


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    train = read_csv_str(args.train_csv)
    test = read_csv_str(args.test_csv)
    train = train.loc[
        train["source"].eq("HARn") & train["category"].eq("object_interaction")
    ].copy()
    test = test.loc[
        test["source"].eq("HARn") & test["category"].eq("object_interaction")
    ].copy()
    mapping = canonical_mapping(train)

    skeleton = np.load(args.skeleton_probabilities, allow_pickle=False)
    skeleton_oof = skeleton_predictions(
        train,
        probabilities=skeleton["oof_probabilities"],
        units=skeleton["train_units"].astype(str),
        model_names=skeleton["model_names"].astype(str),
        classes=skeleton["classes"].astype(int),
        mapping_frame=train,
        training=True,
    )
    skeleton_test = skeleton_predictions(
        test,
        probabilities=skeleton["test_probabilities"],
        units=skeleton["test_units"].astype(str),
        model_names=skeleton["model_names"].astype(str),
        classes=skeleton["classes"].astype(int),
        mapping_frame=train,
        training=False,
    )

    kuno = np.load(args.kuno_logits, allow_pickle=False)
    kuno_train = kuno_predictions(
        train,
        logits=kuno["logits"].astype(np.float32),
        units=kuno["units"].astype(str),
        classes=kuno["classes"].astype(int),
        weights=kuno["weights"].astype(np.float32),
        mapping=mapping,
        training=True,
    )
    kuno_test = kuno_predictions(
        test,
        logits=kuno["logits"].astype(np.float32),
        units=kuno["units"].astype(str),
        classes=kuno["classes"].astype(int),
        weights=kuno["weights"].astype(np.float32),
        mapping=mapping,
        training=False,
    )
    test_result = skeleton_test.merge(
        kuno_test,
        on="qa_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_skeleton", "_kuno"),
    )
    parent = read_csv_str(args.parent_submission)
    prediction_column = str(parent.columns[1])
    parent_map = parent.set_index("qa_id")[prediction_column]
    test_result["parent"] = test_result["qa_id"].map(parent_map)
    test_result["selected"] = (
        test_result["consensus"]
        & test_result["margin_skeleton"].ge(args.minimum_skeleton_margin)
        & test_result["fold_consensus"]
        & test_result["margin_kuno"].ge(args.minimum_kuno_margin)
        & test_result["prediction_skeleton"].eq(test_result["prediction_kuno"])
        & test_result["prediction_skeleton"].ne(test_result["parent"])
    )
    overlay = test_result.loc[test_result["selected"]].set_index("qa_id")[
        "prediction_skeleton"
    ]
    submission = parent.copy()
    submission[prediction_column] = submission.apply(
        lambda row: overlay.get(row["qa_id"], row[prediction_column]), axis=1
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = args.output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False, lineterminator="\n")
    skeleton_oof.to_csv(
        args.output_dir / "skeleton_oof_predictions.csv", index=False, lineterminator="\n"
    )
    kuno_train.to_csv(
        args.output_dir / "kuno_train_diagnostic.csv", index=False, lineterminator="\n"
    )
    skeleton_test.to_csv(
        args.output_dir / "skeleton_test_predictions.csv", index=False, lineterminator="\n"
    )
    kuno_test.to_csv(
        args.output_dir / "kuno_test_predictions.csv", index=False, lineterminator="\n"
    )
    test_result.to_csv(args.output_dir / "test_predictions.csv", index=False, lineterminator="\n")
    validation = validate_submission_files(args.test_csv, args.sample_csv, submission_path)
    subject_accuracy = skeleton_oof.groupby("subject")["correct"].mean()
    selected_oof = skeleton_oof["consensus"] & skeleton_oof["margin"].ge(
        args.minimum_skeleton_margin
    )
    report = {
        "method": "HARn action-to-object decoding with Skeleton/K-KUNO consensus",
        "config": {
            "minimum_skeleton_margin": args.minimum_skeleton_margin,
            "minimum_kuno_margin": args.minimum_kuno_margin,
            "canonical_mapping": {key: list(value) for key, value in mapping.items()},
        },
        "validation_protocol": (
            "Skeleton action probabilities are subject-group OOF; action-to-object mapping is "
            "refit without each held-out subject; K-KUNO training numbers are diagnostic only"
        ),
        "skeleton_oof_rows": len(skeleton_oof),
        "skeleton_oof_accuracy": float(skeleton_oof["correct"].mean()),
        "skeleton_oof_worst_subject": float(subject_accuracy.min()),
        "skeleton_oof_confidence": confidence_report(skeleton_oof),
        "skeleton_selected_oof_rows": int(selected_oof.sum()),
        "skeleton_selected_oof_accuracy": float(skeleton_oof.loc[selected_oof, "correct"].mean())
        if selected_oof.any()
        else None,
        "kuno_train_diagnostic_accuracy": float(kuno_train["correct"].mean()),
        "skeleton_probability_archive": {
            "path": str(args.skeleton_probabilities.resolve()),
            "sha256": sha256_file(args.skeleton_probabilities),
        },
        "kuno_archive": {
            "path": str(args.kuno_logits.resolve()),
            "sha256": sha256_file(args.kuno_logits),
        },
        "test_rows": len(test),
        "skeleton_test_rows": len(skeleton_test),
        "consensus_test_rows": len(test_result),
        "overrides": int(test_result["selected"].sum()),
        "override_rows": test_result.loc[
            test_result["selected"],
            [
                "qa_id",
                "parent",
                "prediction_skeleton",
                "margin_skeleton",
                "margin_kuno",
            ],
        ].to_dict("records"),
        "submission": validation.to_dict(),
        "runtime_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
