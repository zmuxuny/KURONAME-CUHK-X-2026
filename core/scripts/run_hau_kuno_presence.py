#!/usr/bin/env python3
"""Calibrate K-KUNO action presence on HAU with subject-group OOF.

The default binary calibration labels come only from HAU single- and
multi-answer questions, leaving combination as an out-of-task validation.
An experimental ``all-actions`` mode can additionally deduplicate positive
atoms from combination and sequence questions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_hau_kuno_temporal import load_logit_archives

from cuhkx.actions import hau_action_class
from cuhkx.metadata import feature_unit, option_atoms, subject_id
from cuhkx.submission import available_letters, read_csv_str, validate_submission_files
from cuhkx.temporal import softmax

ACTION_CATEGORIES = {"single", "multi", "combination"}
MODEL_LEAVES = {"extra_leaf1": 1, "extra_leaf2": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--sample-csv", type=Path, required=True)
    parser.add_argument("--logits", type=Path, nargs="+", required=True)
    parser.add_argument("--parent-submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--top-windows", type=int, default=2)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--supervision",
        choices=("single-multi", "all-actions"),
        default="single-multi",
        help="labels used to fit the binary action-presence calibrators",
    )
    parser.add_argument(
        "--base-blend",
        type=float,
        default=0.0,
        help="weight of the frozen direct-presence score in the selected prediction",
    )
    parser.add_argument("--minimum-agreement", type=int, default=3)
    parser.add_argument("--override-categories", default="single,multi,combination")
    parser.add_argument("--override-min-margin", type=float, default=0.50)
    return parser.parse_args()


def feature_cube(
    logits: np.ndarray,
    weights: np.ndarray,
    temperature: float,
    top_windows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-(unit, action) calibration features and direct presence."""

    probabilities = softmax(logits, temperature)
    normalized_weights = weights / weights.sum()
    ensemble = np.tensordot(normalized_weights, probabilities, axes=(0, 0))
    if not 1 <= top_windows <= ensemble.shape[1]:
        raise ValueError("top-windows must be within the segment count")
    presence = np.partition(ensemble, -top_windows, axis=1)[:, -top_windows:].mean(axis=1)

    folds, units, segments, classes = logits.shape
    transpose = (1, 3, 0, 2)
    raw_target = logits.transpose(transpose).reshape(units, classes, folds * segments)
    probability_target = probabilities.transpose(transpose).reshape(
        units, classes, folds * segments
    )
    ranks = np.argsort(np.argsort(-probabilities, axis=-1), axis=-1).astype(np.float32)
    ranks /= max(1, classes - 1)
    rank_target = ranks.transpose(transpose).reshape(units, classes, folds * segments)
    logit_margin = logits - logits.max(axis=-1, keepdims=True)
    margin_target = logit_margin.transpose(transpose).reshape(
        units, classes, folds * segments
    )
    ensemble_target = ensemble.transpose(0, 2, 1)
    sorted_target = np.sort(ensemble_target, axis=-1)
    global_presence = np.broadcast_to(presence[:, None, :], (units, classes, classes))
    identity = np.broadcast_to(np.eye(classes, dtype=np.float32), (units, classes, classes))
    features = np.concatenate(
        (
            raw_target,
            probability_target,
            rank_target,
            margin_target,
            ensemble_target,
            sorted_target,
            global_presence,
            identity,
        ),
        axis=-1,
    ).astype(np.float32)
    return features, presence.astype(np.float32)


def training_options(
    train: pd.DataFrame,
    unit_to_row: dict[str, int],
    supervision: str = "single-multi",
) -> pd.DataFrame:
    if supervision not in {"single-multi", "all-actions"}:
        raise ValueError(f"unknown supervision mode: {supervision}")
    records = []
    eligible = train.loc[
        train["source"].eq("HAU") & train["category"].isin({"single", "multi"})
    ]
    for _, row in eligible.iterrows():
        unit = feature_unit(row["path"], training=True)
        if unit not in unit_to_row:
            continue
        for letter in available_letters(row):
            records.append(
                {
                    "qa_id": row["qa_id"],
                    "unit": unit,
                    "unit_row": unit_to_row[unit],
                    "category": row["category"],
                    "letter": letter,
                    "class_id": hau_action_class(row[letter]),
                    "label": int(
                        letter in row["answer"]
                        if row["category"] == "multi"
                        else letter == row["answer"]
                    ),
                    "subject": subject_id(row["path"]),
                }
            )
    if supervision == "single-multi":
        return pd.DataFrame(records)

    labels: dict[tuple[str, int, str], set[int]] = defaultdict(set)
    for record in records:
        labels[(record["unit"], record["class_id"], record["subject"])].add(
            record["label"]
        )
    extra = train.loc[
        train["source"].eq("HAU") & train["category"].isin({"combination", "sequence"})
    ]
    for _, row in extra.iterrows():
        unit = feature_unit(row["path"], training=True)
        if unit not in unit_to_row:
            continue
        if row["category"] == "combination":
            atoms = option_atoms(row[row["answer"]])
        else:
            atoms = tuple(row[letter] for letter in available_letters(row))
        for atom in atoms:
            labels[(unit, hau_action_class(atom), subject_id(row["path"]))].add(1)
    conflicts = [key for key, values in labels.items() if len(values) != 1]
    if conflicts:
        raise ValueError(f"contradictory action-presence labels: {conflicts[:3]}")
    return pd.DataFrame(
        [
            {
                "qa_id": None,
                "unit": unit,
                "unit_row": unit_to_row[unit],
                "category": "presence",
                "letter": None,
                "class_id": class_id,
                "label": next(iter(values)),
                "subject": subject,
            }
            for (unit, class_id, subject), values in labels.items()
        ]
    )


def build_model(args: argparse.Namespace, leaf_size: int, seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=args.trees,
        max_features="sqrt",
        min_samples_leaf=leaf_size,
        class_weight="balanced",
        n_jobs=args.workers,
        random_state=seed,
    )


def option_scores(row: pd.Series, action_scores: np.ndarray) -> dict[str, float]:
    result = {}
    for letter in available_letters(row):
        atoms = option_atoms(row[letter]) if row["category"] == "combination" else (row[letter],)
        values = np.asarray(
            [action_scores[hau_action_class(atom)] for atom in atoms], dtype=np.float64
        ).clip(1e-8, 1.0)
        result[letter] = float(np.exp(np.log(values).mean()))
    return result


def decode_scores(scores: dict[str, float], cardinality: int) -> tuple[str, float, float]:
    if not 1 <= cardinality <= len(scores):
        raise ValueError("answer cardinality is outside the available options")
    labels = sorted(scores)
    values = np.asarray([scores[label] for label in labels], dtype=np.float64).clip(1e-12)
    probabilities = values / values.sum()
    ranked = sorted(zip(labels, probabilities, strict=True), key=lambda item: (-item[1], item[0]))
    selected = sorted(label for label, _ in ranked[:cardinality])
    next_score = ranked[cardinality][1] if cardinality < len(ranked) else 0.0
    margin = float(ranked[cardinality - 1][1] - next_score)
    return "".join(selected), float(sum(value for _, value in ranked[:cardinality])), margin


def prediction_frame(
    qa: pd.DataFrame,
    unit_to_row: dict[str, int],
    family_scores: dict[str, np.ndarray],
    selected_scores: np.ndarray,
    *,
    training: bool,
    parent_map: pd.Series | None = None,
) -> pd.DataFrame:
    records = []
    for _, row in qa.iterrows():
        if row["source"] != "HAU" or row["category"] not in ACTION_CATEGORIES:
            continue
        unit = feature_unit(row["path"], training=training)
        if unit not in unit_to_row:
            continue
        source_row = unit_to_row[unit]
        parent = None if parent_map is None else parent_map.get(row["qa_id"])
        cardinality = (
            len(row["answer"])
            if training and row["category"] == "multi"
            else len(str(parent))
            if row["category"] == "multi" and parent is not None
            else 1
        )
        predictions = []
        record: dict[str, object] = {
            "qa_id": row["qa_id"],
            "unit": unit,
            "category": row["category"],
        }
        for name, scores in family_scores.items():
            family_option_scores = option_scores(row, scores[source_row])
            prediction, score, margin = decode_scores(family_option_scores, cardinality)
            predictions.append(prediction)
            record[f"prediction_{name}"] = prediction
            record[f"score_{name}"] = score
            record[f"margin_{name}"] = margin
            for letter, value in family_option_scores.items():
                record[f"option_score_{name}_{letter}"] = value
        selected_option_scores = option_scores(row, selected_scores[source_row])
        prediction, score, margin = decode_scores(selected_option_scores, cardinality)
        for letter, value in selected_option_scores.items():
            record[f"option_score_{letter}"] = value
        record.update(
            {
                "prediction": prediction,
                "score": score,
                "margin": margin,
                "agreement_count": max(Counter(predictions).values()),
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


def confidence_report(frame: pd.DataFrame, family_count: int) -> list[dict[str, object]]:
    result = []
    for category in sorted(frame["category"].unique()):
        for agreement in sorted({max(2, family_count - 1), family_count}):
            for threshold in (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
                selected = (
                    frame["category"].eq(category)
                    & frame["agreement_count"].ge(agreement)
                    & frame["margin"].ge(threshold)
                )
                result.append(
                    {
                        "category": category,
                        "minimum_agreement": agreement,
                        "margin": threshold,
                        "rows": int(selected.sum()),
                        "accuracy": float(frame.loc[selected, "correct"].mean())
                        if selected.any()
                        else None,
                    }
                )
    return result


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    if not 0 <= args.base_blend <= 1:
        raise ValueError("base-blend must be in [0, 1]")
    train = read_csv_str(args.train_csv)
    test = read_csv_str(args.test_csv)
    parent = read_csv_str(args.parent_submission)
    prediction_column = str(parent.columns[1])
    parent_map = parent.set_index("qa_id")[prediction_column]

    logits, units, weights, archive_metadata = load_logit_archives(args.logits)
    features, direct_presence = feature_cube(
        logits, weights, args.temperature, args.top_windows
    )
    unit_to_row = {unit: index for index, unit in enumerate(units)}
    options = training_options(train, unit_to_row, args.supervision)
    option_matrix = features[
        options["unit_row"].to_numpy(), options["class_id"].to_numpy()
    ]
    labels = options["label"].to_numpy()
    groups = options["subject"].to_numpy()
    splits = list(GroupKFold(n_splits=args.folds).split(option_matrix, labels, groups))

    train_unit_subject = {
        feature_unit(row["path"], training=True): subject_id(row["path"])
        for _, row in train.loc[train["source"].eq("HAU")].iterrows()
    }
    train_score_rows = np.asarray(
        [index for index, unit in enumerate(units) if unit in train_unit_subject]
    )
    test_score_rows = np.flatnonzero(np.char.startswith(units, "Testing/"))
    family_oof: dict[str, np.ndarray] = {}
    family_test: dict[str, np.ndarray] = {}
    fitted: dict[str, ExtraTreesClassifier] = {}
    binary_auc = {}

    for model_index, (name, leaf_size) in enumerate(MODEL_LEAVES.items()):
        oof_scores = np.full((len(units), 40), np.nan, dtype=np.float32)
        for fold, (fit_index, valid_index) in enumerate(splits, start=1):
            held_out = set(groups[valid_index])
            valid_unit_rows = np.asarray(
                [
                    row_index
                    for row_index in train_score_rows
                    if train_unit_subject[units[row_index]] in held_out
                ]
            )
            model = build_model(
                args, leaf_size, args.seed + 1000 * model_index + fold
            )
            model.fit(option_matrix[fit_index], labels[fit_index])
            oof_scores[valid_unit_rows] = model.predict_proba(
                features[valid_unit_rows].reshape(-1, features.shape[-1])
            )[:, 1].reshape(len(valid_unit_rows), 40)
            print(
                f"model={name} fold={fold}/{len(splits)} "
                f"held_out={','.join(sorted(held_out))}",
                flush=True,
            )
        option_oof = oof_scores[
            options["unit_row"].to_numpy(), options["class_id"].to_numpy()
        ]
        binary_auc[name] = float(roc_auc_score(labels, option_oof))
        family_oof[name] = oof_scores

        model = build_model(args, leaf_size, args.seed + 1000 * model_index)
        model.fit(option_matrix, labels)
        fitted[name] = model
        test_scores = np.full((len(units), 40), np.nan, dtype=np.float32)
        test_scores[test_score_rows] = model.predict_proba(
            features[test_score_rows].reshape(-1, features.shape[-1])
        )[:, 1].reshape(len(test_score_rows), 40)
        family_test[name] = test_scores

    family_oof["direct"] = direct_presence
    family_test["direct"] = direct_presence
    learned_oof = np.mean(
        np.stack([family_oof[name] for name in MODEL_LEAVES]), axis=0
    )
    learned_test = np.mean(
        np.stack([family_test[name] for name in MODEL_LEAVES]), axis=0
    )
    selected_oof = (1 - args.base_blend) * learned_oof + args.base_blend * direct_presence
    selected_test = (1 - args.base_blend) * learned_test + args.base_blend * direct_presence

    oof = prediction_frame(
        train,
        unit_to_row,
        family_oof,
        selected_oof,
        training=True,
    )
    subject_accuracy = oof.groupby("subject")["correct"].mean().sort_values()
    category_accuracy = oof.groupby("category")["correct"].agg(["size", "mean"])
    individual_accuracy = {
        name: float(oof[f"prediction_{name}"].eq(oof["answer"]).mean())
        for name in family_oof
    }
    report: dict[str, object] = {
        "method": "subject-group OOF calibration of frozen K-KUNO HAU action presence",
        "config": {
            "temperature": args.temperature,
            "top_windows": args.top_windows,
            "folds": args.folds,
            "trees": args.trees,
            "workers": args.workers,
            "seed": args.seed,
            "supervision": args.supervision,
            "base_blend": args.base_blend,
            "minimum_agreement": args.minimum_agreement,
            "override_categories": sorted(
                value.strip()
                for value in args.override_categories.split(",")
                if value.strip()
            ),
            "override_min_margin": args.override_min_margin,
        },
        "validation_protocol": (
            "six-fold subject-group OOF; "
            + (
                "calibrators train only on single/multi option labels"
                if args.supervision == "single-multi"
                else "calibrators train on deduplicated action labels from all action questions"
            )
            + "; multi validation receives oracle answer cardinality"
        ),
        "validation_rows": len(oof),
        "validation_accuracy": float(oof["correct"].mean()),
        "validation_worst_subject": float(subject_accuracy.iloc[0]),
        "validation_category": {
            name: {"rows": int(row["size"]), "accuracy": float(row["mean"])}
            for name, row in category_accuracy.iterrows()
        },
        "individual_accuracy": individual_accuracy,
        "binary_option_auc": binary_auc,
        "confidence": confidence_report(oof, len(family_oof)),
        "logit_archives": archive_metadata,
        "logit_shape": list(logits.shape),
        "feature_shape": list(features.shape),
        "training_option_rows": len(options),
    }
    print(json.dumps(report, indent=2), flush=True)

    test_result = prediction_frame(
        test,
        unit_to_row,
        family_test,
        selected_test,
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
        & test_result["agreement_count"].ge(args.minimum_agreement)
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
    oof.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    test_result.to_csv(args.output_dir / "test_predictions.csv", index=False)
    joblib.dump(
        {
            "models": fitted,
            "config": report["config"],
            "feature_shape": features.shape,
        },
        args.output_dir / "models.joblib",
        compress=3,
    )
    validation = validate_submission_files(args.test_csv, args.sample_csv, submission_path)
    report["test_rows"] = len(test_result)
    report["parent_disagreements"] = int(test_result["disagrees_parent"].sum())
    report["overrides"] = int(test_result["override"].sum())
    report["override_rows"] = test_result.loc[
        test_result["override"],
        ["qa_id", "category", "parent", "prediction", "margin", "agreement_count"],
    ].to_dict("records")
    report["submission"] = validation.to_dict()
    report["runtime_seconds"] = time.perf_counter() - started
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"submission={submission_path.resolve()} overrides={report['overrides']} "
        f"sha256={validation.sha256} runtime={report['runtime_seconds']:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
