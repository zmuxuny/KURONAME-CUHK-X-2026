#!/usr/bin/env python3
"""Learn subject-disjoint pairwise HAU action order from frozen temporal logits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuhkx.actions import hau_action_class
from cuhkx.metadata import feature_unit, subject_id
from cuhkx.pairwise_temporal import (
    PAIR_INDICES,
    before_counts,
    decode_pairwise_order,
    pair_curve_features,
)
from cuhkx.submission import read_csv_str, validate_submission_files
from cuhkx.temporal import softmax

LOCKED_SUBJECTS = frozenset(
    {"user3", "user7", "user9", "user17", "user19", "user21", "user22"}
)
FAMILY_CONFIGS: dict[str, tuple[int, str | float]] = {
    "leaf2_sqrt": (2, "sqrt"),
    "leaf5_sqrt": (5, "sqrt"),
    "leaf10_sqrt": (10, "sqrt"),
    "leaf5_mf30": (5, 0.30),
    "leaf10_mf30": (10, 0.30),
}
SELECTED_FAMILY = "leaf5_mf30"


@dataclass(frozen=True)
class PairRows:
    question: np.ndarray
    left: np.ndarray
    right: np.ndarray
    target: np.ndarray | None
    forward: np.ndarray
    reverse: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--sample-csv", type=Path, required=True)
    parser.add_argument("--train-logits", type=Path, action="append", required=True)
    parser.add_argument("--test-logits", type=Path, action="append", required=True)
    parser.add_argument("--parent-submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--prior-weight", type=float, default=1.0)
    parser.add_argument("--minimum-family-agreement", type=int, default=5)
    parser.add_argument("--minimum-margin", type=float, default=3.0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_frame(path: Path, *, training: bool) -> pd.DataFrame:
    frame = read_csv_str(path)
    frame = frame.loc[
        frame["source"].eq("HAU") & frame["category"].eq("sequence")
    ].copy()
    frame["unit"] = frame["path"].map(
        lambda value: feature_unit(value, training=training)
    )
    if training:
        frame["subject"] = frame["path"].map(subject_id)
    return frame.reset_index(drop=True)


def probability_maps(
    train_paths: list[Path],
    test_paths: list[Path],
    temperature: float,
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, object]]]:
    if len(train_paths) != len(test_paths):
        raise ValueError("train and test temporal scales must be aligned")
    sources: list[dict[str, np.ndarray]] = []
    metadata = []
    for train_path, test_path in zip(train_paths, test_paths, strict=True):
        combined: dict[str, np.ndarray] = {}
        shapes = []
        for path in (train_path, test_path):
            with np.load(path, allow_pickle=False) as archive:
                logits = archive["logits"].astype(np.float32)
                units = archive["units"].astype(str)
                weights = archive["weights"].astype(np.float64)
            if logits.ndim != 4 or logits.shape[1] != len(units) or logits.shape[-1] != 40:
                raise ValueError(f"unexpected temporal archive shape: {path}")
            weights /= weights.sum()
            probabilities = np.tensordot(
                weights, softmax(logits, temperature), axes=(0, 0)
            ).astype(np.float32)
            overlap = set(combined) & set(units)
            if overlap:
                raise ValueError(f"train/test unit overlap in temporal scale: {sorted(overlap)[:3]}")
            combined.update(dict(zip(units, probabilities, strict=True)))
            shapes.append(list(logits.shape))
            metadata.append(
                {"path": str(path.resolve()), "sha256": sha256_file(path), "shape": shapes[-1]}
            )
        sources.append(combined)
    return sources, metadata


def question_classes(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        [[hau_action_class(row[letter]) for letter in "ABCD"] for _, row in frame.iterrows()],
        dtype=int,
    )


def make_pair_rows(
    frame: pd.DataFrame,
    classes: np.ndarray,
    sources: list[dict[str, np.ndarray]],
    *,
    training: bool,
) -> PairRows:
    questions: list[int] = []
    left_options: list[int] = []
    right_options: list[int] = []
    targets: list[int] = []
    forward: list[np.ndarray] = []
    reverse: list[np.ndarray] = []
    for question, (_, row) in enumerate(frame.iterrows()):
        unit_sources = [source[str(row["unit"])] for source in sources]
        positions = (
            {letter: index for index, letter in enumerate(str(row["answer"]))}
            if training
            else {}
        )
        for left, right in PAIR_INDICES:
            left_class = int(classes[question, left])
            right_class = int(classes[question, right])
            questions.append(question)
            left_options.append(left)
            right_options.append(right)
            if training:
                targets.append(int(positions["ABCD"[left]] < positions["ABCD"[right]]))
            forward.append(pair_curve_features(unit_sources, left_class, right_class))
            reverse.append(pair_curve_features(unit_sources, right_class, left_class))
    return PairRows(
        question=np.asarray(questions, dtype=int),
        left=np.asarray(left_options, dtype=int),
        right=np.asarray(right_options, dtype=int),
        target=np.asarray(targets, dtype=int) if training else None,
        forward=np.stack(forward),
        reverse=np.stack(reverse),
    )


def build_model(
    leaf: int,
    max_features: str | float,
    *,
    trees: int,
    workers: int,
    seed: int,
) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=trees,
        min_samples_leaf=leaf,
        max_features=max_features,
        class_weight="balanced",
        n_jobs=workers,
        random_state=seed,
    )


def ordered_class_rows(frame: pd.DataFrame, classes: np.ndarray) -> list[list[int]]:
    return [
        [int(classes[index, "ABCD".index(letter)]) for letter in str(row["answer"])]
        for index, (_, row) in enumerate(frame.iterrows())
    ]


def decode_questions(
    frame: pd.DataFrame,
    classes: np.ndarray,
    pairs: PairRows,
    probabilities: np.ndarray,
    counts_by_question: list[Counter[tuple[int, int]]],
    prior_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = []
    margins = []
    for question in range(len(frame)):
        pair_rows = np.flatnonzero(pairs.question == question)
        prediction, margin = decode_pairwise_order(
            probabilities[pair_rows],
            classes[question],
            counts_by_question[question],
            prior_weight=prior_weight,
        )
        predictions.append(prediction)
        margins.append(margin)
    return np.asarray(predictions), np.asarray(margins)


def accuracy_summary(
    predictions: np.ndarray, truth: np.ndarray, subjects: np.ndarray
) -> dict[str, float]:
    correct = predictions == truth
    subject_accuracy = pd.Series(correct).groupby(subjects).mean()
    locked = np.isin(subjects, list(LOCKED_SUBJECTS))
    return {
        "accuracy": float(correct.mean()),
        "development_accuracy": float(correct[~locked].mean()),
        "locked_accuracy": float(correct[locked].mean()),
        "worst_subject": float(subject_accuracy.min()),
    }


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    if len(args.train_logits) < 2:
        raise ValueError("pairwise temporal decoding requires at least two temporal scales")
    if not 1 <= args.minimum_family_agreement <= len(FAMILY_CONFIGS):
        raise ValueError("minimum family agreement is outside the family count")
    if args.minimum_margin < 0 or args.prior_weight < 0:
        raise ValueError("margins and prior weight must be nonnegative")

    train = sequence_frame(args.train_csv, training=True)
    test = sequence_frame(args.test_csv, training=False)
    sources, source_metadata = probability_maps(
        args.train_logits, args.test_logits, args.temperature
    )
    expected_units = set(train["unit"]) | set(test["unit"])
    for source in sources:
        missing = expected_units - set(source)
        if missing:
            raise ValueError(f"temporal source misses sequence units: {sorted(missing)[:3]}")
    train_classes = question_classes(train)
    test_classes = question_classes(test)
    train_pairs = make_pair_rows(train, train_classes, sources, training=True)
    test_pairs = make_pair_rows(test, test_classes, sources, training=False)
    assert train_pairs.target is not None
    truth = train["answer"].to_numpy()
    subjects = train["subject"].to_numpy()

    splits = list(GroupKFold(n_splits=args.folds).split(train, groups=subjects))
    ordered = ordered_class_rows(train, train_classes)
    oof_counts: list[Counter[tuple[int, int]]] = [Counter() for _ in range(len(train))]
    for fit_questions, valid_questions in splits:
        counts = before_counts([ordered[index] for index in fit_questions])
        for question in valid_questions:
            oof_counts[question] = counts
    test_counts = [before_counts(ordered) for _ in range(len(test))]

    family_oof: dict[str, np.ndarray] = {}
    family_test: dict[str, np.ndarray] = {}
    family_oof_pair_probabilities: dict[str, np.ndarray] = {}
    family_test_pair_probabilities: dict[str, np.ndarray] = {}
    fitted = {}
    validation = {}
    oof_frame = train[["qa_id", "unit", "subject", "answer"]].copy()
    test_frame = test[["qa_id", "unit"]].copy()
    for family_index, (name, (leaf, max_features)) in enumerate(FAMILY_CONFIGS.items()):
        oof_probabilities = np.zeros(len(train_pairs.question), dtype=np.float32)
        for fold, (fit_questions, valid_questions) in enumerate(splits, start=1):
            fit = np.isin(train_pairs.question, fit_questions)
            valid = np.isin(train_pairs.question, valid_questions)
            model = build_model(
                leaf,
                max_features,
                trees=args.trees,
                workers=args.workers,
                seed=args.seed + family_index * 1000 + fold,
            )
            model.fit(
                np.concatenate((train_pairs.forward[fit], train_pairs.reverse[fit])),
                np.concatenate((train_pairs.target[fit], 1 - train_pairs.target[fit])),
            )
            forward = model.predict_proba(train_pairs.forward[valid])[:, 1]
            reverse = model.predict_proba(train_pairs.reverse[valid])[:, 1]
            oof_probabilities[valid] = 0.5 * (forward + 1.0 - reverse)
            print(f"family={name} fold={fold}/{len(splits)}", flush=True)

        model = build_model(
            leaf,
            max_features,
            trees=args.trees,
            workers=args.workers,
            seed=args.seed + family_index * 1000,
        )
        model.fit(
            np.concatenate((train_pairs.forward, train_pairs.reverse)),
            np.concatenate((train_pairs.target, 1 - train_pairs.target)),
        )
        fitted[name] = model
        test_forward = model.predict_proba(test_pairs.forward)[:, 1]
        test_reverse = model.predict_proba(test_pairs.reverse)[:, 1]
        test_probabilities = 0.5 * (test_forward + 1.0 - test_reverse)

        oof_prediction, oof_margin = decode_questions(
            train,
            train_classes,
            train_pairs,
            oof_probabilities,
            oof_counts,
            args.prior_weight,
        )
        test_prediction, test_margin = decode_questions(
            test,
            test_classes,
            test_pairs,
            test_probabilities,
            test_counts,
            args.prior_weight,
        )
        family_oof[name] = oof_prediction
        family_test[name] = test_prediction
        family_oof_pair_probabilities[name] = oof_probabilities
        family_test_pair_probabilities[name] = test_probabilities
        oof_frame[f"prediction_{name}"] = oof_prediction
        oof_frame[f"margin_{name}"] = oof_margin
        test_frame[f"prediction_{name}"] = test_prediction
        test_frame[f"margin_{name}"] = test_margin
        validation[name] = {
            **accuracy_summary(oof_prediction, truth, subjects),
            "pair_accuracy": float(
                np.mean((oof_probabilities >= 0.5) == train_pairs.target)
            ),
        }

    prediction_columns = [f"prediction_{name}" for name in FAMILY_CONFIGS]
    margin_columns = [f"margin_{name}" for name in FAMILY_CONFIGS]
    oof_frame["prediction"] = oof_frame[f"prediction_{SELECTED_FAMILY}"]
    test_frame["prediction"] = test_frame[f"prediction_{SELECTED_FAMILY}"]
    oof_frame["agreement_count"] = oof_frame[prediction_columns].apply(
        lambda row: int(row.value_counts().iloc[0]), axis=1
    )
    test_frame["agreement_count"] = test_frame[prediction_columns].apply(
        lambda row: int(row.value_counts().iloc[0]), axis=1
    )
    oof_frame["minimum_margin"] = oof_frame[margin_columns].min(axis=1)
    test_frame["minimum_margin"] = test_frame[margin_columns].min(axis=1)
    oof_frame["correct"] = oof_frame["prediction"].eq(oof_frame["answer"])

    confidence = []
    for agreement in (4, 5):
        for threshold in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
            selected = oof_frame["agreement_count"].ge(agreement) & oof_frame[
                "minimum_margin"
            ].ge(threshold)
            locked = selected & oof_frame["subject"].isin(LOCKED_SUBJECTS)
            confidence.append(
                {
                    "minimum_family_agreement": agreement,
                    "minimum_margin": threshold,
                    "rows": int(selected.sum()),
                    "accuracy": float(oof_frame.loc[selected, "correct"].mean())
                    if selected.any()
                    else None,
                    "subjects": int(oof_frame.loc[selected, "subject"].nunique()),
                    "locked_rows": int(locked.sum()),
                    "locked_accuracy": float(oof_frame.loc[locked, "correct"].mean())
                    if locked.any()
                    else None,
                }
            )

    selected_oof = oof_frame["agreement_count"].ge(args.minimum_family_agreement)
    selected_oof &= oof_frame["minimum_margin"].ge(args.minimum_margin)
    selected_test = test_frame["agreement_count"].ge(args.minimum_family_agreement)
    selected_test &= test_frame["minimum_margin"].ge(args.minimum_margin)
    parent = read_csv_str(args.parent_submission)
    prediction_column = str(parent.columns[1])
    parent_map = parent.set_index("qa_id")[prediction_column]
    test_frame["parent"] = test_frame["qa_id"].map(parent_map)
    if test_frame["parent"].isna().any():
        raise ValueError("parent submission lacks sequence IDs")
    test_frame["selected"] = selected_test & test_frame["prediction"].ne(
        test_frame["parent"]
    )
    submission = parent.copy()
    overlay = test_frame.loc[test_frame["selected"]].set_index("qa_id")["prediction"]
    submission[prediction_column] = submission.apply(
        lambda row: overlay.get(row["qa_id"], row[prediction_column]), axis=1
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_score_path = args.output_dir / "pair_scores.npz"
    family_names = np.asarray(list(FAMILY_CONFIGS))
    np.savez_compressed(
        pair_score_path,
        family_names=family_names,
        pair_indices=np.asarray(PAIR_INDICES, dtype=np.int8),
        oof_qa_ids=train["qa_id"].to_numpy(dtype=str),
        oof_classes=train_classes,
        oof_probabilities=np.stack(
            [family_oof_pair_probabilities[name] for name in FAMILY_CONFIGS]
        ).reshape(len(FAMILY_CONFIGS), len(train), len(PAIR_INDICES)),
        test_qa_ids=test["qa_id"].to_numpy(dtype=str),
        test_classes=test_classes,
        test_probabilities=np.stack(
            [family_test_pair_probabilities[name] for name in FAMILY_CONFIGS]
        ).reshape(len(FAMILY_CONFIGS), len(test), len(PAIR_INDICES)),
    )
    submission_path = args.output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False, lineterminator="\n")
    oof_frame.assign(selected=selected_oof).to_csv(
        args.output_dir / "oof_predictions.csv", index=False, lineterminator="\n"
    )
    test_frame.to_csv(
        args.output_dir / "test_predictions.csv", index=False, lineterminator="\n"
    )
    joblib.dump(
        {
            "models": fitted,
            "family_configs": FAMILY_CONFIGS,
            "selected_family": SELECTED_FAMILY,
            "ordered_class_counts": before_counts(ordered),
            "config": vars(args),
        },
        args.output_dir / "models.joblib",
        compress=3,
    )
    submission_validation = validate_submission_files(
        args.test_csv, args.sample_csv, submission_path
    )
    locked_oof = selected_oof & oof_frame["subject"].isin(LOCKED_SUBJECTS)
    report = {
        "method": "subject-disjoint pairwise temporal classification and permutation decoding",
        "config": {
            "temperature": args.temperature,
            "trees": args.trees,
            "folds": args.folds,
            "seed": args.seed,
            "prior_weight": args.prior_weight,
            "symmetric_pair_augmentation": True,
            "family_configs": FAMILY_CONFIGS,
            "selected_family": SELECTED_FAMILY,
            "minimum_family_agreement": args.minimum_family_agreement,
            "minimum_margin": args.minimum_margin,
        },
        "validation_protocol": (
            "six-fold subject-group OOF; action-order counts and pair models exclude every "
            "held-out subject; temporal inputs are frozen external-transfer logits"
        ),
        "validation_rows": len(train),
        "validation": validation,
        "confidence": confidence,
        "selected_oof_rows": int(selected_oof.sum()),
        "selected_oof_accuracy": float(oof_frame.loc[selected_oof, "correct"].mean())
        if selected_oof.any()
        else None,
        "selected_oof_subjects": int(oof_frame.loc[selected_oof, "subject"].nunique()),
        "selected_locked_rows": int(locked_oof.sum()),
        "selected_locked_accuracy": float(oof_frame.loc[locked_oof, "correct"].mean())
        if locked_oof.any()
        else None,
        "test_rows": len(test),
        "eligible_test_rows": int(selected_test.sum()),
        "overrides": int(test_frame["selected"].sum()),
        "override_rows": test_frame.loc[
            test_frame["selected"],
            ["qa_id", "parent", "prediction", "agreement_count", "minimum_margin"],
        ].to_dict("records"),
        "temporal_archives": source_metadata,
        "parent": {
            "path": str(args.parent_submission.resolve()),
            "sha256": sha256_file(args.parent_submission),
        },
        "pair_score_archive": {
            "path": str(pair_score_path.resolve()),
            "sha256": sha256_file(pair_score_path),
        },
        "submission": submission_validation.to_dict(),
        "runtime_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
