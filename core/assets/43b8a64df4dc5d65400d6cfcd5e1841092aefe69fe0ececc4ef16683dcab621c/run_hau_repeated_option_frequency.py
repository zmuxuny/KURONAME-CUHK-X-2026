#!/usr/bin/env python3
"""Train a repeated-recording option-frequency expert and gate joint answers."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuhkx.option_frequency import (
    build_option_frequency_features,
    decode_option_probabilities,
)
from cuhkx.submission import read_csv_str, validate_submission_files

LOCKED_SUBJECTS = frozenset(
    {"user3", "user7", "user9", "user17", "user19", "user21", "user22"}
)
FAMILY_CONFIGS = {
    "leaf2": {"min_samples_leaf": 2, "max_features": "sqrt"},
    "leaf5": {"min_samples_leaf": 5, "max_features": "sqrt"},
    "leaf10": {"min_samples_leaf": 10, "max_features": "sqrt"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--sample-csv", type=Path, required=True)
    parser.add_argument("--parent-submission", type=Path, required=True)
    parser.add_argument("--joint-complete-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--multi-threshold", type=float, default=0.475)
    parser.add_argument("--minimum-margin", type=float, default=0.50)
    parser.add_argument("--minimum-direct-margin", type=float, default=0.80)
    parser.add_argument("--required-joint-agreement", type=int, default=7)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(values: pd.Series) -> pd.Series:
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false", "1", "0"}).all():
        raise ValueError("eligible column contains an invalid boolean")
    return normalized.isin({"true", "1"})


def load_joint(path: Path, filename: str, qa_ids: pd.Series) -> pd.DataFrame:
    frame = read_csv_str(path / filename).set_index("qa_id")
    required = {"prediction", "agreement_count", "eligible"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"joint predictions lack columns: {sorted(missing)}")
    if set(frame.index) != set(qa_ids):
        raise ValueError("joint and option-frequency action question sets differ")
    result = frame.loc[qa_ids].reset_index()
    result["agreement_count"] = pd.to_numeric(
        result["agreement_count"], errors="raise"
    )
    result["eligible"] = as_bool(result["eligible"])
    return result


def model(config: dict[str, object], trees: int, seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=trees,
        min_samples_leaf=int(config["min_samples_leaf"]),
        max_features=config["max_features"],
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )


def metrics(frame: pd.DataFrame) -> dict[str, object]:
    correct = frame["prediction"].eq(frame["answer"])
    locked = frame["subject"].isin(LOCKED_SUBJECTS)
    return {
        "accuracy": float(correct.mean()),
        "development_accuracy": float(correct[~locked].mean()),
        "locked_accuracy": float(correct[locked].mean()),
        "worst_subject": float(correct.groupby(frame["subject"]).mean().min()),
        "by_category": {
            category: {
                "rows": len(group),
                "accuracy": float(group["prediction"].eq(group["answer"]).mean()),
            }
            for category, group in frame.groupby("category", sort=True)
        },
    }


def gate_predictions(
    family_frames: dict[str, pd.DataFrame],
    joint: pd.DataFrame,
    *,
    minimum_margin: float,
    minimum_direct_margin: float,
    required_joint_agreement: int,
) -> pd.DataFrame:
    first_name = next(iter(family_frames))
    output = family_frames[first_name][
        ["qa_id", "category", "prediction", "margin"]
    ].rename(
        columns={
            "prediction": f"prediction_{first_name}",
            "margin": f"margin_{first_name}",
        }
    )
    for name, frame in list(family_frames.items())[1:]:
        output = output.merge(
            frame[["qa_id", "prediction", "margin"]].rename(
                columns={
                    "prediction": f"prediction_{name}",
                    "margin": f"margin_{name}",
                }
            ),
            on="qa_id",
            how="inner",
            validate="one_to_one",
        )
    output = output.merge(
        joint[["qa_id", "prediction", "agreement_count", "eligible"]].rename(
            columns={"prediction": "joint_prediction"}
        ),
        on="qa_id",
        how="inner",
        validate="one_to_one",
    )
    prediction_columns = [f"prediction_{name}" for name in family_frames]
    margin_columns = [f"margin_{name}" for name in family_frames]
    output["frequency_prediction"] = output[prediction_columns[0]]
    output["family_consensus"] = output[prediction_columns].nunique(axis=1).eq(1)
    output["minimum_margin"] = output[margin_columns].min(axis=1)
    output["joint_expansion_gate"] = (
        output["family_consensus"]
        & output["minimum_margin"].ge(minimum_margin)
        & output["agreement_count"].ge(required_joint_agreement)
        & output["frequency_prediction"].eq(output["joint_prediction"])
        & ~output["eligible"]
    )
    output["direct_frequency_gate"] = (
        output["family_consensus"]
        & output["minimum_margin"].ge(minimum_direct_margin)
        & output["category"].isin({"single", "combination"})
    )
    output["selected"] = output[
        ["joint_expansion_gate", "direct_frequency_gate"]
    ].any(axis=1)
    output["selected_prediction"] = output["joint_prediction"].where(
        ~output["direct_frequency_gate"], output["frequency_prediction"]
    )
    return output


def main() -> int:
    args = parse_args()
    if args.folds < 2 or args.trees <= 0:
        raise ValueError("folds must be at least two and trees must be positive")
    if (
        args.minimum_margin < 0
        or args.minimum_direct_margin < 0
        or args.required_joint_agreement < 1
    ):
        raise ValueError("gate thresholds must be nonnegative")

    train_all = read_csv_str(args.train_csv)
    test_all = read_csv_str(args.test_csv)
    train_x, train_meta = build_option_frequency_features(train_all)
    test_x, test_meta = build_option_frequency_features(test_all)
    if train_x.shape[1] != test_x.shape[1]:
        raise ValueError("training and test option-frequency dimensions differ")
    labels = train_meta["label"].to_numpy(dtype=int)
    subjects = train_meta["subject"].to_numpy()

    oof_families: dict[str, pd.DataFrame] = {}
    test_families: dict[str, pd.DataFrame] = {}
    oof_option_scores = train_meta.copy()
    test_option_scores = test_meta.copy()
    family_reports = {}
    for family_index, (name, config) in enumerate(FAMILY_CONFIGS.items()):
        oof_probability = np.zeros(len(train_x), dtype=np.float64)
        splitter = GroupKFold(n_splits=args.folds)
        for fold, (fit, valid) in enumerate(
            splitter.split(train_x, labels, groups=subjects)
        ):
            estimator = model(
                config,
                args.trees,
                args.seed + 1000 * family_index + fold,
            )
            estimator.fit(train_x[fit], labels[fit])
            oof_probability[valid] = estimator.predict_proba(train_x[valid])[:, 1]
        fitted = model(config, args.trees, args.seed + 1000 * family_index)
        fitted.fit(train_x, labels)
        test_probability = fitted.predict_proba(test_x)[:, 1]
        oof_option_scores[f"probability_{name}"] = oof_probability
        test_option_scores[f"probability_{name}"] = test_probability
        oof_frame = decode_option_probabilities(
            train_meta,
            oof_probability,
            multi_threshold=args.multi_threshold,
        )
        test_frame = decode_option_probabilities(
            test_meta,
            test_probability,
            multi_threshold=args.multi_threshold,
        )
        oof_families[name] = oof_frame
        test_families[name] = test_frame
        family_reports[name] = metrics(oof_frame)

    oof_joint = load_joint(
        args.joint_complete_run,
        "oof_predictions.csv",
        oof_families[next(iter(oof_families))]["qa_id"],
    )
    test_joint = load_joint(
        args.joint_complete_run,
        "test_predictions.csv",
        test_families[next(iter(test_families))]["qa_id"],
    )
    oof_gate = gate_predictions(
        oof_families,
        oof_joint,
        minimum_margin=args.minimum_margin,
        minimum_direct_margin=args.minimum_direct_margin,
        required_joint_agreement=args.required_joint_agreement,
    )
    test_gate = gate_predictions(
        test_families,
        test_joint,
        minimum_margin=args.minimum_margin,
        minimum_direct_margin=args.minimum_direct_margin,
        required_joint_agreement=args.required_joint_agreement,
    )
    truth = oof_families[next(iter(oof_families))][
        ["qa_id", "answer", "subject"]
    ]
    oof_gate = oof_gate.merge(truth, on="qa_id", validate="one_to_one")
    selected_oof = oof_gate["selected"]
    selected_correct = oof_gate.loc[selected_oof, "selected_prediction"].eq(
        oof_gate.loc[selected_oof, "answer"]
    )
    if not selected_oof.any() or not selected_correct.all():
        raise ValueError("option-frequency expansion gate is not perfect in OOF validation")

    parent = read_csv_str(args.parent_submission)
    prediction_column = str(parent.columns[1])
    parent_map = parent.set_index("qa_id")[prediction_column]
    test_gate["parent"] = test_gate["qa_id"].map(parent_map)
    if test_gate["parent"].isna().any():
        raise ValueError("parent submission lacks a HAU action qa_id")
    test_gate["override"] = test_gate["selected"] & test_gate[
        "selected_prediction"
    ].ne(test_gate["parent"])
    overlay = test_gate.loc[test_gate["override"]].set_index("qa_id")[
        "selected_prediction"
    ]
    submission = parent.copy()
    submission[prediction_column] = submission.apply(
        lambda row: overlay.get(row["qa_id"], row[prediction_column]), axis=1
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = args.output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False, lineterminator="\n")
    oof_gate.to_csv(args.output_dir / "oof_predictions.csv", index=False, lineterminator="\n")
    test_gate.to_csv(
        args.output_dir / "test_predictions.csv", index=False, lineterminator="\n"
    )
    oof_option_scores.to_csv(
        args.output_dir / "oof_option_scores.csv", index=False, lineterminator="\n"
    )
    test_option_scores.to_csv(
        args.output_dir / "test_option_scores.csv", index=False, lineterminator="\n"
    )
    validation = validate_submission_files(args.test_csv, args.sample_csv, submission_path)

    locked = oof_gate["subject"].isin(LOCKED_SUBJECTS)
    report = {
        "method": "repeated-recording answer-free option-frequency expert",
        "validation_protocol": (
            "features use only official option text; all models are subject-group OOF; the "
            "multi threshold and final gate were fixed on development subjects before the "
            "seven locked subjects were reported"
        ),
        "config": {
            "families": FAMILY_CONFIGS,
            "folds": args.folds,
            "trees": args.trees,
            "seed": args.seed,
            "multi_threshold": args.multi_threshold,
            "minimum_margin": args.minimum_margin,
            "minimum_direct_margin": args.minimum_direct_margin,
            "required_joint_agreement": args.required_joint_agreement,
            "requires_all_family_consensus": True,
            "expands_only_rows_outside_existing_joint_gate": True,
        },
        "features": {
            "training_option_rows": len(train_meta),
            "test_option_rows": len(test_meta),
            "dimensions": train_x.shape[1],
        },
        "family_validation": family_reports,
        "joint_expansion_gate": {
            "rows": int(oof_gate["joint_expansion_gate"].sum()),
            "accuracy": float(
                oof_gate.loc[
                    oof_gate["joint_expansion_gate"], "joint_prediction"
                ].eq(
                    oof_gate.loc[oof_gate["joint_expansion_gate"], "answer"]
                ).mean()
            ),
        },
        "direct_frequency_gate": {
            "rows": int(oof_gate["direct_frequency_gate"].sum()),
            "accuracy": float(
                oof_gate.loc[
                    oof_gate["direct_frequency_gate"], "frequency_prediction"
                ].eq(
                    oof_gate.loc[oof_gate["direct_frequency_gate"], "answer"]
                ).mean()
            ),
        },
        "selected_union": {
            "rows": int(selected_oof.sum()),
            "accuracy": float(selected_correct.mean()),
            "development_rows": int((selected_oof & ~locked).sum()),
            "development_accuracy": float(
                selected_correct[~locked[selected_oof].to_numpy()].mean()
            ),
            "locked_rows": int((selected_oof & locked).sum()),
            "locked_accuracy": float(
                oof_gate.loc[
                    selected_oof & locked, "selected_prediction"
                ].eq(oof_gate.loc[selected_oof & locked, "answer"]).mean()
            ),
            "worst_subject": float(
                oof_gate.loc[selected_oof]
                .assign(correct=selected_correct.to_numpy())
                .groupby("subject")["correct"]
                .mean()
                .min()
            ),
        },
        "test": {
            "selected_rows": int(test_gate["selected"].sum()),
            "overrides": int(test_gate["override"].sum()),
            "override_rows": test_gate.loc[
                test_gate["override"],
                [
                    "qa_id",
                    "category",
                    "parent",
                    "selected_prediction",
                    "direct_frequency_gate",
                    "joint_expansion_gate",
                    "minimum_margin",
                    "agreement_count",
                ],
            ].to_dict("records"),
        },
        "inputs": {
            "parent": {
                "path": str(args.parent_submission.resolve()),
                "sha256": sha256_file(args.parent_submission),
            },
            "joint_complete_run": str(args.joint_complete_run.resolve()),
        },
        "submission": validation.to_dict(),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
