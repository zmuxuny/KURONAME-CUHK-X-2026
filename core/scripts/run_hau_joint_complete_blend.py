#!/usr/bin/env python3
"""Blend complete-set calibration into full HAU joint-action decoding."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_hau_joint_actions import (
    blend_vectors,
    calibrated_presence_map,
    combine_unit_maps,
    skeleton_presence_map,
)

from cuhkx.actions import HAU_CLASS_TEXT
from cuhkx.clip_constraints import apply_sequence_constraints
from cuhkx.joint_actions import (
    JOINT_CATEGORIES,
    decode_joint_state,
    decode_joint_state_with_option_evidence,
)
from cuhkx.metadata import clip_key, feature_unit, subject_id
from cuhkx.submission import read_csv_str, validate_submission_files

LOCKED_SUBJECTS = frozenset(
    {"user3", "user7", "user9", "user17", "user19", "user21", "user22"}
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--sample-csv", type=Path, required=True)
    parser.add_argument("--train-logits-s8", type=Path, required=True)
    parser.add_argument("--test-logits-s8", type=Path, required=True)
    parser.add_argument("--train-logits-s16", type=Path, required=True)
    parser.add_argument("--test-logits-s16", type=Path, required=True)
    parser.add_argument("--presence-run", type=Path, required=True)
    parser.add_argument("--skeleton-action-run", type=Path, action="append", required=True)
    parser.add_argument("--complete-set-run", type=Path, action="append", required=True)
    parser.add_argument("--option-frequency-run", type=Path)
    parser.add_argument("--parent-submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibrated-weight", type=float, default=0.5)
    parser.add_argument("--visual16-weight", type=float, default=0.5)
    parser.add_argument("--complete-set-weight", type=float, default=1.5)
    parser.add_argument("--skeleton-weight", type=float, default=0.2)
    parser.add_argument("--option-frequency-weight", type=float, default=0.0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def complete_set_maps(paths: list[Path]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    train_values: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    test_values: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    for path in paths:
        archive_path = path / "action_scores.npz"
        archive = np.load(archive_path, allow_pickle=False)
        names = archive["view_names"].astype(str)
        selected = np.flatnonzero(names == "extra_leaf2")
        if len(selected) != 1:
            raise ValueError(f"{archive_path} lacks exactly one extra_leaf2 view")
        index = int(selected[0])
        for prefix, destination in (("train", train_values), ("test", test_values)):
            units = archive[f"{prefix}_units"].astype(str)
            scores = archive[f"{prefix}_scores"][index].astype(np.float32)
            if scores.shape != (len(units), len(HAU_CLASS_TEXT)):
                raise ValueError(f"unexpected complete-set score shape in {archive_path}")
            for unit, score in zip(units, scores, strict=True):
                destination[unit].append(score)

    def average(values: defaultdict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
        if any(len(items) != len(paths) for items in values.values()):
            raise ValueError("complete-set runs do not have identical unit coverage")
        return {
            unit: np.mean(np.stack(items), axis=0).astype(np.float32)
            for unit, items in values.items()
        }

    return average(train_values), average(test_values)


def option_frequency_scores(
    path: Path,
    *,
    training: bool,
) -> dict[str, dict[str, float]]:
    filename = "oof_option_scores.csv" if training else "test_option_scores.csv"
    frame = read_csv_str(path / filename)
    probability_columns = [
        column for column in frame if column.startswith("probability_")
    ]
    required = {"qa_id", "letter"}
    if not probability_columns or not required <= set(frame):
        raise ValueError(f"{path / filename} lacks option-frequency probabilities")
    probabilities = frame[probability_columns].astype(float).clip(1e-5, 1 - 1e-5)
    logits = np.log(probabilities / (1.0 - probabilities)).mean(axis=1)
    frame = frame.copy()
    frame["probability"] = 1.0 / (1.0 + np.exp(-logits))
    output = {
        str(qa_id): dict(zip(group["letter"], group["probability"], strict=True))
        for qa_id, group in frame.groupby("qa_id", sort=False)
    }
    if any(set(values) != set("ABCD") for values in output.values()):
        raise ValueError("option-frequency score rows do not exactly cover ABCD")
    return output


def decode_frame(
    qa: pd.DataFrame,
    sources: dict[str, dict[str, np.ndarray]],
    weights: dict[str, float],
    option_scores: dict[str, dict[str, float]] | None,
    option_weight: float,
    *,
    training: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = qa.loc[qa["source"].eq("HAU")].copy()
    work["_clip"] = work["path"].map(clip_key)
    row_records: list[dict[str, object]] = []
    clip_records: list[dict[str, object]] = []
    for clip, group in work.groupby("_clip", sort=False):
        unit = feature_unit(group.iloc[0]["path"], training=training)
        if unit not in sources["calibrated"]:
            raise ValueError(f"calibrated scores lack {unit}")
        fallback = sources["calibrated"][unit]
        vectors = [sources[name].get(unit, fallback) for name in weights]
        probabilities = blend_vectors(vectors, list(weights.values()))
        action_probabilities = {
            HAU_CLASS_TEXT[class_id]: float(probabilities[class_id])
            for class_id in HAU_CLASS_TEXT
        }
        decision = (
            decode_joint_state(group, action_probabilities)
            if option_weight == 0
            else decode_joint_state_with_option_evidence(
                group,
                action_probabilities,
                option_scores or {},
                option_weight=option_weight,
            )
        )
        answer_map = decision.state.answer_map()
        clip_record: dict[str, object] = {
            "clip": clip,
            "unit": unit,
            "margin": decision.margin,
            "candidate_count": decision.candidate_count,
        }
        if training:
            truth = {
                str(row["category"]): str(row["answer"])
                for _, row in group.loc[group["category"].isin(JOINT_CATEGORIES)].iterrows()
            }
            clip_record["correct"] = answer_map == truth
            clip_record["subject"] = subject_id(group.iloc[0]["path"])
        clip_records.append(clip_record)
        for _, row in group.loc[group["category"].isin(JOINT_CATEGORIES)].iterrows():
            category = str(row["category"])
            record: dict[str, object] = {
                "qa_id": row["qa_id"],
                "clip": clip,
                "unit": unit,
                "category": category,
                "prediction": answer_map[category],
                "margin": decision.margin,
            }
            if training:
                record["answer"] = row["answer"]
                record["correct"] = record["prediction"] == row["answer"]
                record["subject"] = subject_id(row["path"])
            row_records.append(record)
    return pd.DataFrame(row_records), pd.DataFrame(clip_records)


def validation_report(frame: pd.DataFrame) -> dict[str, object]:
    by_category = frame.groupby("category")["correct"].agg(["size", "mean"])
    by_subject = frame.groupby("subject")["correct"].mean()
    locked = frame["subject"].isin(LOCKED_SUBJECTS)
    return {
        "rows": len(frame),
        "accuracy": float(frame["correct"].mean()),
        "worst_subject": float(by_subject.min()),
        "locked_accuracy": float(frame.loc[locked, "correct"].mean()),
        "category": {
            name: {"rows": int(row["size"]), "accuracy": float(row["mean"])}
            for name, row in by_category.iterrows()
        },
    }


def main() -> int:
    args = parse_args()
    weights = {
        "calibrated": args.calibrated_weight,
        "visual16": args.visual16_weight,
        "complete_set": args.complete_set_weight,
        "skeleton": args.skeleton_weight,
    }
    if (
        any(value < 0 for value in weights.values())
        or sum(weights.values()) <= 0
        or args.option_frequency_weight < 0
    ):
        raise ValueError("source weights must be nonnegative with a positive sum")
    if args.option_frequency_weight > 0 and args.option_frequency_run is None:
        raise ValueError("option-frequency-run is required when its weight is positive")

    train = read_csv_str(args.train_csv)
    test = read_csv_str(args.test_csv)
    v8 = combine_unit_maps(
        args.train_logits_s8,
        args.test_logits_s8,
        temperature=4.0,
        top_windows=2,
    )
    v16 = combine_unit_maps(
        args.train_logits_s16,
        args.test_logits_s16,
        temperature=4.0,
        top_windows=4,
    )
    calibrated = calibrated_presence_map(
        train, args.presence_run, v8, training=True
    ) | calibrated_presence_map(test, args.presence_run, v8, training=False)
    skeleton = skeleton_presence_map(args.skeleton_action_run)
    complete_train, complete_test = complete_set_maps(args.complete_set_run)
    sources = {
        "calibrated": calibrated,
        "visual16": v16,
        "complete_set": complete_train | complete_test,
        "skeleton": skeleton,
    }
    train_option_scores = (
        option_frequency_scores(args.option_frequency_run, training=True)
        if args.option_frequency_run is not None
        else None
    )
    test_option_scores = (
        option_frequency_scores(args.option_frequency_run, training=False)
        if args.option_frequency_run is not None
        else None
    )

    oof, oof_clips = decode_frame(
        train,
        sources,
        weights,
        train_option_scores,
        args.option_frequency_weight,
        training=True,
    )
    test_predictions, test_clips = decode_frame(
        test,
        sources,
        weights,
        test_option_scores,
        args.option_frequency_weight,
        training=False,
    )
    parent = read_csv_str(args.parent_submission)
    prediction_column = str(parent.columns[1])
    parent_map = parent.set_index("qa_id")[prediction_column]
    test_predictions["parent"] = test_predictions["qa_id"].map(parent_map)
    if test_predictions["parent"].isna().any():
        raise ValueError("parent submission lacks HAU joint-action qa_ids")
    test_predictions["override"] = test_predictions["prediction"].ne(
        test_predictions["parent"]
    )

    intermediate = parent.copy()
    overlay = test_predictions.set_index("qa_id")["prediction"]
    intermediate[prediction_column] = intermediate.apply(
        lambda row: overlay.get(row["qa_id"], row[prediction_column]), axis=1
    )
    constrained, constraint_decisions = apply_sequence_constraints(
        test, intermediate.set_index("qa_id")[prediction_column].to_dict()
    )
    submission = intermediate.copy()
    submission[prediction_column] = constrained.to_numpy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = args.output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False, lineterminator="\n")
    oof.to_csv(args.output_dir / "oof_predictions.csv", index=False, lineterminator="\n")
    oof_clips.to_csv(args.output_dir / "oof_clips.csv", index=False, lineterminator="\n")
    test_predictions.to_csv(
        args.output_dir / "test_predictions.csv", index=False, lineterminator="\n"
    )
    test_clips.to_csv(args.output_dir / "test_clips.csv", index=False, lineterminator="\n")
    constraint_decisions.to_csv(
        args.output_dir / "constraint_decisions.csv", index=False, lineterminator="\n"
    )
    submission_validation = validate_submission_files(
        args.test_csv, args.sample_csv, submission_path
    )

    report = {
        "method": "full HAU joint decoding with complete-set calibration blend",
        "validation_protocol": (
            "six-fold subject-group OOF for calibrated, complete-set, and Skeleton sources; "
            "frozen temporal views receive no HAU fit; no oracle multi cardinality"
        ),
        "weights_log_odds": weights,
        "option_frequency_weight": args.option_frequency_weight,
        "validation": validation_report(oof),
        "validation_exact_clip_accuracy": float(oof_clips["correct"].mean()),
        "test_overrides": int(test_predictions["override"].sum()),
        "inputs": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for path in [
                args.train_csv,
                args.test_csv,
                args.sample_csv,
                args.train_logits_s8,
                args.test_logits_s8,
                args.train_logits_s16,
                args.test_logits_s16,
                args.parent_submission,
                args.presence_run / "oof_predictions.csv",
                args.presence_run / "test_predictions.csv",
                *(run / "action_scores.npz" for run in args.skeleton_action_run),
                *(run / "action_scores.npz" for run in args.complete_set_run),
                *(
                    [
                        args.option_frequency_run / "oof_option_scores.csv",
                        args.option_frequency_run / "test_option_scores.csv",
                    ]
                    if args.option_frequency_run is not None
                    else []
                ),
            ]
        ],
        "submission": submission_validation.to_dict(),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
