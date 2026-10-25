#!/usr/bin/env python3
"""Apply exact HARn action/object compatibility constraints."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuhkx.harn import action_text_classes, object_text_classes, paired_answer_candidates
from cuhkx.metadata import clip_key, subject_id
from cuhkx.submission import read_csv_str, validate_submission_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--sample-csv", type=Path, required=True)
    parser.add_argument("--parent-submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_rows(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.loc[
        frame["source"].eq("HARn")
        & frame["category"].isin({"single", "object_interaction"})
    ].copy()
    selected["clip_key"] = selected["path"].map(clip_key)
    return selected


def pair_members(rows: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    action = rows.loc[rows["category"].eq("single")]
    objects = rows.loc[rows["category"].eq("object_interaction")]
    if len(action) != 1 or len(objects) != 1:
        raise ValueError("pair_members requires exactly one action and one object question")
    return action.iloc[0], objects.iloc[0]


def training_audit(frame: pd.DataFrame) -> pd.DataFrame:
    subjects = sorted(value for value in frame["path"].map(subject_id).unique() if value)
    records: list[dict[str, object]] = []
    for subject in subjects:
        reference = frame.loc[frame["path"].map(subject_id).ne(subject)]
        action_mapping = action_text_classes(reference)
        object_mapping = object_text_classes(reference)
        held_out = frame.loc[frame["path"].map(subject_id).eq(subject)]
        for key, rows in held_out.groupby("clip_key", sort=True):
            if (
                rows["category"].eq("single").sum() != 1
                or rows["category"].eq("object_interaction").sum() != 1
            ):
                continue
            action_row, object_row = pair_members(rows)
            candidates = paired_answer_candidates(rows, action_mapping, object_mapping)
            truth = (str(action_row["answer"]), str(object_row["answer"]))
            records.append(
                {
                    "subject": subject,
                    "clip_key": key,
                    "action_qa_id": action_row["qa_id"],
                    "object_qa_id": object_row["qa_id"],
                    "truth_action": truth[0],
                    "truth_object": truth[1],
                    "candidate_count": len(candidates),
                    "candidates": json.dumps(candidates),
                    "truth_contained": truth in candidates,
                    "unique": len(candidates) == 1,
                    "unique_correct": len(candidates) == 1 and candidates[0] == truth,
                }
            )
    return pd.DataFrame(records)


def test_decisions(
    training_frame: pd.DataFrame, test_frame: pd.DataFrame, parent: pd.DataFrame
) -> pd.DataFrame:
    action_mapping = action_text_classes(training_frame)
    object_mapping = object_text_classes(training_frame)
    prediction_column = str(parent.columns[1])
    parent_predictions = parent.set_index("qa_id")[prediction_column]
    records: list[dict[str, object]] = []
    for key, rows in test_frame.groupby("clip_key", sort=True):
        if (
            rows["category"].eq("single").sum() != 1
            or rows["category"].eq("object_interaction").sum() != 1
        ):
            continue
        action_row, object_row = pair_members(rows)
        candidates = paired_answer_candidates(rows, action_mapping, object_mapping)
        action_parent = str(parent_predictions.loc[action_row["qa_id"]])
        object_parent = str(parent_predictions.loc[object_row["qa_id"]])
        unique = len(candidates) == 1
        action_prediction = candidates[0][0] if unique else action_parent
        object_prediction = candidates[0][1] if unique else object_parent
        records.append(
            {
                "clip_key": key,
                "action_qa_id": action_row["qa_id"],
                "object_qa_id": object_row["qa_id"],
                "candidate_count": len(candidates),
                "candidates": json.dumps(candidates),
                "unique": unique,
                "action_parent": action_parent,
                "object_parent": object_parent,
                "action_prediction": action_prediction,
                "object_prediction": object_prediction,
                "action_changed": action_prediction != action_parent,
                "object_changed": object_prediction != object_parent,
            }
        )
    return pd.DataFrame(records)


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    train = paired_rows(read_csv_str(args.train_csv))
    test = paired_rows(read_csv_str(args.test_csv))
    parent = read_csv_str(args.parent_submission)
    parent_validation = validate_submission_files(
        args.test_csv, args.sample_csv, args.parent_submission
    )

    audit = training_audit(train)
    unique = audit.loc[audit["unique"]]
    if audit.empty or not audit["truth_contained"].all():
        raise RuntimeError("HARn pair compatibility does not contain every OOF truth")
    if unique.empty or not unique["unique_correct"].all():
        raise RuntimeError("a unique HARn OOF pair is incorrect; refusing to generate output")

    decisions = test_decisions(train, test, parent)
    unique_decisions = decisions.loc[decisions["unique"]]
    overlays: dict[str, str] = {}
    for row in unique_decisions.to_dict("records"):
        overlays[str(row["action_qa_id"])] = str(row["action_prediction"])
        overlays[str(row["object_qa_id"])] = str(row["object_prediction"])

    prediction_column = str(parent.columns[1])
    submission = parent.copy()
    submission[prediction_column] = submission.apply(
        lambda row: overlays.get(str(row["qa_id"]), str(row[prediction_column])), axis=1
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = args.output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False, lineterminator="\n")
    audit.to_csv(args.output_dir / "train_audit.csv", index=False, lineterminator="\n")
    decisions.to_csv(args.output_dir / "test_decisions.csv", index=False, lineterminator="\n")
    validation = validate_submission_files(args.test_csv, args.sample_csv, submission_path)

    subject_unique = unique.groupby("subject")["unique_correct"].agg(["size", "mean"])
    changed_rows = decisions.loc[decisions["action_changed"] | decisions["object_changed"]]
    report = {
        "method": "exact HARn action/object option compatibility",
        "validation_protocol": (
            "option-text dictionaries are refit without the held-out subject; unique compatible "
            "pairs must be correct and all truth pairs must remain feasible"
        ),
        "training": {
            "paired_paths": len(audit),
            "truth_contained": int(audit["truth_contained"].sum()),
            "unique_paths": len(unique),
            "unique_correct": int(unique["unique_correct"].sum()),
            "subjects_with_unique_pairs": int(subject_unique.shape[0]),
            "worst_subject_unique_accuracy": float(subject_unique["mean"].min()),
        },
        "test": {
            "paired_paths": len(decisions),
            "unique_paths": len(unique_decisions),
            "overridden_rows": int(
                decisions["action_changed"].sum() + decisions["object_changed"].sum()
            ),
            "changed_pairs": changed_rows.to_dict("records"),
        },
        "parent": {
            "path": str(args.parent_submission.resolve()),
            "sha256": sha256_file(args.parent_submission),
            "validation": parent_validation.to_dict(),
        },
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
