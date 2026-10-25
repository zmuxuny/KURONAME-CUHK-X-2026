#!/usr/bin/env python3
"""Expand the strict HAU joint-action gate with complete-set calibrators."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuhkx.clip_constraints import apply_sequence_constraints
from cuhkx.submission import read_csv_str, validate_submission_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--sample-csv", type=Path, required=True)
    parser.add_argument("--parent-submission", type=Path, required=True)
    parser.add_argument("--joint-run", type=Path, required=True)
    parser.add_argument("--complete-set-run", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strict-joint-agreement", type=int, default=7)
    parser.add_argument("--strict-joint-margin", type=float, default=0.05)
    parser.add_argument("--complete-margin", type=float, default=0.20)
    parser.add_argument("--skeleton-complete-margin", type=float, default=0.15)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_predictions(
    joint_run: Path,
    complete_runs: list[Path],
    *,
    training: bool,
) -> pd.DataFrame:
    filename = "oof_predictions.csv" if training else "test_predictions.csv"
    frame = read_csv_str(joint_run / filename)
    required = {
        "qa_id",
        "prediction",
        "agreement_count",
        "margin",
        "prediction_skeleton_direct",
    }
    if training:
        required |= {"answer", "subject"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"joint predictions lack columns: {sorted(missing)}")
    frame = frame.copy()
    frame["agreement_count"] = pd.to_numeric(frame["agreement_count"])
    frame["margin"] = pd.to_numeric(frame["margin"])
    for run_index, run in enumerate(complete_runs):
        complete = read_csv_str(run / filename)
        complete = complete[["qa_id", "prediction", "margin"]].rename(
            columns={
                "prediction": f"complete_prediction_{run_index}",
                "margin": f"complete_margin_{run_index}",
            }
        )
        complete[f"complete_margin_{run_index}"] = pd.to_numeric(
            complete[f"complete_margin_{run_index}"]
        )
        frame = frame.merge(complete, on="qa_id", how="inner", validate="one_to_one")
    if not complete_runs:
        raise ValueError("at least one complete-set run is required")
    return frame


def apply_gates(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    result = frame.copy()
    prediction_columns = [
        f"complete_prediction_{index}" for index in range(len(args.complete_set_run))
    ]
    margin_columns = [
        f"complete_margin_{index}" for index in range(len(args.complete_set_run))
    ]
    result["complete_consensus"] = result[prediction_columns].nunique(axis=1).eq(1)
    result["complete_consensus"] &= result[prediction_columns[0]].eq(
        result["prediction"]
    )
    result["minimum_complete_margin"] = result[margin_columns].min(axis=1)
    result["strict_joint_gate"] = result["agreement_count"].ge(
        args.strict_joint_agreement
    ) & result["margin"].ge(args.strict_joint_margin)
    result["complete_gate"] = result["complete_consensus"] & result[
        "minimum_complete_margin"
    ].ge(args.complete_margin)
    result["skeleton_complete_gate"] = (
        result["complete_consensus"]
        & result["minimum_complete_margin"].ge(args.skeleton_complete_margin)
        & result["prediction"].eq(result["prediction_skeleton_direct"])
    )
    result["eligible"] = result[
        ["strict_joint_gate", "complete_gate", "skeleton_complete_gate"]
    ].any(axis=1)
    return result


def gate_report(frame: pd.DataFrame, mask: pd.Series) -> dict[str, object]:
    correct = frame.loc[mask, "prediction"].eq(frame.loc[mask, "answer"])
    return {
        "rows": int(mask.sum()),
        "accuracy": float(correct.mean()) if mask.any() else None,
        "worst_subject": float(correct.groupby(frame.loc[mask, "subject"]).mean().min())
        if mask.any()
        else None,
    }


def main() -> int:
    args = parse_args()
    for value in (
        args.strict_joint_margin,
        args.complete_margin,
        args.skeleton_complete_margin,
    ):
        if value < 0:
            raise ValueError("all margins must be nonnegative")
    if args.strict_joint_agreement < 1:
        raise ValueError("strict-joint-agreement must be positive")

    train = read_csv_str(args.train_csv)
    test = read_csv_str(args.test_csv)
    oof = apply_gates(
        load_predictions(
            args.joint_run, args.complete_set_run, training=True
        ),
        args,
    )
    expected_oof = set(
        train.loc[
            train["source"].eq("HAU")
            & train["category"].isin({"single", "multi", "combination"}),
            "qa_id",
        ]
    )
    if len(oof) != len(expected_oof) or set(oof["qa_id"]) != expected_oof:
        raise ValueError("OOF predictions do not exactly cover HAU joint-action rows")
    test_predictions = apply_gates(
        load_predictions(
            args.joint_run, args.complete_set_run, training=False
        ),
        args,
    )
    expected_test = set(
        test.loc[
            test["source"].eq("HAU")
            & test["category"].isin({"single", "multi", "combination"}),
            "qa_id",
        ]
    )
    if len(test_predictions) != len(expected_test) or set(test_predictions["qa_id"]) != expected_test:
        raise ValueError("test predictions do not exactly cover HAU joint-action rows")

    parent = read_csv_str(args.parent_submission)
    prediction_column = str(parent.columns[1])
    parent_map = parent.set_index("qa_id")[prediction_column]
    test_predictions["parent"] = test_predictions["qa_id"].map(parent_map)
    if test_predictions["parent"].isna().any():
        raise ValueError("parent submission lacks HAU action qa_ids")
    test_predictions["override"] = test_predictions["eligible"] & test_predictions[
        "prediction"
    ].ne(test_predictions["parent"])

    intermediate = parent.copy()
    overlay = test_predictions.loc[test_predictions["override"]].set_index("qa_id")[
        "prediction"
    ]
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
    test_predictions.to_csv(
        args.output_dir / "test_predictions.csv", index=False, lineterminator="\n"
    )
    constraint_decisions.to_csv(
        args.output_dir / "constraint_decisions.csv", index=False, lineterminator="\n"
    )
    validation = validate_submission_files(args.test_csv, args.sample_csv, submission_path)

    strict = oof["strict_joint_gate"]
    complete = oof["complete_gate"]
    skeleton_complete = oof["skeleton_complete_gate"]
    expanded = oof["eligible"] & ~strict
    report = {
        "method": "strict joint action graph plus cross-seed complete-set gates",
        "validation_protocol": (
            "all learned inputs are subject-group OOF; complete-set targets come only from "
            "labeled training combination answers; thresholds are fixed before test overlay"
        ),
        "config": {
            "strict_joint_agreement": args.strict_joint_agreement,
            "strict_joint_margin": args.strict_joint_margin,
            "complete_margin": args.complete_margin,
            "skeleton_complete_margin": args.skeleton_complete_margin,
            "complete_set_runs": len(args.complete_set_run),
        },
        "validation": {
            "strict_joint": gate_report(oof, strict),
            "complete": gate_report(oof, complete),
            "skeleton_complete": gate_report(oof, skeleton_complete),
            "expanded_only": gate_report(oof, expanded),
            "union": gate_report(oof, oof["eligible"]),
        },
        "test": {
            "eligible_rows": int(test_predictions["eligible"].sum()),
            "overrides": int(test_predictions["override"].sum()),
            "expanded_overrides": int(
                (test_predictions["override"] & ~test_predictions["strict_joint_gate"]).sum()
            ),
            "override_rows": test_predictions.loc[
                test_predictions["override"],
                [
                    "qa_id",
                    "category",
                    "parent",
                    "prediction",
                    "strict_joint_gate",
                    "complete_gate",
                    "skeleton_complete_gate",
                    "margin",
                    "minimum_complete_margin",
                ],
            ].to_dict("records"),
        },
        "inputs": {
            "parent_submission": {
                "path": str(args.parent_submission.resolve()),
                "sha256": sha256_file(args.parent_submission),
            },
            "joint_run": str(args.joint_run.resolve()),
            "complete_set_runs": [str(path.resolve()) for path in args.complete_set_run],
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
