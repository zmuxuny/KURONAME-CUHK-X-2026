#!/usr/bin/env python3
"""Propagate high-confidence action positives across repeated HAU recordings."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuhkx.clip_constraints import apply_repeated_sequence_constraints
from cuhkx.metadata import subject_id
from cuhkx.repeated_actions import (
    SOURCE_CATEGORIES,
    apply_repeated_action_decisions,
    repeated_action_decisions,
)
from cuhkx.submission import read_csv_str, validate_submission_files

LOCKED_SUBJECTS = frozenset(
    {"user3", "user7", "user9", "user17", "user19", "user21", "user22"}
)
GATE_COLUMNS = (
    "strict_joint_gate",
    "complete_gate",
    "skeleton_complete_gate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--sample-csv", type=Path, required=True)
    parser.add_argument("--parent-submission", type=Path, required=True)
    parser.add_argument("--joint-complete-run", type=Path, required=True)
    parser.add_argument("--option-frequency-run", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
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
        raise ValueError("gate column contains an invalid boolean")
    return normalized.isin({"true", "1"})


def load_joint_predictions(path: Path, filename: str) -> pd.DataFrame:
    frame = read_csv_str(path / filename)
    required = {
        "qa_id",
        "category",
        "prediction",
        *GATE_COLUMNS,
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"joint-complete predictions lack columns: {sorted(missing)}")
    result = frame.copy()
    for column in GATE_COLUMNS:
        result[column] = as_bool(result[column])
    result["source_eligible"] = result[list(GATE_COLUMNS)].all(axis=1)
    result["source_eligible"] &= result["category"].isin(SOURCE_CATEGORIES)
    return result


def load_frequency_sources(path: Path, filename: str) -> pd.DataFrame:
    frame = read_csv_str(path / filename)
    required = {
        "qa_id",
        "category",
        "frequency_prediction",
        "direct_frequency_gate",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"option-frequency predictions lack columns: {sorted(missing)}")
    result = frame.copy()
    result["direct_frequency_gate"] = as_bool(result["direct_frequency_gate"])
    result["source_eligible"] = result["direct_frequency_gate"] & result[
        "category"
    ].isin(SOURCE_CATEGORIES)
    return result


def merge_source_maps(
    joint: pd.DataFrame,
    frequency: pd.DataFrame | None,
) -> tuple[dict[str, str], set[str], list[dict[str, str]]]:
    joint_eligible = joint["source_eligible"]
    source_map = dict(
        zip(
            joint.loc[joint_eligible, "qa_id"],
            joint.loc[joint_eligible, "prediction"],
            strict=True,
        )
    )
    conflicts = []
    if frequency is not None:
        selected = frequency.loc[frequency["source_eligible"]]
        for row in selected.itertuples():
            if row.qa_id in source_map and source_map[row.qa_id] != row.frequency_prediction:
                conflicts.append(
                    {
                        "qa_id": row.qa_id,
                        "joint_prediction": source_map[row.qa_id],
                        "frequency_prediction": row.frequency_prediction,
                    }
                )
            source_map[row.qa_id] = row.frequency_prediction
    return source_map, set(source_map), conflicts


def audit_decisions(
    frame: pd.DataFrame, decisions: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    indexed = frame.set_index("qa_id")
    audit = decisions.copy()
    audit["answer"] = audit["qa_id"].map(indexed["answer"])
    audit["subject"] = audit["qa_id"].map(indexed["path"].map(subject_id))
    audit["valid"] = audit.apply(
        lambda row: set(row["certified_letters"]) <= set(row["answer"])
        if row["category"] == "multi"
        else len(row["certified_letters"]) == 1
        and row["certified_letters"] == row["answer"],
        axis=1,
    )
    nonmulti = audit["category"].isin({"single", "combination"})
    multi = audit["category"].eq("multi")
    multi_letters = int(audit.loc[multi, "certified_letters"].str.len().sum())
    valid_multi_letters = int(
        sum(
            letter in row.answer
            for row in audit.loc[multi].itertuples()
            for letter in row.certified_letters
        )
    )
    if not audit.loc[nonmulti, "valid"].all():
        raise ValueError("unique sibling matches are not exact on labeled training data")
    if not multi_letters or valid_multi_letters / multi_letters < 0.999:
        raise ValueError("sibling multi-positive precision fell below the fixed safety floor")
    locked = audit["subject"].isin(LOCKED_SUBJECTS)
    summary = {
        "rows": len(audit),
        "subjects": int(audit["subject"].nunique()),
        "single": {
            "rows": int(audit["category"].eq("single").sum()),
            "valid": int(audit.loc[audit["category"].eq("single"), "valid"].sum()),
        },
        "combination": {
            "rows": int(audit["category"].eq("combination").sum()),
            "valid": int(
                audit.loc[audit["category"].eq("combination"), "valid"].sum()
            ),
        },
        "multi": {
            "rows": int(multi.sum()),
            "valid_rows": int(audit.loc[multi, "valid"].sum()),
            "certified_letter_decisions": multi_letters,
            "valid_letter_decisions": valid_multi_letters,
            "letter_precision": valid_multi_letters / multi_letters,
        },
        "locked_rows": int(locked.sum()),
        "locked_valid_rows": int(audit.loc[locked, "valid"].sum()),
    }
    return audit, summary


def oof_overlay_report(
    predictions: pd.DataFrame,
    audit: pd.DataFrame,
) -> dict[str, object]:
    parent = predictions.set_index("qa_id")["prediction"].to_dict()
    output, applied = apply_repeated_action_decisions(parent, audit)
    indexed = predictions.set_index("qa_id")
    qa_ids = indexed.index
    before = indexed["prediction"].eq(indexed["answer"])
    after_values = pd.Series(output).reindex(qa_ids)
    after = after_values.eq(indexed["answer"])
    changed_ids = set(applied.loc[applied["changed"], "qa_id"])
    changed = qa_ids.isin(changed_ids)
    subjects = indexed["subject"]
    locked = subjects.isin(LOCKED_SUBJECTS)
    return {
        "parent_accuracy": float(before.mean()),
        "constrained_accuracy": float(after.mean()),
        "changes": int(changed.sum()),
        "changed_accuracy": float(after[changed].mean()) if changed.any() else None,
        "net_correct": int(after.sum() - before.sum()),
        "locked_parent_accuracy": float(before[locked].mean()),
        "locked_constrained_accuracy": float(after[locked].mean()),
    }


def main() -> int:
    args = parse_args()
    train = read_csv_str(args.train_csv)
    test = read_csv_str(args.test_csv)
    oof = load_joint_predictions(args.joint_complete_run, "oof_predictions.csv")
    test_predictions = load_joint_predictions(
        args.joint_complete_run, "test_predictions.csv"
    )
    frequency_oof = (
        load_frequency_sources(args.option_frequency_run, "oof_predictions.csv")
        if args.option_frequency_run is not None
        else None
    )
    frequency_test = (
        load_frequency_sources(args.option_frequency_run, "test_predictions.csv")
        if args.option_frequency_run is not None
        else None
    )

    eligible_oof = oof["source_eligible"]
    source_correct = oof.loc[eligible_oof, "prediction"].eq(
        oof.loc[eligible_oof, "answer"]
    )
    if not source_correct.all():
        raise ValueError("the learned source gate is not perfect in subject-OOF validation")
    source_map, source_ids, oof_conflicts = merge_source_maps(oof, frequency_oof)
    if oof_conflicts:
        raise ValueError("joint and frequency OOF source gates disagree")
    source_truth = train.set_index("qa_id")["answer"]
    source_union_correct = pd.Series(source_map).eq(source_truth.loc[list(source_map)])
    if not source_union_correct.all():
        raise ValueError("the union source gate is not perfect in OOF validation")
    train_decisions = repeated_action_decisions(train, source_map, source_ids)
    audit, audit_summary = audit_decisions(train, train_decisions)

    eligible_test = test_predictions["source_eligible"]
    test_source_map, test_source_ids, test_conflicts = merge_source_maps(
        test_predictions, frequency_test
    )
    test_decisions = repeated_action_decisions(
        test, test_source_map, test_source_ids
    )

    parent = read_csv_str(args.parent_submission)
    prediction_column = str(parent.columns[1])
    parent_map = parent.set_index("qa_id")[prediction_column].to_dict()
    overlaid, applied = apply_repeated_action_decisions(parent_map, test_decisions)
    constrained, sequence_decisions = apply_repeated_sequence_constraints(test, overlaid)
    submission = parent.copy()
    submission[prediction_column] = constrained.to_numpy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = args.output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False, lineterminator="\n")
    audit.to_csv(args.output_dir / "train_audit.csv", index=False, lineterminator="\n")
    applied.to_csv(args.output_dir / "test_decisions.csv", index=False, lineterminator="\n")
    sequence_decisions.to_csv(
        args.output_dir / "sequence_decisions.csv", index=False, lineterminator="\n"
    )
    validation = validate_submission_files(args.test_csv, args.sample_csv, submission_path)

    changed = applied["changed"]
    final_map = submission.set_index("qa_id")[prediction_column]
    source_by_category = oof.loc[eligible_oof].groupby("category", sort=True)
    report = {
        "method": "cross-recording high-confidence positive-action propagation",
        "validation_protocol": (
            "learned sources require all three joint/complete-set gates and are subject-group "
            "OOF; only source categories with perfect OOF answer accuracy are enabled; target "
            "single/combination matches must be unique and multi changes are additive only"
        ),
        "config": {
            "learned_source_categories": sorted(SOURCE_CATEGORIES),
            "required_source_gates": list(GATE_COLUMNS),
            "same_recording_evidence_excluded": True,
            "sequence_options_are_direct_sources": True,
            "final_repeated_sequence_pass": True,
        },
        "source_gate_audit": {
            "joint_rows": int(eligible_oof.sum()),
            "joint_correct": int(source_correct.sum()),
            "union_rows": len(source_map),
            "union_correct": int(source_union_correct.sum()),
            "union_accuracy": float(source_union_correct.mean()),
            "frequency_rows": int(frequency_oof["source_eligible"].sum())
            if frequency_oof is not None
            else 0,
            "oof_conflicts": oof_conflicts,
            "by_category": {
                category: {
                    "rows": len(group),
                    "correct": int(group["prediction"].eq(group["answer"]).sum()),
                }
                for category, group in source_by_category
            },
        },
        "target_rule_audit": audit_summary,
        "oof_overlay_diagnostic": oof_overlay_report(oof, audit),
        "test": {
            "eligible_joint_sources": int(eligible_test.sum()),
            "eligible_frequency_sources": int(
                frequency_test["source_eligible"].sum()
            )
            if frequency_test is not None
            else 0,
            "union_sources": len(test_source_map),
            "source_conflicts_resolved_by_frequency": test_conflicts,
            "linked_target_rows": len(applied),
            "overrides": int(changed.sum()),
            "override_rows": [
                {
                    "qa_id": row.qa_id,
                    "category": row.category,
                    "parent": row.parent,
                    "prediction": str(final_map[row.qa_id]),
                    "certified_letters": row.certified_letters,
                    "source_count": row.source_count,
                    "source_kinds": row.source_kinds,
                }
                for row in applied.loc[changed].itertuples()
            ],
            "final_sequence_repairs": int(sequence_decisions["changed"].sum()),
        },
        "inputs": {
            "parent": {
                "path": str(args.parent_submission.resolve()),
                "sha256": sha256_file(args.parent_submission),
            },
            "joint_complete_run": str(args.joint_complete_run.resolve()),
            "option_frequency_run": str(args.option_frequency_run.resolve())
            if args.option_frequency_run is not None
            else None,
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
