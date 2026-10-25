#!/usr/bin/env python3
"""Replay frozen structural constraints over a local-only diagnostic parent."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from cuhkx.clip_constraints import (
    apply_repeated_sequence_constraints,
    audit_repeated_sequence_constraints,
)
from cuhkx.metadata import subject_id
from cuhkx.repeated_actions import (
    SOURCE_CATEGORIES,
    apply_repeated_action_decisions,
    repeated_action_decisions,
)
from cuhkx.submission import validate_submission_frames
from scripts.apply_harn_paired_constraints import (
    paired_rows,
)
from scripts.apply_harn_paired_constraints import (
    test_decisions as decode_pairs,
)
from scripts.apply_harn_paired_constraints import (
    training_audit as audit_pairs,
)
from scripts.apply_repeated_action_evidence import (
    GATE_COLUMNS,
    as_bool,
    audit_decisions,
    merge_source_maps,
)
from scripts.build_local_evidence_baseline import (
    PROTOCOL,
    QA_COLUMNS,
    Inputs,
    evidence_path,
    sha256_file,
    validate_evidence,
    validate_ids,
    validation_report,
)

ACTION_CATEGORIES = ["single", "multi", "combination"]
LOCAL_COLUMNS = ["qa_id", "prediction", "evidence_source", "evidence_history"]
JOINT_COLUMNS = ["qa_id", "category", "prediction", *GATE_COLUMNS]
FREQUENCY_COLUMNS = [
    "qa_id", "category", "direct_frequency_gate", "joint_expansion_gate",
    "frequency_prediction", "joint_prediction",
]
STAGES = (
    "frequency", "repeated_sequence_1", "repeated_actions",
    "repeated_sequence_2", "harn_unique_pair",
)
FROZEN_RULES = {
    "stage_order": list(STAGES),
    "frequency": {
        "selection": "direct_frequency_gate OR joint_expansion_gate",
        "prediction": "frequency_prediction if direct_frequency_gate else joint_prediction",
        "original_expansion_margin": 0.5, "original_direct_margin": 0.8,
        "original_required_joint_agreement": 7,
        "cached_masks_recomputed_from_parent_independent_gate_columns": True,
    },
    "sibling_action_sources": {
        "joint_required_gates": list(GATE_COLUMNS), "all_joint_gates_required": True,
        "joint_thresholds": {
            "agreement": 7, "margin": 0.05, "complete_margin": 0.20,
            "skeleton_complete_margin": 0.15,
        },
        "frequency_gate": "direct_frequency_gate",
        "source_categories": sorted(SOURCE_CATEGORIES),
        "same_recording_excluded": True,
        "test_source_conflict_precedence": "frequency",
        "minimum_oof_multi_letter_precision": 0.999,
    },
    "harn_pair": {
        "apply_only_unique_compatible_pairs": True,
        "oof_dictionary": "refit excluding each target subject",
        "test_dictionary": "all training subjects",
    },
    "search": False, "public_feedback_used": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "train-csv", "test-csv", "sample-csv", "local-run",
        "joint-complete-run", "option-frequency-run", "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser.parse_args()


def load_gates(
    inputs: Inputs, qa: pd.DataFrame, joint_path: Path, frequency_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected = qa.loc[qa["source"].eq("HAU") & qa["category"].isin(ACTION_CATEGORIES)]
    joint = inputs.read(evidence_path(joint_path), JOINT_COLUMNS, "joint source gates")
    frequency = inputs.read(
        evidence_path(frequency_path), FREQUENCY_COLUMNS, "frequency direct and expansion gates"
    )
    expected_category = expected.set_index("qa_id")["category"]
    for name, frame, prediction_columns in (
        ("joint", joint, ["prediction"]),
        ("frequency", frequency, ["frequency_prediction", "joint_prediction"]),
    ):
        for column in prediction_columns:
            validate_evidence(
                expected, frame[["qa_id", column]].rename(columns={column: "prediction"}),
                name, complete=True, source="HAU", categories=ACTION_CATEGORIES,
            )
        categories = frame["qa_id"].map(expected_category)
        if not frame["category"].eq(categories).all():
            raise ValueError(f"{name} cached category differs from QA metadata")
    for column in GATE_COLUMNS:
        joint[column] = as_bool(joint[column])
    for column in ("direct_frequency_gate", "joint_expansion_gate"):
        frequency[column] = as_bool(frequency[column])
    if (
        frequency["direct_frequency_gate"]
        & ~frequency["category"].isin(SOURCE_CATEGORIES)
    ).any():
        raise ValueError("direct frequency gate includes an unsupported source category")
    joint["source_eligible"] = (
        joint[list(GATE_COLUMNS)].all(axis=1) & joint["category"].isin(SOURCE_CATEGORIES)
    )
    frequency["source_eligible"] = (
        frequency["direct_frequency_gate"] & frequency["category"].isin(SOURCE_CATEGORIES)
    )
    return joint, frequency


def frequency_updates(frequency: pd.DataFrame) -> pd.Series:
    chosen = frequency["direct_frequency_gate"] | frequency["joint_expansion_gate"]
    values = frequency["joint_prediction"].where(
        ~frequency["direct_frequency_gate"], frequency["frequency_prediction"]
    )
    return pd.Series(values.loc[chosen].to_numpy(), index=frequency.loc[chosen, "qa_id"])


def source_maps(
    joint: pd.DataFrame, frequency: pd.DataFrame,
) -> tuple[dict[str, str], set[str], list[dict[str, str]]]:
    # merge_source_maps implements the already frozen frequency conflict precedence.
    return merge_source_maps(joint, frequency)


def pair_updates(
    training: pd.DataFrame, target: pd.DataFrame, predictions: pd.DataFrame, *, oof: bool,
) -> tuple[pd.Series, pd.DataFrame]:
    reference = paired_rows(training)
    target_pairs = paired_rows(target)
    if oof:
        target_subjects = target_pairs["path"].map(subject_id)
        reference_subjects = reference["path"].map(subject_id)
        if target_subjects.isna().any() or reference_subjects.isna().any():
            raise ValueError("HARn OOF pairs require identifiable subjects")
        pieces = []
        for subject in sorted(target_subjects.unique()):
            held_out = target_pairs.loc[target_subjects.eq(subject)]
            fit = reference.loc[reference_subjects.ne(subject)]
            decisions = decode_pairs(fit, held_out, predictions[["qa_id", "prediction"]])
            if not decisions.empty:
                decisions["subject"] = subject
                pieces.append(decisions)
        decisions = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    else:
        decisions = decode_pairs(reference, target_pairs, predictions[["qa_id", "prediction"]])
    updates: dict[str, str] = {}
    if not decisions.empty:
        for row in decisions.loc[decisions["unique"]].itertuples():
            updates[row.action_qa_id] = row.action_prediction
            updates[row.object_qa_id] = row.object_prediction
    return pd.Series(updates, dtype=str), decisions


def applicable_action_updates(decisions: pd.DataFrame) -> pd.Series:
    if decisions.empty:
        return pd.Series(dtype=str)
    applicable = decisions["category"].eq("multi") | (
        decisions["category"].isin(SOURCE_CATEGORIES)
        & decisions["certified_letters"].str.len().eq(1)
    )
    return decisions.loc[applicable].set_index("qa_id")["prediction"]


def apply_pipeline(
    qa: pd.DataFrame, local: pd.DataFrame, joint: pd.DataFrame, frequency: pd.DataFrame,
    training_reference: pd.DataFrame, *, oof: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    validate_evidence(qa, local[["qa_id", "prediction"]], "local parent", complete=True)
    result = qa[QA_COLUMNS[:4]].set_index("qa_id").copy()
    indexed_local = local.set_index("qa_id")
    result["parent_prediction"] = indexed_local["prediction"]
    result["prediction"] = indexed_local["prediction"]
    result["base_evidence_source"] = indexed_local["evidence_source"]
    result["evidence_source"] = indexed_local["evidence_source"]
    result["evidence_history"] = indexed_local["evidence_history"]
    traces = []
    audit_frames = {}

    def update(stage: str, values: pd.Series) -> None:
        evidence = values.rename("prediction").rename_axis("qa_id").reset_index()
        validate_evidence(qa, evidence, stage)
        previous = result.loc[values.index, "prediction"]
        trace = evidence.assign(
            stage=stage, previous_prediction=previous.to_numpy(),
            changed=previous.ne(values).to_numpy(),
        )
        traces.append(trace)
        result.loc[values.index, "prediction"] = values
        result.loc[values.index, "evidence_source"] = stage
        result.loc[values.index, "evidence_history"] += "|" + stage
        result[f"prediction_{stage}"] = result["prediction"]

    update("frequency", frequency_updates(frequency))
    _, decisions = apply_repeated_sequence_constraints(qa, result["prediction"].to_dict())
    audit_frames["repeated_sequence_1"] = decisions
    update("repeated_sequence_1", applicable_action_updates(decisions))
    sources, eligible, conflicts = source_maps(joint, frequency)
    action_decisions = repeated_action_decisions(qa, sources, eligible)
    _, applied = apply_repeated_action_decisions(result["prediction"].to_dict(), action_decisions)
    audit_frames["repeated_actions"] = applied
    update("repeated_actions", applicable_action_updates(applied))
    _, decisions = apply_repeated_sequence_constraints(qa, result["prediction"].to_dict())
    audit_frames["repeated_sequence_2"] = decisions
    update("repeated_sequence_2", applicable_action_updates(decisions))
    updates, pairs = pair_updates(
        training_reference, qa, result.reset_index(), oof=oof
    )
    audit_frames["harn_pairs"] = pairs
    update("harn_unique_pair", updates)
    result["changed_from_parent"] = result["prediction"].ne(result["parent_prediction"])
    metadata = {
        "joint_source_rows": int(joint["source_eligible"].sum()),
        "frequency_source_rows": int(frequency["source_eligible"].sum()),
        "union_source_rows": len(sources), "source_conflicts": conflicts,
    }
    return result.reset_index(), pd.concat(traces, ignore_index=True), audit_frames, metadata


def rename_metric_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            ("parent_accuracy" if key == "fallback_accuracy" else key): rename_metric_keys(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [rename_metric_keys(item) for item in value]
    return value


def metrics(frame: pd.DataFrame, *, prediction: str = "prediction",
            parent: str = "parent_prediction") -> dict[str, Any]:
    evaluated = frame.copy()
    evaluated["prediction"] = frame[prediction]
    evaluated["fallback_prediction"] = frame[parent]
    evaluated["changed_from_fallback"] = frame[prediction].ne(frame[parent])
    report = rename_metric_keys(validation_report(evaluated))
    if prediction != "prediction":
        # The stored provenance names describe the final chain, not an earlier stage.
        report.pop("by_evidence_source")
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    # Record the complete fixed policy before any prediction or validation is computed.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    frozen = {
        "version": 2, "status": "frozen_before_run", "rules": FROZEN_RULES,
        "paths": {name: str(getattr(args, name).resolve()) for name in (
            "train_csv", "test_csv", "sample_csv", "local_run",
            "joint_complete_run", "option_frequency_run",
        )},
    }
    config_path = args.output_dir / "frozen_config.json"
    config_path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    inputs = Inputs()
    inputs.register(config_path, "policy frozen before prediction")
    train = inputs.read(args.train_csv, QA_COLUMNS, "training metadata")
    test = inputs.read(args.test_csv, QA_COLUMNS, "unlabeled test metadata")
    labels = inputs.read(args.train_csv, ["qa_id", "answer"], "OOF audits and HARn dictionaries")
    validate_ids(train, "training metadata")
    validate_ids(test, "test metadata")
    training_reference = train.merge(labels, on="qa_id", validate="one_to_one")
    local, gates = {}, {}
    for split, qa in (("oof", train), ("test", test)):
        filename = f"{split}_predictions.csv"
        local[split] = inputs.read(
            evidence_path(args.local_run / filename), LOCAL_COLUMNS, f"{split} local parent"
        )
        gates[split] = load_gates(
            inputs, qa, args.joint_complete_run / filename, args.option_frequency_run / filename
        )
    oof_joint, oof_frequency = gates["oof"]
    sources, _, conflicts = source_maps(oof_joint, oof_frequency)
    truth = labels.set_index("qa_id")["answer"]
    if conflicts:
        raise ValueError("joint and frequency OOF source gates disagree")
    source_correct = pd.Series(sources).eq(truth.reindex(list(sources)))
    if not sources or not source_correct.all():
        raise ValueError("the frozen sibling source gate is not perfect in OOF validation")
    sequence_audit = audit_repeated_sequence_constraints(training_reference)
    if sequence_audit.empty or not sequence_audit["valid"].all():
        raise ValueError("repeated sequence invariant failed the training audit")
    pair_audit = audit_pairs(paired_rows(training_reference))
    if (
        pair_audit.empty or not pair_audit["truth_contained"].all()
        or not pair_audit["unique"].any()
        or not pair_audit.loc[pair_audit["unique"], "unique_correct"].all()
    ):
        raise ValueError("HARn subject-excluded pair invariant failed the training audit")
    outputs, traces, audits, source_info = {}, {}, {}, {}
    for split, qa in (("oof", train), ("test", test)):
        outputs[split], traces[split], audits[split], source_info[split] = apply_pipeline(
            qa, local[split], *gates[split], training_reference, oof=split == "oof"
        )
    action_audit, action_summary = audit_decisions(
        training_reference, audits["oof"]["repeated_actions"]
    )
    oof = outputs["oof"].merge(labels, on="qa_id", validate="one_to_one", sort=False)
    oof["subject"] = oof["path"].map(subject_id)
    if oof["subject"].isna().any():
        raise ValueError("every OOF row must have an identifiable subject")
    oof["correct"] = oof["prediction"].eq(oof["answer"])
    oof["parent_correct"] = oof["parent_prediction"].eq(oof["answer"])
    outputs["oof"] = oof
    sample_path = inputs.register(args.sample_csv, "sample schema and ID order", ["qa_id"])
    if pd.read_csv(sample_path, nrows=0).columns.tolist() != ["qa_id", "prediction"]:
        raise ValueError("sample schema must be qa_id,prediction")
    sample = inputs.read(args.sample_csv, ["qa_id"], "sample ID order").assign(prediction="")
    submission = outputs["test"][["qa_id", "prediction"]]
    validate_submission_frames(test, sample, submission)
    stage_metrics = {}
    previous = "parent_prediction"
    for stage in STAGES:
        column = f"prediction_{stage}"
        stage_metrics[stage] = {
            "oof_vs_previous_stage": metrics(oof, prediction=column, parent=previous),
            "oof_vs_local_parent": metrics(oof, prediction=column),
            "test_changed_from_previous_stage": int(
                outputs["test"][column].ne(outputs["test"][previous]).sum()
            ),
        }
        previous = column
    files = {
        "oof_predictions.csv": outputs["oof"], "test_predictions.csv": outputs["test"],
        "submission.csv": submission, "oof_trace.csv": traces["oof"],
        "test_trace.csv": traces["test"], "sequence_training_audit.csv": sequence_audit,
        "action_training_audit.csv": action_audit, "harn_training_audit.csv": pair_audit,
        **{f"{split}_{name}_decisions.csv": frame
           for split, pieces in audits.items() for name, frame in pieces.items()},
    }
    report: dict[str, Any] = {
        "status": "diagnostic_only", "validation_protocol": PROTOCOL,
        "method": "frozen structural replay over local expert predictions",
        "limitations": [
            (
                "Previously selected component cross-fits and exposed validation subjects; this is "
                "neither a fully nested pipeline evaluation nor a new blind holdout."
            ),
            "This diagnostic still consumes frozen expert caches; it is not new-media inference.",
        ],
        "config": frozen, "validation": metrics(oof), "stage_metrics": stage_metrics,
        "source_gate_audit": {
            **source_info, "oof_correct_sources": int(source_correct.sum()),
        },
        "sibling_target_audit": action_summary,
        "harn_pair_audit": {
            "pairs": len(pair_audit), "truth_contained": int(pair_audit["truth_contained"].sum()),
            "unique_pairs": int(pair_audit["unique"].sum()),
            "unique_correct": int(pair_audit["unique_correct"].sum()),
        },
        "test": {
            "rows": len(outputs["test"]),
            "changed_from_local_parent": int(outputs["test"]["changed_from_parent"].sum()),
            "evidence_source_rows": outputs["test"]["evidence_source"].value_counts().to_dict(),
        },
        "inputs": inputs.verified_records(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "source_code": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in sorted([*ROOT.glob("scripts/*.py"), *ROOT.glob("src/**/*.py")])
        ],
    }
    for name, frame in files.items():
        frame.to_csv(args.output_dir / name, index=False, lineterminator="\n")
    report["outputs"] = {
        name: {"path": str((args.output_dir / name).resolve()), "rows": len(frame),
               "sha256": sha256_file(args.output_dir / name)}
        for name, frame in files.items()
    }
    report["submission_validation"] = {
        "valid": True, "rows": len(submission), "columns": submission.columns.tolist(),
        "sha256": report["outputs"]["submission.csv"]["sha256"],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    report = build(parse_args())
    print(json.dumps({
        "status": report["status"], "validation_protocol": report["validation_protocol"],
        "validation": report["validation"]["overall"],
        "worst_subject_accuracy": report["validation"]["worst_subject_accuracy"],
        "test": report["test"], "submission_validation": report["submission_validation"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
