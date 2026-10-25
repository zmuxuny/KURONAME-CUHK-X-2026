#!/usr/bin/env python3
"""Decode HAU single/multi/combination questions as one latent action set."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuhkx.actions import HAU_CLASS_TEXT, hau_action_class
from cuhkx.clip_constraints import apply_sequence_constraints
from cuhkx.joint_actions import JOINT_CATEGORIES, decode_joint_state, enumerate_joint_states
from cuhkx.metadata import clip_key, feature_unit, option_atoms, subject_id
from cuhkx.submission import available_letters, read_csv_str, validate_submission_files
from cuhkx.temporal import softmax

LOCKED_SUBJECTS = frozenset(
    {"user3", "user7", "user9", "user17", "user19", "user21", "user22"}
)
FAMILY_WEIGHTS: dict[str, dict[str, float]] = {
    "calibrated": {"calibrated": 1.0},
    "direct_v16_peak": {"v16_peak": 1.0},
    "cal_v8": {"calibrated": 0.5, "v8": 0.5},
    "cal_v16": {"calibrated": 0.5, "v16": 0.5},
    "cal_multiscale": {"calibrated": 0.5, "v8": 0.25, "v16": 0.25},
    "cal_skeleton": {"calibrated": 0.75, "skeleton": 0.25},
    "skeleton_direct": {"skeleton": 1.0},
}
SELECTED_FAMILY = "cal_v16"


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
    parser.add_argument("--parent-submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("full", "consensus"), default="consensus")
    parser.add_argument("--minimum-family-agreement", type=int, default=7)
    parser.add_argument("--minimum-margin", type=float, default=0.0)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def temporal_presence_map(
    path: Path,
    *,
    temperature: float,
    top_windows: int,
) -> dict[str, np.ndarray]:
    archive = np.load(path)
    required = {"logits", "units", "weights", "classes"}
    if not required <= set(archive.files):
        raise ValueError(f"temporal archive lacks required keys: {path}")
    logits = np.asarray(archive["logits"], dtype=np.float64)
    units = np.asarray(archive["units"]).astype(str)
    weights = np.asarray(archive["weights"], dtype=np.float64)
    classes = np.asarray(archive["classes"], dtype=int)
    if logits.ndim != 4 or logits.shape[0] != len(weights) or logits.shape[1] != len(units):
        raise ValueError(f"unexpected temporal logits shape in {path}: {logits.shape}")
    if not np.array_equal(classes, np.arange(40)):
        raise ValueError(f"unexpected action class order in {path}")
    if not 1 <= top_windows <= logits.shape[2]:
        raise ValueError(f"invalid top-window count for {path}: {top_windows}")
    probabilities = softmax(logits, temperature)
    weights = weights / weights.sum()
    ensemble = np.tensordot(weights, probabilities, axes=(0, 0))
    presence = np.partition(ensemble, -top_windows, axis=1)[:, -top_windows:].mean(axis=1)
    return dict(zip(units, presence.astype(np.float32), strict=True))


def combine_unit_maps(
    training_path: Path,
    test_path: Path,
    *,
    temperature: float,
    top_windows: int,
) -> dict[str, np.ndarray]:
    training = temporal_presence_map(
        training_path, temperature=temperature, top_windows=top_windows
    )
    testing = temporal_presence_map(
        test_path, temperature=temperature, top_windows=top_windows
    )
    overlap = set(training) & set(testing)
    if overlap:
        raise ValueError(f"train/test temporal unit overlap: {sorted(overlap)[:3]}")
    return training | testing


def calibrated_presence_map(
    qa: pd.DataFrame,
    presence_run: Path,
    direct_fallback: dict[str, np.ndarray],
    *,
    training: bool,
) -> dict[str, np.ndarray]:
    filename = "oof_predictions.csv" if training else "test_predictions.csv"
    predictions = read_csv_str(presence_run / filename).set_index("qa_id")
    output: dict[str, np.ndarray] = {}
    values: dict[tuple[str, int], list[float]] = defaultdict(list)
    rows = qa.loc[qa["source"].eq("HAU") & qa["category"].isin({"single", "multi"})]
    for _, row in rows.iterrows():
        qa_id = str(row["qa_id"])
        if qa_id not in predictions.index:
            raise ValueError(f"presence output lacks {qa_id!r} in {filename}")
        unit = feature_unit(row["path"], training=training)
        for letter in available_letters(row):
            atoms = option_atoms(row[letter])
            if len(atoms) != 1:
                raise ValueError(f"expected a singleton option at {qa_id}/{letter}")
            column = f"option_score_{letter}"
            if column not in predictions:
                raise ValueError(f"presence output lacks {column!r}; rerun current pipeline")
            values[(unit, hau_action_class(atoms[0]))].append(
                float(predictions.at[qa_id, column])
            )

    expected_units = {
        feature_unit(row["path"], training=training)
        for _, row in qa.loc[qa["source"].eq("HAU")].iterrows()
    }
    for unit in expected_units:
        if unit not in direct_fallback:
            raise ValueError(f"direct visual scores lack HAU unit {unit!r}")
        vector = direct_fallback[unit].copy()
        for class_id in range(40):
            candidates = values.get((unit, class_id))
            if candidates:
                if max(candidates) - min(candidates) > 1e-5:
                    raise ValueError(f"inconsistent calibrated score for {unit}/class {class_id}")
                vector[class_id] = float(np.mean(candidates))
        output[unit] = vector
    return output


def skeleton_presence_map(paths: list[Path]) -> dict[str, np.ndarray]:
    by_unit: dict[str, list[np.ndarray]] = defaultdict(list)
    for path in paths:
        archive_path = path / "action_scores.npz"
        archive = np.load(archive_path)
        classes = np.asarray(archive["classes"], dtype=int)
        if not np.array_equal(classes, np.arange(40)):
            raise ValueError(f"unexpected Skeleton class order in {archive_path}")
        for prefix in ("train", "test"):
            units = np.asarray(archive[f"{prefix}_units"]).astype(str)
            scores = np.asarray(archive[f"{prefix}_scores"], dtype=np.float32)
            if scores.ndim != 3 or scores.shape[1:] != (len(units), 40):
                raise ValueError(f"unexpected Skeleton score shape in {archive_path}")
            mean_views = scores.mean(axis=0)
            for unit, vector in zip(units, mean_views, strict=True):
                by_unit[unit].append(vector)
    return {
        unit: np.mean(np.stack(vectors), axis=0).astype(np.float32)
        for unit, vectors in by_unit.items()
    }


def blend_vectors(vectors: list[np.ndarray], weights: list[float]) -> np.ndarray:
    if len(vectors) != len(weights) or not vectors:
        raise ValueError("vectors and weights must be nonempty and aligned")
    normalized = np.asarray(weights, dtype=np.float64)
    normalized /= normalized.sum()
    logits = np.zeros_like(np.asarray(vectors[0], dtype=np.float64))
    for weight, vector in zip(normalized, vectors, strict=True):
        probability = np.asarray(vector, dtype=np.float64).clip(1e-5, 1 - 1e-5)
        logits += weight * np.log(probability / (1 - probability))
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def family_probabilities(
    unit: str,
    sources: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, float]]:
    if unit not in sources["calibrated"]:
        raise ValueError(f"calibrated scores lack {unit!r}")
    fallback = sources["calibrated"][unit]
    result: dict[str, dict[str, float]] = {}
    for family, source_weights in FAMILY_WEIGHTS.items():
        vectors = [sources[name].get(unit, fallback) for name in source_weights]
        blended = blend_vectors(vectors, list(source_weights.values()))
        result[family] = {
            HAU_CLASS_TEXT[class_id]: float(blended[class_id]) for class_id in range(40)
        }
    return result


def decode_frame(
    qa: pd.DataFrame,
    sources: dict[str, dict[str, np.ndarray]],
    *,
    training: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = qa.loc[qa["source"].eq("HAU")].copy()
    work["_clip"] = work["path"].map(clip_key)
    row_records: list[dict[str, object]] = []
    clip_records: list[dict[str, object]] = []
    for clip, group in work.groupby("_clip", sort=False):
        unit = feature_unit(group.iloc[0]["path"], training=training)
        probability_families = family_probabilities(unit, sources)
        decisions = {
            family: decode_joint_state(group, probabilities)
            for family, probabilities in probability_families.items()
        }
        answer_maps = {
            family: decision.state.answer_map() for family, decision in decisions.items()
        }
        selected = decisions[SELECTED_FAMILY]
        selected_answers = answer_maps[SELECTED_FAMILY]
        clip_record: dict[str, object] = {
            "clip": clip,
            "unit": unit,
            "candidate_count": selected.candidate_count,
            "atom_count": selected.atom_count,
            "selected_margin": selected.margin,
        }
        if training:
            truth = {
                str(row["category"]): str(row["answer"])
                for _, row in group.loc[group["category"].isin(JOINT_CATEGORIES)].iterrows()
            }
            feasible = any(
                all(state.answer_map().get(category) == answer for category, answer in truth.items())
                for state in enumerate_joint_states(group)
            )
            clip_record["truth_feasible"] = feasible
            clip_record["selected_state_correct"] = selected_answers == truth
            clip_record["subject"] = subject_id(group.iloc[0]["path"])
        clip_records.append(clip_record)

        for _, row in group.loc[group["category"].isin(JOINT_CATEGORIES)].iterrows():
            category = str(row["category"])
            predictions = {
                family: answers[category] for family, answers in answer_maps.items()
            }
            agreement = max(Counter(predictions.values()).values())
            record: dict[str, object] = {
                "qa_id": row["qa_id"],
                "clip": clip,
                "unit": unit,
                "category": category,
                "prediction": predictions[SELECTED_FAMILY],
                "agreement_count": agreement,
                "margin": selected.margin,
                "candidate_count": selected.candidate_count,
                **{f"prediction_{family}": value for family, value in predictions.items()},
                **{
                    f"margin_{family}": decisions[family].margin
                    for family in FAMILY_WEIGHTS
                },
            }
            if training:
                record["answer"] = row["answer"]
                record["correct"] = predictions[SELECTED_FAMILY] == row["answer"]
                record["subject"] = subject_id(row["path"])
            row_records.append(record)
    return pd.DataFrame(row_records), pd.DataFrame(clip_records)


def accuracy_report(frame: pd.DataFrame, prediction_column: str) -> dict[str, object]:
    correct = frame[prediction_column].eq(frame["answer"])
    subject_accuracy = correct.groupby(frame["subject"]).mean().sort_values()
    category = (
        frame.assign(_correct=correct)
        .groupby("category")["_correct"]
        .agg(["size", "mean"])
    )
    return {
        "accuracy": float(correct.mean()),
        "worst_subject": float(subject_accuracy.iloc[0]),
        "category": {
            name: {"rows": int(row["size"]), "accuracy": float(row["mean"])}
            for name, row in category.iterrows()
        },
    }


def confidence_report(frame: pd.DataFrame) -> list[dict[str, object]]:
    result = []
    for agreement in range(4, len(FAMILY_WEIGHTS) + 1):
        for margin in (0.0, 0.01, 0.02, 0.05, 0.10, 0.20):
            selected = frame["agreement_count"].ge(agreement) & frame["margin"].ge(margin)
            locked = selected & frame["subject"].isin(LOCKED_SUBJECTS)
            result.append(
                {
                    "minimum_agreement": agreement,
                    "minimum_margin": margin,
                    "rows": int(selected.sum()),
                    "accuracy": float(frame.loc[selected, "correct"].mean())
                    if selected.any()
                    else None,
                    "locked_rows": int(locked.sum()),
                    "locked_accuracy": float(frame.loc[locked, "correct"].mean())
                    if locked.any()
                    else None,
                }
            )
    return result


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    if not 1 <= args.minimum_family_agreement <= len(FAMILY_WEIGHTS):
        raise ValueError("minimum family agreement is outside the family count")
    if args.minimum_margin < 0:
        raise ValueError("minimum margin must be nonnegative")

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
    v16_peak = combine_unit_maps(
        args.train_logits_s16,
        args.test_logits_s16,
        temperature=5.0,
        top_windows=1,
    )
    calibrated = calibrated_presence_map(
        train, args.presence_run, v8, training=True
    ) | calibrated_presence_map(test, args.presence_run, v8, training=False)
    sources = {
        "calibrated": calibrated,
        "v8": v8,
        "v16": v16,
        "v16_peak": v16_peak,
        "skeleton": skeleton_presence_map(args.skeleton_action_run),
    }

    oof, oof_clips = decode_frame(train, sources, training=True)
    test_predictions, test_clips = decode_frame(test, sources, training=False)
    parent = read_csv_str(args.parent_submission)
    prediction_column = str(parent.columns[1])
    parent_map = parent.set_index("qa_id")[prediction_column]
    test_predictions["parent"] = test_predictions["qa_id"].map(parent_map)
    if test_predictions["parent"].isna().any():
        raise ValueError("parent submission lacks joint-decoder qa_id values")
    if args.mode == "full":
        eligible = pd.Series(True, index=test_predictions.index)
    else:
        eligible = test_predictions["agreement_count"].ge(args.minimum_family_agreement)
        eligible &= test_predictions["margin"].ge(args.minimum_margin)
    test_predictions["eligible"] = eligible
    test_predictions["override"] = eligible & test_predictions["prediction"].ne(
        test_predictions["parent"]
    )

    intermediate = parent.copy()
    overlay = test_predictions.loc[test_predictions["override"]].set_index("qa_id")["prediction"]
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
    validation = validate_submission_files(args.test_csv, args.sample_csv, submission_path)

    locked = oof["subject"].isin(LOCKED_SUBJECTS)
    input_paths = [
        args.train_logits_s8,
        args.test_logits_s8,
        args.train_logits_s16,
        args.test_logits_s16,
        args.presence_run / "oof_predictions.csv",
        args.presence_run / "test_predictions.csv",
        *(path / "action_scores.npz" for path in args.skeleton_action_run),
        args.parent_submission,
    ]
    report = {
        "method": "clip-level latent action-set graph for HAU single/multi/combination",
        "config": {
            "selected_family": SELECTED_FAMILY,
            "family_weights_log_odds": FAMILY_WEIGHTS,
            "mode": args.mode,
            "minimum_family_agreement": args.minimum_family_agreement,
            "minimum_margin": args.minimum_margin,
        },
        "validation_protocol": (
            "six-fold subject-group OOF calibrated visual scores; frozen direct visual views; "
            "HARn-only Skeleton transfer nested by held-out HAU subject; no oracle multi cardinality"
        ),
        "validation_rows": len(oof),
        "validation_clips": len(oof_clips),
        "truth_feasible_clips": int(oof_clips["truth_feasible"].sum()),
        "selected_state_exact_accuracy": float(oof_clips["selected_state_correct"].mean()),
        "selected": accuracy_report(oof, "prediction"),
        "families": {
            family: accuracy_report(oof, f"prediction_{family}")
            for family in FAMILY_WEIGHTS
        },
        "locked_transfer_slice": accuracy_report(oof.loc[locked], "prediction"),
        "confidence": confidence_report(oof),
        "test_rows": len(test_predictions),
        "test_clips": len(test_clips),
        "eligible_rows": int(test_predictions["eligible"].sum()),
        "overrides": int(test_predictions["override"].sum()),
        "override_rows": test_predictions.loc[
            test_predictions["override"],
            ["qa_id", "category", "parent", "prediction", "agreement_count", "margin"],
        ].to_dict("records"),
        "sequence_constraint_changes_after_joint_decode": int(
            constraint_decisions["changed"].sum()
        ),
        "inputs": [
            {"path": str(path.resolve()), "sha256": file_sha256(path)} for path in input_paths
        ],
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
