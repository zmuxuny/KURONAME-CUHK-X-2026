#!/usr/bin/env python3
"""Compose frozen local experts into a complete, unsubmitted diagnostic baseline.

This is a cache-bound component-cross-fitted diagnostic, not a nested evaluation or a
media-to-answer inference entry point. Source CSVs are parsed with explicit column
allowlists; historical parent predictions and override masks never enter decoding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from cuhkx.metadata import subject_id
from cuhkx.submission import (
    LETTERS,
    available_letters,
    validate_prediction,
    validate_submission_frames,
)
from scripts.run_harn_object_interaction import skeleton_predictions

QA_COLUMNS = ["qa_id", "source", "category", "path", *LETTERS]
PREDICTION_COLUMNS = frozenset({"prediction", "prediction_motion"})
PROBABILITY_KEYS = (
    "model_names", "classes", "train_units", "test_units",
    "oof_probabilities", "test_probabilities",
)
PROTOCOL = "component-cross-fitted diagnostic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train-csv", "test-csv", "sample-csv", "fallback-oof", "fallback-test"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/local_evidence_v1.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Inputs:
    """Record the bytes used for provenance, separately from inference column selection."""

    def __init__(self) -> None:
        self.records: dict[Path, dict[str, Any]] = {}

    def register(self, path: Path, role: str, columns: list[str] | None = None) -> Path:
        path = path.resolve()
        if path not in self.records:
            self.records[path] = {
                "path": str(path), "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size, "uses": [],
            }
        self.records[path]["uses"].append({"role": role, "parsed_columns": columns})
        return path

    def read(self, path: Path, columns: list[str], role: str) -> pd.DataFrame:
        path = self.register(path, role, columns)
        return pd.read_csv(
            path, usecols=columns, dtype=str, keep_default_na=False, encoding="utf-8-sig"
        ).loc[:, columns]

    def verified_records(self) -> list[dict[str, Any]]:
        for path, record in self.records.items():
            if sha256_file(path) != record["sha256"]:
                raise ValueError(f"input changed during composition: {path}")
        return list(self.records.values())


def evidence_path(value: str | Path) -> Path:
    path = Path(value)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if path.name == "submission.csv" or "submissions" in path.parts:
        raise ValueError(f"submission artifacts are forbidden as evidence inputs: {path}")
    return path


def read_prediction(
    inputs: Inputs, path: Path, column: str, role: str
) -> pd.DataFrame:
    if column not in PREDICTION_COLUMNS:
        raise ValueError(f"prediction column is not allowlisted: {column!r}")
    return inputs.read(evidence_path(path), ["qa_id", column], role).rename(
        columns={column: "prediction"}
    )


def validate_ids(frame: pd.DataFrame, name: str) -> None:
    if frame["qa_id"].eq("").any() or not frame["qa_id"].is_unique:
        raise ValueError(f"{name} must have nonempty, unique qa_id values")


def validate_evidence(
    qa: pd.DataFrame,
    evidence: pd.DataFrame,
    name: str,
    *,
    complete: bool = False,
    source: str | None = None,
    categories: list[str] | None = None,
) -> None:
    validate_ids(evidence, name)
    unknown = set(evidence["qa_id"]) - set(qa["qa_id"])
    if unknown:
        raise ValueError(f"{name} contains unknown qa_id values: {sorted(unknown)[:5]}")
    if complete and set(evidence["qa_id"]) != set(qa["qa_id"]):
        raise ValueError(f"{name} does not exactly cover the QA ID set")
    metadata = qa.set_index("qa_id").loc[evidence["qa_id"]]
    if source is not None and not metadata["source"].eq(source).all():
        raise ValueError(f"{name} has rows outside its source scope {source!r}")
    if categories is not None and not metadata["category"].isin(categories).all():
        raise ValueError(f"{name} has rows outside its category scope {categories!r}")
    for (_, row), prediction in zip(
        metadata.iterrows(), evidence["prediction"], strict=True
    ):
        error = validate_prediction(prediction, row["category"], available_letters(row))
        if prediction != prediction.upper():
            error = "prediction must use uppercase letters"
        if error:
            raise ValueError(f"{name} has invalid prediction for {row.name}: {error}")


def compose_predictions(
    qa: pd.DataFrame,
    fallback: pd.DataFrame,
    layers: list[tuple[dict[str, Any], pd.DataFrame]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Replace every covered row, including agreement with the preceding prediction."""

    validate_ids(qa, "QA metadata")
    validate_evidence(qa, fallback, "structural fallback", complete=True)
    result = qa[QA_COLUMNS[:4]].copy().set_index("qa_id")
    result["fallback_prediction"] = fallback.set_index("qa_id")["prediction"]
    result["prediction"] = result["fallback_prediction"]
    result["evidence_source"] = "structural_fallback"
    result["evidence_history"] = "structural_fallback"
    stages = []
    for spec, evidence in layers:
        name = spec["name"]
        validate_evidence(
            qa, evidence, name, source=spec["source"], categories=spec["categories"]
        )
        values = evidence.set_index("qa_id")["prediction"]
        previous = result.loc[values.index, "prediction"]
        stages.append({
            "name": name, "covered_rows": len(values),
            "changed_rows": int(previous.ne(values).sum()),
        })
        result.loc[values.index, "prediction"] = values
        result.loc[values.index, "evidence_source"] = name
        result.loc[values.index, "evidence_history"] += "|" + name
    result["changed_from_fallback"] = result["prediction"].ne(result["fallback_prediction"])
    return result.reset_index(), stages


def object_predictions(
    train: pd.DataFrame, test: pd.DataFrame, labels: pd.DataFrame,
    archive_path: Path, inputs: Inputs,
) -> dict[str, pd.DataFrame]:
    """Reuse the existing subject-excluded object dictionary and class-posterior decoder."""

    archive_path = inputs.register(
        evidence_path(archive_path), "HARn object action probabilities", list(PROBABILITY_KEYS)
    )
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in PROBABILITY_KEYS}
    classes = arrays["classes"]
    names = arrays["model_names"]
    if classes.ndim != 1 or len(set(classes)) != len(classes) or not len(classes):
        raise ValueError("object probability archive has invalid classes")
    if names.ndim != 1 or len(set(names)) != len(names) or not len(names):
        raise ValueError("object probability archive has invalid model names")
    mapping_frame = train.loc[
        train["source"].eq("HARn") & train["category"].eq("object_interaction")
    ].merge(labels, on="qa_id", validate="one_to_one")
    output = {}
    for split, frame, training, unit_key in (
        ("oof", mapping_frame, True, "train_units"),
        ("test", test, False, "test_units"),
    ):
        units = arrays[unit_key].astype(str)
        probabilities = arrays[f"{split}_probabilities"]
        if units.ndim != 1 or len(set(units)) != len(units):
            raise ValueError(f"object archive has invalid {unit_key}")
        if probabilities.shape != (len(names), len(units), len(classes)):
            raise ValueError(f"object archive has invalid {split} probability shape")
        if (
            not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=-1), 1, atol=1e-5)
        ):
            raise ValueError(f"object archive has invalid {split} probabilities")
        rows = frame.loc[
            frame["source"].eq("HARn") & frame["category"].eq("object_interaction")
        ]
        decoded = skeleton_predictions(
            rows, probabilities=probabilities, units=units, model_names=names,
            classes=classes, mapping_frame=mapping_frame, training=training,
        )
        output[split] = decoded.loc[:, ["qa_id", "prediction"]]
    return output


def score_rows(frame: pd.DataFrame) -> dict[str, Any]:
    correct = frame["prediction"].eq(frame["answer"])
    parent_correct = frame["fallback_prediction"].eq(frame["answer"])
    changed = frame["changed_from_fallback"]
    return {
        "rows": len(frame), "accuracy": float(correct.mean()) if len(frame) else None,
        "fallback_accuracy": float(parent_correct.mean()) if len(frame) else None,
        "net_correct": int(correct.sum() - parent_correct.sum()),
        "corrected_rows": int((correct & ~parent_correct).sum()),
        "harmed_rows": int((~correct & parent_correct).sum()),
        "changed_subset": {
            "rows": int(changed.sum()),
            "accuracy": float(correct[changed].mean()) if changed.any() else None,
            "fallback_accuracy": float(parent_correct[changed].mean()) if changed.any() else None,
        },
    }


def validation_report(frame: pd.DataFrame) -> dict[str, Any]:
    subjects = {
        name: score_rows(group) for name, group in frame.groupby("subject", sort=True)
    }
    return {
        "overall": score_rows(frame),
        "by_source_category": {
            f"{source}/{category}": score_rows(group)
            for (source, category), group in frame.groupby(["source", "category"], sort=True)
        },
        "by_subject": subjects,
        "worst_subject_accuracy": min(value["accuracy"] for value in subjects.values()),
        "by_evidence_source": {
            name: score_rows(group) for name, group in frame.groupby("evidence_source", sort=True)
        },
    }


def check_count(actual: int, expected: int, description: str) -> None:
    if actual != expected:
        raise ValueError(f"{description}: expected {expected} rows, got {actual}")


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    inputs = Inputs()
    config_path = inputs.register(args.config, "fixed composition configuration")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("version") != 1 or not config.get("experts"):
        raise ValueError("configuration must contain version 1 and nonempty experts")
    names = [spec["name"] for spec in config["experts"]]
    if len(set(names)) != len(names) or "structural_fallback" in names:
        raise ValueError("expert names must be unique and differ from structural_fallback")
    train = inputs.read(args.train_csv, QA_COLUMNS, "training metadata")
    test = inputs.read(args.test_csv, QA_COLUMNS, "test metadata; prediction column excluded")
    labels = inputs.read(args.train_csv, ["qa_id", "answer"], "OOF audit and object dictionary")
    for split, frame in (("oof", train), ("test", test)):
        validate_ids(frame, f"{split} QA metadata")
        check_count(len(frame), config["expected_qa_rows"][split], f"{split} QA")
    sample_path = inputs.register(args.sample_csv, "sample schema and ID order", ["qa_id"])
    header = pd.read_csv(sample_path, nrows=0, encoding="utf-8-sig").columns.tolist()
    if header != ["qa_id", "prediction"]:
        raise ValueError(f"unexpected sample schema: {header}")
    sample = inputs.read(args.sample_csv, ["qa_id"], "sample ID order")
    sample["prediction"] = ""  # Sample values are never used as evidence.
    fallbacks = {
        split: read_prediction(inputs, path, "prediction", f"{split} structural fallback")
        for split, path in (("oof", args.fallback_oof), ("test", args.fallback_test))
    }
    layers: dict[str, list[tuple[dict[str, Any], pd.DataFrame]]] = {"oof": [], "test": []}
    for spec in config["experts"]:
        if spec["kind"] == "prediction_csv":
            predictions = {
                split: read_prediction(
                    inputs, evidence_path(spec[split]), spec["column"], f"{spec['name']} {split}"
                )
                for split in layers
            }
        elif spec["kind"] == "action_probability_object":
            if spec["source"] != "HARn" or spec["categories"] != ["object_interaction"]:
                raise ValueError("action_probability_object must be restricted to HARn objects")
            predictions = object_predictions(train, test, labels, Path(spec["archive"]), inputs)
        else:
            raise ValueError(f"unknown expert kind: {spec['kind']!r}")
        for split, split_layers in layers.items():
            check_count(len(predictions[split]), spec["expected_rows"][split], spec["name"] + split)
            split_layers.append((spec, predictions[split]))
    oof, oof_stages = compose_predictions(train, fallbacks["oof"], layers["oof"])
    decisions, test_stages = compose_predictions(test, fallbacks["test"], layers["test"])
    for split, frame in (("oof", oof), ("test", decisions)):
        check_count(
            int(frame["evidence_source"].eq("structural_fallback").sum()),
            config["expected_final_fallback_rows"][split], split + " final structural fallback",
        )
    oof = oof.merge(labels, on="qa_id", validate="one_to_one", sort=False)
    oof["subject"] = oof["path"].map(subject_id)
    if oof["subject"].isna().any():
        raise ValueError("every OOF row must have an identifiable validation subject")
    oof["correct"] = oof["prediction"].eq(oof["answer"])
    oof["fallback_correct"] = oof["fallback_prediction"].eq(oof["answer"])
    submission = decisions[["qa_id", "prediction"]]
    validate_submission_frames(test, sample, submission)
    code_paths = sorted([*ROOT.glob("scripts/*.py"), *ROOT.glob("src/**/*.py")])
    source_code = [
        {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in code_paths
    ]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    report: dict[str, Any] = {
        "method": "fixed local evidence composition over an independent structural fallback",
        "status": "diagnostic_only",
        "validation_protocol": PROTOCOL,
        "limitations": [
            (
                "Experts use previously selected component cross-fits with differing subject folds; "
                "this is not a fully nested evaluation or a new blind holdout."
            ),
            "Cached expert predictions do not supply media-to-answer inference on new data.",
            (
                "Raw input hashes cover complete artifact bytes; only listed columns or arrays "
                "participate in inference, never historical parent or override fields."
            ),
        ],
        "config": config, "validation": validation_report(oof),
        "test": {
            "rows": len(decisions),
            "changed_from_fallback": int(decisions["changed_from_fallback"].sum()),
            "evidence_source_rows": decisions["evidence_source"].value_counts().to_dict(),
            "source_category_rows": {
                f"{source}/{category}": len(group)
                for (source, category), group in decisions.groupby(["source", "category"])
            },
        },
        "stages": {"oof": oof_stages, "test": test_stages},
        "git_commit": revision, "source_code": source_code,
        "inputs": inputs.verified_records(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "oof_predictions.csv": oof, "test_predictions.csv": decisions,
        "submission.csv": submission,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / name, index=False, lineterminator="\n")
    report["outputs"] = {
        name: {"path": str((args.output_dir / name).resolve()), "rows": len(frame),
               "sha256": sha256_file(args.output_dir / name)}
        for name, frame in outputs.items()
    }
    report["submission_validation"] = {
        "valid": True, "rows": len(submission), "columns": submission.columns.tolist(),
        "sha256": report["outputs"]["submission.csv"]["sha256"],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    report = build(parse_args())
    print(json.dumps({
        "status": report["status"], "validation_protocol": report["validation_protocol"],
        "validation": report["validation"]["overall"],
        "worst_subject_accuracy": report["validation"]["worst_subject_accuracy"],
        "test": report["test"], "outputs": report["outputs"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
