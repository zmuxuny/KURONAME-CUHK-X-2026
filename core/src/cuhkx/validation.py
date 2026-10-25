"""Local accuracy, paired subject bootstrap, and declared split provenance checks.

The split checker validates a manifest, not the historical execution of its training
code. In particular, old component cross-fits never become a nested validation by
passing their prediction CSV through this module.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cuhkx.metadata import subject_id
from cuhkx.submission import LETTERS, available_letters, validate_prediction

QA_COLUMNS = ["qa_id", "source", "category", "path", *LETTERS, "answer"]
VALIDATION_KINDS = {"component_diagnostic", "subject_excluded", "nested_subject"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prediction_path(path: Path) -> Path:
    resolved = path.resolve()
    if (
        "submission" in resolved.name.lower()
        or "submissions" in resolved.parts
        or "external" in resolved.parts
        or any(part.startswith("public_") for part in resolved.parts)
    ):
        raise ValueError(f"public or submission artifacts are forbidden evaluation inputs: {path}")
    return resolved


def _unique_strings(values: Any, name: str, *, nonempty: bool = True) -> set[str]:
    if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values):
        raise ValueError(f"{name} must be a list of nonempty strings")
    if len(set(values)) != len(values) or (nonempty and not values):
        raise ValueError(f"{name} must be nonempty and unique")
    return set(values)


def prepare_qa(qa: pd.DataFrame, scopes: list[str] | None = None) -> pd.DataFrame:
    missing = set(QA_COLUMNS) - set(qa.columns)
    if missing:
        raise ValueError(f"training QA missing columns: {sorted(missing)}")
    if qa.empty or qa["qa_id"].eq("").any() or not qa["qa_id"].is_unique:
        raise ValueError("training QA must have nonempty unique qa_id values")
    frame = qa.loc[:, QA_COLUMNS].copy()
    frame["subject"] = frame["path"].map(subject_id)
    if frame["subject"].isna().any():
        raise ValueError("every training QA path must identify its subject")
    if not frame["source"].isin(["HAU", "HARn"]).all():
        raise ValueError("unknown training QA source")
    if scopes:
        if len(set(scopes)) != len(scopes):
            raise ValueError("duplicate source:category scope")
        available = frame["source"] + ":" + frame["category"]
        unknown = set(scopes) - set(available)
        if unknown:
            raise ValueError(f"empty or unknown source:category scope: {sorted(unknown)}")
        frame = frame.loc[available.isin(scopes)].copy()
    for row in frame.to_dict("records"):
        _validate_answer(row["answer"], row, "official answer")
    return frame.reset_index(drop=True)


def _validate_answer(prediction: Any, row: dict[str, Any], name: str) -> None:
    if not isinstance(prediction, str) or prediction != prediction.upper():
        raise ValueError(f"{name} must be an uppercase string for {row['qa_id']}")
    error = validate_prediction(prediction, row["category"], available_letters(row))
    if error:
        raise ValueError(f"invalid {name} for {row['qa_id']}: {error}")


def attach_predictions(
    qa: pd.DataFrame, predictions: pd.DataFrame, *, column: str = "prediction",
    require_order: bool = False,
) -> tuple[pd.DataFrame, bool]:
    if not {"qa_id", "prediction"}.issubset(predictions.columns):
        raise ValueError("OOF predictions require qa_id,prediction columns")
    ids = predictions["qa_id"]
    if ids.eq("").any() or not ids.is_unique:
        raise ValueError("OOF predictions must have nonempty unique qa_id values")
    if set(ids) != set(qa["qa_id"]):
        raise ValueError("OOF predictions must exactly cover the declared QA scope")
    matches_order = ids.tolist() == qa["qa_id"].tolist()
    if require_order and not matches_order:
        raise ValueError("OOF prediction order does not match official QA scope order")
    frame = qa.copy()
    frame[column] = predictions.set_index("qa_id").loc[qa["qa_id"], "prediction"].to_numpy()
    for row in frame.to_dict("records"):
        _validate_answer(row[column], row, column)
    return frame, matches_order


def _split_sets(fold: dict[str, Any], name: str, allowed: set[str]) -> tuple[set, set]:
    train = _unique_strings(fold.get("train_subjects"), name + " train_subjects")
    valid = _unique_strings(fold.get("validation_subjects"), name + " validation_subjects")
    if train & valid:
        raise ValueError(f"{name} train/validation subjects overlap")
    if (train | valid) - allowed:
        raise ValueError(f"{name} contains subjects outside its allowed training universe")
    return train, valid


def validate_protocol(
    protocol: dict[str, Any], qa: pd.DataFrame, all_subjects: set[str], *, base_dir: Path,
) -> dict[str, Any]:
    if protocol.get("schema_version") != 1:
        raise ValueError("protocol schema_version must be 1")
    kind = protocol.get("validation_kind")
    if kind not in VALIDATION_KINDS:
        raise ValueError(f"unknown validation_kind: {kind!r}")
    folds = protocol.get("folds", [])
    if not isinstance(folds, list) or (kind != "component_diagnostic" and not folds):
        raise ValueError("subject-excluded or nested claims require nonempty folds")
    expected_components = _unique_strings(
        protocol.get("fitted_components", []), "fitted_components", nonempty=False
    )
    unverified = []
    if not expected_components:
        unverified.append("The complete list of fitted upstream components was not supplied.")
    seen_ids: set[str] = set()
    seen_fold_ids: set[str] = set()
    seen_subjects: set[str] = set()
    checked_files: dict[str, dict[str, str]] = {}
    for fold in folds:
        name = fold.get("fold_id")
        if not isinstance(name, str) or not name or name in seen_fold_ids:
            raise ValueError("fold_id must be a unique nonempty string")
        seen_fold_ids.add(name)
        train, valid = _split_sets(fold, name, all_subjects)
        if valid & seen_subjects:
            raise ValueError("an outer validation subject occurs in multiple folds")
        seen_subjects.update(valid)
        ids = _unique_strings(fold.get("prediction_qa_ids"), name + " prediction_qa_ids")
        expected_ids = set(qa.loc[qa["subject"].isin(valid), "qa_id"])
        if ids != expected_ids or ids & seen_ids:
            raise ValueError(f"{name} prediction rows do not exactly match its validation subjects")
        seen_ids.update(ids)
        components = fold.get("upstream_training", [])
        if not isinstance(components, list):
            raise TypeError("upstream_training must be a list")
        component_names: set[str] = set()
        for component in components:
            component_name = component.get("name")
            if not isinstance(component_name, str) or not component_name:
                raise ValueError("upstream component needs a nonempty name")
            if component_name in component_names:
                raise ValueError("duplicate upstream component name")
            component_names.add(component_name)
            subjects = _unique_strings(
                component.get("train_subjects"), name + "/" + component_name + " subjects"
            )
            if subjects - train:
                raise ValueError(f"{name}/{component_name} trained outside outer training subjects")
            evidence = component.get("provenance")
            if not evidence:
                unverified.append(f"{name}/{component_name}: training provenance is missing.")
                continue
            if not isinstance(evidence, dict) or not {"path", "sha256"} <= evidence.keys():
                raise ValueError("provenance requires path and sha256")
            path = Path(evidence["path"])
            path = (base_dir / path).resolve() if not path.is_absolute() else path.resolve()
            if not path.is_file():
                unverified.append(f"{name}/{component_name}: provenance file is unavailable.")
                continue
            if str(path) not in checked_files:
                checked_files[str(path)] = {"path": str(path), "sha256": sha256_file(path)}
            if checked_files[str(path)]["sha256"] != evidence["sha256"]:
                raise ValueError(f"{name}/{component_name}: training provenance SHA256 mismatch")
        if component_names != expected_components or not component_names:
            unverified.append(f"{name}: complete fitted-component training provenance is unverified.")
        if kind == "nested_subject":
            inner = fold.get("inner_folds", [])
            if not isinstance(inner, list) or not inner:
                raise ValueError(f"{name}: nested_subject requires inner_folds")
            inner_seen: set[str] = set()
            for i, split in enumerate(inner):
                _, inner_valid = _split_sets(split, f"{name}/inner{i}", train)
                if inner_seen & inner_valid:
                    raise ValueError(f"{name}: inner validation subjects are duplicated")
                inner_seen.update(inner_valid)
            if inner_seen != train:
                raise ValueError(f"{name}: inner validation must cover all outer training subjects")
    if folds and seen_ids != set(qa["qa_id"]):
        raise ValueError("outer folds do not cover all evaluated predictions")
    if kind == "component_diagnostic":
        status = "component_diagnostic"
        unverified.append("Historical component cross-fits are not a full-chain nested evaluation.")
    else:
        status = "unverified" if unverified else "declared_provenance_verified"
    return {
        "validation_kind": kind, "status": status,
        "declared_split_checks_passed": bool(folds), "outer_folds": len(folds),
        "training_provenance": "unverified" if unverified else "declared_files_verified",
        "unverified_reasons": unverified, "provenance_files": list(checked_files.values()),
        "selection_history": protocol.get("selection_history", "unspecified"),
        "limitation": (
            "Checks validate declared subject splits and artifact hashes; they do not establish "
            "that training code used only those rows, that all upstream models were declared, "
            "or that previously observed subjects form a fresh blind holdout."
        ),
    }


def score_rows(frame: pd.DataFrame) -> dict[str, Any]:
    correct = frame["prediction"].eq(frame["answer"])
    result = {"rows": len(frame), "correct_rows": int(correct.sum()),
              "accuracy": float(correct.mean()) if len(frame) else None}
    if "baseline_prediction" in frame:
        parent = frame["baseline_prediction"].eq(frame["answer"])
        changed = frame["prediction"].ne(frame["baseline_prediction"])
        result.update({
            "baseline_accuracy": float(parent.mean()) if len(frame) else None,
            "accuracy_delta": float(correct.mean() - parent.mean()) if len(frame) else None,
            "corrected_rows": int((correct & ~parent).sum()),
            "harmed_rows": int((~correct & parent).sum()),
            "net_correct": int(correct.sum() - parent.sum()),
            "changed_subset": {
                "rows": int(changed.sum()),
                "accuracy": float(correct[changed].mean()) if changed.any() else None,
                "baseline_accuracy": float(parent[changed].mean()) if changed.any() else None,
            },
        })
    return result


def subject_bootstrap(
    frame: pd.DataFrame, *, samples: int = 10000, seed: int = 20260905,
    confidence: float = 0.95,
) -> dict[str, Any]:
    if samples < 100 or not 0 < confidence < 1:
        raise ValueError("bootstrap needs at least 100 samples and confidence between zero and one")
    subjects = [score_rows(group) for _, group in frame.groupby("subject", sort=True)]
    if not subjects:
        raise ValueError("cannot bootstrap an empty evaluation")
    size = np.array([s["rows"] for s in subjects], dtype=float)
    count = np.array([s["correct_rows"] for s in subjects], dtype=float)
    draw = np.random.default_rng(seed).integers(0, len(subjects), (samples, len(subjects)))
    pooled = count[draw].sum(axis=1) / size[draw].sum(axis=1)
    macro = (count / size)[draw].mean(axis=1)
    quantiles = [(1 - confidence) / 2, 1 - (1 - confidence) / 2]

    def interval(values: np.ndarray, estimate: float) -> dict[str, Any]:
        low, high = np.quantile(values, quantiles)
        return {"estimate": float(estimate), "lower": float(low), "upper": float(high)}

    result = {
        "method": "paired percentile cluster bootstrap; resampling complete subjects",
        "sampling_unit": "subject", "subjects": len(subjects), "samples": samples,
        "seed": seed, "confidence": confidence,
        "accuracy": interval(pooled, count.sum() / size.sum()),
        "subject_macro_accuracy": interval(macro, (count / size).mean()),
        "limitation": (
            "Conditional uncertainty for these fixed predictions, not a correction for "
            "historical model selection, upstream leakage, or training-set variation."
        ),
    }
    if "baseline_prediction" in frame:
        delta = np.array([s["net_correct"] for s in subjects], dtype=float)
        bootstrap_delta = delta[draw].sum(axis=1) / size[draw].sum(axis=1)
        result["accuracy_delta"] = interval(bootstrap_delta, delta.sum() / size.sum())
        result["subject_macro_accuracy_delta"] = interval(
            (delta / size)[draw].mean(axis=1), (delta / size).mean()
        )
    return result


def evaluation_report(frame: pd.DataFrame, **bootstrap_kwargs: Any) -> dict[str, Any]:
    by_subject = {name: score_rows(group) for name, group in frame.groupby("subject", sort=True)}
    values = [group["accuracy"] for group in by_subject.values()]
    worst = min(values)
    report = {
        "overall": score_rows(frame), "subjects": len(by_subject),
        "subject_macro_accuracy": float(np.mean(values)), "worst_subject_accuracy": worst,
        "worst_subjects": [name for name, group in by_subject.items() if group["accuracy"] == worst],
        "by_subject": by_subject,
        "by_source_category": {
            f"{source}/{category}": score_rows(group)
            for (source, category), group in frame.groupby(["source", "category"], sort=True)
        },
        "by_source": {name: score_rows(group) for name, group in frame.groupby("source")},
        "by_category": {name: score_rows(group) for name, group in frame.groupby("category")},
        "cluster_bootstrap": subject_bootstrap(frame, **bootstrap_kwargs),
    }
    if "baseline_prediction" in frame:
        baseline = [group["baseline_accuracy"] for group in by_subject.values()]
        report["baseline_subject_macro_accuracy"] = float(np.mean(baseline))
        report["baseline_worst_subject_accuracy"] = min(baseline)
    return report
