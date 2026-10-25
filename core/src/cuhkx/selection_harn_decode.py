"""HARn fitting and decoding with explicit, auditable subject exclusion.

Held-out paths are opaque feature keys here. Numeric action identities are read
only from the official training feature units selected for a particular fit.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from cuhkx.harn import (
    action_class,
    action_text_classes,
    constrained_scores,
    decode_scores,
    object_text_classes,
    paired_answer_candidates,
)
from cuhkx.metadata import feature_unit, normalize_text, subject_id
from cuhkx.submission import available_letters, validate_prediction

QA_COLUMNS = ["qa_id", "source", "category", "path", "A", "B", "C", "D"]
OFFICIAL_SUBJECTS = {f"user{number}" for number in [*range(1, 10), *range(16, 25)]}
CLASSES = np.arange(44, dtype=np.int64)
CHECKPOINT_KIND = "cuhkx-harn-nested-fusion-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def harn_rows(qa: pd.DataFrame) -> pd.DataFrame:
    rows = qa.loc[
        qa["source"].eq("HARn")
        & qa["category"].isin(["single", "object_interaction"])
    ].copy()
    if rows.empty or not rows["qa_id"].is_unique:
        raise ValueError("HARn QA must be nonempty with unique IDs")
    if "answer" in rows and any(subject_id(path) is None for path in rows["path"]):
        raise ValueError("labeled HARn QA has an unparseable training subject")
    return rows.reset_index(drop=True)


@dataclass
class FeatureBank:
    skeleton_units: np.ndarray
    skeleton: np.ndarray
    skeleton_names: np.ndarray
    visual_units: np.ndarray
    visual: np.ndarray
    visual_spec: dict[str, Any]

    @classmethod
    def load(cls, skeleton_path: Path, visual_path: Path) -> FeatureBank:
        with np.load(skeleton_path, allow_pickle=False) as archive:
            skeleton_units = archive["units"].astype(str)
            skeleton = archive["X"].astype(np.float32)
            skeleton_names = archive["feature_names"].astype(str)
        if skeleton.shape != (len(skeleton_units), len(skeleton_names)):
            raise ValueError("Skeleton feature axes mismatch")
        if len(set(skeleton_units)) != len(skeleton_units):
            raise ValueError("duplicate Skeleton unit")
        with np.load(visual_path, allow_pickle=False) as archive:
            embeddings = archive["X"].astype(np.float32)
            units = archive["units"].astype(str)
            modalities = archive["modalities"].astype(str)
            spec = {key: str(archive[key]) for key in ("backbone", "weights", "frames")}
        if spec != {
            "backbone": "r3d_18",
            "weights": "R3D_18_Weights.KINETICS400_V1",
            "frames": "16",
        }:
            raise ValueError(f"unexpected visual extractor: {spec}")
        if embeddings.shape != (len(units), 512) or len(modalities) != len(units):
            raise ValueError("visual feature axes mismatch")
        keys = list(zip(units, modalities, strict=True))
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate visual unit/modality")
        lookup = dict(zip(keys, embeddings, strict=True))
        visual_units = np.asarray(sorted(
            {unit for unit, modality in keys if modality == "Depth"}
            & {unit for unit, modality in keys if modality == "IR"}
        ), dtype=str)
        visual = (
            np.stack([np.concatenate([lookup[(unit, m)] for m in ("Depth", "IR")])
                      for unit in visual_units])
            if len(visual_units) else np.zeros((0, 1024), dtype=np.float32)
        )
        if not np.isfinite(skeleton).all() or not np.isfinite(embeddings).all():
            raise ValueError("non-finite feature value")
        for axis in (skeleton_units, visual_units):
            train_units = [unit for unit in axis if unit.startswith("Training/HARn/")]
            if not {subject_id(unit) for unit in train_units} <= OFFICIAL_SUBJECTS:
                raise ValueError("unofficial subject in HARn training feature units")
        return cls(skeleton_units, skeleton, skeleton_names, visual_units, visual, spec)

    def training_rows(self, family: str, subjects: set[str]) -> np.ndarray:
        if not subjects or not subjects <= OFFICIAL_SUBJECTS:
            raise ValueError("fitting subjects must be official training subjects")
        units = self.skeleton_units if family == "skeleton" else self.visual_units
        rows = np.asarray([
            index for index, unit in enumerate(units)
            if unit.startswith("Training/HARn/") and subject_id(unit) in subjects
        ], dtype=np.int64)
        if not len(rows):
            raise ValueError(f"no {family} feature rows for fitting subjects")
        return rows


def fit_prior(qa: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Training-only option-text correctness frequency with a fixed Beta prior."""
    offered: Counter[tuple[str, str]] = Counter()
    correct: Counter[tuple[str, str]] = Counter()
    for _, row in qa.iterrows():
        category = str(row["category"])
        answer = str(row["answer"])
        if validate_prediction(answer, category, available_letters(row)):
            raise ValueError(f"invalid training answer: {row['qa_id']}")
        for letter in available_letters(row):
            key = category, normalize_text(row[letter])
            offered[key] += 1
            correct[key] += letter == answer
    result: dict[str, dict[str, float]] = {}
    for (category, text), count in offered.items():
        result.setdefault(category, {})[text] = (correct[(category, text)] + 1) / (count + 2)
    return result


def training_manifest(
    bank: FeatureBank, qa: pd.DataFrame, subjects: set[str], config: dict[str, Any]
) -> dict[str, Any]:
    qa_subjects = {subject_id(path) for path in qa["path"]}
    if qa_subjects != subjects or not subjects <= OFFICIAL_SUBJECTS:
        raise ValueError("QA fitting subjects do not exactly match declared official subjects")
    result: dict[str, Any] = {
        "train_subjects": sorted(subjects),
        "training_qa_ids": qa["qa_id"].tolist(),
        "seed": config["seed"],
        "dictionary_train_subjects": sorted(qa_subjects),
        "features": {},
    }
    for family in ("skeleton", "visual"):
        indices = bank.training_rows(family, subjects)
        axis = bank.skeleton_units if family == "skeleton" else bank.visual_units
        units = axis[indices].tolist()
        labels = [action_class(unit) for unit in units]
        if not set(labels) <= set(CLASSES.tolist()):
            raise ValueError("unexpected official action class")
        result["features"][family] = {
            "train_subjects": sorted({subject_id(unit) for unit in units}),
            "units": units,
            "labels": labels,
        }
    return result


def fit_bundle(
    bank: FeatureBank,
    qa: pd.DataFrame,
    subjects: set[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    training_manifest(bank, qa, subjects, config)
    workers = int(config["workers"])
    if not 1 <= workers <= 3:
        raise ValueError("workers must be within 1..3")
    skeleton_rows = bank.training_rows("skeleton", subjects)
    skeleton_labels = np.asarray([action_class(unit) for unit in bank.skeleton_units[skeleton_rows]])
    aggregate = ~np.char.startswith(bank.skeleton_names, "resampled_pose_")
    models: dict[str, Any] = {}
    masks = {"extra_full": np.ones(len(aggregate), dtype=bool), "extra_aggregate": aggregate}
    for offset, (name, mask) in enumerate(masks.items()):
        model = ExtraTreesClassifier(
            n_estimators=config["skeleton"]["trees"], max_features="sqrt",
            min_samples_leaf=1, class_weight="balanced", n_jobs=workers,
            random_state=config["seed"] + offset * 1000,
        )
        model.fit(bank.skeleton[skeleton_rows][:, mask], skeleton_labels)
        models[name] = model
    visual_rows = bank.training_rows("visual", subjects)
    visual_labels = np.asarray([action_class(unit) for unit in bank.visual_units[visual_rows]])
    visual_model = make_pipeline(StandardScaler(), SVC(
        C=config["visual"]["C"], kernel="rbf", class_weight="balanced", probability=True,
        random_state=config["seed"] + 2000,
    ))
    visual_model.fit(bank.visual[visual_rows], visual_labels)
    return {
        "kind": CHECKPOINT_KIND,
        "status": "diagnostic_only",
        "config": config,
        "train_subjects": sorted(subjects),
        "models": models,
        "masks": masks,
        "visual_model": visual_model,
        "skeleton_names": bank.skeleton_names,
        "visual_spec": bank.visual_spec,
        "skeleton_classes": np.unique(skeleton_labels),
        "visual_classes": np.unique(visual_labels),
        "action_mapping": action_text_classes(qa),
        "object_mapping": object_text_classes(qa),
        "prior": fit_prior(qa),
    }


def aligned_probability(model: Any, matrix: np.ndarray) -> np.ndarray:
    result = np.zeros((len(matrix), len(CLASSES)), dtype=np.float64)
    if len(matrix):
        values = model.predict_proba(matrix)
        result[:, np.asarray(model.classes_, dtype=int)] = values
    return result


def predict_evidence(
    bundle: dict[str, Any], bank: FeatureBank, qa: pd.DataFrame, *, training_paths: bool
) -> dict[str, Any]:
    if not np.array_equal(bundle["skeleton_names"], bank.skeleton_names):
        raise ValueError("Skeleton feature names/order differ from checkpoint")
    if bank.visual_spec != bundle["visual_spec"]:
        raise ValueError("visual extractor differs from checkpoint")
    units = np.asarray(sorted({feature_unit(path, training=training_paths) for path in qa["path"]}))
    result: dict[str, Any] = {"units": units}
    for family, axis, matrix in (
        ("skeleton", bank.skeleton_units, bank.skeleton),
        ("visual", bank.visual_units, bank.visual),
    ):
        lookup = {unit: index for index, unit in enumerate(axis)}
        present = np.asarray([unit in lookup for unit in units])
        rows = [lookup[unit] for unit in units if unit in lookup]
        probabilities = np.zeros((len(units), len(CLASSES)), dtype=np.float64)
        if rows and family == "skeleton":
            probabilities[present] = np.mean([
                aligned_probability(model, matrix[rows][:, bundle["masks"][name]])
                for name, model in bundle["models"].items()
            ], axis=0)
        elif rows:
            probabilities[present] = aligned_probability(bundle["visual_model"], matrix[rows])
        result[f"{family}_probabilities"] = probabilities
        result[f"{family}_present"] = present
        result[f"{family}_classes"] = bundle[f"{family}_classes"]
    return result


def normalized_option_scores(scores: dict[str, float]) -> dict[str, float]:
    total = sum(scores.values())
    return {letter: value / total if total > 1e-12 else 0.0 for letter, value in scores.items()}


def decode_questions(
    qa: pd.DataFrame,
    evidence: dict[str, Any],
    bundle: dict[str, Any],
    skeleton_weight: float,
    *,
    training_paths: bool,
    visual_option_scores: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Decode metadata without using any target labels or path action identity."""
    if not 0 <= skeleton_weight <= 1:
        raise ValueError("invalid Skeleton weight")
    metadata = qa.loc[:, QA_COLUMNS].copy()
    lookup = {unit: index for index, unit in enumerate(evidence["units"])}
    records = []
    for _, row in metadata.iterrows():
        unit = feature_unit(row["path"], training=training_paths)
        index = lookup[unit]
        mapping = bundle["action_mapping" if row["category"] == "single" else "object_mapping"]
        vectors: dict[str, dict[str, float]] = {}
        supported: dict[str, bool] = {}
        record: dict[str, Any] = {
            "qa_id": row["qa_id"], "source": "HARn", "category": row["category"],
            "subject": subject_id(row["path"]), "unit": unit,
            "skeleton_weight": skeleton_weight,
        }
        for family in ("skeleton", "visual"):
            present = bool(evidence[f"{family}_present"][index])
            family_classes = evidence[f"{family}_classes"]
            scores, supported[family] = constrained_scores(
                row, evidence[f"{family}_probabilities"][index, family_classes],
                family_classes, mapping,
            )
            vector = normalized_option_scores(scores)
            if family == "visual" and visual_option_scores is not None:
                vector = visual_option_scores[row["qa_id"]]
            usable = present and sum(vector.values()) > 0
            vectors[family] = vector if usable else {}
            record[f"{family}_present"] = present
            record[f"{family}_fully_supported"] = supported[family]
            record[f"prediction_{family}"] = decode_scores(vector)[0] if usable else ""
        if vectors["visual"] and (not vectors["skeleton"] or not supported["skeleton"]):
            scores = vectors["visual"]
            source = "visual_missing_or_unsupported_skeleton"
        elif vectors["skeleton"] and vectors["visual"]:
            scores = {
                letter: skeleton_weight * vectors["skeleton"][letter]
                + (1 - skeleton_weight) * vectors["visual"][letter]
                for letter in available_letters(row)
            }
            source = "nested_modality_fusion"
        elif vectors["skeleton"]:
            scores = vectors["skeleton"]
            source = "skeleton_without_visual"
        else:
            prior = bundle["prior"].get(row["category"], {})
            scores = normalized_option_scores({
                letter: prior.get(normalize_text(row[letter]), 0.5)
                for letter in available_letters(row)
            })
            source = "training_option_prior_fallback"
        prediction, score, margin = decode_scores(scores)
        record.update({
            "prediction": prediction, "raw_prediction": prediction,
            "raw_score": score, "raw_margin": margin, "evidence_source": source,
            "pair_candidates": 0, "unique_pair_applied": False, "pair_changed": False,
        })
        records.append(record)
    result = pd.DataFrame(records).set_index("qa_id", drop=False)
    metadata["unit"] = [feature_unit(path, training=training_paths) for path in metadata["path"]]
    for _, rows in metadata.groupby("unit", sort=True):
        candidates = paired_answer_candidates(rows, bundle["action_mapping"], bundle["object_mapping"])
        result.loc[rows["qa_id"], "pair_candidates"] = len(candidates)
        if len(candidates) != 1:
            continue
        for category, prediction in zip(("single", "object_interaction"), candidates[0], strict=True):
            qa_id = rows.loc[rows["category"].eq(category), "qa_id"].iloc[0]
            result.loc[qa_id, "unique_pair_applied"] = True
            result.loc[qa_id, "pair_changed"] = result.loc[qa_id, "prediction"] != prediction
            result.loc[qa_id, "prediction"] = prediction
            result.loc[qa_id, "evidence_source"] = "training_dictionary_unique_pair"
    result = result.loc[metadata["qa_id"]].reset_index(drop=True)
    for row, prediction in zip(metadata.to_dict("records"), result["prediction"], strict=True):
        error = validate_prediction(prediction, str(row["category"]), available_letters(row))
        if error:
            raise ValueError(f"invalid decoded prediction: {row['qa_id']}: {error}")
    return result


def evaluate_predictions(qa: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, Any]:
    if set(qa["qa_id"]) != set(predictions["qa_id"]) or not predictions["qa_id"].is_unique:
        raise ValueError("prediction QA coverage mismatch")
    merged = qa.loc[:, ["qa_id", "category", "path", "answer"]].merge(
        predictions.loc[:, ["qa_id", "prediction"]], on="qa_id", validate="one_to_one"
    )
    merged["correct"] = merged["prediction"].eq(merged["answer"])
    merged["subject"] = merged["path"].map(subject_id)
    # Ground-truth path-derived action identities are used only for reporting.
    merged["action_class"] = merged["path"].map(action_class)
    result: dict[str, Any] = {
        "rows": len(merged), "correct": int(merged["correct"].sum()),
        "accuracy": float(merged["correct"].mean()),
    }
    for grouping in ("category", "subject", "action_class"):
        result[f"by_{grouping}"] = {
            str(key): {"rows": len(rows), "correct": int(rows["correct"].sum()),
                       "accuracy": float(rows["correct"].mean())}
            for key, rows in merged.groupby(grouping)
        }
    result["worst_subject_accuracy"] = min(x["accuracy"] for x in result["by_subject"].values())
    result["macro_subject_accuracy"] = float(np.mean([
        x["accuracy"] for x in result["by_subject"].values()
    ]))
    return result


def select_weight(candidate_reports: list[dict[str, Any]], tie_order: list[float]) -> float:
    ranks = {float(value): index for index, value in enumerate(tie_order)}
    return float(max(candidate_reports, key=lambda report: (
        report["metrics"]["accuracy"], report["metrics"]["worst_subject_accuracy"],
        -ranks[float(report["skeleton_weight"])],
    ))["skeleton_weight"])
