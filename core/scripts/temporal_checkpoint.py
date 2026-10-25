#!/usr/bin/env python3
"""Load the existing HAU pairwise models and decode complete sequence questions.

Export references the original trained weights and records the audited release
layout. Inference reads question metadata and temporal logits, never training
answers, past prediction files, or a parent submission. A changed clip layout
requires an explicit acquisition-group manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from cuhkx.clip_constraints import repeated_recording_groups
from cuhkx.inference import QUESTION_COLUMNS, question_metadata
from cuhkx.metadata import clip_key, feature_unit
from cuhkx.pairwise_temporal import PAIR_INDICES
from cuhkx.repeated_temporal import (
    decode_repeated_pair_probabilities,
    share_repeated_pair_probabilities,
)
from cuhkx.submission import validate_submission_frames
from cuhkx.temporal import softmax
from scripts.run_hau_pairwise_temporal import make_pair_rows, question_classes

KIND = "cuhkx_repeated_pairwise_temporal_v1"
PROFILE = {"families": ["leaf2_sqrt", "leaf5_sqrt"], "sibling_weight": 0.75,
           "prior_weight": 1.0, "temperature": 4.0}
SOURCE_PATHS = (
    "scripts/temporal_checkpoint.py", "scripts/run_hau_pairwise_temporal.py",
    "src/cuhkx/actions.py", "src/cuhkx/clip_constraints.py", "src/cuhkx/inference.py",
    "src/cuhkx/metadata.py", "src/cuhkx/pairwise_temporal.py",
    "src/cuhkx/repeated_temporal.py", "src/cuhkx/submission.py",
    "src/cuhkx/temporal.py", "src/cuhkx/structural.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sources() -> dict[str, str]:
    return {name: sha256(ROOT / name) for name in SOURCE_PATHS}


def read_questions(path: Path) -> pd.DataFrame:
    return question_metadata(pd.read_csv(
        path, usecols=QUESTION_COLUMNS, dtype=str, keep_default_na=False,
    ))


def load_logits(
    paths: list[Path], temperature: float, contracts: list[dict] | None = None,
) -> tuple[list[dict[str, np.ndarray]], list[dict]]:
    if len(paths) != 2 or (contracts is not None and len(contracts) != 2):
        raise ValueError("the frozen model requires exactly two ordered temporal scales")
    maps, descriptions = [], []
    for index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as archive:
            logits = archive["logits"].astype(np.float32)
            units = archive["units"].astype(str)
            weights = archive["weights"].astype(np.float64)
            description = {
                "folds": archive["folds"].astype(int).tolist(),
                "weights": weights.tolist(), "classes": archive["classes"].astype(int).tolist(),
                "segments": int(archive["segments"]), "frames": int(archive["frames"]),
                "window_seconds": float(archive["window_seconds"]),
            }
        if logits.ndim != 4 or logits.shape != (
            len(weights), len(units), description["segments"], 40,
        ):
            raise ValueError("unexpected temporal logit shape")
        if (not len(units) or len(set(units)) != len(units)
                or any(not unit.startswith("Testing/") for unit in units)):
            raise ValueError("temporal inputs require unique, nonempty Testing units")
        if (description["classes"] != list(range(40))
                or len(description["folds"]) != len(weights)
                or not np.isfinite(logits).all() or not np.isfinite(weights).all()
                or (weights < 0).any() or weights.sum() <= 0):
            raise ValueError("invalid logit classes, values, or ensemble weights")
        if contracts is not None and description != contracts[index]:
            raise ValueError(f"temporal scale {index} differs from frozen feature contract")
        weights /= weights.sum()
        probabilities = np.tensordot(
            weights, softmax(logits, temperature), axes=(0, 0),
        ).astype(np.float32)
        maps.append(dict(zip(units, probabilities, strict=True)))
        descriptions.append(description)
    return maps, descriptions


def resolve_groups(
    questions: pd.DataFrame, layout: dict, explicit: dict | None = None,
) -> dict[str, str]:
    clips = set(questions.loc[questions.source.eq("HAU"), "path"].map(clip_key))
    if explicit is None:
        if clips != set(layout["groups"]):
            raise ValueError("changed HAU clip layout requires an explicit acquisition-group manifest")
        groups = repeated_recording_groups(questions)
        if groups != layout["groups"]:
            raise ValueError("release grouping differs from the audited layout")
    else:
        if explicit.get("schema_version") != 1:
            raise ValueError("unsupported acquisition-group manifest schema")
        groups = explicit.get("groups", {})
        if (set(groups) != clips or not all(isinstance(g, str) and g.strip() for g in groups.values())
                or max(pd.Series(list(groups.values())).value_counts(), default=0) > 3):
            raise ValueError("acquisition groups must cover exactly the HAU clips, at most three each")
    return groups


def predict(
    bundle: dict, manifest: dict, questions: pd.DataFrame,
    probability_sources: list[dict[str, np.ndarray]], *, workers: int = 3,
    explicit_groups: dict | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    questions = question_metadata(questions)
    groups = resolve_groups(questions, manifest["layout"], explicit_groups)
    frame = questions.loc[questions.source.eq("HAU") & questions.category.eq("sequence")].copy()
    if frame.empty:
        raise ValueError("no HAU sequence questions to decode")
    frame = frame.reset_index(drop=True)
    frame["unit"] = frame.path.map(lambda path: feature_unit(path, training=False))
    frame["repeat_group"] = frame.path.map(lambda path: groups[clip_key(path)])
    if any(set(frame.unit) - set(source) for source in probability_sources):
        raise ValueError("temporal features do not cover all sequence questions")
    classes = question_classes(frame)
    pairs = make_pair_rows(frame, classes, probability_sources, training=False)
    family_names = list(bundle["models"])
    probabilities = []
    for name in family_names:
        estimator = bundle["models"][name]
        if list(estimator.classes_) != [0, 1] or estimator.n_features_in_ != pairs.forward.shape[1]:
            raise ValueError("temporal classifier schema differs from pair features")
        estimator.n_jobs = workers
        forward = estimator.predict_proba(pairs.forward)[:, 1]
        reverse = estimator.predict_proba(pairs.reverse)[:, 1]
        probabilities.append((0.5 * (forward + 1.0 - reverse)).reshape(len(frame), 6))
    values = np.asarray(probabilities, dtype=np.float32)
    config = manifest["profile"]
    pooled, shared = share_repeated_pair_probabilities(
        values, classes, frame.repeat_group.to_numpy(),
        family_indices=[family_names.index(name) for name in config["families"]],
        sibling_weight=config["sibling_weight"],
    )
    predictions, margins = decode_repeated_pair_probabilities(
        pooled, classes, [bundle["ordered_class_counts"]] * len(frame),
        prior_weight=config["prior_weight"],
    )
    output = frame[["qa_id", "source", "category", "unit", "repeat_group"]].copy()
    output["prediction"], output["margin"], output["shared_pairs"] = predictions, margins, shared
    submission = output[["qa_id", "prediction"]]
    validate_submission_frames(frame, submission, submission)
    arrays = {"qa_ids": frame.qa_id.to_numpy(dtype=str), "classes": classes,
              "family_names": np.asarray(family_names), "probabilities": values,
              "pair_indices": np.asarray(PAIR_INDICES), "pooled": pooled}
    return output, arrays


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["export", "infer"])
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="original models.joblib for export; exported checkpoint.json for infer")
    parser.add_argument("--qa-csv", type=Path, required=True)
    parser.add_argument("--logits", type=Path, action="append", required=True)
    parser.add_argument("--groups-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.output_dir.exists() or args.workers < 1:
        raise ValueError("output directory must be new and workers positive")
    questions = read_questions(args.qa_csv)
    inputs = [args.qa_csv, *args.logits, args.checkpoint]
    if args.operation == "export":
        bundle = joblib.load(args.checkpoint)
        if float(bundle["config"]["temperature"]) != PROFILE["temperature"]:
            raise ValueError("source temperature differs from the fixed decoder profile")
        maps, contracts = load_logits(args.logits, PROFILE["temperature"])
        # Assert original feature contract using the model's own extraction metadata.
        if [c["segments"] for c in contracts] != [8, 16] or [
            c["window_seconds"] for c in contracts
        ] != [2.5, 1.25]:
            raise ValueError("export expects the production s8/2.5s then s16/1.25s scales")
        manifest = {
            "schema_version": 1, "model_kind": KIND, "profile": PROFILE,
            "model": {"path": os.path.relpath(args.checkpoint.resolve(), args.output_dir.resolve()),
                      "sha256": sha256(args.checkpoint)},
            "feature_contracts": contracts, "source_hashes": sources(),
            "layout": {"kind": "audited_original_release",
                       "groups": repeated_recording_groups(questions)},
            "scope": "fixed historical expert; no new OOF or backbone subject-exclusion claim",
        }
    else:
        if sha256(args.checkpoint) != args.checkpoint.with_suffix(".json.sha256").read_text().strip():
            raise ValueError("manifest SHA256 mismatch")
        manifest = json.loads(args.checkpoint.read_text())
        if (manifest.get("model_kind") != KIND or manifest.get("schema_version") != 1
                or manifest["source_hashes"] != sources()):
            raise ValueError("checkpoint kind, schema, or sources changed")
        weights = (args.checkpoint.parent / manifest["model"]["path"]).resolve()
        if sha256(weights) != manifest["model"]["sha256"]:
            raise ValueError("model weight SHA256 mismatch")
        bundle = joblib.load(weights)
        inputs.append(weights)
        maps, _ = load_logits(args.logits, manifest["profile"]["temperature"], manifest["feature_contracts"])
    explicit = None
    if args.groups_json:
        explicit = json.loads(args.groups_json.read_text())
        inputs.append(args.groups_json)
    output, arrays = predict(bundle, manifest, questions, maps, workers=args.workers, explicit_groups=explicit)
    args.output_dir.mkdir(parents=True)
    if args.operation == "export":
        checkpoint = args.output_dir / "checkpoint.json"
        checkpoint.write_text(json.dumps(manifest, indent=2) + "\n")
        checkpoint.with_suffix(".json.sha256").write_text(sha256(checkpoint) + "\n")
        for name in SOURCE_PATHS:
            target = args.output_dir / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
    output.to_csv(args.output_dir / "test_predictions.csv", index=False, lineterminator="\n")
    np.savez_compressed(args.output_dir / "pair_scores.npz", **arrays)
    report = {
        "operation": args.operation, "model_kind": KIND, "profile": manifest["profile"],
        "rows": len(output), "source_hashes": sources(),
        "layout_source": str(args.groups_json.resolve()) if args.groups_json else "audited_original_release",
        "inputs": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in inputs],
        "outputs": {p.name: sha256(p) for p in args.output_dir.iterdir() if p.is_file()},
        "environment": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib")},
        "limitations": ["test-feature inference; raw media not extracted here",
                        "K-KUNO backbone fold subject memberships remain unverified",
                        "no training QA or historical prediction file read during inference"],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rows": len(output), "outputs": report["outputs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
