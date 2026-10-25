"""Portable inference for the independently trained metadata fallback.

This profile does not reproduce the scored multimodal submission. Its JSON
checkpoint contains learned counts, never a qa_id-to-answer lookup table.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from cuhkx.metadata import option_atoms
from cuhkx.structural import StructuralConfig, StructuralDecoder
from cuhkx.submission import LETTERS, available_letters, validate_submission_frames

MODEL_KIND = "cuhkx_structural_fallback_v1"
QUESTION_COLUMNS = ["qa_id", "source", "path", "category", *LETTERS]
SOURCE_CATEGORIES = {
    "HAU": {"single", "multi", "combination", "sequence", "emotion"},
    "HARn": {"single", "object_interaction"},
}
COUNTER_KEYS = {
    "atom_seen": (str, str, str),
    "atom_correct": (str, str, str),
    "multi_pattern_seen": (int, int, int),
    "multi_pattern_correct": (int, int, int),
    "multi_sizes": (int,),
    "before": (str, str),
}


def question_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Whitelist inputs so optional answer/prediction columns cannot affect inference."""

    missing = set(QUESTION_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"questions lack columns: {sorted(missing)}")
    result = frame[QUESTION_COLUMNS].copy()
    if result.empty or result.isna().any().any():
        raise ValueError("questions must be nonempty with no null metadata")
    if not result.qa_id.is_unique or result.qa_id.str.strip().eq("").any():
        raise ValueError("question IDs must be nonempty and unique")
    for row in result.to_dict("records"):
        if not str(row["path"]).strip():
            raise ValueError(f"empty clip path for {row['qa_id']}")
        if row["category"] not in SOURCE_CATEGORIES.get(row["source"], set()):
            raise ValueError(f"unsupported source/category for {row['qa_id']}")
        letters = available_letters(pd.Series(row))
        if len(letters) < 2 or any(not option_atoms(row[letter]) for letter in letters):
            raise ValueError(f"invalid option text for {row['qa_id']}")
    return result


def save_structural_checkpoint(
    model: StructuralDecoder, path: Path, *, provenance: dict[str, object]
) -> str:
    if not model._fitted:
        raise ValueError("cannot export an unfitted model")
    state = {}
    for name in COUNTER_KEYS:
        values = getattr(model, name)
        state[name] = [
            {"key": list(key) if isinstance(key, tuple) else [key], "count": count}
            for key, count in sorted(values.items())
        ]
    payload = {
        "schema_version": 1,
        "model_kind": MODEL_KIND,
        "scope": "metadata-only fallback; not the scored multimodal system",
        "config": asdict(model.config),
        "state": state,
        "provenance": provenance,
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    digest = hashlib.sha256(encoded).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(encoded)
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n")
    return digest


def load_structural_checkpoint(path: Path) -> tuple[StructuralDecoder, dict[str, object]]:
    encoded = path.read_bytes()
    expected = path.with_suffix(path.suffix + ".sha256").read_text().strip()
    if hashlib.sha256(encoded).hexdigest() != expected:
        raise ValueError("checkpoint SHA256 mismatch")
    payload = json.loads(encoded)
    if payload.get("schema_version") != 1 or payload.get("model_kind") != MODEL_KIND:
        raise ValueError("unsupported checkpoint schema or model kind")
    if set(payload["state"]) != set(COUNTER_KEYS):
        raise ValueError("checkpoint has incomplete or unexpected learned state")
    model = StructuralDecoder(StructuralConfig(**payload["config"]))
    for name, types in COUNTER_KEYS.items():
        counts: Counter = Counter()
        for entry in payload["state"][name]:
            key, count = entry["key"], entry["count"]
            if len(key) != len(types) or any(type(v) is not t for v, t in zip(key, types)):
                raise ValueError(f"invalid checkpoint key in {name}")
            if type(count) is not int or count < 0:
                raise ValueError(f"invalid checkpoint count in {name}")
            canonical = key[0] if name == "multi_sizes" else tuple(key)
            if canonical in counts:
                raise ValueError(f"duplicate checkpoint key in {name}")
            counts[canonical] = count
        setattr(model, name, counts)
    model._fitted = True
    return model, payload


def infer_structural(
    model: StructuralDecoder, questions: pd.DataFrame, sample: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = question_metadata(questions)
    if sample.columns.tolist() != ["qa_id", "prediction"]:
        raise ValueError("sample schema must be qa_id,prediction")
    if sample.qa_id.tolist() != metadata.qa_id.tolist():
        raise ValueError("sample IDs/order differ from questions")
    decisions = model.predict(metadata)
    submission = decisions[["qa_id", "prediction"]].copy()
    validate_submission_frames(metadata, sample, submission)
    decisions["evidence_source"] = MODEL_KIND
    return submission, decisions
