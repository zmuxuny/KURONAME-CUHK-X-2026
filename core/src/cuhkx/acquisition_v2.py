"""Select frozen finalist components from an arbitrary nonempty QA batch."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from cuhkx.acquisition import Acquisition
from cuhkx.inference import QUESTION_COLUMNS, question_metadata
from cuhkx.metadata import clip_key, feature_unit
from cuhkx.submission import validate_submission_frames

EXPERT_SCOPES = {
    "action": {("HAU", "single"), ("HAU", "multi"), ("HAU", "combination")},
    "emotion": {("HAU", "emotion")}, "sequence": {("HAU", "sequence")},
    "harn": {("HARn", "single"), ("HARn", "object_interaction")},
}
EXPERT_FEATURES = {
    "action": {"logits8", "logits16", "windows16", "windows32", "windows48"},
    "emotion": {"imu", "skeleton", "radar", "emotion_visual"},
    "sequence": {"logits8", "logits16"}, "harn": {"skeleton", "harn_middle"},
}
FEATURE_ORDER = ("imu", "skeleton", "radar", "windows16", "windows32", "windows48",
                 "logits8", "logits16", "emotion_visual", "harn_middle")
STAGE_ORDER = (*FEATURE_ORDER[:6], "roi", *FEATURE_ORDER[6:], "inference")


@dataclass(frozen=True)
class BatchPlan:
    experts: tuple[str, ...]
    features: tuple[str, ...]
    stages: tuple[str, ...]
    units: tuple[str, ...]
    hau_units: tuple[str, ...]
    harn_units: tuple[str, ...]
    feature_units: dict[str, list[str]]

    def to_dict(self):
        return {key: list(value) if isinstance(value, tuple) else value for key, value in asdict(self).items()}


def batch_plan(questions):
    qa = question_metadata(questions)
    scope = set(zip(qa.source, qa.category, strict=True))
    experts = tuple(name for name, categories in EXPERT_SCOPES.items() if scope & categories)
    selected = set().union(*(EXPERT_FEATURES[name] for name in experts))
    features = tuple(name for name in FEATURE_ORDER if name in selected)
    stages = tuple(name for name in STAGE_ORDER if name in selected or name in {"roi", "inference"})
    units_by_source = {}
    for source in ("HAU", "HARn"):
        rows = qa.loc[qa.source.eq(source)]
        for value in rows.path:
            path = Path(value.replace("\\", "/"))
            if path.is_absolute() or ".." in path.parts or "training" in {part.lower() for part in path.parts}:
                raise ValueError("QA media paths must be relative Testing paths")
        units_by_source[source] = sorted({feature_unit(value, training=False) for value in rows.path})
    hau, harn = units_by_source["HAU"], units_by_source["HARn"]
    if set(hau) & set(harn):
        raise ValueError("the same recording cannot be declared both HAU and HARn")
    feature_units = {}
    for name in features:
        if name == "harn_middle":
            required = harn
        elif name == "skeleton":
            required = sorted(set((hau if "emotion" in experts else []) + (harn if "harn" in experts else [])))
        else:
            required = hau
        feature_units[name] = list(required)
    return BatchPlan(experts, features, stages, tuple(sorted(hau + harn)), tuple(hau), tuple(harn), feature_units)


def read_batch(directory):
    qa = question_metadata(pd.read_csv(directory / "test_qa.csv", usecols=QUESTION_COLUMNS,
                                       dtype=str, keep_default_na=False))
    sample = pd.read_csv(directory / "sample_submission.csv", dtype=str, keep_default_na=False)
    if sample.columns.tolist() != ["qa_id", "prediction"] or sample.qa_id.tolist() != qa.qa_id.tolist():
        raise ValueError("sample schema, IDs or order differ from the current questions")
    return qa, sample, batch_plan(qa)


def read_acquisition(qa, scene_path):
    if not qa.source.eq("HAU").any():
        return None
    if scene_path is None:
        raise ValueError("HAU questions require an explicit acquisition.csv for this batch")
    return Acquisition.parse(qa, pd.read_csv(scene_path, dtype=str, keep_default_na=False))


def expert_questions(questions, name):
    """Keep a target expert's clips and their actual same-clip question context.

    Frozen HAU decoders assume every supplied clip has a question for that
    expert.  A batch can also contain unrelated clips with other question
    types; those must not enter this expert's metadata preparation.
    """
    if name not in EXPERT_SCOPES:
        raise ValueError("unknown expert")
    qa = question_metadata(questions)
    scope = pd.Series(list(zip(qa.source, qa.category, strict=True)), index=qa.index)
    target = qa.loc[scope.isin(EXPERT_SCOPES[name])]
    if target.empty:
        raise ValueError("cannot prepare an inactive expert")
    if name == "harn":
        return target.reset_index(drop=True)
    clips = set(target.path.map(clip_key))
    return qa.loc[qa.source.eq("HAU") & qa.path.map(clip_key).isin(clips)].reset_index(drop=True)


def expert_acquisition(questions, acquisition):
    clips = set(questions.loc[questions.source.eq("HAU"), "path"].map(clip_key))
    if not clips:
        return None
    if acquisition is None:
        raise ValueError("active HAU expert requires explicit acquisition metadata")
    return Acquisition.parse(questions, acquisition.scenes.loc[acquisition.scenes["clip"].isin(clips)])


def raw_modalities(plan, unit):
    """Return required visual, optional visual and useful sensors for this source."""
    if unit in plan.harn_units:
        return ("Depth", "IR"), (), ("Skeleton",)
    if unit not in plan.hau_units:
        raise ValueError("raw unit is outside the planned batch")
    required = ("Depth", "IR") if "emotion" in plan.experts else ("IR",)
    optional = ("Depth_Color",) if {"action", "sequence"} & set(plan.experts) else ()
    sensors = ("IMU", "Radar", "Skeleton") if "emotion" in plan.experts else (
        ("Skeleton",) if "action" in plan.experts else ())
    return required, optional, sensors


def assemble(questions, sample, frames):
    qa, plan = question_metadata(questions), batch_plan(questions)
    if set(frames) != set(plan.experts):
        raise ValueError("expert outputs must match exactly the active question scopes")
    parts = []
    scope = pd.Series(list(zip(qa.source, qa.category, strict=True)), index=qa.index)
    for name in plan.experts:
        frame = frames[name]
        target = qa.loc[scope.isin(EXPERT_SCOPES[name])]
        if (not {"qa_id", "prediction"} <= set(frame) or not frame.qa_id.is_unique
                or set(frame.qa_id) != set(target.qa_id)):
            raise ValueError(f"expert {name} omitted, duplicated or added question IDs")
        for column in ("source", "category"):
            if column in frame and not frame[column].eq(frame.qa_id.map(target.set_index("qa_id")[column])).all():
                raise ValueError(f"expert {name} output metadata differs")
        parts.append(frame[["qa_id", "prediction"]].assign(expert=name))
    evidence = pd.concat(parts, ignore_index=True).set_index("qa_id")
    decisions = qa[["qa_id", "source", "category"]].copy()
    decisions["prediction"] = decisions.qa_id.map(evidence.prediction)
    decisions["expert"] = decisions.qa_id.map(evidence.expert)
    submission = decisions[["qa_id", "prediction"]].copy()
    validate_submission_frames(qa, sample, submission)
    return submission, decisions
