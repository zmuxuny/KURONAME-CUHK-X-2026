"""Unlabelled acquisition groups for inference on a different recording layout."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cuhkx.inference import question_metadata
from cuhkx.metadata import clip_key, subject_id


@dataclass(frozen=True)
class Acquisition:
    scenes: pd.DataFrame

    @classmethod
    def parse(cls, questions, manifest):
        qa = question_metadata(questions)
        columns = set(manifest)
        if columns not in ({"clip", "group", "position"},
                           {"clip", "group", "position", "subject"}):
            raise ValueError("acquisition columns must be clip, group, position, and optional subject")
        scenes = manifest.copy()
        for name in ("clip", "group"):
            if scenes[name].isna().any() or scenes[name].astype(str).str.strip().eq("").any():
                raise ValueError(f"acquisition {name} must be nonempty")
            scenes[name] = scenes[name].astype(str)
        if not scenes["clip"].is_unique:
            raise ValueError("each acquisition clip must occur exactly once")
        expected = set(qa.loc[qa.source.eq("HAU"), "path"].map(clip_key))
        if set(scenes["clip"]) != expected:
            raise ValueError("acquisition must cover exactly the requested HAU clips")
        positions = pd.to_numeric(scenes.position, errors="raise")
        if not positions.isin([0, 1, 2]).all():
            raise ValueError("acquisition positions must be 0, 1, or 2")
        scenes["position"] = positions.astype(int)
        for _, group in scenes.groupby("group"):
            if not group.position.is_unique or len(group) > 3:
                raise ValueError("a scene has at most three distinct recording positions")
            identified = set(group["clip"].map(subject_id).dropna())
            if len(identified) > 1:
                raise ValueError("a scene cannot span identified subjects")
            if "subject" in scenes:
                if (group.subject.isna().any() or group.subject.astype(str).str.strip().eq("").any()
                        or group.subject.astype(str).nunique() != 1):
                    raise ValueError("a scene requires one nonempty declared subject")
                if identified and identified != set(group.subject.astype(str)):
                    raise ValueError("declared subject contradicts a subject encoded in the clip path")
        return cls(scenes.sort_values(["group", "position"]).reset_index(drop=True))

    @property
    def groups(self):
        return dict(zip(self.scenes["clip"], self.scenes.group, strict=True))

    def groups_for(self, questions):
        clips = set(questions.loc[questions.source.eq("HAU"), "path"].map(clip_key))
        mapping = self.groups
        if not clips <= set(mapping):
            raise ValueError("decoder requested clips outside its acquisition manifest")
        return {clip: mapping[clip] for clip in sorted(clips)}

    def temporal_manifest(self):
        return {"schema_version": 1, "groups": self.groups}
