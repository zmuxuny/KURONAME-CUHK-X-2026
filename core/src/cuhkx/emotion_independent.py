"""Emotion inference from one recording, without acquisition order or peers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cuhkx.emotion_nested import base_evidence, predict_rows, prepare
from cuhkx.emotion_scene_robustness import inference_frame
from cuhkx.metadata import clip_key


def prepare_independent(questions, *, training):
    emotion = questions.loc[questions.source.eq("HAU") & questions.category.eq("emotion")]
    clips = emotion.path.map(clip_key)
    manifest = pd.DataFrame({"clip": clips, "group": clips, "position": 0})
    # Other recordings in the incoming batch must never contribute context.
    visible = questions.loc[questions.path.map(clip_key).isin(set(clips))]
    frame, contexts = prepare(visible, training=training, manifest=manifest)
    if training:
        # This is a supervised latent manner target, never an inference input.
        frame["position"] = frame.path.str.rsplit("-", n=1).str[1].astype(int) - 1
        if not frame.position.isin([0, 1, 2]).all():
            raise ValueError("invalid official Train manner target")
    if not frame.group.is_unique:
        raise ValueError("independent expert requires one emotion question per recording")
    return frame, contexts


def absolute_matrix(frame, source):
    available = frame.unit.isin(source["index"]).to_numpy()
    matrix = np.zeros((len(frame), len(source["names"])), dtype=np.float32)
    rows = np.flatnonzero(available)
    matrix[rows] = source["matrix"][[source["index"][frame.iloc[i].unit] for i in rows]]
    return matrix, available


def evidence(bundle, frame, contexts, matrices, masks, rows, *, poison=False):
    safe = inference_frame(frame, poison=poison)
    safe["position"] = 0
    return base_evidence(bundle, safe, contexts, matrices, masks, rows)


def predict(frame, scores, decoder):
    if decoder["use_position"] or not frame.group.is_unique:
        raise ValueError("independent decoding cannot use order or recording peers")
    return predict_rows(inference_frame(frame).assign(position=0), scores, decoder)


def choose(scores, decoders):
    ranking = []
    for order, decoder in enumerate(decoders):
        selected = scores.loc[scores.decoder.eq(decoder["name"])]
        if selected.empty or not selected.qa_id.is_unique:
            raise ValueError("inner scores must cover each question exactly once")
        ranking.append({"name": decoder["name"], "correct": int(selected.correct.sum()),
                        "rows": len(selected), "accuracy": float(selected.correct.mean()),
                        "worst_subject": float(selected.groupby("subject").correct.mean().min()),
                        "order": order})
    best = max(ranking, key=lambda item: (item["accuracy"], item["worst_subject"], -item["order"]))
    return decoders[best["order"]], ranking
