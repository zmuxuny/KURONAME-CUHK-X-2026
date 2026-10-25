"""Fixed emotion fusion alternatives for nested subject evaluation.

All model fitting and word/context priors are inherited from emotion_nested.
Only the order of modality fusion, manner projection, and word scoring changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cuhkx import emotion_nested as legacy
from cuhkx.emotion_relative import triplet_assignment_probabilities

FUSION_FIELDS = {
    "v1_manner_mean": "sensor",
    "option_log_sum": "sensor_option_log_sum",
    "projected_option_log_sum": "sensor_projected_option_log_sum",
}


def option_log_probabilities(probabilities, groups, association, *, projected):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    association = np.asarray(association, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError("manner probabilities must have shape (rows, 3)")
    if association.ndim != 2 or association.shape[1] != 3:
        raise ValueError("word association must have shape (words, 3)")
    for values in (probabilities, association):
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("probabilities and associations must be finite and nonnegative")
        if not np.allclose(values.sum(axis=1), 1, atol=1e-6):
            raise ValueError("probabilities and associations must sum to one")
    if len(groups) != len(probabilities):
        raise ValueError("scene IDs and manner probability rows differ")
    if projected:
        probabilities = triplet_assignment_probabilities(probabilities, np.asarray(groups))
    return np.log(np.clip(probabilities @ association.T, 1e-8, None))


def base_evidence(bundle, frame, contexts, matrices, masks, target_rows):
    rows = np.asarray(target_rows, dtype=int)
    if len(set(rows)) != len(rows):
        raise ValueError("target rows must be unique")
    target = frame.iloc[rows]
    if not target.qa_id.is_unique:
        raise ValueError("target question IDs must be unique")
    if "subject" in target and target.groupby("group").subject.nunique().gt(1).any():
        raise ValueError("a target recording scene cannot span subjects")
    # This call preserves the exact v1 fusion and its target-subject exclusion guard.
    evidence = legacy.base_evidence(bundle, frame, contexts, matrices, masks, rows)
    fields = [FUSION_FIELDS[mode] for mode in ("option_log_sum", "projected_option_log_sum")]
    for row in evidence.values():
        for item in row.values():
            for field in fields:
                item[field] = 0.0
    association = np.asarray(bundle["association"], dtype=np.float64)
    association = np.vstack([association, np.full((1, 3), 1 / 3)])
    fallback_word = len(bundle["words"])
    for name, model in bundle["models"].items():
        valid = np.asarray(masks[name][rows], dtype=bool)
        if not valid.any():
            continue
        selected = target.iloc[np.flatnonzero(valid)]
        probabilities = np.full((len(selected), 3), 1e-8, dtype=np.float64)
        probabilities[:, model.classes_.astype(int)] = model.predict_proba(matrices[name][rows[valid]])
        weight = float(bundle["config"]["sensor_weights"][name])
        logs = [option_log_probabilities(
            probabilities, selected.group.to_numpy(), association, projected=projected
        ) for projected in (False, True)]
        for index, row in enumerate(selected.itertuples()):
            for word in row.options:
                word_index = bundle["word_index"].get(word, fallback_word)
                for field, values in zip(fields, logs, strict=True):
                    evidence[row.qa_id][word][field] += weight * float(values[index, word_index])
    return evidence


def adapted_evidence(evidence, decoder):
    mode = decoder.get("fusion_mode")
    if mode not in FUSION_FIELDS:
        raise ValueError(f"unknown emotion fusion mode: {mode!r}")
    if not decoder.get("use_position"):
        raise ValueError("these fixed fusion candidates require the declared position prior")
    if mode == "v1_manner_mean":
        return evidence
    field = FUSION_FIELDS[mode]
    return {
        qa_id: {word: {**item, "sensor": item[field]} for word, item in options.items()}
        for qa_id, options in evidence.items()
    }


def predict_rows(frame, evidence, decoder):
    return legacy.predict_rows(frame, adapted_evidence(evidence, decoder), decoder)


def validation_views(frame, evidence, decoders):
    return pd.concat([
        legacy.validation_views(frame, adapted_evidence(evidence, decoder), [decoder])
        for decoder in decoders
    ], ignore_index=True)
