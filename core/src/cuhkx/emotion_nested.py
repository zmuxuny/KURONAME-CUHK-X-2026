"""Emotion inference on complete or partial question sets within audited scenes."""

from __future__ import annotations

import itertools
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.ensemble import ExtraTreesClassifier

from cuhkx.actions import HAU_CLASS_TEXT
from cuhkx.clip_constraints import repeated_recording_groups
from cuhkx.inference import question_metadata
from cuhkx.metadata import clip_key, feature_unit, normalize_text, option_atoms, subject_id
from cuhkx.submission import available_letters

ACTION_CATEGORIES = ("single", "multi", "combination", "sequence")
OFFICIAL_SUBJECTS = {f"user{i}" for i in (*range(1, 10), *range(16, 25))}


def prepare(qa: pd.DataFrame, *, training: bool, manifest: pd.DataFrame | None = None):
    metadata = question_metadata(qa)
    frame = metadata.loc[metadata.source.eq("HAU") & metadata.category.eq("emotion")].copy()
    frame["clip"] = frame.path.map(clip_key)
    frame["unit"] = frame.path.map(lambda value: feature_unit(value, training=training))
    if manifest is not None:
        if not {"clip", "group", "position"} <= set(manifest) or not manifest["clip"].is_unique:
            raise ValueError("scene manifest requires unique clip/group/position rows")
        positions = pd.to_numeric(manifest["position"], errors="raise")
        if positions.isna().any() or not positions.isin([0, 1, 2]).all():
            raise ValueError("scene manifest positions must be integers 0, 1, or 2")
        if manifest["group"].isna().any() or manifest["group"].astype(str).str.strip().eq("").any():
            raise ValueError("scene manifest groups must be nonempty")
        mapping = manifest.set_index("clip")
        if not set(frame["clip"]) <= set(mapping.index):
            raise ValueError("scene manifest does not cover every emotion clip")
        frame["group"] = frame["clip"].map(mapping.group)
        frame["position"] = frame["clip"].map(mapping.position).astype(int)
    else:
        mapping = repeated_recording_groups(metadata)
        frame["group"] = frame["clip"].map(mapping)
        if training:
            frame["position"] = frame.path.str.rsplit("-", n=1).str[1].astype(int) - 1
        else:
            frame["number"] = frame["clip"].str.extract(r"LM_test_(\d+)$", expand=False).astype(int)
            frame = frame.sort_values("number")
            if not frame.groupby("group").size().eq(3).all():
                raise ValueError("incomplete emotion questions require an explicit scene manifest")
            frame["position"] = frame.groupby("group", sort=False).cumcount()
    if not frame.position.isin([0, 1, 2]).all():
        raise ValueError("recording position must be 0, 1, or 2")
    if any(not group.position.is_unique or len(group) > 3 for _, group in frame.groupby("group")):
        raise ValueError("scene recording positions must be distinct, with at most three rows")
    frame["options"] = frame.apply(
        lambda row: tuple(normalize_text(row[letter]) for letter in available_letters(row)), axis=1
    )
    frame["letters"] = frame.apply(lambda row: tuple(available_letters(row)), axis=1)
    if training:
        answer_map = qa.set_index("qa_id").answer
        frame["answer"] = frame.qa_id.map(answer_map)
        frame["truth_word"] = frame.apply(lambda row: normalize_text(row[row.answer]), axis=1)
        frame["subject"] = frame.path.map(subject_id)
        if not set(frame.subject) <= OFFICIAL_SUBJECTS:
            raise ValueError("emotion training includes non-training subjects")
        if frame.groupby("group").subject.nunique().gt(1).any():
            raise ValueError("a recording scene cannot span training subjects")
    frame = frame.reset_index(drop=True)
    action_words = sorted({normalize_text(value) for value in HAU_CLASS_TEXT.values()})
    action_index = {word: i for i, word in enumerate(action_words)}
    clip_info = {row.clip: (row.group, row.position) for row in frame.itertuples()}
    counts = defaultdict(lambda: np.zeros((4, 3, len(action_words)), dtype=np.float64))
    for _, row in metadata.loc[
        metadata.source.eq("HAU") & metadata.category.isin(ACTION_CATEGORIES)
    ].iterrows():
        key = clip_key(row.path)
        if key not in clip_info:
            raise ValueError("HAU action clip lacks an emotion scene mapping")
        group, position = clip_info[key]
        category = ACTION_CATEGORIES.index(row.category)
        for letter in available_letters(row):
            for action in set(option_atoms(row[letter])):
                counts[group][category, position, action_index[action]] += 1
    contexts = {}
    for group in frame.group.unique():
        raw = counts[group]
        vector = np.concatenate((raw.sum(axis=(0, 1)), *raw.sum(axis=0), *raw.sum(axis=1)))
        contexts[group] = (vector / max(np.linalg.norm(vector), 1e-12)).astype(np.float32)
    return frame, contexts


def read_sensor(path: Path, pattern: str):
    with np.load(path, allow_pickle=False) as archive:
        names = archive["feature_names"].astype(str)
        mask = np.asarray([re.search(pattern, name) is not None for name in names])
        matrix = archive["X"][:, mask].astype(np.float32)
        units = archive["units"].astype(str)
    if not mask.any() or matrix.shape != (len(units), int(mask.sum())):
        raise ValueError("invalid sensor feature schema")
    if not np.isfinite(matrix).all() or len(set(units)) != len(units):
        raise ValueError("sensor features must be finite and units unique")
    return {"matrix": matrix, "index": {unit: i for i, unit in enumerate(units)},
            "names": names[mask]}


def sensor_matrix(frame: pd.DataFrame, source: dict):
    available = frame.unit.isin(source["index"]).to_numpy()
    raw = np.zeros((len(frame), len(source["names"])), dtype=np.float32)
    indices = np.flatnonzero(available)
    raw[indices] = source["matrix"][[source["index"][frame.iloc[i].unit] for i in indices]]
    selected = np.zeros(len(frame), dtype=bool)
    matrix = np.zeros_like(raw)
    for _, group in frame.groupby("group", sort=False):
        rows = group.index.to_numpy()
        if len(rows) == 3 and available[rows].all():
            values = raw[rows]
            matrix[rows] = (values - values.mean(axis=0)) / (values.std(axis=0) + 1e-4)
            selected[rows] = True
    return matrix, selected


def fit_bundle(frame, contexts, matrices, masks, fit_subjects, config, seed):
    fit_subjects = set(fit_subjects)
    if not fit_subjects or not fit_subjects <= OFFICIAL_SUBJECTS:
        raise ValueError("invalid fitted subject set")
    train = frame.loc[frame.subject.isin(fit_subjects)]
    words = sorted({word for options in train.options for word in options})
    word_index = {word: i for i, word in enumerate(words)}
    counts = np.ones((len(words), 2, 3), dtype=np.float64)
    association = np.ones((len(words), 3), dtype=np.float64)
    for row in train.itertuples():
        for word in row.options:
            counts[word_index[word], int(word != row.truth_word), row.position] += 1
        association[word_index[row.truth_word], row.position] += 1
    association /= association.sum(axis=1, keepdims=True)
    references = []
    for group_id, group in train.groupby("group", sort=False):
        references.append({"context": contexts[group_id],
                           "word_states": [(row.truth_word, row.position) for row in group.itertuples()]})
    models = {}
    fitted_units = {}
    for index, name in enumerate(matrices):
        rows = frame.subject.isin(fit_subjects).to_numpy() & masks[name]
        if not rows.any():
            raise ValueError(f"no training units for {name}")
        model = ExtraTreesClassifier(
            n_estimators=config["trees"], max_features="sqrt", min_samples_leaf=config["minimum_leaf"],
            max_depth=config["max_depth"], class_weight="balanced", n_jobs=config["workers"],
            random_state=seed + 1000 * index,
        )
        model.fit(matrices[name][rows], frame.loc[rows, "position"].to_numpy())
        models[name] = model
        fitted_units[name] = frame.loc[rows, "unit"].tolist()
    return {"models": models, "words": words, "word_index": word_index,
            "lexical": np.log(counts[:, 0]) - np.log(counts[:, 1]),
            "association": association, "references": references, "config": config,
            "training_subjects": sorted(fit_subjects), "fitted_units": fitted_units}


def base_evidence(bundle, frame, contexts, matrices, masks, target_rows):
    """Compute reusable scores without reading target answers or target semantic labels."""
    rows = np.asarray(target_rows, dtype=int)
    target = frame.iloc[rows]
    if "subject" in target and set(target.subject) & set(bundle["training_subjects"]):
        raise ValueError("target subject was used to train an upstream emotion model")
    log_sensors = np.zeros((len(rows), 3), dtype=np.float64)
    total_weight = np.zeros(len(rows), dtype=np.float64)
    for name, model in bundle["models"].items():
        valid = masks[name][rows]
        if not valid.any():
            continue
        probabilities = np.full((int(valid.sum()), 3), 1e-8, dtype=np.float64)
        probabilities[:, model.classes_.astype(int)] = model.predict_proba(matrices[name][rows[valid]])
        weight = bundle["config"]["sensor_weights"][name]
        log_sensors[valid] += weight * np.log(np.clip(probabilities, 1e-8, 1))
        total_weight[valid] += weight
    log_sensors /= np.maximum(total_weight[:, None], 1)
    log_sensors -= logsumexp(log_sensors, axis=1, keepdims=True)
    probabilities = np.exp(log_sensors)
    context_scores = {}
    references = bundle["references"]
    reference_vectors = np.stack([item["context"] for item in references])
    for group_id in target.group.unique():
        words = sorted({word for options in target.loc[target.group.eq(group_id), "options"] for word in options})
        counts = {word: np.full(3, bundle["config"]["context_smoothing"]) for word in words}
        similarities = reference_vectors @ contexts[group_id]
        weights = np.exp(bundle["config"]["context_beta"] * (similarities - 1))
        for reference, weight in zip(references, weights, strict=True):
            for word, state in reference["word_states"]:
                if word in counts:
                    counts[word][state] += weight
        context_scores[group_id] = {
            word: np.log(value / value.sum()) for word, value in counts.items()
        }
    evidence = {}
    for index, (_, row) in enumerate(target.iterrows()):
        local = {}
        for word in row.options:
            key = bundle["word_index"].get(word)
            lexical = np.zeros(3) if key is None else bundle["lexical"][key]
            association = np.full(3, 1 / 3) if key is None else bundle["association"][key]
            local[word] = {"lexical": lexical, "context": context_scores[row.group][word],
                           "sensor": float(np.log(max(probabilities[index] @ association, 1e-8))),
                           "log_manner": log_sensors[index]}
        evidence[row.qa_id] = local
    return evidence


def feasible_assignments(group: pd.DataFrame):
    """Return shared-core/different-word choices inside one externally supplied scene."""
    if len(group) not in (1, 2, 3) or group.group.nunique() != 1:
        raise ValueError("decode one scene containing one to three emotion rows")
    options = list(group.options)
    allowed = [set(value) for value in options]
    common = set.intersection(*allowed)
    if len(group) > 1 and len(common) == 3:
        allowed = [common] * len(group)
    elif len(group) == 3:
        pairs = [(i, j, allowed[i] & allowed[j]) for i, j in itertools.combinations(range(3), 2)]
        pairs = [(i, j, core) for i, j, core in pairs if len(core) == 3]
        if len(pairs) == 1:
            i, j, core = pairs[0]
            allowed[i], allowed[j] = core, core
    result = [words for words in itertools.product(*options)
              if len(set(words)) == len(words)
              and all(word in permitted for word, permitted in zip(words, allowed, strict=True))]
    if not result:
        raise ValueError("no feasible distinct-word emotion assignment")
    return result


def decode_group(group, evidence, config):
    assignments = feasible_assignments(group)
    rows = list(group.itertuples())
    option_values = []
    for row in rows:
        scores = {}
        for word in row.options:
            item = evidence[row.qa_id][word]
            semantic = config["lexical"] * item["lexical"] + config["context"] * item["context"]
            if config["use_position"]:
                scores[word] = semantic[row.position] + config["sensor"] * item["sensor"]
            else:
                scores[word] = logsumexp(semantic + config["sensor"] * item["log_manner"])
        option_values.append(scores)
    values = [sum(scores[word] for scores, word in zip(option_values, words, strict=True))
              for words in assignments]
    order = np.argsort(-np.asarray(values), kind="stable")
    words = assignments[int(order[0])]
    margin = float(values[order[0]] - values[order[1]]) if len(order) > 1 else 0.0
    return {row.qa_id: (row.letters[row.options.index(word)], margin)
            for row, word in zip(rows, words, strict=True)}


def predict_rows(frame, evidence, decoder):
    decisions = {}
    for _, group in frame.groupby("group", sort=False):
        decisions.update(decode_group(group, evidence, decoder))
    return pd.DataFrame([{"qa_id": row.qa_id, "prediction": decisions[row.qa_id][0],
                          "margin": decisions[row.qa_id][1]} for row in frame.itertuples()])


def validation_views(frame, evidence, decoders):
    """Score complete scenes and masked question subsets; retain full unlabeled sensor context."""
    records = []
    for group_id, group in frame.groupby("group", sort=False):
        views = [("complete", group)]
        views += [("pair", group.loc[list(indices)]) for indices in itertools.combinations(group.index, 2)]
        views += [("singleton", group.loc[[index]]) for index in group.index]
        for view_name, visible in views:
            for decoder in decoders:
                decisions = decode_group(visible, evidence, decoder)
                for row in visible.itertuples():
                    records.append({"qa_id": row.qa_id, "group": group_id, "subject": row.subject,
                                    "view": view_name, "decoder": decoder["name"],
                                    "correct": decisions[row.qa_id][0] == row.answer})
    return pd.DataFrame(records)


def choose_decoder(scores, config):
    results = []
    weights = config["validation_view_weights"]
    for order, decoder in enumerate(config["decoders"]):
        selected = scores.loc[scores.decoder.eq(decoder["name"])]
        by_view = selected.groupby("view").correct.mean().to_dict()
        by_subject = selected.groupby(["subject", "view"]).correct.mean().unstack()
        weighted = sum(weights[view] * by_view[view] for view in weights)
        worst = float(sum(weights[view] * by_subject[view] for view in weights).min())
        results.append({"name": decoder["name"], "weighted_accuracy": float(weighted),
                        "worst_subject_weighted_accuracy": worst, "by_view": by_view, "order": order})
    chosen = max(results, key=lambda row: (row["weighted_accuracy"], row["worst_subject_weighted_accuracy"], -row["order"]))
    decoder = next(item for item in config["decoders"] if item["name"] == chosen["name"])
    return decoder, results
