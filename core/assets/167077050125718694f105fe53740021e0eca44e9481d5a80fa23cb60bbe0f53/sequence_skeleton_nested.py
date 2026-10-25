"""Subject-excluded Skeleton window mass and constrained HAU order decoding."""

from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from cuhkx.actions import hau_action_class
from cuhkx.clip_constraints import repeated_recording_groups
from cuhkx.harn import action_class
from cuhkx.metadata import clip_key, feature_unit, subject_id
from cuhkx.pairwise_temporal import PAIR_INDICES, before_counts, decode_pairwise_order
from cuhkx.repeated_temporal import share_repeated_pair_probabilities
from cuhkx.submission import available_letters


def array_digest(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        value = np.ascontiguousarray(value)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def prepare_sequences(qa, *, training):
    """Construct inference metadata without touching answer/prediction columns."""
    selected = qa.loc[qa.source.eq("HAU") & qa.category.eq("sequence")]
    groups = repeated_recording_groups(qa)
    records = []
    for _, row in selected[["qa_id", "source", "path", "category", "A", "B", "C", "D"]].iterrows():
        if available_letters(row) != "ABCD":
            raise ValueError("sequence requires exactly four valid ordered option columns")
        classes = tuple(hau_action_class(row[letter]) for letter in "ABCD")
        if len(set(classes)) != 4:
            raise ValueError("sequence options must describe four distinct actions")
        clip = clip_key(row.path)
        subject = subject_id(row.path)
        if training and subject is None:
            raise ValueError("training subject must be independently parsed from the official path")
        records.append({"qa_id": str(row.qa_id), "path": row.path, "clip": clip,
                        "unit": feature_unit(row.path, training=training), "subject": subject,
                        "repeat_group": groups[clip], "classes": classes})
    frame = pd.DataFrame(records)
    if frame.empty or frame.qa_id.duplicated().any() or frame.unit.duplicated().any():
        raise ValueError("sequence QA must be nonempty and unique by QA ID and recording")
    validate_target(frame)
    return frame


def validate_target(frame, training_subjects=()):
    if frame.qa_id.duplicated().any() or frame.unit.duplicated().any():
        raise ValueError("target rows must be unique by QA ID and recording")
    actual = frame.path.map(subject_id)
    if actual.fillna("").tolist() != frame.subject.fillna("").tolist():
        raise ValueError("target subject field differs from its independently parsed path")
    if set(actual.dropna()) & set(training_subjects):
        raise ValueError("target includes a subject used to fit the classifier or prior")
    for _, group in frame.groupby("repeat_group", sort=False):
        if len(set(group.subject.fillna("test"))) != 1:
            raise ValueError("a repeated-scene group crosses a subject boundary")
        if len(group) > 3:
            raise ValueError("a repeated scene has more than three recordings")


def read_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in
                ("X", "units", "starts", "stops", "centers", "feature_names")}
    data["units"] = data["units"].astype(str)
    data["feature_names"] = data["feature_names"].astype(str)
    n = len(data["units"])
    if data["X"].shape != (n, len(data["feature_names"])):
        raise ValueError("archive feature dimensions differ from its metadata")
    if any(len(data[key]) != n for key in ("starts", "stops", "centers")):
        raise ValueError("archive window metadata is misaligned")
    if not np.isfinite(data["X"]).all() or not np.isfinite(data["centers"]).all():
        raise ValueError("archive contains nonfinite features or centers")
    if (data["centers"] < 0).any() or (data["centers"] > 1).any():
        raise ValueError("window centers must be normalized physical times in [0, 1]")
    if (data["stops"] < data["starts"]).any():
        raise ValueError("archive has reversed window bounds")
    windows = list(zip(data["units"], data["starts"], data["stops"], strict=True))
    if len(set(windows)) != n:
        raise ValueError("archive has duplicate windows")
    by_unit = {}
    for index, unit in enumerate(data["units"]):
        by_unit.setdefault(str(unit), []).append(index)
    data["unit_rows"] = {unit: np.asarray(rows, dtype=int) for unit, rows in by_unit.items()}
    data["path"] = str(path.resolve())
    return data


def temporal_before_probability(centers, left, right, *, mode, slope=4.0):
    """P(left before right), with physical-time sorting and half weight for ties."""
    centers, left, right = (np.asarray(item, dtype=np.float64) for item in (centers, left, right))
    if centers.ndim != 1 or not len(centers) or left.shape != centers.shape or right.shape != centers.shape:
        raise ValueError("time and action probability vectors must be nonempty and aligned")
    if not all(np.isfinite(item).all() for item in (centers, left, right)):
        raise ValueError("action time mass must be finite")
    if (centers < 0).any() or (centers > 1).any() or (left < 0).any() or (right < 0).any():
        raise ValueError("invalid physical times or negative probability mass")
    if left.sum() <= 0 or right.sum() <= 0:
        raise ValueError("an action has no temporal probability mass")
    order = np.argsort(centers, kind="stable")
    centers = centers[order]
    left, right = left[order] / left.sum(), right[order] / right.sum()
    if mode == "mass_before":
        relation = ((centers[:, None] < centers[None, :]).astype(float)
                    + 0.5 * (centers[:, None] == centers[None, :]))
        return float(np.clip(left @ relation @ right, 0.0, 1.0))
    if mode == "expected_time":
        delta = float(right @ centers - left @ centers)
        return float(1.0 / (1.0 + np.exp(-slope * delta)))
    raise ValueError(f"unknown temporal probability mode: {mode}")


def unit_weights(units):
    counts = Counter(units)
    return np.asarray([1.0 / counts[unit] for unit in units], dtype=np.float64)


def training_plan(qa, archives, train_subjects, validation_subjects):
    """Select actual fitting rows before reading any QA answer or fitting a model."""
    train, valid = set(train_subjects), set(validation_subjects)
    sequence = prepare_sequences(qa, training=True)
    universe = set(sequence.subject)
    if not train or train & valid or (train | valid) - universe:
        raise ValueError("fitting and validation subjects overlap, are empty, or are unknown")
    prior_frame = sequence.loc[sequence.subject.isin(train)].reset_index(drop=True)
    truth = qa.loc[qa.qa_id.isin(prior_frame.qa_id), ["qa_id", "answer"]].set_index("qa_id").answer
    orders = []
    for row in prior_frame.itertuples():
        answer = str(truth[row.qa_id]).strip().upper()
        if sorted(answer) != list("ABCD"):
            raise ValueError("training sequence answer is not a legal four-action permutation")
        orders.append([row.classes["ABCD".index(letter)] for letter in answer])
    counts = before_counts(orders)
    selected = {}
    for name, archive in archives.items():
        units = archive["units"]
        rows = np.asarray([index for index, unit in enumerate(units)
                           if unit.startswith("Training/HARn/") and subject_id(unit) in train], dtype=int)
        if not len(rows):
            raise ValueError("a Skeleton scale has no eligible HARn training windows")
        fitted_units = units[rows]
        if {subject_id(unit) for unit in fitted_units} != train:
            raise ValueError("HARn training windows do not cover exactly the requested subjects")
        labels = np.asarray([action_class(unit) for unit in fitted_units], dtype=np.int64)
        weights = unit_weights(fitted_units)
        selected[name] = {"rows": rows, "labels": labels, "weights": weights}
    return sequence, prior_frame, counts, selected


def scope_contract(qa, archives, train_subjects, validation_subjects, config, seed):
    sequence, prior, counts, selected = training_plan(qa, archives, train_subjects, validation_subjects)
    scales = {}
    all_units = set()
    for name, selection in selected.items():
        archive, rows = archives[name], selection["rows"]
        units = sorted(set(archive["units"][rows]))
        all_units.update(units)
        scales[name] = {
            "archive_path": archive["path"], "window_indices": rows.tolist(),
            "window_indices_sha256": array_digest(rows), "training_units": units,
            "actual_subjects": sorted({subject_id(unit) for unit in units}),
            "window_content_sha256": array_digest(
                archive["X"][rows], archive["units"][rows], archive["starts"][rows],
                archive["stops"][rows], archive["centers"][rows], selection["labels"], selection["weights"]),
            "window_rows": len(rows), "action_classes": sorted(set(map(int, selection["labels"]))),
        }
    harn = qa.loc[qa.source.eq("HARn") & qa.path.map(subject_id).isin(train_subjects)]
    harn = harn.loc[harn.path.map(lambda p: feature_unit(p, training=True)).isin(all_units)]
    record = {
        "train_subjects": sorted(train_subjects), "validation_subjects": sorted(validation_subjects),
        "prediction_qa_ids": sequence.loc[sequence.subject.isin(validation_subjects), "qa_id"].tolist(),
        "prior_training_qa_ids": prior.qa_id.tolist(),
        "harn_metadata_qa_ids": harn.qa_id.tolist(),
        "training_qa_ids": prior.qa_id.tolist() + harn.qa_id.tolist(),
        "label_source_note": "HARn targets come from official Training/HARn class paths, including assets without QA rows. HARn QA IDs document matching metadata, not consumed answer labels.",
        "prior_counts": [[left, right, count] for (left, right), count in sorted(counts.items())],
        "scales": scales, "config": config, "seed": seed,
    }
    return record, counts, selected


def fit_bundle(qa, archives, train_subjects, validation_subjects, config, seed):
    record, counts, selections = scope_contract(
        qa, archives, train_subjects, validation_subjects, config, seed
    )
    models = {}
    for scale_index, name in enumerate(config["scales"]):
        selected = selections[name]
        model = ExtraTreesClassifier(
            n_estimators=config["trees"], min_samples_leaf=config["minimum_leaf"],
            max_features=config["max_features"], class_weight=config["class_weight"],
            n_jobs=config["workers"], random_state=seed + 1000 * scale_index,
        )
        model.fit(archives[name]["X"][selected["rows"]], selected["labels"],
                  sample_weight=selected["weights"])
        models[name] = model
    bundle = {"models": models, "counts": counts, "config": config,
              "training_subjects": sorted(train_subjects),
              "feature_names": {name: archives[name]["feature_names"] for name in config["scales"]}}
    return bundle, record


def base_evidence(bundle, target, archives):
    """Predict only from target paths/options/features; target answers are not accessed."""
    validate_target(target, bundle["training_subjects"])
    config = bundle["config"]
    modes = ("mass_before", "expected_time")
    evidence = {mode: np.full((len(config["scales"]), len(target), 6), 0.5) for mode in modes}
    failures = [[] for _ in range(len(target))]
    for scale_index, name in enumerate(config["scales"]):
        archive, model = archives[name], bundle["models"][name]
        if not np.array_equal(archive["feature_names"], bundle["feature_names"][name]):
            raise ValueError("feature names/order differ from the fitted scale")
        rows_by_unit = {unit: archive["unit_rows"].get(unit, np.asarray([], dtype=int)) for unit in target.unit}
        present = [rows for rows in rows_by_unit.values() if len(rows)]
        all_rows = np.concatenate(present) if present else np.asarray([], dtype=int)
        probabilities = model.predict_proba(archive["X"][all_rows]) if len(all_rows) else np.empty((0, 0))
        offsets, start = {}, 0
        for unit, rows in rows_by_unit.items():
            offsets[unit] = (start, start + len(rows))
            start += len(rows)
        columns = {int(label): index for index, label in enumerate(model.classes_)}
        for index, row in enumerate(target.itertuples()):
            rows = rows_by_unit[row.unit]
            if not len(rows):
                failures[index].append(name + ":missing_windows")
                continue
            if any(label not in columns for label in row.classes):
                failures[index].append(name + ":unseen_action_class")
                continue
            lo, hi = offsets[row.unit]
            values = probabilities[lo:hi][:, [columns[label] for label in row.classes]]
            if (values.sum(axis=0) <= 0).any():
                failures[index].append(name + ":zero_action_mass")
                continue
            for mode in modes:
                for pair_index, (left, right) in enumerate(PAIR_INDICES):
                    evidence[mode][scale_index, index, pair_index] = temporal_before_probability(
                        archive["centers"][rows], values[:, left], values[:, right],
                        mode=mode, slope=config["expectation_slope"],
                    )
    usable = np.asarray([not failure for failure in failures])
    pooled, shared = {}, None
    for mode in modes:
        evidence[mode][:, ~usable] = 0.5
        pooled[mode], shared = share_repeated_pair_probabilities(
            evidence[mode], np.asarray(target.classes.tolist()), target.repeat_group.tolist(),
            family_indices=range(len(config["scales"])), sibling_weight=config["sibling_weight"],
        )
        pooled[mode][~usable] = 0.5
    return {"qa_ids": target.qa_id.tolist(), "pooled": pooled, "usable": usable,
            "fallback_reasons": [";".join(items) for items in failures], "shared_pairs": shared}


def predict_rows(bundle, target, evidence, decoder):
    validate_target(target, bundle["training_subjects"])
    if evidence["qa_ids"] != target.qa_id.tolist():
        raise ValueError("evidence rows differ from target QA ID order")
    predictions, margins = [], []
    mode = decoder["mode"]
    for index, row in enumerate(target.itertuples()):
        probabilities = np.full(6, 0.5) if mode == "prior_only" else evidence["pooled"][mode][index]
        prediction, margin = decode_pairwise_order(
            probabilities, row.classes, bundle["counts"],
            prior_weight=decoder["prior_weight"], prior_smoothing=bundle["config"]["prior_smoothing"],
        )
        predictions.append(prediction)
        margins.append(margin)
    return pd.DataFrame({"qa_id": target.qa_id.tolist(), "prediction": predictions,
                         "margin": margins, "subject": target.subject.tolist(),
                         "selected_decoder": decoder["name"],
                         "evidence_available": evidence["usable"],
                         "fallback_reason": evidence["fallback_reasons"],
                         "shared_pair_count": evidence["shared_pairs"]})
