"""Physical-time pair curves from exactly scoped, jointly trained VideoMAE heads."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import ExtraTreesClassifier

from cuhkx.action_window_head import unpack
from cuhkx.pairwise_temporal import (
    PAIR_INDICES,
    before_counts,
    decode_pairwise_order,
    pair_curve_features,
)
from cuhkx.repeated_temporal import share_repeated_pair_probabilities
from cuhkx.sequence_skeleton_nested import array_digest, temporal_before_probability


def read_centers(bank_dir, raw_records):
    records = json.loads(Path(raw_records).read_text())["records"]
    lookup = {(row["unit"], row["modality"]): row for row in records}
    result = {}
    with np.load(Path(bank_dir) / "metadata.npz", allow_pickle=False) as data:
        for unit, modality, indices, available in zip(data["units"], data["modalities"], data["frame_indices"], data["available"], strict=True):
            row = lookup[(str(unit), str(modality))]
            assert bool(available) == row["available"]
            if available:
                count = row["metadata"]["frame_count"]
                assert count > 1 and indices.min() >= 0 and indices.max() < count
                times = indices.mean(axis=1) / (count - 1)
                assert (np.diff(times) >= 0).all(), "physical windows are not ordered"
                result[(str(unit), str(modality))] = times
    return result


def window_curves(head, matrix, units, centers):
    """Average available modalities only after physical-time interpolation."""
    import torch

    features, mask = unpack(matrix)
    normalized = head._standardized(features, mask)
    with torch.inference_mode():
        logits = torch.nn.functional.linear(torch.from_numpy(normalized), torch.from_numpy(head.state["weight"]),
                                             torch.from_numpy(head.state["bias"])).numpy()
    logits = logits.astype(np.float64) - np.log(head.positive_weight)
    probability = expit(logits)
    probability[..., ~head.supported] = .5
    result = np.zeros((len(units), 16, 40), dtype=np.float64)
    for index, unit in enumerate(units):
        views = []
        for view, modality in enumerate(("Depth_Color", "IR")):
            present = mask[index, :, view]
            if not present.any():
                continue
            times = centers[(str(unit), modality)][present]
            values = probability[index, present, view]
            unique = np.unique(times)
            values = np.stack([values[times == time].mean(axis=0) for time in unique])
            times = unique
            views.append(np.stack([np.interp(np.linspace(0, 1, 16), times, values[:, action])
                                   for action in range(40)], axis=1))
        if not views:
            raise ValueError("sequence curves require an available aligned view")
        result[index] = np.mean(views, axis=0)
    return result


def pair_inputs(frame, curves):
    forward, reverse, mass = [], [], []
    for row, curve in zip(frame.itertuples(), curves, strict=True):
        local = []
        for left, right in PAIR_INDICES:
            a, b = row.classes[left], row.classes[right]
            forward.append(pair_curve_features([curve], a, b))
            reverse.append(pair_curve_features([curve], b, a))
            local.append(temporal_before_probability(np.linspace(0, 1, 16), curve[:, a], curve[:, b], mode="mass_before"))
        mass.append(local)
    return {"forward": np.asarray(forward), "reverse": np.asarray(reverse), "mass": np.asarray(mass)}


def training_arrays(qa, frame, pairs):
    truth = qa.loc[qa.qa_id.isin(frame.qa_id), ["qa_id", "answer"]].set_index("qa_id").answer
    labels, orders = [], []
    for row in frame.itertuples():
        answer = str(truth[row.qa_id])
        if sorted(answer) != list("ABCD"):
            raise ValueError("sequence training label is not a permutation")
        orders.append([row.classes["ABCD".index(letter)] for letter in answer])
        labels.extend(int(answer.index("ABCD"[left]) < answer.index("ABCD"[right])) for left, right in PAIR_INDICES)
    labels = np.asarray(labels, dtype=np.int64)
    features = np.concatenate([pairs["forward"], pairs["reverse"]])
    targets = np.concatenate([labels, 1 - labels])
    assert len(features) == len(targets) == len(frame) * 12
    return features, targets, before_counts(orders)


def fit(qa, frame, pairs, config, seed):
    matrix, targets, counts = training_arrays(qa, frame, pairs)
    models = {}
    for name, features in (("curve", matrix), ("identity", matrix[:, -80:])):
        models[name] = ExtraTreesClassifier(**config["forest"], n_jobs=3, random_state=seed + 7000000).fit(features, targets)
    bundle = {"models": models, "counts": counts, "training_subjects": sorted(set(frame.subject)), "config": config}
    scope = {"training_qa_ids": frame.qa_id.tolist(), "training_units": frame.unit.tolist(),
             "actual_training_subjects": sorted(set(frame.subject)), "training_matrix_shape": list(matrix.shape),
             "actual_pair_features_labels_sha256": array_digest(matrix, targets),
             "prior_counts": [[a, b, count] for (a, b), count in sorted(counts.items())]}
    return bundle, scope


def evidence(bundle, frame, pairs):
    if set(frame.subject) & set(bundle["training_subjects"]):
        raise ValueError("held-out prediction includes a fitted subject")
    result = {"prior_only": np.full((len(frame), 6), .5), "mass_before": pairs["mass"]}
    for name in ("curve", "identity"):
        model = bundle["models"][name]
        column = list(model.classes_).index(1)
        x, y = pairs["forward"], pairs["reverse"]
        if name == "identity":
            x, y = x[:, -80:], y[:, -80:]
        result["learned_" + name] = (.5 * (model.predict_proba(x)[:, column] + 1 - model.predict_proba(y)[:, column])).reshape(-1, 6)
    for name in ("mass_before", "learned_curve", "learned_identity"):
        result[name], _ = share_repeated_pair_probabilities(result[name][None], np.asarray(frame.classes.tolist()),
                                                           frame.repeat_group.tolist(), family_indices=[0],
                                                           sibling_weight=bundle["config"]["sibling_weight"])
    left, right = [np.clip(result[name], 1e-5, 1 - 1e-5) for name in ("mass_before", "learned_curve")]
    result["curve_mass_blend"] = expit(.5 * (np.log(left / (1 - left)) + np.log(right / (1 - right))))
    return result


def predict(bundle, frame, values, recipe):
    rows = []
    for row, probability in zip(frame.itertuples(), values[recipe["mode"]], strict=True):
        prediction, margin = decode_pairwise_order(probability, row.classes, bundle["counts"],
                                                   prior_weight=recipe["prior_weight"], prior_smoothing=1.5)
        rows.append({"qa_id": row.qa_id, "prediction": prediction, "margin": margin})
    return pd.DataFrame(rows)
