"""Fixed unlabeled generic-video descriptors for the relative emotion expert."""

from __future__ import annotations

import hashlib

import numpy as np

VISUAL_SPEC = {
    "backbone": "r2plus1d_18", "weights": "R2Plus1D_18_Weights.KINETICS400_V1",
    "frames": 16, "windows": 8, "window_seconds": 2.5,
}


def array_digest(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def training_content(frame, matrices, masks, fit_subjects):
    records = {}
    for name, matrix in matrices.items():
        rows = np.flatnonzero(frame.subject.isin(fit_subjects).to_numpy() & masks[name])
        records[name] = {
            "row_indices": rows.tolist(), "row_indices_sha256": array_digest(rows),
            "matrix_positions_units_sha256": array_digest(matrix[rows], frame.iloc[rows].position.to_numpy(),
                                                            frame.iloc[rows].unit.to_numpy(dtype=str)),
            "feature_count": matrix.shape[1], "rows": len(rows),
        }
    return records


def fit_bundle(frame, contexts, matrices, masks, fit_subjects, config, seed):
    """Retain exactly the matrices/positions selected by the unchanged v3 fitter."""
    from cuhkx.emotion_nested import fit_bundle as legacy_fit

    content = training_content(frame, matrices, masks, fit_subjects)
    bundle = legacy_fit(frame, contexts, matrices, masks, fit_subjects, config, seed)
    for index, (name, record) in enumerate(content.items()):
        units = frame.iloc[record["row_indices"]].unit.tolist()
        if bundle["fitted_units"][name] != units:
            raise ValueError("recorded content differs from the actual legacy fitting units")
        model = bundle["models"][name]
        if model.n_features_in_ != record["feature_count"] or model.random_state != seed + 1000 * index:
            raise ValueError("sensor feature count or seed order differs from the frozen v3 fitter")
    bundle["training_content"] = content
    return bundle


def verify_training_content(bundle, scope, frame, matrices, masks):
    expected = training_content(frame, matrices, masks, bundle["training_subjects"])
    if bundle.get("training_content") != expected or scope.get("training_content") != expected:
        raise ValueError("retained fitting matrix/target hashes differ from the subject-excluded inputs")


def aggregate_visual(archive):
    """Produce one keyed mean/std vector only for physically valid paired media.

    Missing units are omitted from this sensor archive. The established
    sensor_matrix then disables this modality for their entire three-row scene.
    No questions, scene IDs, recording positions or targets enter aggregation.
    """
    if any(str(archive[name]) != str(value) for name, value in VISUAL_SPEC.items()):
        raise ValueError("generic visual specification differs from the frozen experiment")
    units = archive["units"].astype(str)
    n = len(units)
    if archive["X"].shape != (n, 512) or not np.isfinite(archive["X"]).all():
        raise ValueError("invalid generic visual embeddings")
    for field in ("available", "alignment_ok"):
        if archive[field].shape != (n,) or archive[field].dtype != np.bool_:
            raise ValueError("generic visual masks must be explicit boolean rows")
    for field in ("fps", "frame_count", "duration_seconds", "starts_seconds", "stops_seconds",
                  "centers_seconds", "starts", "stops", "centers"):
        if archive[field].shape != (n,):
            raise ValueError("generic visual metadata axes differ")
    if archive["frame_indices"].shape != (n, 16):
        raise ValueError("generic visual sampling requires sixteen frame indices")
    keys = list(zip(units, archive["modalities"].astype(str), archive["window_indices"], strict=True))
    index = {key: row for row, key in enumerate(keys)}
    if len(index) != n or n != len(set(units)) * 16:
        raise ValueError("generic visual keys must form a complete unique modality/window grid")
    output_units, vectors, rejected, repeated_windows = [], [], [], []
    for unit in sorted(set(units)):
        if not unit.startswith(("Training/HAU/", "Testing/")):
            raise ValueError("only official HAU unit paths are allowed")
        required = [(unit, modality, window) for modality in ("Depth", "IR") for window in range(8)]
        if not set(required) <= set(index):
            raise ValueError("generic visual grid is incomplete")
        rows = np.asarray([index[key] for key in required], dtype=int)
        if not archive["available"][rows].all():
            rejected.append({"unit": unit, "reason": "missing_modality"})
            continue
        if not archive["alignment_ok"][rows].all():
            rejected.append({"unit": unit, "reason": "physical_alignment_mask_false"})
            continue
        fps, counts, frames = (archive[field][rows] for field in ("fps", "frame_count", "frame_indices"))
        if (fps <= 0).any() or (counts <= 0).any() or not np.isfinite(fps).all():
            raise ValueError("invalid available generic video metadata")
        if (frames < 0).any() or (frames >= counts[:, None]).any() or (np.diff(frames, axis=1) < 0).any():
            raise ValueError("generic video sampling indices are outside the recording")
        duration = counts / fps
        left, right = frames[:, 0] / fps, (frames[:, -1] + 1) / fps
        center = (left + right) / 2
        for field, expected in (("duration_seconds", duration), ("starts_seconds", left),
                                ("stops_seconds", right), ("centers_seconds", center),
                                ("starts", left / duration), ("stops", right / duration),
                                ("centers", center / duration)):
            if not np.array_equal(archive[field][rows], expected):
                raise ValueError("stored generic video timestamps do not match frame indices")
        tolerance = max(1 / fps) + 1e-6
        if any(np.abs(value[:8] - value[8:]).max() > tolerance for value in (duration, left, right, center)):
            raise ValueError("generic video alignment mask contradicts physical time")
        values = archive["X"][rows].astype(np.float32).reshape(2, 8, 512)
        if not np.allclose(np.linalg.norm(values, axis=2), 1, atol=1e-4, rtol=1e-4):
            raise ValueError("generic visual embeddings must be L2 normalized")
        vector = np.concatenate([part for modality in values for part in
                                 (modality.mean(axis=0), modality.std(axis=0, ddof=0))])
        vectors.append(vector)
        output_units.append(unit)
        if any(len(set(zip(left[half * 8:(half + 1) * 8], right[half * 8:(half + 1) * 8], strict=True))) < 8
               for half in range(2)):
            repeated_windows.append(unit)
    names = [f"visual_{modality}_{statistic}_{index:03d}" for modality in ("Depth", "IR")
             for statistic in ("mean", "std") for index in range(512)]
    if not vectors:
        raise ValueError("no physically valid paired HAU units")
    arrays = {"X": np.asarray(vectors, dtype=np.float32), "units": np.asarray(output_units),
              "feature_names": np.asarray(names)}
    report = {"shape": list(arrays["X"].shape), "input_units": len(set(units)),
              "accepted_units": len(output_units), "rejected_units": rejected,
              "accepted_units_with_repeated_physical_windows": repeated_windows,
              "aggregation": "Depth mean512,std512,IR mean512,std512; population std over eight fixed windows; no additional L2 normalization",
              "missing_policy": "Omit invalid visual unit; unchanged sensor_matrix requires all three units of a scene, disabling visual for the whole scene only",
              "scope": "Official HAU training/test media; no QA answers, scene positions, classifier logits or stored predictions"}
    return arrays, report
