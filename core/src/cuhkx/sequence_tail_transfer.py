"""Window inference from an exactly subject-matched retained HARn video model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cuhkx.harn_tail import configure_torch, tail_network
from cuhkx.pairwise_temporal import PAIR_INDICES
from cuhkx.repeated_temporal import share_repeated_pair_probabilities
from cuhkx.sequence_skeleton_nested import temporal_before_probability, validate_target
from cuhkx.validation import sha256_file
from cuhkx.video_midfeatures import MID_SHAPE
from cuhkx.video_midfeatures import SPEC as CORE_SPEC

SPEC = {"core": CORE_SPEC, "windows": 8, "window_seconds": 2.5,
        "layout": "unit,Depth_then_IR,window,channel,time,height,width"}


def paired_batch(middle, rows):
    """Keep the Depth/IR views adjacent within each independently keyed window."""
    return np.stack([middle[unit, :, window] for unit, window in rows]).reshape(-1, *MID_SHAPE)


def temporal_unit_metadata(metadata, unit):
    keys = list(zip(metadata["units"].astype(str), metadata["modalities"].astype(str),
                    metadata["window_indices"].astype(int), strict=True))
    lookup = {key: index for index, key in enumerate(keys)}
    if len(keys) != len(lookup):
        raise ValueError("duplicate physical window metadata key")
    pairs = []
    for window in range(8):
        if any((unit, modality, window) not in lookup for modality in ("Depth", "IR")):
            raise ValueError("explicit eight-window modality grid is incomplete")
        rows = [lookup[unit, modality, window] for modality in ("Depth", "IR")]
        if not metadata["available"][rows].all():
            return None, "missing_modality"
        fps = metadata["fps"][rows]
        if not np.isfinite(fps).all() or (fps <= 0).any():
            raise ValueError("invalid available-video FPS")
        tolerance = float(np.max(1.0 / fps)) + 1e-6
        for field in ("duration_seconds", "starts_seconds", "stops_seconds", "centers_seconds"):
            values = metadata[field][rows]
            if not np.isfinite(values).all():
                raise ValueError("nonfinite video timing")
            if abs(float(values[0] - values[1])) > tolerance:
                return None, f"physical_alignment:{field}"
        if not metadata["alignment_ok"][rows].all():
            return None, "physical_alignment_mask"
        times = [float(metadata[field][rows].mean()) for field in ("starts", "stops", "centers")]
        if not np.isfinite(times).all() or not 0 <= times[0] <= times[2] <= times[1] <= 1:
            raise ValueError("invalid normalized physical window bounds")
        pairs.append(times)
    if len({(start, stop) for start, stop, _ in pairs}) != 8:
        return None, "duplicate_physical_windows"
    return np.asarray(pairs, dtype=np.float64), ""


class WindowMidBank:
    def __init__(self, manifest_path):
        self.path = Path(manifest_path).resolve()
        manifest = json.loads(self.path.read_text())
        if manifest["spec"] != SPEC:
            raise ValueError("window middle-feature specification differs")
        self.manifest = manifest
        paths = {}
        for name in ("features", "metadata"):
            entry = manifest[name]
            path = (self.path.parent / entry["path"]).resolve()
            if sha256_file(path) != entry["sha256"]:
                raise ValueError("window feature/metadata SHA differs")
            paths[name] = path
        with np.load(paths["metadata"], allow_pickle=False) as source:
            self.metadata = {key: source[key].copy() for key in source.files}
        self.units = np.asarray(manifest["units"], dtype=str)
        if (len(set(self.units)) != len(self.units)
                or any(not unit.startswith(("Training/HAU/", "Testing/")) for unit in self.units)):
            raise ValueError("window bank must have unique official HAU or Testing units")
        if set(self.metadata["units"].astype(str)) != set(self.units):
            raise ValueError("metadata and middle unit axes differ")
        self.middle = np.load(paths["features"], mmap_mode="r", allow_pickle=False)
        if self.middle.shape != (len(self.units), 2, 8, *MID_SHAPE) or self.middle.dtype != np.float16:
            raise ValueError("paired middle window shape/dtype differs")
        for start in range(0, len(self.units), 8):
            if not np.isfinite(self.middle[start:start + 8]).all():
                raise ValueError("nonfinite middle window features")
        self.lookup = {unit: index for index, unit in enumerate(self.units)}
        self.timing, self.invalid = {}, {}
        for unit in self.units:
            times, reason = temporal_unit_metadata(self.metadata, unit)
            if reason:
                self.invalid[unit] = reason
            else:
                self.timing[unit] = times


def window_probabilities(model, bank, target, training_subjects):
    validate_target(target, training_subjects)
    selected = [unit for unit in target.unit if unit in bank.timing]
    indices = [(bank.lookup[unit], window) for unit in selected for window in range(8)]
    probabilities = np.empty((len(selected), 8, len(model.classes_)), dtype=np.float32)
    if not indices:
        return {"units": np.asarray(selected, dtype=str), "classes": model.classes_, "probabilities": probabilities}
    torch = configure_torch()
    network = tail_network()
    network.load_state_dict({key: torch.from_numpy(value) for key, value in model.state.items()})
    network.eval().cuda()
    pieces = []
    with torch.inference_mode():
        for start in range(0, len(indices), model.settings["batch_units"]):
            values = paired_batch(bank.middle, indices[start:start + model.settings["batch_units"]])
            tensor = torch.from_numpy(values).to("cuda", dtype=torch.float32)
            with torch.autocast(device_type="cuda"):
                logits = network(tensor)
            raw = logits.float()[:, model.classes_].softmax(dim=1)
            pieces.append(raw.reshape(-1, 2, len(model.classes_)).mean(dim=1).cpu().numpy())
    probabilities[:] = np.concatenate(pieces).reshape(probabilities.shape)
    del network
    torch.cuda.empty_cache()
    return {"units": np.asarray(selected, dtype=str), "classes": model.classes_, "probabilities": probabilities}


def temporal_evidence(bundle, target, bank, scored):
    validate_target(target, bundle["training_subjects"])
    config = bundle["config"]
    unit_lookup = {unit: index for index, unit in enumerate(scored["units"].astype(str))}
    columns = {int(label): index for index, label in enumerate(scored["classes"])}
    raw = {mode: np.full((1, len(target), 6), 0.5) for mode in ("mass_before", "expected_time")}
    reasons = []
    for index, row in enumerate(target.itertuples()):
        if row.unit not in unit_lookup:
            reasons.append(bank.invalid.get(row.unit, "missing_windows"))
            continue
        if any(label not in columns for label in row.classes):
            reasons.append("unseen_action_class")
            continue
        probabilities = scored["probabilities"][unit_lookup[row.unit]][:, [columns[label] for label in row.classes]]
        if (probabilities.sum(axis=0) <= 0).any():
            reasons.append("zero_action_mass")
            continue
        reasons.append("")
        for mode, values in raw.items():
            for pair, (left, right) in enumerate(PAIR_INDICES):
                values[0, index, pair] = temporal_before_probability(
                    bank.timing[row.unit][:, 2], probabilities[:, left], probabilities[:, right],
                    mode=mode, slope=config["expectation_slope"],
                )
    usable = np.asarray([not reason for reason in reasons])
    pooled, shared = {}, None
    for mode, values in raw.items():
        pooled[mode], shared = share_repeated_pair_probabilities(
            values, np.asarray(target.classes.tolist()), target.repeat_group.tolist(),
            family_indices=[0], sibling_weight=config["sibling_weight"],
        )
        pooled[mode][~usable] = 0.5
    return {"qa_ids": target.qa_id.tolist(), "pooled": pooled, "usable": usable,
            "fallback_reasons": reasons, "shared_pairs": shared}
