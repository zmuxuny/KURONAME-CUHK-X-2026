"""Pre-registered separate-view A1 / adjacent-delta B1 heads for the 16-window HAU bank."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from cuhkx.action_mil import positive_weights
from cuhkx.action_window_head import CHANNELS, VIEWS, WindowPartialActionClassifier
from cuhkx.sequence_skeleton_nested import array_digest
from cuhkx.validation import sha256_file

MODALITIES = ("Depth_Color", "IR")


def _row_digest(row):
    value = np.ascontiguousarray(row, dtype=np.float32)
    return hashlib.sha256(value.view(np.uint8)).hexdigest()


def read_server_window_bank(directory, expected_windows=16):
    """Read each view by availability only; alignment is metadata, not a masking rule."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if expected_windows != 16 or manifest["kind"] != "cuhkx_videomae_ssv2_hau_train_local_windows_v1":
        raise ValueError("server temporal work requires the frozen 16-window bank")
    if manifest["labels_used"] or manifest["windows"] != 16:
        raise ValueError("bank is not the declared label-free 16-window representation")
    for entry in manifest["files"]:
        path = directory / entry["path"]
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError("frozen VideoMAE bank changed")
    completed = np.load(directory / "completed.npy", allow_pickle=False)
    features = np.load(directory / "features.npy", mmap_mode="r", allow_pickle=False)
    with np.load(directory / "metadata.npz", allow_pickle=False) as archive:
        meta = {key: archive[key] for key in archive.files}
    units = meta["units"].astype(str)
    modalities = meta["modalities"].astype(str)
    available = meta["available"]
    alignment = meta["alignment_ok"]
    if (features.shape != (len(units), 16, CHANNELS) or not completed.all()
            or available.shape != (len(units),) or alignment.shape != (len(units),)
            or meta["frame_indices"].shape != (len(units), 16, 16)
            or not np.isfinite(features).all()):
        raise ValueError("malformed 16-window bank")
    lookup = {(unit, modality): i for i, (unit, modality) in enumerate(zip(units, modalities, strict=True))}
    if len(lookup) != len(units) or set(modalities) != set(MODALITIES):
        raise ValueError("duplicated or unknown modality keys")
    ordered = sorted(set(units))
    values = np.zeros((len(ordered), 16, VIEWS, CHANNELS), dtype=np.float32)
    mask = np.zeros((len(ordered), 16, VIEWS), dtype=bool)
    aligned = np.zeros((len(ordered), VIEWS), dtype=bool)
    for i, unit in enumerate(ordered):
        for view, modality in enumerate(MODALITIES):
            row = lookup[(unit, modality)]
            values[i, :, view] = features[row]
            mask[i, :, view] = bool(available[row])
            aligned[i, view] = bool(alignment[row])
    if not mask.any(axis=(1, 2)).all():
        raise ValueError("every recording needs one available view")
    matrix = np.concatenate((values.reshape(len(ordered), -1), mask.reshape(len(ordered), -1)), axis=1)
    names = np.asarray([f"videomae_w{w}_{view}_{c:03d}" for w in range(16)
                        for view in ("depthcolor", "ir") for c in range(CHANNELS)]
                       + [f"available_w{w}_{view}" for w in range(16) for view in ("depthcolor", "ir")])
    digests = [_row_digest(row) for row in matrix]
    if len(set(digests)) != len(digests):
        raise ValueError("window matrix rows are not uniquely keyed")
    return {"X": matrix, "units": np.asarray(ordered), "feature_names": names,
            "index": {unit: i for i, unit in enumerate(ordered)},
            "row_digest_to_unit": dict(zip(digests, ordered, strict=True)),
            "alignment_ok": aligned, "availability": mask,
            "source_manifest_sha256": sha256_file(directory / "manifest.json")}


def unpack16(matrix):
    values = np.asarray(matrix, dtype=np.float32)
    dimension = 16 * VIEWS * (CHANNELS + 1)
    if values.ndim != 2 or values.shape[1] != dimension or not np.isfinite(values).all():
        raise ValueError("server temporal head requires finite 16-window matrices")
    features = values[:, :16 * VIEWS * CHANNELS].reshape(-1, 16, VIEWS, CHANNELS)
    raw = values[:, 16 * VIEWS * CHANNELS:]
    if not np.isin(raw, [0, 1]).all():
        raise ValueError("availability masks must be binary")
    mask = raw.reshape(-1, 16, VIEWS).astype(bool)
    if not mask.any(axis=(1, 2)).all():
        raise ValueError("each recording requires an available view")
    return features, mask


def pool_per_view(logits, available, temperature):
    import torch
    if temperature <= 0 or logits.shape[:3] != available.shape:
        raise ValueError("invalid separate-view pooling axes")
    scaled = (logits / temperature).masked_fill(~available[..., None], -torch.inf)
    counts = available.sum(dim=1)
    pooled = temperature * (torch.logsumexp(scaled, dim=1) - counts.clamp_min(1).log()[..., None])
    present = counts > 0
    pooled = torch.where(present[..., None], pooled, torch.zeros_like(pooled))
    return pooled, present


def fuse_views(view_logits, present):
    if view_logits.ndim != 3 or present.shape != view_logits.shape[:2]:
        raise ValueError("invalid view fusion axes")
    return (view_logits * present[..., None]).sum(dim=1) / present.sum(dim=1).clamp_min(1)[..., None]


def partial_loss(pooled, targets, weights):
    from torch.nn import functional
    known = targets != -1
    if targets.shape != pooled.shape or not known.any(dim=1).all():
        raise ValueError("partial action loss requires known labels")
    cells = functional.binary_cross_entropy_with_logits(
        pooled, targets.clamp_min(0), pos_weight=weights, reduction="none")
    return ((cells * known).sum(dim=1) / known.sum(dim=1)).mean()


class ServerTemporalJointClassifier(WindowPartialActionClassifier):
    """A1 separates views; B1 adds same-view physical adjacent deltas."""
    def __init__(self, config, seed, auxiliary, visual, centers, arm):
        super().__init__(config, seed)
        if arm not in {"a1", "b1"}:
            raise ValueError("arm must be a1 or b1")
        self.auxiliary = auxiliary
        self.arm = arm
        self.row_digest_to_unit = dict(visual["row_digest_to_unit"])
        self.centers = {(str(unit), str(view)): np.asarray(axis, dtype=np.float32)
                        for (unit, view), axis in centers.items()}
        self.windows = 16

    def _units(self, matrix):
        result = []
        for row in np.asarray(matrix, dtype=np.float32):
            key = _row_digest(row)
            if key not in self.row_digest_to_unit:
                raise ValueError("matrix row is outside the frozen bank")
            result.append(self.row_digest_to_unit[key])
        return result

    def _times(self, units, mask):
        result = np.full((len(units), 16, VIEWS), np.nan, dtype=np.float32)
        for i, unit in enumerate(units):
            for view, modality in enumerate(MODALITIES):
                if not mask[i, :, view].any():
                    continue
                key = (str(unit), modality)
                if key not in self.centers:
                    raise ValueError(f"missing physical centers: {key}")
                axis = self.centers[key]
                if axis.shape != (16,) or not np.isfinite(axis).all() or (np.diff(axis) < 0).any():
                    raise ValueError("invalid physical time axis")
                result[i, :, view] = axis
        return result

    def _residual_view(self, hidden, available, times, residual):
        import torch
        difference = hidden[:, 1:] - hidden[:, :-1]
        dt = times[:, 1:] - times[:, :-1]
        valid = available[:, 1:] & available[:, :-1] & torch.isfinite(dt) & (dt > 1e-6)
        safe_dt = torch.where(valid, dt, torch.ones_like(dt))
        delta = difference / safe_dt[..., None]
        local = residual(delta)
        local = torch.where(valid[..., None], local, torch.zeros_like(local))
        counts = valid.sum(dim=1)
        pooled = local.sum(dim=1) / counts.clamp_min(1)[..., None]
        return torch.where((counts > 0)[..., None], pooled, torch.zeros_like(pooled)), valid

    def _forward_pooled(self, base, residual, data, visible, times=None):
        import torch
        hidden = torch.nn.functional.gelu(base[0](data))
        logits = base[3](base[2](hidden))
        views, present = pool_per_view(logits, visible, self.config["temperature"])
        pair_mask = None
        if self.arm == "b1":
            extra, pair_mask = self._residual_view(hidden, visible, times, residual)
            views = views + extra
        return fuse_views(views, present), hidden, logits, pair_mask

    def fit(self, matrix, targets):
        import torch
        features, mask = unpack16(matrix)
        if len(features) != len(targets):
            raise ValueError("window descriptors and targets differ")
        units = self._units(matrix)
        auxiliary, auxiliary_mask, auxiliary_labels, self.auxiliary_scope = self.auxiliary
        del self.auxiliary
        weights, self.supported = positive_weights(targets)
        self.fitting_content_sha256 = array_digest(np.asarray(matrix), targets)
        self.training_targets = np.asarray(targets).copy()
        self.n_features_in_, self.n_outputs_ = matrix.shape[1], 40
        self.mean = np.zeros((VIEWS, CHANNELS), dtype=np.float32)
        self.scale = np.ones_like(self.mean)
        for view in range(VIEWS):
            visible = features[:, :, view][mask[:, :, view]]
            if len(visible):
                self.mean[view] = visible.mean(axis=0)
                self.scale[view] = np.maximum(visible.std(axis=0), self.config["std_floor"])
        device = self.config["device"]
        torch.manual_seed(self.seed)
        base = torch.nn.Sequential(torch.nn.Linear(CHANNELS, self.config["hidden"]), torch.nn.GELU(),
                                   torch.nn.Dropout(self.config["dropout"]),
                                   torch.nn.Linear(self.config["hidden"], 40)).to(device)
        residual = None
        if self.arm == "b1":
            residual = torch.nn.Sequential(torch.nn.Linear(self.config["hidden"], 64), torch.nn.GELU(),
                                           torch.nn.Linear(64, 40)).to(device)
            torch.nn.init.zeros_(residual[2].weight)
            torch.nn.init.zeros_(residual[2].bias)
        parameters = list(base.parameters()) + ([] if residual is None else list(residual.parameters()))
        optimizer = torch.optim.AdamW(parameters, lr=self.config["lr"], weight_decay=self.config["weight_decay"])
        steps = self.config["epochs"] * ((len(features) + self.config["batch_size"] - 1) // self.config["batch_size"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        # Reset stochastic training state after construction so A1/B1 base dropout streams match.
        torch.manual_seed(self.seed + 424242)
        random = np.random.default_rng(self.seed)
        auxiliary_random = np.random.default_rng(self.seed + 9973)
        data = torch.from_numpy(self._standardized(features, mask)).to(device)
        visible = torch.from_numpy(mask).to(device)
        times = None if self.arm == "a1" else torch.from_numpy(self._times(units, mask)).to(device)
        labels = torch.as_tensor(targets, device=device, dtype=torch.float32)
        positive_weight = torch.from_numpy(weights).to(device)
        aux_data = torch.from_numpy(self._standardized(auxiliary, auxiliary_mask)).to(device)
        aux_mask = torch.from_numpy(auxiliary_mask[:, 0]).to(device)
        aux_labels = torch.from_numpy(auxiliary_labels).to(device)
        counts = np.bincount(auxiliary_labels, minlength=40)
        sampling = 1 / counts[auxiliary_labels].astype(np.float64)
        sampling /= sampling.sum()
        started, self.history = time.perf_counter(), []
        total_valid_pairs = 0
        for epoch in range(self.config["epochs"]):
            main_total = aux_total = 0.0
            seen = epoch_pairs = 0
            order = random.permutation(len(features))
            for start in range(0, len(order), self.config["batch_size"]):
                rows = torch.as_tensor(order[start:start + self.config["batch_size"]], device=device)
                extra = torch.as_tensor(auxiliary_random.choice(len(auxiliary), size=len(rows), replace=True, p=sampling), device=device)
                optimizer.zero_grad(set_to_none=True)
                pooled, _, _, pair_mask = self._forward_pooled(
                    base, residual, data[rows], visible[rows], None if times is None else times[rows])
                main_loss = partial_loss(pooled, labels[rows], positive_weight)
                # HARn remains a whole-recording base-head auxiliary; no synthetic temporal residual path.
                aux_hidden = torch.nn.functional.gelu(base[0](aux_data[extra]))
                aux_logits = base[3](base[2](aux_hidden))[:, 0]
                aux_pooled = (aux_logits * aux_mask[extra, :, None]).sum(dim=1) / aux_mask[extra].sum(dim=1)[:, None]
                auxiliary_loss = torch.nn.functional.cross_entropy(aux_pooled, aux_labels[extra])
                loss = main_loss + self.config["auxiliary_weight"] * auxiliary_loss
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite server temporal training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, self.config["gradient_clip"])
                optimizer.step()
                scheduler.step()
                main_total += float(main_loss.detach()) * len(rows)
                aux_total += float(auxiliary_loss.detach()) * len(rows)
                seen += len(rows)
                if pair_mask is not None:
                    epoch_pairs += int(pair_mask.sum().detach().cpu())
            total_valid_pairs = epoch_pairs
            self.history.append({"epoch": epoch + 1, "mean_training_loss": main_total / seen,
                                 "auxiliary_ce": aux_total / seen, "valid_adjacent_pairs": epoch_pairs})
        self.state = {name: value.detach().cpu().numpy().copy() for name, value in base.state_dict().items()}
        if residual is not None:
            self.state.update({"residual." + name: value.detach().cpu().numpy().copy()
                               for name, value in residual.state_dict().items()})
        if any(not np.isfinite(value).all() for value in self.state.values()):
            raise ValueError("nonfinite saved server temporal head")
        self.positive_weight = weights
        self.fit_seconds = time.perf_counter() - started
        self.valid_adjacent_pairs_last_epoch = total_valid_pairs
        return self

    def _inference_parts(self, matrix):
        import torch
        features, mask = unpack16(matrix)
        units = self._units(matrix)
        data = torch.from_numpy(self._standardized(features, mask))
        visible = torch.from_numpy(mask)
        with torch.inference_mode():
            hidden = torch.nn.functional.linear(data, torch.from_numpy(self.state["0.weight"]),
                                                torch.from_numpy(self.state["0.bias"]))
            hidden = torch.nn.functional.gelu(hidden)
            logits = torch.nn.functional.linear(hidden, torch.from_numpy(self.state["3.weight"]),
                                                torch.from_numpy(self.state["3.bias"]))
            times = None
            residual_pair = valid = None
            if self.arm == "b1":
                times = torch.from_numpy(self._times(units, mask))
                difference = hidden[:, 1:] - hidden[:, :-1]
                dt = times[:, 1:] - times[:, :-1]
                valid = visible[:, 1:] & visible[:, :-1] & torch.isfinite(dt) & (dt > 1e-6)
                safe = torch.where(valid, dt, torch.ones_like(dt))
                z = torch.nn.functional.linear(difference / safe[..., None],
                                               torch.from_numpy(self.state["residual.0.weight"]),
                                               torch.from_numpy(self.state["residual.0.bias"]))
                z = torch.nn.functional.gelu(z)
                residual_pair = torch.nn.functional.linear(z, torch.from_numpy(self.state["residual.2.weight"]),
                                                           torch.from_numpy(self.state["residual.2.bias"]))
                residual_pair = torch.where(valid[..., None], residual_pair, torch.zeros_like(residual_pair))
        return units, visible, hidden, logits, times, residual_pair, valid

    def predict(self, matrix):
        import torch
        _, visible, _, logits, _, residual_pair, valid = self._inference_parts(matrix)
        with torch.inference_mode():
            views, present = pool_per_view(logits, visible, self.config["temperature"])
            if self.arm == "b1":
                counts = valid.sum(dim=1)
                extra = residual_pair.sum(dim=1) / counts.clamp_min(1)[..., None]
                views = views + torch.where((counts > 0)[..., None], extra, torch.zeros_like(extra))
            pooled = fuse_views(views, present)
            probability = (pooled - torch.from_numpy(np.log(self.positive_weight))).sigmoid().numpy()
        probability[:, ~self.supported] = .5
        return probability

    def sequence_window_probabilities(self, matrix):
        """Fixed preregistered endpoint projection used only by the downstream 308-sequence decoder."""
        from scipy.special import expit
        units, visible, _, logits, _, residual_pair, valid = self._inference_parts(matrix)
        values = logits.numpy().astype(np.float64)
        if self.arm == "b1":
            pair = residual_pair.numpy().astype(np.float64)
            pair_valid = valid.numpy()
            addition = np.zeros_like(values)
            incident = np.zeros(values.shape[:3], dtype=np.float64)
            for t in range(15):
                active = pair_valid[:, t]
                contribution = pair[:, t] * active[..., None]
                addition[:, t] += contribution
                addition[:, t + 1] += contribution
                incident[:, t] += active
                incident[:, t + 1] += active
            values += addition / np.maximum(incident[..., None], 1.0)
        values -= np.log(self.positive_weight)[None, None, None, :]
        probability = expit(values)
        probability[..., ~self.supported] = .5
        return units, probability, visible.numpy()

    def predict_prior(self, matrix):
        targets = self.training_targets
        prior = ((targets == 1).sum(axis=0) + 1) / ((targets != -1).sum(axis=0) + 2)
        prior[~self.supported] = .5
        return np.repeat(prior[None], len(matrix), axis=0)

    def temporal_summary(self, matrix):
        units, visible, _, _, _, _, valid = self._inference_parts(matrix)
        return {"arm": self.arm, "units": units,
                "view_window_counts": visible.sum(dim=1).numpy().tolist(),
                "valid_adjacent_pair_counts": None if valid is None else valid.sum(dim=1).numpy().tolist()}
