"""Joint HARn primary-action and HAU partial-presence supervision of one window head."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from cuhkx.action_mil import masked_loss, positive_weights
from cuhkx.action_window_head import CHANNELS, VIEWS, WindowPartialActionClassifier, unpack
from cuhkx.harn import action_class
from cuhkx.metadata import subject_id
from cuhkx.sequence_skeleton_nested import array_digest
from cuhkx.validation import sha256_file


class HarnWindowBank:
    def __init__(self, directory):
        directory = Path(directory)
        self.manifest_path = str(directory / "manifest.json")
        self.manifest_sha256 = sha256_file(Path(self.manifest_path))
        manifest = json.loads(Path(self.manifest_path).read_text())
        if manifest["kind"] != "cuhkx_videomae_ssv2_harn_train_whole_v1" or manifest["windows"] != 1 or manifest["labels_used"]:
            raise ValueError("joint supervision requires the fixed generic HARn whole-video bank")
        for entry in manifest["files"]:
            path = directory / entry["path"]
            if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
                raise ValueError("HARn bank changed")
        with np.load(directory / "metadata.npz", allow_pickle=False) as archive:
            metadata = {key: archive[key] for key in archive.files}
        raw = np.load(directory / "features.npy", allow_pickle=False)
        if raw.shape != (len(metadata["units"]), 1, 768) or not np.load(directory / "completed.npy", allow_pickle=False).all():
            raise ValueError("incomplete HARn descriptors")
        keys = list(zip(metadata["units"].astype(str), metadata["modalities"].astype(str), strict=True))
        lookup = {key: i for i, key in enumerate(keys)}
        if len(lookup) != len(keys):
            raise ValueError("duplicated HARn media keys")
        self.units = np.asarray(sorted({key[0] for key in keys}))
        if any(not unit.startswith("Training/HARn/") for unit in self.units):
            raise ValueError("HARn auxiliary fitting bank must contain official Training units only")
        self.features = np.zeros((len(self.units), 1, 2, 768), dtype=np.float32)
        self.mask = np.zeros((len(self.units), 1, 2), dtype=bool)
        for index, unit in enumerate(self.units):
            rows = [lookup[(unit, modality)] for modality in ("Depth_Color", "IR")]
            self.features[index, 0] = raw[rows, 0]
            self.mask[index, 0] = metadata["available"][rows]
            if self.mask[index].all() and not metadata["alignment_ok"][rows].all():
                self.mask[index, 0, 0] = False
        self.subjects = np.asarray([subject_id(unit) for unit in self.units])

    def scope(self, train, valid):
        train, valid = set(train), set(valid)
        if train & valid:
            raise ValueError("HARn auxiliary train/validation people overlap")
        rows = np.flatnonzero(np.isin(self.subjects, sorted(train)) & self.mask.any(axis=(1, 2)))
        labels = np.asarray([action_class(self.units[row]) for row in rows], dtype=np.int64)
        rows, labels = rows[labels < 40], labels[labels < 40]
        if set(self.subjects[rows]) != train or (labels < 0).any():
            raise ValueError("actual HARn primary-action scope does not match fitting people")
        features, mask = self.features[rows], self.mask[rows]
        record = {"training_subjects": sorted(train), "validation_subjects": sorted(valid),
                  "units": self.units[rows].tolist(), "source_rows": rows.tolist(), "labels": labels.tolist(),
                  "actual_features_masks_labels_sha256": array_digest(features, mask, labels),
                  "bank_manifest": self.manifest_path, "bank_manifest_sha256": self.manifest_sha256,
                  "label_semantics": "Official HARn primary class0..39; auxiliary cross-entropy does not assert all other HAU actions are absent."}
        return features, mask, labels, record


class JointHarnWindowClassifier(WindowPartialActionClassifier):
    def __init__(self, config, seed, auxiliary):
        super().__init__(config, seed)
        self.auxiliary = auxiliary

    def fit(self, matrix, targets):
        import torch

        features, mask = unpack(matrix)
        auxiliary, auxiliary_mask, auxiliary_labels, self.auxiliary_scope = self.auxiliary
        del self.auxiliary  # Retain scope/digests, not another copy of the raw feature bank in every model.
        weights, self.supported = positive_weights(targets)
        self.fitting_content_sha256 = array_digest(np.asarray(matrix), targets)
        self.training_targets = np.asarray(targets).copy()
        self.n_features_in_, self.n_outputs_ = matrix.shape[1], 40
        self.mean = np.zeros((VIEWS, CHANNELS), dtype=np.float32)
        self.scale = np.ones_like(self.mean)
        # Preserve the HAU-only normalization control; auxiliary labels do not alter statistics.
        for view in range(VIEWS):
            visible = features[:, :, view][mask[:, :, view]]
            if len(visible):
                self.mean[view] = visible.mean(axis=0)
                self.scale[view] = np.maximum(visible.std(axis=0), self.config["std_floor"])
        device = self.config["device"]
        torch.manual_seed(self.seed)
        random = np.random.default_rng(self.seed)
        auxiliary_random = np.random.default_rng(self.seed + 9973)
        network = torch.nn.Linear(CHANNELS, 40).to(device)
        optimizer = torch.optim.AdamW(network.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"])
        steps = self.config["epochs"] * ((len(features) + self.config["batch_size"] - 1) // self.config["batch_size"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        data = torch.from_numpy(self._standardized(features, mask)).to(device)
        visible = torch.from_numpy(mask).to(device)
        labels = torch.as_tensor(targets, device=device, dtype=torch.float32)
        positive_weight = torch.from_numpy(weights).to(device)
        aux_data = torch.from_numpy(self._standardized(auxiliary, auxiliary_mask)).to(device)
        aux_mask = torch.from_numpy(auxiliary_mask[:, 0]).to(device)
        aux_labels = torch.from_numpy(auxiliary_labels).to(device)
        counts = np.bincount(auxiliary_labels, minlength=40)
        sampling = 1 / counts[auxiliary_labels].astype(np.float64)
        sampling /= sampling.sum()
        started, self.history = time.perf_counter(), []
        for epoch in range(self.config["epochs"]):
            total, auxiliary_total, seen = 0., 0., 0
            order = random.permutation(len(features))
            for start in range(0, len(order), self.config["batch_size"]):
                rows = torch.as_tensor(order[start:start + self.config["batch_size"]], device=device)
                extra = torch.as_tensor(auxiliary_random.choice(len(auxiliary), size=len(rows), replace=True, p=sampling), device=device)
                optimizer.zero_grad(set_to_none=True)
                main_loss = masked_loss(network(data[rows]), visible[rows], labels[rows], positive_weight,
                                        self.config["temperature"])
                aux_logits = network(aux_data[extra])[:, 0]
                pooled = (aux_logits * aux_mask[extra, :, None]).sum(dim=1) / aux_mask[extra].sum(dim=1)[:, None]
                aux_loss = torch.nn.functional.cross_entropy(pooled, aux_labels[extra])
                loss = main_loss + self.config["auxiliary_weight"] * aux_loss
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite joint HARn/HAU training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), self.config["gradient_clip"])
                optimizer.step()
                scheduler.step()
                total += float(main_loss.detach()) * len(rows)
                auxiliary_total += float(aux_loss.detach()) * len(rows)
                seen += len(rows)
            self.history.append({"epoch": epoch + 1, "mean_training_loss": total / seen,
                                 "auxiliary_ce": auxiliary_total / seen, "auxiliary_samples": seen})
        self.state = {name: value.detach().cpu().numpy().copy() for name, value in network.state_dict().items()}
        if any(not np.isfinite(value).all() for value in self.state.values()):
            raise ValueError("nonfinite joint window head")
        self.positive_weight = weights
        self.fit_seconds = time.perf_counter() - started
        return self
