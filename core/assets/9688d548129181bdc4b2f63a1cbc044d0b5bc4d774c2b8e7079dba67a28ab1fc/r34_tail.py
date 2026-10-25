"""Train the last generic R34 block in explicit HARn subject scopes.

Missing modalities never become zero-valued training examples. The cache keeps
its flat physical row axis; paired batches gather views by explicit metadata.
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from cuhkx.harn import action_class, action_text_classes, object_text_classes
from cuhkx.harn_nested import FeatureBank, fit_prior, training_manifest
from cuhkx.harn_tail import configure_torch, training_rows

KIND = "cuhkx-harn-r34-last-block-nested-v1"
MID_SHAPE = (512, 4, 7, 7)
WEIGHTS_SHA = "ade133f18548b59da3a36d6a70d44a4a7cc4b90460654c338cb8a023e9a3e0e1"
MODALITIES = ("Depth_Color", "IR")


@dataclass
class ViewBatch:
    values: np.ndarray
    available: np.ndarray

    def __post_init__(self):
        if self.values.shape != (len(self.available), 2, *MID_SHAPE):
            raise ValueError("R34 paired feature axes differ")
        if self.available.shape != (len(self.values), 2) or self.available.dtype != bool:
            raise ValueError("R34 view mask axes or dtype differ")
        if len(self.values) and not self.available.any(axis=1).all():
            raise ValueError("R34 visual family requires at least one available view per unit")

    def __len__(self):
        return len(self.values)

    @property
    def shape(self):
        return len(self), 2 * int(np.prod(MID_SHAPE))

    @property
    def ndim(self):
        return 2

    def __getitem__(self, indices):
        if np.isscalar(indices):
            indices = [indices]
        return ViewBatch(self.values[indices], self.available[indices])


class PairedRows:
    """Index-only adapter; never makes a second whole-cache feature copy."""

    def __init__(self, features, indices, available):
        self.features = features
        self.indices = np.asarray(indices, dtype=np.int64)
        self.available = np.asarray(available, dtype=bool)
        if self.indices.shape != self.available.shape or self.indices.shape[1:] != (2,):
            raise ValueError("R34 paired cache row axes differ")
        if not self.available.any(axis=1).all() or np.any(self.indices[self.available] < 0):
            raise ValueError("R34 paired cache contains unavailable visual units")
        self.shape = (len(self.indices), 2 * int(np.prod(MID_SHAPE)))
        self.dtype = np.dtype(np.float16)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, rows):
        if np.isscalar(rows):
            rows = [rows]
        selected, available = self.indices[rows], self.available[rows]
        values = np.zeros((len(selected), 2, *MID_SHAPE), dtype=np.float16)
        values[available] = self.features[selected[available]]
        return ViewBatch(values, available.copy())


class R34PairedBank(FeatureBank):
    @classmethod
    def load(cls, skeleton_path, visual_path):
        from cuhkx.r34_video import R34FeatureBank

        raw = R34FeatureBank.load(visual_path)
        metadata = raw.metadata
        whole = (metadata["sources"] == "HARn") & (metadata["regimes"] == "whole")
        lookup = {}
        for index in np.flatnonzero(whole):
            key = (str(metadata["units"][index]), str(metadata["modalities"][index]),
                   int(metadata["window_indices"][index]))
            if key in lookup or key[1] not in MODALITIES or key[2] != 0:
                raise ValueError("R34 whole-clip unit/modality/window keys differ")
            lookup[key] = int(index)
        units, indices, availability = [], [], []
        for unit in sorted({key[0] for key in lookup}):
            rows = [lookup.get((unit, modality, 0), -1) for modality in MODALITIES]
            present = [index >= 0 and bool(metadata["available"][index]) for index in rows]
            if any(present):
                units.append(unit)
                indices.append(rows)
                availability.append(present)
        with np.load(skeleton_path, allow_pickle=False) as archive:
            skeleton_units = archive["units"].astype(str)
            skeleton = archive["X"].astype(np.float32)
            names = archive["feature_names"].astype(str)
        if (skeleton.shape != (len(skeleton_units), len(names))
                or len(set(skeleton_units)) != len(skeleton_units)
                or not np.isfinite(skeleton).all()):
            raise ValueError("Skeleton feature axes or values differ")
        visual = PairedRows(raw.features, indices, availability)
        bank = cls(skeleton_units, skeleton, names, np.asarray(units, dtype=str),
                   visual, raw.manifest["spec"])
        bank.raw_bank = raw
        return bank


def row_digest(matrix, rows, labels):
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": [len(rows), 2, *MID_SHAPE], "dtype": "float16",
                              "mask_dtype": "bool", "labels_dtype": labels.dtype.str},
                             sort_keys=True).encode())
    for start in range(0, len(rows), 16):
        batch = matrix[rows[start:start + 16]]
        digest.update(np.ascontiguousarray(batch.values).tobytes())
        digest.update(np.ascontiguousarray(batch.available).tobytes())
    digest.update(np.ascontiguousarray(labels).tobytes())
    return digest.hexdigest()


def tail_network(weights_path=None):
    import torch

    from cuhkx.r34_video import build_r34, pretrained_model

    original = pretrained_model(weights_path) if weights_path else build_r34()
    return torch.nn.Sequential(original.layer4[2], torch.nn.AdaptiveAvgPool3d((1, 1, 1)),
                               torch.nn.Flatten(1), torch.nn.Linear(512, 44))


def reduce_views(values, owners, units):
    """Average available views independently within each unit."""
    result = np.zeros((units, values.shape[1]), dtype=values.dtype)
    np.add.at(result, owners, values)
    counts = np.bincount(owners, minlength=units)
    if np.any(counts == 0):
        raise ValueError("cannot average a unit with no available view")
    return result / counts[:, None]


class R34Classifier:
    def __init__(self, settings, seed):
        self.settings = settings
        self.seed = seed

    def fit(self, bank, rows, labels):
        torch = configure_torch()
        if len(rows) != len(labels) or len(set(rows)) != len(rows):
            raise ValueError("visual fitting rows and labels must be unique and aligned")
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        generator = np.random.default_rng(self.seed)
        network = tail_network(self.settings["initial_weights"]).cuda()
        optimizer = torch.optim.AdamW([
            {"params": network[0].parameters(), "lr": self.settings["last_block_lr"]},
            {"params": network[3].parameters(), "lr": self.settings["classifier_lr"]},
        ], weight_decay=self.settings["weight_decay"])
        counts = np.bincount(labels, minlength=44)
        self.classes_ = np.flatnonzero(counts).astype(np.int64)
        weights = np.zeros(44, dtype=np.float32)
        weights[self.classes_] = len(labels) / (len(self.classes_) * counts[self.classes_])
        loss_function = torch.nn.CrossEntropyLoss(
            weight=torch.as_tensor(weights, device="cuda"), reduction="none")
        scaler = torch.amp.GradScaler("cuda")
        epochs, batch_size = self.settings["epochs"], self.settings["batch_units"]
        steps = epochs * ((len(rows) + batch_size - 1) // batch_size)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        self.history, self.training_rows, self.training_labels = [], rows.copy(), labels.copy()
        started = time.monotonic()
        for epoch in range(epochs):
            network.train()
            order = generator.permutation(len(rows))
            total_loss, seen, views_seen, skipped = 0.0, 0, 0, 0
            for start in range(0, len(order), batch_size):
                selected = order[start:start + batch_size]
                batch = bank.visual[rows[selected]]
                owners = np.nonzero(batch.available)[0]
                values = batch.values[batch.available]
                tensor = torch.from_numpy(values.copy()).to("cuda", dtype=torch.float32)
                truth = torch.as_tensor(labels[selected][owners], device="cuda", dtype=torch.long)
                # Weight each clip equally even when only its IR view is present.
                view_weights = torch.as_tensor(
                    1.0 / batch.available.sum(axis=1)[owners], device="cuda", dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda"):
                    logits = network(tensor)
                    loss = (loss_function(logits, truth) * view_weights).sum() / len(selected)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite R34 adaptation loss")
                previous_scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(network.parameters(), self.settings["gradient_clip"])
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    schedule.step()
                else:
                    skipped += 1
                total_loss += float(loss.detach()) * len(selected)
                seen += len(selected)
                views_seen += len(owners)
            self.history.append({"epoch": epoch + 1, "mean_training_loss": total_loss / seen,
                                 "training_units": seen, "training_views": views_seen,
                                 "skipped_nonfinite_gradient_steps": skipped})
            print(f"r34_epoch={epoch+1}/{epochs} units={seen} views={views_seen} "
                  f"loss={total_loss/seen:.5f} seconds={time.monotonic()-started:.1f}", flush=True)
        self.state = {key: value.detach().cpu().numpy().astype(np.float16)
                      if value.is_floating_point() else value.detach().cpu().numpy().copy()
                      for key, value in network.state_dict().items()}
        if any(not np.isfinite(value).all() for value in self.state.values()):
            raise ValueError("nonfinite retained R34 weights")
        self.fit_seconds = time.monotonic() - started
        self.n_features_in_ = bank.visual.shape[1]
        self.weights_pretraining_sha256 = WEIGHTS_SHA
        del network, optimizer, scaler, schedule, loss, logits, tensor
        gc.collect()
        torch.cuda.empty_cache()
        return self

    def predict_proba(self, matrix):
        torch = configure_torch()
        if not isinstance(matrix, ViewBatch) or matrix.shape[1] != self.n_features_in_:
            raise ValueError("R34 classifier requires paired features with explicit availability")
        network = tail_network()
        network.load_state_dict({key: torch.from_numpy(value) for key, value in self.state.items()})
        network.eval().cuda()
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(matrix), self.settings["batch_units"]):
                batch = matrix[start:start + self.settings["batch_units"]]
                owners = np.nonzero(batch.available)[0]
                tensor = torch.from_numpy(batch.values[batch.available].copy()).to("cuda", dtype=torch.float32)
                with torch.autocast(device_type="cuda"):
                    logits = network(tensor)
                probabilities = logits.float()[:, self.classes_].softmax(dim=1).cpu().numpy()
                outputs.append(reduce_views(probabilities, owners, len(batch)))
        del network
        torch.cuda.empty_cache()
        return np.concatenate(outputs) if outputs else np.empty((0, len(self.classes_)), dtype=np.float32)


def fit_bundle(bank, qa, subjects, config):
    training_manifest(bank, qa, subjects, config)
    skeleton_rows = bank.training_rows("skeleton", subjects)
    labels = np.asarray([action_class(unit) for unit in bank.skeleton_units[skeleton_rows]])
    aggregate = ~np.char.startswith(bank.skeleton_names, "resampled_pose_")
    masks = {"extra_full": np.ones(len(aggregate), dtype=bool), "extra_aggregate": aggregate}
    models = {}
    for offset, (name, mask) in enumerate(masks.items()):
        estimator = ExtraTreesClassifier(
            n_estimators=config["skeleton"]["trees"], max_features="sqrt", min_samples_leaf=1,
            class_weight="balanced", n_jobs=config["workers"], random_state=config["seed"] + offset * 1000)
        estimator.fit(bank.skeleton[skeleton_rows][:, mask], labels)
        models[name] = estimator
    rows, visual_labels = training_rows(bank, subjects)
    visual = R34Classifier(config["visual"]["adaptation"], config["seed"] + 2000).fit(bank, rows, visual_labels)
    return {"kind": KIND, "status": "diagnostic_only", "config": config,
            "train_subjects": sorted(subjects), "models": models, "masks": masks,
            "visual_model": visual, "skeleton_names": bank.skeleton_names, "visual_spec": bank.visual_spec,
            "skeleton_classes": np.unique(labels), "visual_classes": visual.classes_,
            "action_mapping": action_text_classes(qa), "object_mapping": object_text_classes(qa),
            "prior": fit_prior(qa)}
