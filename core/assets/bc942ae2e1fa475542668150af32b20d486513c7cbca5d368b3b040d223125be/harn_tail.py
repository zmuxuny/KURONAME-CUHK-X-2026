"""Subject-scoped adaptation of the last generic video residual stage."""

from __future__ import annotations

import gc
import os
import time

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from cuhkx.harn import action_class, action_text_classes, object_text_classes
from cuhkx.harn_nested import fit_prior, training_manifest
from cuhkx.video_midfeatures import MID_SHAPE, WEIGHTS_SHA, generic_model

KIND = "cuhkx-harn-tail-nested-v1"


def configure_torch():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("this fixed adaptation and replay protocol requires CUDA")
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return torch


def tail_network(weights_path=None):
    import torch
    from torchvision.models.video import r2plus1d_18

    original = generic_model(weights_path) if weights_path else r2plus1d_18(weights=None)
    network = torch.nn.Sequential(original.layer4, torch.nn.AdaptiveAvgPool3d((1, 1, 1)),
                                  torch.nn.Flatten(1), torch.nn.Linear(512, 44))
    return network


def training_rows(bank, subjects):
    rows = bank.training_rows("visual", set(subjects))
    units = bank.visual_units[rows]
    from cuhkx.metadata import subject_id

    if {subject_id(unit) for unit in units} != set(subjects):
        raise ValueError("middle-feature training rows must cover exactly the fitted subjects")
    labels = np.asarray([action_class(unit) for unit in units], dtype=np.int64)
    if not set(labels) <= set(range(44)):
        raise ValueError("unknown official HARn action identity")
    return rows, labels


class TailClassifier:
    def __init__(self, settings, seed):
        self.settings = settings
        self.seed = seed

    def fit(self, bank, rows, labels):
        torch = configure_torch()
        if len(rows) != len(labels) or len(set(rows)) != len(rows):
            raise ValueError("visual training rows/labels must be unique and aligned")
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        generator = np.random.default_rng(self.seed)
        network = tail_network(self.settings["initial_weights"]).cuda()
        # The frozen stem/layer1..3 have no fit path here. BatchNorm updates in
        # layer4 and all learned parameters see only this fitted subject scope.
        optimizer = torch.optim.AdamW([
            {"params": network[0].parameters(), "lr": self.settings["layer4_lr"]},
            {"params": network[3].parameters(), "lr": self.settings["classifier_lr"]},
        ], weight_decay=self.settings["weight_decay"])
        counts = np.bincount(labels, minlength=44)
        self.classes_ = np.flatnonzero(counts).astype(np.int64)
        weights = np.zeros(44, dtype=np.float32)
        weights[self.classes_] = len(labels) / (len(self.classes_) * counts[self.classes_])
        loss_function = torch.nn.CrossEntropyLoss(weight=torch.as_tensor(weights, device="cuda"))
        scaler = torch.amp.GradScaler("cuda")
        epochs = self.settings["epochs"]
        batch_size = self.settings["batch_units"]
        steps = epochs * ((len(rows) + batch_size - 1) // batch_size)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        self.history = []
        self.training_rows = rows.copy()
        self.training_labels = labels.copy()
        started = time.monotonic()
        for epoch in range(epochs):
            network.train()
            order = generator.permutation(len(rows))
            total_loss, seen, skipped = 0.0, 0, 0
            for start in range(0, len(order), batch_size):
                selected = order[start:start + batch_size]
                values = np.asarray(bank.visual[rows[selected]]).reshape(-1, *MID_SHAPE)
                target = np.repeat(labels[selected], 2)
                tensor = torch.from_numpy(values.copy()).to("cuda", dtype=torch.float32)
                truth = torch.as_tensor(target, device="cuda", dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda"):
                    logits = network(tensor)
                    loss = loss_function(logits, truth)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite adapted visual loss")
                previous_scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(network.parameters(), self.settings["gradient_clip"])
                scaler.step(optimizer)
                scaler.update()
                did_step = scaler.get_scale() >= previous_scale
                if did_step:
                    schedule.step()
                else:
                    skipped += 1
                total_loss += float(loss.detach()) * len(target)
                seen += len(target)
            self.history.append({"epoch": epoch + 1, "mean_training_loss": total_loss / seen,
                                 "training_views": seen, "skipped_nonfinite_gradient_steps": skipped})
            print(f"tail_epoch={epoch+1}/{epochs} views={seen} loss={total_loss/seen:.5f} seconds={time.monotonic()-started:.1f}", flush=True)
        # Quantization is fixed before any validation prediction. Every fit and
        # reload predicts from these same retained arrays, not pre-quantized weights.
        self.state = {key: tensor.detach().cpu().numpy().astype(np.float16)
                      if tensor.is_floating_point() else tensor.detach().cpu().numpy().copy()
                      for key, tensor in network.state_dict().items()}
        if any(not np.isfinite(value).all() for value in self.state.values()):
            raise ValueError("nonfinite retained adapted weights")
        self.fit_seconds = time.monotonic() - started
        self.n_features_in_ = bank.visual.shape[1]
        self.weights_pretraining_sha256 = WEIGHTS_SHA
        del network, optimizer, scaler, schedule, loss, logits, tensor
        gc.collect()
        torch.cuda.empty_cache()
        return self

    def predict_proba(self, matrix):
        torch = configure_torch()
        if matrix.ndim != 2 or matrix.shape[1] != self.n_features_in_:
            raise ValueError("adapted classifier expects paired fixed middle features")
        network = tail_network()
        network.load_state_dict({key: torch.from_numpy(value) for key, value in self.state.items()})
        network.eval().cuda()
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(matrix), self.settings["batch_units"]):
                values = np.asarray(matrix[start:start + self.settings["batch_units"]]).reshape(-1, *MID_SHAPE)
                tensor = torch.from_numpy(values.copy()).to("cuda", dtype=torch.float32)
                with torch.autocast(device_type="cuda"):
                    logits = network(tensor)
                probabilities = logits.float()[:, self.classes_].softmax(dim=1)
                outputs.append(probabilities.reshape(-1, 2, len(self.classes_)).mean(dim=1).cpu().numpy())
        del network
        torch.cuda.empty_cache()
        return np.concatenate(outputs) if outputs else np.empty((0, len(self.classes_)), dtype=np.float32)


def fit_bundle(bank, qa, subjects, config):
    training_manifest(bank, qa, subjects, config)
    skeleton_rows = bank.training_rows("skeleton", subjects)
    skeleton_labels = np.asarray([action_class(unit) for unit in bank.skeleton_units[skeleton_rows]])
    aggregate = ~np.char.startswith(bank.skeleton_names, "resampled_pose_")
    masks = {"extra_full": np.ones(len(aggregate), dtype=bool), "extra_aggregate": aggregate}
    models = {}
    for offset, (name, mask) in enumerate(masks.items()):
        estimator = ExtraTreesClassifier(n_estimators=200, max_features="sqrt", min_samples_leaf=1,
                                         class_weight="balanced", n_jobs=3,
                                         random_state=config["seed"] + offset * 1000)
        estimator.fit(bank.skeleton[skeleton_rows][:, mask], skeleton_labels)
        models[name] = estimator
    rows, labels = training_rows(bank, subjects)
    visual = TailClassifier(config["visual"]["adaptation"], config["seed"] + 2000).fit(bank, rows, labels)
    return {"kind": KIND, "status": "diagnostic_only", "config": config,
            "train_subjects": sorted(subjects), "models": models, "masks": masks,
            "visual_model": visual, "skeleton_names": bank.skeleton_names, "visual_spec": bank.visual_spec,
            "skeleton_classes": np.unique(skeleton_labels), "visual_classes": visual.classes_,
            "action_mapping": action_text_classes(qa), "object_mapping": object_text_classes(qa),
            "prior": fit_prior(qa)}
