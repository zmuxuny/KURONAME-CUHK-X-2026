"""Subject-scoped partial-label MIL over retained VideoMAE window descriptors."""

from __future__ import annotations

import time

import numpy as np

from cuhkx.action_mil import masked_loss, pool_logits, positive_weights
from cuhkx.sequence_skeleton_nested import array_digest

WINDOWS, VIEWS, CHANNELS = 8, 2, 768
FEATURE_DIMENSION = WINDOWS * VIEWS * CHANNELS + WINDOWS * VIEWS


def unpack(matrix):
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != FEATURE_DIMENSION or not np.isfinite(values).all():
        raise ValueError("window classifier requires finite keyed descriptors and explicit view masks")
    features = values[:, :WINDOWS * VIEWS * CHANNELS].reshape(-1, WINDOWS, VIEWS, CHANNELS)
    raw_mask = values[:, WINDOWS * VIEWS * CHANNELS:]
    if not np.isin(raw_mask, [0, 1]).all():
        raise ValueError("window availability must be binary")
    mask = raw_mask.reshape(-1, WINDOWS, VIEWS).astype(bool)
    if not mask.any(axis=(1, 2)).all():
        raise ValueError("each training/prediction recording needs an available window")
    return features, mask


class WindowPartialActionClassifier:
    """Fit only known labels; pool windows after shared linear classification.

    Normalization is fitted separately per modality on visible training windows.
    The stored object contains NumPy parameters, avoiding a live CUDA dependency.
    """

    def __init__(self, config, seed):
        self.config, self.seed = config, seed

    def get_params(self):
        return {**self.config, "seed": self.seed}

    def _standardized(self, features, mask):
        return np.where(mask[..., None], (features - self.mean) / self.scale, 0).astype(np.float32)

    def fit(self, matrix, targets):
        import torch

        features, mask = unpack(matrix)
        if len(features) != len(targets):
            raise ValueError("window descriptors and partial-label targets differ")
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
        random = np.random.default_rng(self.seed)
        network = torch.nn.Linear(CHANNELS, 40).to(device)
        optimizer = torch.optim.AdamW(network.parameters(), lr=self.config["lr"],
                                      weight_decay=self.config["weight_decay"])
        steps = self.config["epochs"] * ((len(features) + self.config["batch_size"] - 1) // self.config["batch_size"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        data = torch.from_numpy(self._standardized(features, mask)).to(device)
        visible = torch.from_numpy(mask).to(device)
        labels = torch.as_tensor(targets, device=device, dtype=torch.float32)
        positive_weight = torch.from_numpy(weights).to(device)
        started, self.history = time.perf_counter(), []
        for epoch in range(self.config["epochs"]):
            total = 0.
            order = random.permutation(len(features))
            for start in range(0, len(order), self.config["batch_size"]):
                rows = torch.as_tensor(order[start:start + self.config["batch_size"]], device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = masked_loss(network(data[rows]), visible[rows], labels[rows], positive_weight,
                                   self.config["temperature"])
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite window MIL training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), self.config["gradient_clip"])
                optimizer.step()
                scheduler.step()
                total += float(loss.detach()) * len(rows)
            self.history.append({"epoch": epoch + 1, "mean_training_loss": total / len(features)})
        self.state = {name: value.detach().cpu().numpy().copy() for name, value in network.state_dict().items()}
        if any(not np.isfinite(value).all() for value in self.state.values()):
            raise ValueError("nonfinite saved window head")
        self.positive_weight = weights
        self.fit_seconds = time.perf_counter() - started
        return self

    def predict(self, matrix):
        import torch

        features, mask = unpack(matrix)
        data = self._standardized(features, mask)
        # A small CPU linear head keeps exported inference independent of CUDA state.
        with torch.inference_mode():
            logits = torch.nn.functional.linear(torch.from_numpy(data), torch.from_numpy(self.state["weight"]),
                                                torch.from_numpy(self.state["bias"]))
            pooled = pool_logits(logits, torch.from_numpy(mask), self.config["temperature"])
            probability = (pooled - torch.from_numpy(np.log(self.positive_weight))).sigmoid().numpy()
        probability[:, ~self.supported] = .5
        return probability

    def predict_prior(self, matrix):
        targets = self.training_targets
        prior = ((targets == 1).sum(axis=0) + 1) / ((targets != -1).sum(axis=0) + 2)
        prior[~self.supported] = .5
        return np.repeat(prior[None], len(matrix), axis=0)
