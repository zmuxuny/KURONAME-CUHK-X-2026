"""Same nonlinear joint fitting with explicit eight/sixteen physical window axes."""
from __future__ import annotations

import time

import numpy as np

from cuhkx.action_mil import positive_weights
from cuhkx.action_window_head import CHANNELS, VIEWS, WindowPartialActionClassifier
from cuhkx.sequence_skeleton_nested import array_digest


def pool_logits(logits, available, temperature):
    """Average aligned views, then normalized smooth maximum over visible windows."""
    import torch

    if logits.ndim != 4 or (logits.shape[1] not in (8, 16) or logits.shape[2:] != (2, 40)) or available.shape != logits.shape[:3]:
        raise ValueError("HAU MIL needs batch/window/view/action axes")
    if temperature <= 0 or not available.any(dim=(1, 2)).all():
        raise ValueError("HAU MIL pooling needs positive temperature and visible windows")
    views = available.sum(dim=2)
    pooled_views = (logits * available[..., None]).sum(dim=2) / views.clamp_min(1)[..., None]
    window_mask = views > 0
    scaled = (pooled_views / temperature).masked_fill(~window_mask[..., None], -torch.inf)
    return temperature * (torch.logsumexp(scaled, dim=1) - window_mask.sum(dim=1).log()[:, None])


def masked_loss(logits, available, targets, weights, temperature):
    from torch.nn import functional

    pooled = pool_logits(logits, available, temperature)
    known = targets != -1
    if targets.shape != pooled.shape or not known.any(dim=1).all():
        raise ValueError("HAU MIL loss needs aligned nonempty partial labels")
    per_cell = functional.binary_cross_entropy_with_logits(pooled, targets.clamp_min(0),
                                                          pos_weight=weights, reduction="none")
    return ((per_cell * known).sum(dim=1) / known.sum(dim=1)).mean()



def unpack(matrix, channels=CHANNELS):
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('window inputs must be a finite feature matrix')
    if channels not in (512, 768):
        raise ValueError('only declared R34/VideoMAE pooled channel counts are supported')
    windows, remainder = divmod(values.shape[1], VIEWS * (channels + 1))
    if remainder or windows not in (8, 16):
        raise ValueError('declared descriptor layout requires eight or sixteen windows')
    features = values[:, :windows * VIEWS * channels].reshape(-1, windows, VIEWS, channels)
    raw_mask = values[:, windows * VIEWS * channels:]
    if not np.isin(raw_mask, [0, 1]).all():
        raise ValueError('window masks must be binary')
    mask = raw_mask.reshape(-1, windows, VIEWS).astype(bool)
    if not mask.any(axis=(1, 2)).all():
        raise ValueError('each recording needs at least one visible window')
    return features, mask


class VariableWindowJointClassifier(WindowPartialActionClassifier):
    def __init__(self, config, seed, auxiliary):
        super().__init__(config, seed)
        self.auxiliary = auxiliary

    def fit(self, matrix, targets):
        import torch

        self.channels = self.config.get('channels', CHANNELS)
        features, mask = unpack(matrix, self.channels)
        self.windows = features.shape[1]
        auxiliary, auxiliary_mask, auxiliary_labels, self.auxiliary_scope = self.auxiliary
        if auxiliary.shape[1:] != (1, VIEWS, self.channels) or auxiliary_mask.shape != auxiliary.shape[:3]:
            raise ValueError('Auxiliary feature dimensions differ from the HAU descriptor')
        del self.auxiliary  # Retain scope/digests, not another copy of the raw feature bank in every model.
        weights, self.supported = positive_weights(targets)
        self.fitting_content_sha256 = array_digest(np.asarray(matrix), targets)
        self.training_targets = np.asarray(targets).copy()
        self.n_features_in_, self.n_outputs_ = matrix.shape[1], 40
        self.mean = np.zeros((VIEWS, self.channels), dtype=np.float32)
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
        network = torch.nn.Sequential(torch.nn.Linear(self.channels, self.config["hidden"]), torch.nn.GELU(),
                                      torch.nn.Dropout(self.config["dropout"]),
                                      torch.nn.Linear(self.config["hidden"], 40)).to(device)
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

    def predict(self, matrix):
        import torch

        features, mask = unpack(matrix, getattr(self, 'channels', CHANNELS))
        if features.shape[1] != self.windows:
            raise ValueError('prediction window count differs from the trained descriptor layout')
        data = self._standardized(features, mask)
        with torch.inference_mode():
            hidden = torch.nn.functional.linear(torch.from_numpy(data), torch.from_numpy(self.state["0.weight"]),
                                                torch.from_numpy(self.state["0.bias"]))
            hidden = torch.nn.functional.gelu(hidden)
            logits = torch.nn.functional.linear(hidden, torch.from_numpy(self.state["3.weight"]),
                                                torch.from_numpy(self.state["3.bias"]))
            pooled = pool_logits(logits, torch.from_numpy(mask), self.config["temperature"])
            probability = (pooled - torch.from_numpy(np.log(self.positive_weight))).sigmoid().numpy()
        probability[:, ~self.supported] = .5
        return probability
