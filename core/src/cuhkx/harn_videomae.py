"""Forty-four HARn classes from generic VideoMAE descriptors with scoped view normalization."""
from __future__ import annotations

import time

import numpy as np

from cuhkx.action_joint_harn_head import HarnWindowBank
from cuhkx.harn_nested import FeatureBank
from cuhkx.sequence_skeleton_nested import array_digest

SPEC = {'kind': 'generic_videomae_ssv2_whole16_harn', 'shape': [2, 768],
        'modalities': ['Depth_Color', 'IR'], 'availability': 'explicit aligned view mask',
        'backbone_fitting': 'fixed generic SSv2 e3f23b7; no CUHK-X fitting'}


class VideoMAEFeatureBank(FeatureBank):
    @classmethod
    def load(cls, skeleton_path, visual_path):
        from pathlib import Path

        with np.load(skeleton_path, allow_pickle=False) as z:
            units, matrix, names = z['units'].astype(str), z['X'].astype(np.float32), z['feature_names'].astype(str)
        if matrix.shape != (len(units), len(names)) or len(set(units)) != len(units) or not np.isfinite(matrix).all():
            raise ValueError('invalid skeleton descriptor archive')
        raw = HarnWindowBank(Path(visual_path).parent)
        rows = np.flatnonzero(raw.mask.any(axis=(1, 2)))
        visual = np.concatenate((raw.features[rows, 0].reshape(len(rows), -1), raw.mask[rows, 0]), axis=1)
        return cls(units, matrix, names, raw.units[rows], visual, SPEC)


def unpack(matrix):
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 1538 or not np.isfinite(values).all():
        raise ValueError('HARn VideoMAE descriptors require two768D views plus masks')
    raw = values[:, -2:]
    if not np.isin(raw, [0, 1]).all() or not raw.any(axis=1).all():
        raise ValueError('each recording needs at least one available view')
    return values[:, :-2].reshape(-1, 2, 768), raw.astype(bool)


class VideoMAEClassifier:
    def __init__(self, settings, seed):
        self.settings, self.seed = settings, seed

    def fit(self, bank, rows, labels):
        import torch

        values, visible = unpack(bank.visual[rows])
        if len(rows) != len(labels) or len(set(rows)) != len(rows) or not np.isin(labels, np.arange(44)).all():
            raise ValueError('invalid scoped HARn class training rows')
        self.training_rows, self.training_labels = np.asarray(rows).copy(), np.asarray(labels).copy()
        self.fitting_content_sha256 = array_digest(np.asarray(bank.visual[rows]), self.training_labels)
        self.classes_ = np.unique(labels)
        self.n_features_in_ = bank.visual.shape[1]
        self.mean, self.scale = np.zeros((2, 768), np.float32), np.ones((2, 768), np.float32)
        for view in range(2):
            x = values[:, view][visible[:, view]]
            if len(x):
                self.mean[view] = x.mean(axis=0)
                self.scale[view] = np.maximum(x.std(axis=0), self.settings['std_floor'])
        standardized = np.where(visible[..., None], (values - self.mean) / self.scale, 0).astype(np.float32)
        data, mask = torch.from_numpy(standardized), torch.from_numpy(visible)
        targets = torch.as_tensor(labels, dtype=torch.long)
        counts = np.bincount(labels, minlength=44)
        self.class_weights = np.zeros(44, dtype=np.float32)
        self.class_weights[self.classes_] = len(labels) / (len(self.classes_) * counts[self.classes_])
        weights = torch.from_numpy(self.class_weights)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        network = torch.nn.Sequential(torch.nn.Linear(768, self.settings['hidden']), torch.nn.GELU(),
                                      torch.nn.Dropout(self.settings['dropout']), torch.nn.Linear(self.settings['hidden'], 44))
        optimizer = torch.optim.AdamW(network.parameters(), lr=self.settings['lr'], weight_decay=self.settings['weight_decay'])
        steps = self.settings['epochs'] * ((len(rows) + self.settings['batch_units'] - 1) // self.settings['batch_units'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        start, self.history = time.perf_counter(), []
        for epoch in range(self.settings['epochs']):
            total, seen = 0., 0
            order = rng.permutation(len(rows))
            for offset in range(0, len(order), self.settings['batch_units']):
                selected = order[offset:offset + self.settings['batch_units']]
                optimizer.zero_grad(set_to_none=True)
                logits = network(data[selected])
                active = mask[selected]
                truth = targets[selected, None].expand(-1, 2)[active]
                loss = torch.nn.functional.cross_entropy(logits[active], truth, weight=weights)
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite HARn VideoMAE training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), self.settings['gradient_clip'])
                optimizer.step()
                scheduler.step()
                total += float(loss.detach()) * len(truth)
                seen += len(truth)
            self.history.append({'epoch': epoch + 1, 'mean_training_loss': total / seen, 'training_views': seen})
        self.state = {name: value.detach().cpu().numpy().copy() for name, value in network.state_dict().items()}
        self.fit_seconds = time.perf_counter() - start
        return self

    def predict_proba(self, matrix):
        import torch

        values, visible = unpack(matrix)
        x = np.where(visible[..., None], (values - self.mean) / self.scale, 0).astype(np.float32)
        with torch.inference_mode():
            hidden = torch.nn.functional.linear(torch.from_numpy(x), torch.from_numpy(self.state['0.weight']),
                                                torch.from_numpy(self.state['0.bias']))
            logits = torch.nn.functional.linear(torch.nn.functional.gelu(hidden), torch.from_numpy(self.state['3.weight']),
                                                torch.from_numpy(self.state['3.bias']))
            probability = logits[:, :, self.classes_].softmax(dim=-1).numpy()
        return (probability * visible[..., None]).sum(axis=1) / visible.sum(axis=1)[:, None]
