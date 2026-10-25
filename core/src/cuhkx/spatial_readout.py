"""Matched spatial-mean and action-query readouts on frozen video tokens.

Inputs keep independent time and spatial axes. No subject, group, question ID,
or labels enter forward. The mean control has less effective function freedom
despite identical parameter tensors; it is an information-access control.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SpatialReadout(nn.Module):
    def __init__(self, channels: int = 768, projection: int = 64, classes: int = 40):
        super().__init__()
        self.key = nn.Linear(channels, projection)
        self.value = nn.Linear(channels, projection)
        self.query = nn.Parameter(torch.zeros(classes, projection))
        self.readout = nn.Parameter(torch.empty(classes, projection))
        self.bias = nn.Parameter(torch.zeros(classes))
        nn.init.normal_(self.readout, std=projection**-0.5)

    def forward(self, tokens, *, arm="spatial", valid=None, uniform=False):
        """Return [batch,time,action] logits; invalid positions have zero mass.

        ``uniform`` is a frozen-model contribution diagnostic. Permuting spatial
        rows together with their masks must preserve outputs for either arm.
        All-missing bins emit zero logits and must be excluded by the caller.
        """
        if arm not in {"mean", "spatial"}:
            raise ValueError("arm must be mean or spatial")
        if tokens.ndim != 4 or tokens.shape[-1] != self.key.in_features:
            raise ValueError("expected [batch,time,space,channel] frozen tokens")
        if valid is None:
            valid = torch.ones(tokens.shape[:-1], device=tokens.device, dtype=torch.bool)
        if valid.shape != tokens.shape[:-1] or valid.dtype != torch.bool:
            raise ValueError("invalid spatial mask")
        clean = tokens.masked_fill(~valid[..., None], 0)
        key, value = self.key(clean), self.value(clean)
        weight = valid.to(value.dtype)[..., None]
        denominator = weight.sum(dim=-2, keepdim=True).clamp_min(1)
        if arm == "mean":
            key = ((key * weight).sum(-2, keepdim=True) / denominator).expand_as(key)
            value = ((value * weight).sum(-2, keepdim=True) / denominator).expand_as(value)
        scores = torch.einsum("btsd,cd->btcs", key, self.query) / math.sqrt(key.shape[-1])
        if uniform:
            scores = torch.zeros_like(scores)
        # Finite minimum handles an entirely absent bin without a NaN softmax.
        scores = scores.masked_fill(~valid[:, :, None, :], torch.finfo(scores.dtype).min)
        attention = scores.softmax(dim=-1) * valid[:, :, None, :]
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1)
        evidence = torch.einsum("btcs,btsd->btcd", attention, value)
        logits = (evidence * self.readout).sum(-1) + self.bias
        return logits.masked_fill(~valid.any(-1)[..., None], 0)
