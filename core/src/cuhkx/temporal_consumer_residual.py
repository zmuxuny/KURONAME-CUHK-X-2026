"""Odd pair corrections trained against fixed full-order consumer scores."""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize

from cuhkx.server_temporal_d6_residual import ANSWERS, BITS


def design(raw, scale):
    x = np.asarray(raw, dtype=np.float64) / scale
    x = x.copy()
    x[..., 1:] /= np.sqrt(14.)
    return x


def correction(theta, x, probability, arm):
    if arm == 'linear':
        value = torch.einsum('npd,d->np', x, theta)
    elif arm == 'nonlinear':
        w = theta[:120].reshape(15, 8)
        bias, out = theta[120:128], theta[128:]
        z = x @ w
        value = ((torch.tanh(z + bias) - torch.tanh(-z + bias)) * .5) @ out
    else:
        raise ValueError(arm)
    return value * (4 * probability * (1 - probability))


def fit(raw, probability, offset_scores, answers, groups, arm, seed=20260915):
    torch.set_num_threads(2)
    scale = np.maximum(np.sqrt(np.mean(np.asarray(raw, float) ** 2, axis=(0, 1))), .05)
    scale[0] = 1.
    x = torch.from_numpy(design(raw, scale))
    p = torch.as_tensor(probability, dtype=torch.float64)
    offset = torch.as_tensor(offset_scores, dtype=torch.float64)
    targets = torch.tensor([np.flatnonzero(ANSWERS == a).item() for a in answers])
    _, inverse, counts = np.unique(np.asarray(groups, str), return_inverse=True, return_counts=True)
    weights = torch.from_numpy(1. / counts[inverse])
    bits = torch.from_numpy(BITS)
    initial = np.zeros(15 if arm == 'linear' else 136)
    if arm == 'nonlinear':
        initial[:120] = np.random.default_rng(seed).normal(size=120) / np.sqrt(15)
    center = torch.from_numpy(initial.copy())

    def objective(values):
        theta = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        scores = offset + correction(theta, x, p, arm) @ bits.T
        loss = (weights * (torch.logsumexp(scores, 1) - scores[torch.arange(len(targets)), targets])).sum()
        loss = loss + .5 * (theta - center).square().sum()
        gradient, = torch.autograd.grad(loss, theta)
        return float(loss.detach()), gradient.detach().numpy()

    initial_loss, initial_gradient = objective(initial)
    # Check the implemented derivative on a deterministic direction before optimization.
    direction = np.random.default_rng(seed + 1).normal(size=len(initial))
    direction /= np.linalg.norm(direction)
    eps = 1e-5
    finite_difference = (objective(initial + eps * direction)[0] -
                         objective(initial - eps * direction)[0]) / (2 * eps)
    np.testing.assert_allclose(finite_difference, initial_gradient @ direction, rtol=1e-5, atol=1e-6)
    result = minimize(objective, initial.copy(), jac=True, method='L-BFGS-B',
                      options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-7})
    if not np.isfinite(result.x).all() or not np.isfinite(result.fun) or result.fun > initial_loss + 1e-8:
        raise RuntimeError('Residual optimization did not produce a valid objective')
    model = {'theta': result.x, 'initial': initial, 'scale': scale, 'arm': np.asarray(arm)}
    report = {'initial_loss': initial_loss, 'final_loss': float(result.fun),
              'success': bool(result.success), 'message': str(result.message),
              'iterations': int(result.nit), 'gradient_norm': float(np.linalg.norm(result.jac)),
              'effective_group_weight': float(weights.sum()), 'regularization': 1.,
              'gradient_direction_check': True, 'parameters': len(initial), 'seed': seed}
    return model, report


def predict(model, raw, probability, offset_scores):
    with torch.inference_mode():
        theta = torch.from_numpy(model['theta'])
        x = torch.from_numpy(design(raw, model['scale']))
        p = torch.as_tensor(probability, dtype=torch.float64)
        delta = correction(theta, x, p, str(model['arm'])).numpy()
    scores = np.asarray(offset_scores, float) + delta @ BITS.T
    return ANSWERS[scores.argmax(1)], scores, delta
