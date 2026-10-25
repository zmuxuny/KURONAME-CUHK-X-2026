"""Hierarchical order prior conditioned only on the queried four actions."""

import numpy as np
from scipy.special import logsumexp

from cuhkx.server_temporal_d6_residual import ANSWERS, BITS, ORDERS, logit


def key(classes):
    c = tuple(int(x) for x in classes)
    if len(c) != 4 or len(set(c)) != 4:
        raise ValueError("Four distinct actions required")
    return ",".join(map(str, sorted(c)))


def order_key(order):
    return ",".join(map(str, order))


def fit(classes, answers, groups):
    rows = []
    count = {}
    for c, a, g in zip(classes, answers, groups):
        if sorted(a) != list("ABCD"):
            raise ValueError("Invalid order target")
        k = key(c)
        o = order_key([int(c["ABCD".index(l)]) for l in a])
        rows.append((k, o, str(g)))
    group_n = {}
    for k, o, g in rows:
        group_n[(k, g)] = group_n.get((k, g), 0) + 1
    for k, o, g in rows:
        dest = count.setdefault(k, {})
        dest[o] = dest.get(o, 0.0) + 1.0 / group_n[(k, g)]
    return {
        "schema": "full-action-set-order-prior/v1",
        "alpha": 3.0,
        "counts": count,
        "scope_rows": len(rows),
        "grouped_contexts": len(group_n),
    }


def predict(model, probability, prior_logits, classes):
    p = np.asarray(probability, float)
    prior = np.asarray(prior_logits, float)
    if p.shape != prior.shape or p.shape != (len(classes), 6):
        raise ValueError("Shape mismatch")
    old_scores = (logit(p) + prior) @ BITS.T
    out = old_scores.copy()
    support = []
    prior_scores = prior @ BITS.T
    for i, c in enumerate(classes):
        counts = model["counts"].get(key(c), {})
        mass = sum(counts.values())
        support.append(mass)
        if not mass:
            continue
        q = np.exp(prior_scores[i] - logsumexp(prior_scores[i]))
        empirical = np.array(
            [counts.get(order_key([int(c[j]) for j in perm]), 0.0) for perm in ORDERS]
        )
        dist = (empirical + model["alpha"] * q) / (mass + model["alpha"])
        assert np.isfinite(dist).all() and np.isclose(dist.sum(), 1.0)
        out[i] = logit(p[i]) @ BITS.T + np.log(np.maximum(dist, 1e-300))
    return ANSWERS[out.argmax(1)], out, np.asarray(support, float), old_scores
