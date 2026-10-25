"""One three-action backoff extension of the fixed D9 context prior."""

import itertools

import numpy as np
from scipy.special import logsumexp

from cuhkx.server_temporal_d6_residual import BITS
from cuhkx.server_temporal_d9_context_prior import ANSWERS, ORDERS, order_key
from cuhkx.server_temporal_d9_context_prior import fit as fit4
from cuhkx.server_temporal_d9_context_prior import predict as predict4


def fit(classes, answers, groups):
    m = fit4(classes, answers, groups)
    rows = []
    gn = {}
    tab = {}
    for c, a, g in zip(classes, answers, groups):
        order = [int(c["ABCD".index(l)]) for l in a]
        for subset in itertools.combinations(sorted(map(int, c)), 3):
            k = order_key(subset)
            o = order_key([v for v in order if v in subset])
            rows.append((k, o, str(g)))
            gn[(k, str(g))] = gn.get((k, str(g)), 0) + 1
    for k, o, g in rows:
        d = tab.setdefault(k, {})
        d[o] = d.get(o, 0.0) + 1 / gn[(k, g)]
    m["triplets"] = tab
    m["schema"] = "full-action-set-order-prior/triple-backoff-v1"
    m["backoff_exponent"] = 0.25
    return m


def predict(model, probability, prior_logits, classes):
    _pred, scores, support, old = predict4(model, probability, prior_logits, classes)
    prior_scores = np.asarray(prior_logits, float) @ BITS.T
    for i, c in enumerate(classes):
        if support[i] > 0:
            continue
        q = np.exp(prior_scores[i] - logsumexp(prior_scores[i]))
        correction = np.zeros(24)
        for subset in itertools.combinations(sorted(map(int, c)), 3):
            counts = model["triplets"].get(order_key(subset), {})
            mass = sum(counts.values())
            if not mass:
                continue
            order = [order_key([int(c[j]) for j in perm if int(c[j]) in subset]) for perm in ORDERS]
            oldm = {o: float(q[np.array(order) == o].sum()) for o in set(order)}
            newm = {
                o: (counts.get(o, 0.0) + model["alpha"] * v) / (mass + model["alpha"])
                for o, v in oldm.items()
            }
            correction += np.array(
                [np.log(max(newm[o], 1e-300)) - np.log(max(oldm[o], 1e-300)) for o in order]
            )
            support[i] += mass / 4
        scores[i] = old[i] + model["backoff_exponent"] * correction
    return ANSWERS[scores.argmax(1)], scores, support, old
