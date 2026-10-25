"""Equal R18, R34 and VideoMAE families; R18 retains its fixed two-seed mean."""
import numpy as np

from cuhkx.harn_seed_ensemble import validate_evidence


def three_family_evidence(r18, r34, videomae):
    evidence = (r18, r34, videomae)
    for item in evidence:
        validate_evidence(item)
        np.testing.assert_array_equal(item['units'], r18['units'])
    present = np.stack([item['visual_present'] for item in evidence])
    probabilities = np.stack([item['visual_probabilities'] for item in evidence])
    denominator = present.sum(axis=0)
    weighted = (probabilities * present[..., None]).sum(axis=0)
    result = {key: value.copy() for key, value in r18.items()}
    result['visual_probabilities'] = np.divide(weighted, denominator[:, None], out=np.zeros_like(weighted),
                                               where=denominator[:, None] > 0)
    result['visual_present'] = denominator > 0
    result['visual_classes'] = np.unique(np.concatenate([item['visual_classes'] for item in evidence]))
    validate_evidence(result)
    return result
