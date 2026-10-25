import copy

import numpy as np
import pytest

from cuhkx.harn_three_family import three_family_evidence


def test_equal_families_and_missing_inputs_preserve_skeleton():
    values = np.zeros((4, 44))
    values[:, 0] = 1
    first = {'units': np.asarray(['a', 'b', 'c', 'd']), 'skeleton_probabilities': values.copy(),
             'skeleton_classes': np.asarray([0, 1]), 'skeleton_present': np.ones(4, dtype=bool),
             'visual_probabilities': values.copy(), 'visual_classes': np.asarray([0, 1]),
             'visual_present': np.asarray([True, True, False, False])}
    first['visual_probabilities'][2:] = 0
    second, third = copy.deepcopy(first), copy.deepcopy(first)
    second['visual_present'][:] = [True, False, False, False]
    second['visual_probabilities'][:] = 0
    second['visual_probabilities'][0, 1] = 1
    third['visual_present'][:] = [True, False, True, False]
    third['visual_probabilities'][:] = 0
    third['visual_probabilities'][[0, 2], :2] = [.5, .5]
    result = three_family_evidence(first, second, third)
    np.testing.assert_allclose(result['visual_probabilities'][:, :2], [[.5, .5], [1, 0], [.5, .5], [0, 0]], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(result['visual_present'], [True, True, True, False])
    np.testing.assert_array_equal(result['skeleton_probabilities'], first['skeleton_probabilities'])
    third['units'] = third['units'][::-1]
    with pytest.raises(AssertionError):
        three_family_evidence(first, second, third)
