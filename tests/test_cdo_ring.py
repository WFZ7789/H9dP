import numpy as np

from h9dp.cdo_ring import coldest_ring


def test_ring_finds_a_cold_radially_varying_closed_path():
    radii = [0, 2, 3, 4, 2, 1]
    cost = np.full((2, len(radii), 6), 230., dtype=np.float32)
    cost[0, np.arange(len(radii)), radii] = [201, 199, 203, 202, 201, 202]
    cost[1, :, 0] = 213
    np.testing.assert_array_equal(coldest_ring(cost), [203, 213])


def test_ring_requires_support_and_closure_at_polar_seam():
    cost = np.full((2, 5, 9), 230., dtype=np.float32)
    cost[0, np.arange(5), [0, 1, 2, 4, 6]] = 200
    cost[1, 2] = np.inf
    np.testing.assert_array_equal(coldest_ring(cost), [230, np.inf])
