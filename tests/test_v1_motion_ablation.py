import numpy as np

from h9dp.v1 import ANGLES, RADII, angular_motion
from scripts.v1_motion_ablation import motion_variants, pair_minutes


def test_select_motion_by_actual_pair_and_component():
    times = np.array([[-300, -180, -60, -30, -10, 0], [-300, -180, -60, -30, -10, 0]])
    available = np.ones_like(times, dtype=bool)
    available[1, 4] = False
    gaps = pair_minutes(times, available)
    np.testing.assert_equal(gaps, [[np.nan, 120, 120, 30, 20, 10], [np.nan, 120, 120, 30, np.nan, 30]])
    shape = (*times.shape, len(RADII) - 3, 4)
    values = np.broadcast_to(np.array([1.0, 0.8, 0.1, -0.5]), shape)
    values = np.where(np.isfinite(gaps)[..., None, None], values, np.nan)
    result = {k: v.reshape(shape) for k, v in motion_variants(values.reshape(*times.shape, -1), gaps).items()}
    np.testing.assert_equal(result["only_10min"][0, -1], values[0, -1])
    assert not np.isfinite(result["only_10min"][1]).any()
    assert not np.isfinite(result["no_10min"][0, -1]).any()
    np.testing.assert_equal(result["no_10min"][1, -1], values[1, -1])
    np.testing.assert_equal(result["only_10min_shift"][0, -1, :, 0], values[0, -1, :, 0])
    assert not np.isfinite(result["only_10min_shift"][0, -1, :, 1:]).any()
    assert not np.isfinite(result["only_10min_context"][0, -1, :, 0]).any()
    np.testing.assert_equal(result["only_10min_context"][0, -1, :, 1:], values[0, -1, :, 1:])
    np.testing.assert_equal(result["no_10min_shift"][1], values[1])


def test_10min_rotation_recovers_known_synthetic_shift():
    early = np.random.default_rng(1).normal(240, 4, (4, len(RADII), ANGLES))
    for shift in (-2, 0, 2):
        late = np.roll(early, shift, axis=-1)
        motion = angular_motion(early, late, minutes=10).reshape(-1, 4)
        np.testing.assert_allclose(motion[:, 0], shift * (2 * np.pi / ANGLES) * RADII[2:-1] / 10)
        np.testing.assert_allclose(motion[:, 1], 1)
