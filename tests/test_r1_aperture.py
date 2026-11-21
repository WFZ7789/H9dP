import numpy as np

from h9dp.r1_aperture import aperture_features


def test_equal_peak_eyes_have_distinct_integrals_and_centroid_radii():
    views = np.zeros((1, 3, 10, 64, 64), dtype=np.int16)
    views[:, :, 4] = 100
    views[:, :, :4] = 700  # arbitrary uniform background
    views[0, 0, :4, 32, 32] += 500
    views[0, 1, :4, 31:33, 32:34] += 500
    features = aperture_features(views).reshape(3, 2, 4, 4)
    np.testing.assert_allclose(features[0, 0, 0], [5, 5, 1, 0])
    np.testing.assert_allclose(features[1, 1, 0], [5, 20, 4, np.sqrt(0.5)])
    assert (features[2, :, :, :2] == 0).all()
    assert np.isnan(features[2, :, :, 2:]).all()


def test_missing_data_invalidates_only_affected_apertures():
    views = np.zeros((1, 2, 10, 64, 64), dtype=np.int16)
    views[:, 0, 4] = 100
    views[:, 0, 4, 32, 42] = 0
    views[:, 0, 3, 32, 32] = 1000
    features = aperture_features(views).reshape(2, 2, 4, 4)
    np.testing.assert_allclose(features[0, 0, 3], [10, 10, 1, 0])
    assert np.isnan(features[0, 1]).all() and np.isnan(features[1]).all()
