import csv

import numpy as np
import pytest

from h9dp.r1 import record_folds
from h9dp.r1_column import warm_columns
from h9dp.r1_response import response_component
from scripts.r1_kernel import Cache
from scripts.r1_sampling_probe import pixel_fraction
from scripts.r1_tracking_audit import read_oof, registration_summary, rotation_error


def test_columns_use_colocated_spectra_and_do_not_forge_missing_references():
    views = np.zeros((1, 3, 10, 64, 64), dtype=np.int16)
    views[:, :2, 4] = 100
    views[:, 0, 9] = 100
    views[:, 0, 5:9] = 100
    # Two distinct peaks; if each channel were ranked separately the CO2
    # spectrum would incorrectly read 30 K in the first (water-vapor) band.
    views[:, :2, 0, 32, 30:33] = 3000
    views[:, :2, 3, 34, 30:33] = 2000
    views[:, :2, 0, 34, 30:33] = 200
    views[:, :2, 4, 31, 31] = 0
    views[:, :2, :4, 31, 31] = 19000  # masked values never contribute
    output = warm_columns(views)
    column, anomaly = output['column'], output['column_anomaly']
    np.testing.assert_allclose(column[0, 0, :4], [2, 0, 0, 20])
    np.testing.assert_allclose(column[0, 0, 8:12], [30, 0, 0, 0])
    np.testing.assert_allclose(anomaly[0, 0, 4:8], [1, -1, -1, 19])
    np.testing.assert_allclose(column[0, 1], column[0, 0])
    assert np.isnan(anomaly[0, 1]).all() and np.isnan(column[0, 2]).all()
    with pytest.raises(ValueError, match='views'):
        warm_columns(views[:, :, :5])


@pytest.mark.parametrize('visibility', [True, False])
def test_response_is_psd_and_held_records_cannot_change_training_block(visibility):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(12, 6))
    t = np.r_[np.linspace(230, 279, 11), np.nan]
    train, held = np.arange(9), np.arange(9, 12)
    first = response_component(x, t, train, held, visibility=visibility)
    assert first is not None
    gram, dist = first
    assert np.linalg.eigvalsh(gram).min() > -1e-8
    np.testing.assert_allclose(dist.diagonal(), 0, atol=1e-12)
    x[held] += 100
    t[held] = 230
    changed = response_component(x, t, train, held, visibility=visibility)
    assert changed is not None
    np.testing.assert_allclose(changed[0][:9, :9], gram[:9, :9])
    assert not np.allclose(changed[0][9:, :9], gram[9:, :9])
    assert response_component(x * np.nan, t, train, held, visibility=visibility) is None


def test_gate_counts_supported_but_rejected_scores_and_handles_no_moves():
    trace = np.full((4, 3, 3), np.nan)
    trace[:, 0] = [0, np.nan, 0]
    trace[:3, 1] = [[2, 0.90, 1], [0, 0.99, 0], [2, 0.99, 1]]
    trace[0, 2] = [0, 0.88, 0]
    ncc, w = registration_summary(trace)
    np.testing.assert_allclose(ncc[:3], [0.89, 0.99, 0.99])
    assert np.isnan(ncc[-1])
    np.testing.assert_allclose(w, [0.5, 0, 1, 0])


def test_pairing_is_by_seed_record_and_checks_outer_fold_and_coverage(tmp_path):
    ids = np.array([9, 1, 6])
    cache = Cache(tmp_path, 'digest', [{}, {}, {}], {'record': ids}, np.arange(3.), np.array(['Good'] * 3))
    folds = record_folds(3, 2, 1)
    path = tmp_path / 'predictions.csv'
    rows = [[1, folds[i], ids[i], 'Good', i, i + 0.1] for i in [1, 0, 2]]

    def save(batch):
        with path.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['seed', 'fold', 'record', 'quality', 'target', 'spectral_gp'])
            writer.writerows(batch)

    save(rows)
    np.testing.assert_allclose(read_oof(path, cache, [1], 2)[1], [0.1, 1.1, 2.1])
    for invalid in (rows[:-1], [*rows, rows[2]], [[1, 1 - folds[1], *rows[0][2:]], *rows[1:]]):
        save(invalid)
        with pytest.raises(ValueError):
            read_oof(path, cache, [1], 2)


def test_blurred_pixel_integral_agrees_with_independent_quadrature():
    width, blur = 0.43, 0.60
    offset = np.array([0.35, -0.46])
    # Unit-width square; midpoint quadrature of the flux-conserving blurred Gaussian.
    grid = (np.arange(300) + 0.5) / 300 - 0.5
    r2 = (grid[:, None] + offset[0]) ** 2 + (grid + offset[1]) ** 2
    variance = width ** 2 + blur ** 2
    expected = width ** 2 / variance * np.exp(-r2 / (2 * variance)).mean()
    assert pixel_fraction(offset, width, blur) == pytest.approx(expected, abs=1e-6)
    center = pixel_fraction(np.zeros(2), width, blur)
    assert expected < center < pixel_fraction(np.zeros(2), width, 0) < 1


def test_repeated_record_mse_decomposition_is_exact_not_squared_group_bias():
    result = rotation_error(np.array([[1., 4.], [3., 2.]]), np.array([0., 5.]))
    assert result['mean_mse'] == 5 and result['squared_mean_error'] == 4 and result['split_variance'] == 1
    assert result['top_decile_share_of_persistent_sse'] == 0.5
