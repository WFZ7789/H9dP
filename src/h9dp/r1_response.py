"""Fold-local additive thermal-response kernels for the R1 spectral GP."""

from __future__ import annotations

import numpy as np

from h9dp.r1 import _normalize


def response_component(data: np.ndarray, eye_bt: np.ndarray, train: np.ndarray, held: np.ndarray,
                       *, visibility: bool) -> tuple[np.ndarray, np.ndarray] | None:
    """A sum of univariate smooth responses, optionally different in cold and warm eyes.

    The scalar basis layout is fixed; feature selection, mean and scale use only
    the outer training side. Products with a soft gate are still Gram matrices.
    """
    a, b = _normalize(data[train], data[held])
    if a.shape[1] == 0:
        return None
    x = np.concatenate((a, b)).astype(np.float64, copy=False)
    # Unit-norm local activations retain an intercept at either end of the range;
    # the linear channel allows a slope even beyond the last temperature knot.
    knots = np.array([-2.5, -1.25, 0.0, 1.25, 2.5])
    bumps = np.exp(-0.5 * ((x[..., None] - knots) / 0.85) ** 2)
    bumps /= np.linalg.norm(bumps, axis=-1, keepdims=True)
    basis = bumps.reshape(len(x), -1)
    gram = 1.0 + (basis @ basis.T + 0.25 * (x @ x.T)) / x.shape[1]
    if visibility:
        # B16 is an image-side proxy for obscuration, not a quality label. Use
        # equal memberships if there is no resolved warm-candidate temperature.
        temp = eye_bt[np.r_[train, held]]
        warm = 1 / (1 + np.exp(-np.clip((np.where(np.isfinite(temp), temp, 258.0) - 258) / 6, -40, 40)))
        gates = np.stack((warm, 1 - warm), axis=1)
        gates /= np.linalg.norm(gates, axis=1, keepdims=True)
        gram *= gates @ gates.T
    norm = gram.diagonal()
    return gram, (norm[:, None] + norm[None, :] - 2 * gram).clip(0)
