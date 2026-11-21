"""Synthetic probe: can a warmer subpixel phase recover a blurred eye peak? No dP labels."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def pixel_fraction(offset: np.ndarray, eye_sigma: float, psf_sigma: float) -> np.ndarray:
    """Contrast retained by a unit-area square pixel after a Gaussian optical blur.

    An intrinsic, circular 2D Gaussian warm anomaly has peak 1 and width
    eye_sigma in *native pixels*. offset is the center of the selected pixel
    relative to the intrinsic peak, in the same units.
    """
    if offset.shape[-1:] != (2,) or eye_sigma <= 0 or psf_sigma < 0:
        raise ValueError("Need (..., xy) offsets, positive eye width and nonnegative PSF width")
    sigma = math.hypot(eye_sigma, psf_sigma)
    erf = np.vectorize(math.erf, otypes=[float])
    denom = math.sqrt(2) * sigma
    axes = erf((offset + 0.5) / denom) - erf((offset - 0.5) / denom)
    # Integration of the broadened Gaussian cancels the sigma^2 in the peak attenuation.
    return (math.pi * eye_sigma ** 2 / 2) * axes.prod(axis=-1)


def probe(seed: int, sequences: int) -> dict:
    if sequences < 1:
        raise ValueError("Need at least one synthetic sequence")
    phase = np.random.default_rng(seed).uniform(-0.5, 0.5, size=(sequences, 6, 2))
    rows = []
    for blur in (0.0, 0.55):  # thought experiment, not an AHI instrument calibration
        for width in (0.35, 0.6, 1.2, 2.0):
            ratio = pixel_fraction(phase, width, blur)
            rows.append({'sigma_eye_px': width, 'sigma_psf_px': blur,
                         'mean_one_frame': float(ratio[:, 0].mean()),
                         'mean_best_of_six': float(ratio.max(axis=1).mean()),
                         'ideal_centered_pixel': float(pixel_fraction(np.zeros(2), width, blur))})
    return {'scope': 'synthetic retained peak fraction, not pressure or measured AHI PSF',
            'seed': seed, 'sequences_per_setting': sequences, 'frames': 6, 'results': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--sequences', type=int, default=8192)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.seed, args.sequences)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
