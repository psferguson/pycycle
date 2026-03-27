"""Unit tests for the Lomb-Scargle module."""

import numpy as np
import pytest

from pycycle.scargle import scargle_fast


def _make_sinusoid(period=0.5, n=200, noise=0.05, seed=42):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 10, n))
    mag = np.sin(2 * np.pi * t / period) + rng.normal(0, noise, n)
    return t.astype(np.float64), mag.astype(np.float64)


def test_peak_at_correct_period():
    """scargle_fast should place the highest power near the injected period."""
    period = 0.5
    t, mag = _make_sinusoid(period=period)
    freqs = np.linspace(0.5, 5.0, 500)
    omega = (2 * np.pi * freqs).astype(np.float64)
    px = scargle_fast(t, mag, omega, len(omega))

    best_freq = freqs[np.argmax(px)]
    best_period = 1.0 / best_freq
    assert abs(best_period - period) < 0.02, (
        "Peak period %.4f too far from injected %.4f" % (best_period, period)
    )


def test_output_shape():
    t, mag = _make_sinusoid()
    omega = np.linspace(1, 30, 100, dtype=np.float64)
    px = scargle_fast(t, mag, omega, 100)
    assert px.shape == (100,)
    assert px.dtype == np.float64


def test_power_non_negative():
    t, mag = _make_sinusoid()
    omega = np.linspace(1, 30, 100, dtype=np.float64)
    px = scargle_fast(t, mag, omega, 100)
    assert np.all(px >= 0)
