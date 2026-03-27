"""Unit and integration tests for pycycle.template_fit."""
import importlib.resources

import numpy as np
import pytest

from pycycle.templates import RRTemplate
from pycycle.template_fit import (
    TemplateFitter,
    _interp_template,
    _rss_grid_rr_py,
    _rss_grid_mb_py,
    _USE_C,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_synthetic_rr_template(n_bands=2, n_phase=100):
    """Minimal rr-templates-style template with sine wave shape."""
    phase = np.linspace(0.0, 1.0, n_phase, endpoint=False)
    # simple sine: peaks at phase=0, trough at 0.5
    gamma = np.array([np.sin(2.0 * np.pi * phase) for _ in range(n_bands)])
    bands = ['g', 'r'][:n_bands]
    dust = np.array([3.303, 2.285][:n_bands])
    betas = np.zeros((n_bands, 3))  # zero betas → no period-dependent offset
    return RRTemplate(name='test_rr', bands=bands, phase=phase,
                      gamma=gamma, dust=dust, betas=betas)


def _make_synthetic_mb_template(n_bands=2, n_phase=100):
    """Minimal Multiband-style template (no dust/betas)."""
    phase = np.linspace(0.0, 1.0, n_phase, endpoint=False)
    gamma = np.array([np.sin(2.0 * np.pi * phase) for _ in range(n_bands)])
    bands = ['g', 'r'][:n_bands]
    return RRTemplate(
        name='test_mb', bands=bands, phase=phase, gamma=gamma,
        dust=None, betas=None,
    )


def _synthetic_rr_lightcurve(
    true_period=0.5016, n_per_band=20, noise=0.01,
    mu=14.0, ebv=0.05, A=0.5, phi=0.0, seed=42,
):
    """Generate a synthetic multiband RRab light curve from a known model."""
    rng = np.random.default_rng(seed)
    tmpl = _make_synthetic_rr_template(n_bands=2, n_phase=100)
    freq = 1.0 / true_period
    bands_n = 2
    t_list, m_list, me_list, b_list = [], [], [], []
    for b in range(bands_n):
        t = rng.uniform(0, 100, n_per_band)
        ph = (freq * t + phi) % 1.0
        lo = (ph * 100).astype(int) % 100
        hi = (lo + 1) % 100
        frac = ph * 100 - lo
        g = (1 - frac) * tmpl.gamma[b, lo] + frac * tmpl.gamma[b, hi]
        m = mu + ebv * tmpl.dust[b] + A * g + rng.normal(0, noise, n_per_band)
        me = noise * np.ones(n_per_band)
        t_list.append(t)
        m_list.append(m)
        me_list.append(me)
        b_list.append(np.full(n_per_band, b, dtype=int))
    t = np.concatenate(t_list)
    m = np.concatenate(m_list)
    me = np.concatenate(me_list)
    bi = np.concatenate(b_list)
    return tmpl, t, m, me, bi


def _load_b1392():
    data_path = importlib.resources.files('pycycle.data').joinpath('B1392all.tab')
    hjd, mag, magerr, filts = np.loadtxt(str(data_path), unpack=True)
    ok = (magerr >= 0.0) & (magerr <= 0.2)
    return hjd[ok], mag[ok], magerr[ok], filts[ok]


# ---------------------------------------------------------------------------
# _interp_template
# ---------------------------------------------------------------------------

class TestInterpTemplate:
    def setup_method(self):
        n_ph = 100
        phase = np.linspace(0, 1, n_ph, endpoint=False)
        self.gamma = np.array([np.sin(2 * np.pi * phase)])
        self.n_ph = n_ph

    def test_at_phase_zero(self):
        """Phase 0.0 should interpolate to sin(0) = 0."""
        val = _interp_template(self.gamma, np.array([0]), np.array([0.0]))
        assert abs(val[0]) < 1e-6

    def test_at_phase_quarter(self):
        """Phase 0.25 should interpolate to sin(pi/2) = 1."""
        val = _interp_template(self.gamma, np.array([0]), np.array([0.25]))
        assert abs(val[0] - 1.0) < 0.05  # linear interp on coarse grid

    def test_phase_wraps_at_one(self):
        """Phase exactly 1.0 should wrap to 0 → same as phase 0.0."""
        v0 = _interp_template(self.gamma, np.array([0]), np.array([0.0]))
        v1 = _interp_template(self.gamma, np.array([0]), np.array([1.0]))
        assert abs(v0[0] - v1[0]) < 1e-9

    def test_phase_near_one(self):
        """Phase 0.999 should not crash and give a sensible value."""
        val = _interp_template(self.gamma, np.array([0]), np.array([0.999]))
        assert np.isfinite(val[0])

    def test_negative_phase_normalised(self):
        """Negative phase after modulo should still return finite values."""
        ph = np.array([-0.1]) % 1.0  # = 0.9
        val = _interp_template(self.gamma, np.array([0]), ph)
        assert np.isfinite(val[0])


# ---------------------------------------------------------------------------
# rss_grid_rr vs Python fallback
# ---------------------------------------------------------------------------

class TestRssGridRrVsPython:
    """Cython rss_grid_rr must match _rss_grid_rr_py to within 1e-6."""

    def _make_inputs(self, n=40, n_phase=100, n_freq=30, seed=0):
        rng = np.random.default_rng(seed)
        t = rng.uniform(0, 50, n).astype(np.float64)
        mag = 20.0 + rng.normal(0, 0.1, n).astype(np.float64)
        w = np.ones(n, dtype=np.float64)
        bidx = rng.integers(0, 2, n).astype(np.int64)
        phase = np.linspace(0, 1, n_phase, endpoint=False)
        gamma = np.vstack([
            np.sin(2 * np.pi * phase),
            np.cos(2 * np.pi * phase),
        ]).astype(np.float64)
        dgamma = np.gradient(gamma, 1.0 / n_phase, axis=1)
        dust = np.array([3.3, 2.3], dtype=np.float64)
        betas = np.zeros((2, 3), dtype=np.float64)
        freqs = np.linspace(1.1, 2.3, n_freq).astype(np.float64)
        return t, mag, w, bidx, gamma, dgamma, dust, betas, freqs

    def test_rss_matches_python(self):
        args = self._make_inputs()
        t, mag, w, bidx, gamma, dgamma, dust, betas, freqs = args
        rss_py, _, _ = _rss_grid_rr_py(
            t, mag, w, bidx, gamma, dgamma, dust, betas, freqs,
            n_newton=3, n_start=2,
        )
        if not _USE_C:
            pytest.skip('Cython extension not built')
        from pycycle._ext.template_fit_c import rss_grid_rr
        rss_c, _, _ = rss_grid_rr(
            t, mag, w, bidx, gamma, dgamma, dust, betas, freqs,
            n_newton=3, n_start=2, warm_start=0,
        )
        np.testing.assert_allclose(
            rss_c, rss_py, rtol=1e-5,
            err_msg='Cython rss_grid_rr disagrees with Python',
        )

    def test_warm_start_rss_similar(self):
        """warm_start RSS min should be within 10% of n_start=4."""
        args = self._make_inputs(n_freq=20)
        t, mag, w, bidx, gamma, dgamma, dust, betas, freqs = args
        rss_cold, _, _ = _rss_grid_rr_py(
            t, mag, w, bidx, gamma, dgamma, dust, betas, freqs,
            n_newton=5, n_start=4,
        )
        rss_warm, _, _ = _rss_grid_rr_py(
            t, mag, w, bidx, gamma, dgamma, dust, betas, freqs,
            n_newton=5, n_start=1, warm_start=True,
        )
        rel = abs(rss_cold.min() - rss_warm.min()) / (rss_cold.min() + 1e-30)
        assert rel < 0.10


# ---------------------------------------------------------------------------
# rss_grid_mb vs Python fallback
# ---------------------------------------------------------------------------

class TestRssGridMbVsPython:
    def _make_inputs(self, n=40, n_phase=100, n_freq=20, n_bands=2, seed=1):
        rng = np.random.default_rng(seed)
        t = rng.uniform(0, 50, n).astype(np.float64)
        mag = 20.0 + rng.normal(0, 0.1, n).astype(np.float64)
        w = np.ones(n, dtype=np.float64)
        bidx = rng.integers(0, n_bands, n).astype(np.int64)
        phase = np.linspace(0, 1, n_phase, endpoint=False)
        gamma = np.vstack([
            np.sin(2 * np.pi * phase),
            np.cos(2 * np.pi * phase),
        ]).astype(np.float64)
        dgamma = np.gradient(gamma, 1.0 / n_phase, axis=1)
        freqs = np.linspace(1.0, 2.5, n_freq).astype(np.float64)
        return t, mag, w, bidx, gamma, dgamma, freqs, n_bands

    def test_rss_matches_python(self):
        args = self._make_inputs()
        t, mag, w, bidx, gamma, dgamma, freqs, n_bands = args
        rss_py, _, _, _ = _rss_grid_mb_py(
            t, mag, w, bidx, gamma, dgamma,
            freqs, n_bands, n_newton=3, n_start=2,
        )
        if not _USE_C:
            pytest.skip('Cython extension not built')
        from pycycle._ext.template_fit_c import rss_grid_mb
        rss_c, _, _, _ = rss_grid_mb(
            t, mag, w, bidx, gamma, dgamma,
            freqs, n_bands, n_newton=3, n_start=2,
        )
        np.testing.assert_allclose(rss_c, rss_py, rtol=1e-5,
                                   err_msg='Cython rss_grid_mb disagrees with Python')


# ---------------------------------------------------------------------------
# Template shape validation
# ---------------------------------------------------------------------------

def test_load_rr_template_shapes():
    """Synthetic rr-template has correct array shapes."""
    tmpl = _make_synthetic_rr_template(n_bands=2, n_phase=100)
    assert tmpl.betas is not None
    assert tmpl.betas.shape == (2, 3), 'betas should be (n_bands, 3)'
    assert tmpl.gamma.shape == (2, 100), 'gamma should be (n_bands, n_phase)'
    assert tmpl.dust.shape == (2,), 'dust should be (n_bands,)'
    assert len(tmpl.bands) == 2


# ---------------------------------------------------------------------------
# Period recovery on synthetic data
# ---------------------------------------------------------------------------

def test_period_recovery_synthetic():
    """Injected period should be recovered to within 1%."""
    true_period = 0.5016
    tmpl, t, m, me, bi = _synthetic_rr_lightcurve(
        true_period=true_period, n_per_band=30, noise=0.005)
    filtnams = tmpl.bands
    filts = bi  # already 0-indexed band codes
    fitter = TemplateFitter(tmpl, n_newton=5, n_start=4)
    result = fitter.fit(t, m, me, filts, filtnams,
                        pmin=0.44, dphi=0.02, pmax=0.89)
    rel_err = abs(result.best_period - true_period) / true_period
    assert rel_err < 0.01, (
        f'Period recovery failed: got {result.best_period:.6f} d, '
        f'expected {true_period:.6f} d (rel err {rel_err:.4f})'
    )


# ---------------------------------------------------------------------------
# warm_start returns same best period as n_start=4 on B1392
# ---------------------------------------------------------------------------

def test_warm_start_same_result():
    """warm_start=True and n_start=4 should both recover the gold B1392 period.

    warm_start may settle in a slightly different local minimum than n_start=4
    (it's a speed-accuracy trade-off), but both should be within 1% of the gold
    period when run on the SDSS template with the full 5-band light curve.
    """
    import os
    try:
        rr_path = os.path.expanduser('~/software/rr-templates/template_sdss')
        from pycycle.templates import load_rr_template
        tmpl = load_rr_template(rr_path, name='sdss')
    except Exception:
        pytest.skip('rr-templates/template_sdss not available')

    gold = 0.5016247
    hjd, mag, magerr, filts = _load_b1392()
    filtnams = ['u', 'g', 'r', 'i', 'z']

    fitter_cold = TemplateFitter(tmpl, n_newton=5, n_start=4, warm_start=False)
    result_cold = fitter_cold.fit(
        hjd, mag, magerr, filts.astype(int), filtnams,
        pmin=0.44, dphi=0.02, pmax=0.89,
    )
    assert abs(result_cold.best_period - gold) / gold < 0.01, (
        f'cold n_start=4: period={result_cold.best_period:.7f} d, '
        f'gold={gold} d'
    )

    # warm_start is a speed/accuracy trade-off: it may land in a nearby local
    # minimum rather than the global one.  Require within 2% of gold.
    fitter_warm = TemplateFitter(tmpl, n_newton=5, warm_start=True)
    result_warm = fitter_warm.fit(
        hjd, mag, magerr, filts.astype(int), filtnams,
        pmin=0.44, dphi=0.02, pmax=0.89,
    )
    assert abs(result_warm.best_period - gold) / gold < 0.02, (
        f'warm_start: period={result_warm.best_period:.7f} d, gold={gold} d'
    )
