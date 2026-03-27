"""Unit tests for pycycle.lsdb_utils."""
import numpy as np
import pandas as pd
import pytest

from pycycle.lsdb_utils import (
    flux_to_mag,
    apply_des_to_lsst_correction,
    compute_variability_features,
    make_template_fit_fn,
)
from pycycle.templates import RRTemplate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rr_template(n_bands=2, n_phase=100):
    phase = np.linspace(0.0, 1.0, n_phase, endpoint=False)
    gamma = np.array([np.sin(2.0 * np.pi * phase) for _ in range(n_bands)])
    bands = ['g', 'r'][:n_bands]
    dust = np.array([3.303, 2.285][:n_bands])
    betas = np.zeros((n_bands, 3))
    return RRTemplate(name='test_rr', bands=bands, phase=phase,
                      gamma=gamma, dust=dust, betas=betas)


def _make_mb_template(n_bands=2, n_phase=100):
    phase = np.linspace(0.0, 1.0, n_phase, endpoint=False)
    gamma = np.array([np.sin(2.0 * np.pi * phase) for _ in range(n_bands)])
    bands = ['g', 'r'][:n_bands]
    return RRTemplate(name='test_mb', bands=bands, phase=phase,
                      gamma=gamma, dust=None, betas=None)


# ---------------------------------------------------------------------------
# flux_to_mag
# ---------------------------------------------------------------------------

class TestFluxToMag:
    def test_roundtrip(self):
        """mag → flux → mag should recover original magnitude to 1e-6."""
        zpt = 31.4
        mag_in = np.array([22.0, 23.5, 24.0])
        flux = 10.0 ** ((zpt - mag_in) / 2.5)  # inverse of the formula
        flux_err = 0.01 * flux
        mag_out, _ = flux_to_mag(flux, flux_err, zpt=zpt)
        np.testing.assert_allclose(mag_out, mag_in, atol=1e-6)

    def test_magerr_propagation(self):
        """magerr = (2.5/ln10) * flux_err/flux."""
        flux = np.array([1000.0])
        flux_err = np.array([10.0])
        _, magerr = flux_to_mag(flux, flux_err)
        expected = (2.5 / np.log(10)) * 10.0 / 1000.0
        np.testing.assert_allclose(magerr[0], expected, rtol=1e-9)

    def test_nonpositive_flux_returns_nan_mag(self):
        """Non-positive flux → mag=nan, magerr=9.99."""
        flux = np.array([0.0, -5.0, 100.0])
        flux_err = np.array([1.0, 1.0, 1.0])
        mag, magerr = flux_to_mag(flux, flux_err)
        assert not np.isfinite(mag[0])
        assert not np.isfinite(mag[1])
        assert magerr[0] == pytest.approx(9.99)
        assert magerr[1] == pytest.approx(9.99)
        assert np.isfinite(mag[2])

    def test_scalar_input(self):
        """Scalar inputs should return scalar-shaped output."""
        mag, magerr = flux_to_mag(500.0, 5.0)
        assert mag.shape == ()
        assert magerr.shape == ()

    def test_custom_zeropoint(self):
        """Changing zpt shifts magnitude by the same amount."""
        flux = np.array([1000.0])
        flux_err = np.array([1.0])
        m31, _ = flux_to_mag(flux, flux_err, zpt=31.4)
        m30, _ = flux_to_mag(flux, flux_err, zpt=30.0)
        np.testing.assert_allclose(m31 - m30, 1.4, atol=1e-9)


# ---------------------------------------------------------------------------
# apply_des_to_lsst_correction
# ---------------------------------------------------------------------------

class TestApplyDesToLsstCorrection:
    def test_shifts_c0_in_place(self):
        """c0 column of betas should be shifted by the expected offsets."""
        tmpl = _make_rr_template(n_bands=2)
        tmpl.bands = ['g', 'r']
        c0_before = tmpl.betas[:, 0].copy()
        apply_des_to_lsst_correction(tmpl)
        expected_shifts = {'g': +0.011, 'r': +0.028}
        for bi, band in enumerate(tmpl.bands):
            assert tmpl.betas[bi, 0] == pytest.approx(
                c0_before[bi] + expected_shifts[band], abs=1e-9
            )

    def test_p1_p2_unchanged(self):
        """p1 and p2 columns should be unchanged."""
        tmpl = _make_rr_template(n_bands=2)
        tmpl.bands = ['g', 'r']
        betas_orig = tmpl.betas.copy()
        apply_des_to_lsst_correction(tmpl)
        np.testing.assert_array_equal(tmpl.betas[:, 1], betas_orig[:, 1])
        np.testing.assert_array_equal(tmpl.betas[:, 2], betas_orig[:, 2])

    def test_noop_for_multiband_template(self):
        """Multiband templates (dust=None) should not be modified."""
        tmpl = _make_mb_template(n_bands=2)
        apply_des_to_lsst_correction(tmpl)  # should be a no-op; betas is None
        assert tmpl.betas is None

    def test_correction_does_not_change_period(self):
        """Applying the correction should not change the recovered period."""
        from pycycle.template_fit import TemplateFitter
        import numpy as np

        rng = np.random.default_rng(7)
        true_period = 0.5016
        freq = 1.0 / true_period
        n = 40
        t = rng.uniform(0, 100, n)
        phase = np.linspace(0, 1, 100, endpoint=False)
        gamma = np.sin(2 * np.pi * phase)

        # Build template
        tmpl = RRTemplate(
            name='corr_test', bands=['g', 'r'],
            phase=phase,
            gamma=np.vstack([gamma, gamma]),
            dust=np.array([3.3, 2.3]),
            betas=np.zeros((2, 3)),
        )
        bidx = rng.integers(0, 2, n).astype(int)
        lo = ((freq * t) % 1.0 * 100).astype(int) % 100
        hi = (lo + 1) % 100
        g = tmpl.gamma[bidx, lo]
        mag_clean = 14.0 + 0.05 * tmpl.dust[bidx] + 0.5 * g
        magerr = 0.01 * np.ones(n)

        fitter_before = TemplateFitter(tmpl, n_newton=5, n_start=4)
        res_before = fitter_before.fit(t, mag_clean, magerr, bidx, ['g', 'r'],
                                        pmin=0.44, dphi=0.02, pmax=0.89)

        # Apply correction and refit
        apply_des_to_lsst_correction(tmpl)
        fitter_after = TemplateFitter(tmpl, n_newton=5, n_start=4)
        res_after = fitter_after.fit(t, mag_clean, magerr, bidx, ['g', 'r'],
                                      pmin=0.44, dphi=0.02, pmax=0.89)

        assert abs(res_before.best_period - res_after.best_period) < 0.001, (
            f'Correction changed period: {res_before.best_period:.7f} → '
            f'{res_after.best_period:.7f}'
        )

    def test_mean_colors_override(self):
        """Providing mean_colors should use the full RTN-099 polynomial."""
        tmpl = _make_rr_template(n_bands=2)
        tmpl.bands = ['g', 'r']
        c0_before = tmpl.betas[:, 0].copy()
        apply_des_to_lsst_correction(tmpl, mean_colors={'g-i': 0.3, 'r-i': 0.1, 'i-z': 0.1})
        # g correction: 0.016*0.3 - 0.003*0.09 + 0.006 = 0.004800 - 0.000270 + 0.006 = 0.010530
        # r correction: 0.185*0.1 - 0.015*0.01 + 0.010 = 0.018500 - 0.000150 + 0.010 = 0.028350
        expected = {'g': 0.016 * 0.3 - 0.003 * 0.09 + 0.006,
                    'r': 0.185 * 0.1 - 0.015 * 0.01 + 0.010}
        for bi, band in enumerate(tmpl.bands):
            assert tmpl.betas[bi, 0] == pytest.approx(
                c0_before[bi] + expected[band], abs=1e-9
            )


# ---------------------------------------------------------------------------
# compute_variability_features
# ---------------------------------------------------------------------------

class TestComputeVariabilityFeatures:
    def _make_lc(self, band, mag, magerr):
        return pd.DataFrame({'band': band, 'mag': mag, 'magerr': magerr})

    def test_constant_star_low_lchi(self):
        """A constant star has chi2_nu ≈ 1 → lchi_med close to 0."""
        rng = np.random.default_rng(0)
        n = 30
        bands = np.array(['g', 'r', 'i', 'z'] * (n // 4))
        sigma = 0.02
        mag = 22.0 + rng.normal(0, sigma, len(bands))
        magerr = sigma * np.ones(len(bands))
        lc = self._make_lc(bands, mag, magerr)
        feats = compute_variability_features(lc, ['g', 'r', 'i', 'z'])
        # chi2_nu ≈ 1 → log10(1) = 0; allow generous tolerance for small N
        assert feats['lchi_med'] < 1.0, (
            f'Expected lchi_med < 1.0 for constant star, got {feats["lchi_med"]}'
        )

    def test_variable_star_high_lchi(self):
        """A strongly variable star has lchi_med >> 0."""
        rng = np.random.default_rng(1)
        n = 40
        bands = np.array(['g', 'r', 'i', 'z'] * (n // 4))
        # large 0.5-mag amplitude variations on top of tiny reported errors
        mag = 22.0 + 0.5 * np.sin(np.linspace(0, 4 * np.pi, len(bands)))
        magerr = 0.01 * np.ones(len(bands))
        lc = self._make_lc(bands, mag, magerr)
        feats = compute_variability_features(lc, ['g', 'r', 'i', 'z'])
        assert feats['lchi_med'] >= 0.5, (
            f'Expected lchi_med >= 0.5 for variable star, got {feats["lchi_med"]}'
        )
        assert feats['sig_max'] >= 0.0

    def test_empty_band_skipped(self):
        """A band with no observations should not crash."""
        lc = pd.DataFrame({
            'band': ['g', 'g', 'g'],
            'mag': [22.0, 22.1, 21.9],
            'magerr': [0.02, 0.02, 0.02],
        })
        feats = compute_variability_features(lc, ['g', 'r'])
        assert np.isfinite(feats['lchi_med'])

    def test_single_obs_band_skipped(self):
        """A band with only 1 observation is skipped (need ≥ 2 for chi2_nu)."""
        lc = pd.DataFrame({
            'band': ['g', 'r', 'r'],
            'mag': [22.0, 21.9, 22.1],
            'magerr': [0.02, 0.02, 0.02],
        })
        feats = compute_variability_features(lc, ['g', 'r'])
        # 'g' has 1 obs → skipped; only 'r' contributes
        assert np.isfinite(feats['lchi_med'])

    def test_all_bands_missing_returns_zeros(self):
        """If no band has ≥ 2 observations, return default zeros."""
        lc = pd.DataFrame({'band': ['g'], 'mag': [22.0], 'magerr': [0.02]})
        feats = compute_variability_features(lc, ['g', 'r'])
        assert feats['lchi_med'] == 0.0
        assert feats['sig_max'] == 0.0


# ---------------------------------------------------------------------------
# make_template_fit_fn
# ---------------------------------------------------------------------------

class TestMakeTemplateFitFn:
    def test_returns_callable_and_meta(self):
        tmpl = _make_rr_template(n_bands=2)
        fn, meta = make_template_fit_fn(tmpl, bands=['g', 'r'])
        assert callable(fn)
        assert isinstance(meta, pd.DataFrame)

    def test_meta_has_required_columns_rr(self):
        """rr-templates meta should have id, period, rss, mu, EBV, A."""
        tmpl = _make_rr_template(n_bands=2)
        _, meta = make_template_fit_fn(tmpl, bands=['g', 'r'])
        for col in ['id', 'period', 'rss', 'mu', 'EBV', 'A']:
            assert col in meta.columns, f'Missing column: {col}'

    def test_meta_has_required_columns_mb(self):
        """Multiband meta should have id, period, rss, mu_g, mu_r, A."""
        tmpl = _make_mb_template(n_bands=2)
        _, meta = make_template_fit_fn(tmpl, bands=['g', 'r'])
        for col in ['id', 'period', 'rss', 'mu_g', 'mu_r', 'A']:
            assert col in meta.columns, f'Missing column: {col}'

    def test_fit_fn_on_synthetic_partition(self):
        """fit_fn should return a DataFrame with the correct columns when given valid data."""
        rng = np.random.default_rng(42)
        tmpl = _make_rr_template(n_bands=2)

        true_period = 0.55
        freq = 1.0 / true_period
        n = 30
        t = rng.uniform(0, 100, n)
        bands_arr = np.array(['g', 'r'] * (n // 2))
        bidx = np.array([tmpl.bands.index(b) for b in bands_arr])
        ph = (freq * t) % 1.0
        lo = (ph * 100).astype(int) % 100
        hi = (lo + 1) % 100
        frac = ph * 100 - lo
        g = (1 - frac) * tmpl.gamma[bidx, lo] + frac * tmpl.gamma[bidx, hi]
        # Convert to flux (nJy) for the pipeline: m = -2.5*log10(f) + 31.4
        mag_true = 14.0 + 0.05 * tmpl.dust[bidx] + 0.4 * g
        flux = 10.0 ** ((31.4 - mag_true) / 2.5)
        flux_err = 0.02 * flux

        sources = pd.DataFrame({
            'midpointMjdTai': t,
            'band': bands_arr,
            'psfFlux': flux,
            'psfFluxErr': flux_err,
        })
        # Build a fake partition with one object
        partition = pd.DataFrame({'sources': [sources]}, index=[0])

        fn, meta = make_template_fit_fn(
            tmpl, bands=['g', 'r'],
            pmin=0.44, dphi=0.05, pmax=0.89,
            n_newton=3,
        )
        result_df = fn(partition)
        assert isinstance(result_df, pd.DataFrame)
        # Should have at least the meta columns
        for col in meta.columns:
            assert col in result_df.columns, f'Missing output column: {col}'

    def test_fit_fn_skips_too_few_obs(self):
        """Objects with < 10 valid observations should be skipped (not error)."""
        tmpl = _make_rr_template(n_bands=2)
        sources = pd.DataFrame({
            'midpointMjdTai': np.linspace(0, 10, 5),
            'band': ['g', 'r', 'g', 'r', 'g'],
            'psfFlux': 1000.0 * np.ones(5),
            'psfFluxErr': 10.0 * np.ones(5),
        })
        partition = pd.DataFrame({'sources': [sources]}, index=[0])
        fn, meta = make_template_fit_fn(tmpl, bands=['g', 'r'])
        result_df = fn(partition)
        assert len(result_df) == 0  # skipped due to < 10 valid obs
