"""Unit tests for pycycle.dp2 (Rubin DP2 / LSDB integration).

These exercise the per-object path and the ``map_partitions`` function against
synthetic nested frames, so they run without lsdb or catalog access.
"""
import numpy as np
import pandas as pd
import pytest

from pycycle.dp2 import (
    DP2Config,
    RRAB_PMIN,
    RRAB_PMAX,
    QUALITY_FLAGS,
    _blank_row,
    clean_epochs,
    fit_lightcurve,
    fit_meta,
    make_dp2_fit_fn,
    nest_columns,
    object_columns,
)
from pycycle.templates import RRTemplate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _template(bands=('g', 'r', 'i', 'z'), n_phase=100):
    """rr-templates-style template with a sawtooth-ish shape over `bands`."""
    phase = np.linspace(0.0, 1.0, n_phase, endpoint=False)
    gamma = np.array([np.sin(2.0 * np.pi * phase) for _ in bands])
    _dust = {'u': 4.799, 'g': 3.665, 'r': 2.464, 'i': 1.804,
             'z': 1.38, 'y': 1.196, 'Y': 1.196}
    dust = np.array([_dust.get(b, 2.0) for b in bands])
    betas = np.zeros((len(bands), 3))
    return RRTemplate(name='test_des', bands=list(bands), phase=phase,
                      gamma=gamma, dust=dust, betas=betas)


def _lc_frame(n_per_band=25, bands=('g', 'r', 'i', 'z'), period=0.55,
              mu=18.0, ebv=0.03, A=0.6, noise=0.01, seed=7,
              magerr_col='psfMagErr_corrected', extra_flags=True):
    """A synthetic DP2-style nested light-curve frame for one object."""
    rng = np.random.default_rng(seed)
    tmpl = _template(bands)
    freq = 1.0 / period
    rows = []
    for bi, b in enumerate(bands):
        t = rng.uniform(60000.0, 60400.0, n_per_band)
        ph = (freq * t) % 1.0
        lo = (ph * 100).astype(int) % 100
        hi = (lo + 1) % 100
        frac = ph * 100 - lo
        g = (1 - frac) * tmpl.gamma[bi, lo] + frac * tmpl.gamma[bi, hi]
        m = mu + ebv * tmpl.dust[bi] + A * g + rng.normal(0, noise, n_per_band)
        for j in range(n_per_band):
            rows.append({'midpointMjdTai': t[j], 'band': b,
                         'psfMag': m[j], magerr_col: noise})
    df = pd.DataFrame(rows)
    if extra_flags:
        for col in QUALITY_FLAGS:
            df[col] = False
    return tmpl, df


def _partition(lcs, nest_col='objectForcedSource', start_id=1000):
    """Assemble per-object light curves into a partition-like frame."""
    return pd.DataFrame(
        {
            'objectId': [start_id + i for i in range(len(lcs))],
            'coord_ra': [61.25] * len(lcs),
            'coord_dec': [-48.46] * len(lcs),
            nest_col: lcs,
        },
        index=pd.Index([10**17 + i for i in range(len(lcs))], name='_healpix_29'),
    )


# ---------------------------------------------------------------------------
# Config / column helpers
# ---------------------------------------------------------------------------

class TestConfig:
    def test_rrab_period_grid_is_default(self):
        cfg = DP2Config()
        assert (cfg.pmin, cfg.pmax) == (RRAB_PMIN, RRAB_PMAX) == (0.44, 0.89)

    def test_default_bands_exclude_u_and_y(self):
        cfg = DP2Config()
        assert cfg.bands == ['g', 'r', 'i', 'z']
        assert 'u' not in cfg.bands and 'y' not in cfg.bands

    def test_defaults_target_dp2_nested_schema(self):
        cfg = DP2Config()
        assert cfg.nest_col == 'objectForcedSource'
        assert cfg.magerr_col == 'psfMagErr_corrected'  # not the raw psfMagErr

    def test_nest_columns_deduplicated_and_minimal(self):
        cols = nest_columns(DP2Config())
        assert len(cols) == len(set(cols))
        for required in ('midpointMjdTai', 'band', 'psfMag', 'psfMagErr_corrected'):
            assert required in cols

    def test_object_columns_include_nest(self):
        cfg = DP2Config()
        assert object_columns(cfg)[-1] == cfg.nest_col

    def test_load_template_without_dir_raises(self):
        with pytest.raises(ValueError, match='template_dir'):
            DP2Config().load_template()


# ---------------------------------------------------------------------------
# clean_epochs
# ---------------------------------------------------------------------------

class TestCleanEpochs:
    def test_returns_coaligned_float_arrays(self):
        _, lc = _lc_frame()
        t, m, me, f = clean_epochs(lc, DP2Config())
        assert t.shape == m.shape == me.shape == f.shape
        assert t.dtype == np.float64 and m.dtype == np.float64
        assert len(t) == 100

    def test_output_is_time_sorted(self):
        _, lc = _lc_frame()
        t, _, _, _ = clean_epochs(lc, DP2Config())
        assert np.all(np.diff(t) >= 0)

    def test_flagged_epochs_removed(self):
        _, lc = _lc_frame()
        lc.loc[:9, 'pixelFlags_saturated'] = True
        t, _, _, _ = clean_epochs(lc, DP2Config())
        assert len(t) == 90

    def test_magerr_cut_applied(self):
        _, lc = _lc_frame()
        lc.loc[:4, 'psfMagErr_corrected'] = 0.5   # above magerr_max=0.2
        lc.loc[5:9, 'psfMagErr_corrected'] = -1.0  # non-positive
        t, _, _, _ = clean_epochs(lc, DP2Config())
        assert len(t) == 90

    def test_nan_magnitudes_removed(self):
        _, lc = _lc_frame()
        lc.loc[:2, 'psfMag'] = np.nan
        t, _, _, _ = clean_epochs(lc, DP2Config())
        assert len(t) == 97

    def test_u_and_y_bands_dropped(self):
        _, lc = _lc_frame(bands=('u', 'g', 'r', 'i', 'z', 'y'))
        _, _, _, f = clean_epochs(lc, DP2Config())
        assert set(f.tolist()) == {'g', 'r', 'i', 'z'}

    def test_y_maps_to_capital_Y_when_requested(self):
        cfg = DP2Config(bands=['g', 'r', 'y'])
        _, lc = _lc_frame(bands=('g', 'r', 'y'))
        _, _, _, f = clean_epochs(lc, cfg)
        assert 'Y' in set(f.tolist()) and 'y' not in set(f.tolist())

    def test_undersampled_band_dropped(self):
        cfg = DP2Config(min_band_epochs=5)
        _, lc = _lc_frame(bands=('g', 'r', 'i'))
        lc = pd.concat([lc, lc[lc.band == 'i'].head(2).assign(band='z')])
        _, _, _, f = clean_epochs(lc, cfg)
        assert 'z' not in set(f.tolist())

    def test_empty_lightcurve_returns_empty(self):
        _, lc = _lc_frame()
        t, m, me, f = clean_epochs(lc.iloc[0:0], DP2Config())
        assert len(t) == len(m) == len(me) == len(f) == 0

    def test_none_lightcurve_returns_empty(self):
        t, _, _, _ = clean_epochs(None, DP2Config())
        assert len(t) == 0

    def test_falls_back_to_flux_when_mag_absent(self):
        cfg = DP2Config()
        _, lc = _lc_frame()
        flux = 10.0 ** ((31.4 - lc['psfMag']) / 2.5)
        lc = lc.drop(columns=['psfMag', 'psfMagErr_corrected'])
        lc['psfFlux'] = flux
        lc['psfFluxErr'] = 0.005 * flux
        t, m, _, _ = clean_epochs(lc, cfg)
        assert len(t) == 100 and np.all(np.isfinite(m))

    def test_missing_mag_and_flux_raises(self):
        _, lc = _lc_frame()
        lc = lc.drop(columns=['psfMag', 'psfMagErr_corrected'])
        with pytest.raises(KeyError):
            clean_epochs(lc, DP2Config())

    def test_flag_cols_all_autodetects(self):
        cfg = DP2Config(flag_cols='all')
        _, lc = _lc_frame()
        lc['some_other_flag'] = False
        lc.loc[:4, 'some_other_flag'] = True
        t, _, _, _ = clean_epochs(lc, cfg)
        assert len(t) == 95

    def test_absent_flag_columns_are_skipped(self):
        """A DP1-era light curve lacking DP2 flags must not blow up."""
        _, lc = _lc_frame(extra_flags=False)
        t, _, _, _ = clean_epochs(lc, DP2Config())
        assert len(t) == 100


# ---------------------------------------------------------------------------
# fit_lightcurve
# ---------------------------------------------------------------------------

class TestFitLightcurve:
    def test_recovers_synthetic_rrab_period(self):
        tmpl, lc = _lc_frame(period=0.55, n_per_band=30, noise=0.01)
        cfg = DP2Config(run_period_search=False)
        row = fit_lightcurve(*clean_epochs(lc, cfg), tmpl, cfg)
        assert row['status'] == 'ok'
        assert row['tf_period'] == pytest.approx(0.55, rel=0.02)

    def test_period_search_and_template_agree(self):
        tmpl, lc = _lc_frame(period=0.55, n_per_band=30, noise=0.01)
        cfg = DP2Config(run_period_search=True, n_thresh=0)
        row = fit_lightcurve(*clean_epochs(lc, cfg), tmpl, cfg)
        assert np.isfinite(row['ps_period'])
        assert row['period_ratio'] == pytest.approx(1.0, rel=0.05)

    def test_records_epoch_and_band_counts(self):
        tmpl, lc = _lc_frame()
        cfg = DP2Config(run_period_search=False)
        row = fit_lightcurve(*clean_epochs(lc, cfg), tmpl, cfg)
        assert row['n_epochs'] == 100
        assert row['n_bands'] == 4
        assert row['bands'] == 'g,i,r,z'

    def test_coefficients_populated(self):
        tmpl, lc = _lc_frame(mu=18.0, ebv=0.03, A=0.6)
        cfg = DP2Config(run_period_search=False)
        row = fit_lightcurve(*clean_epochs(lc, cfg), tmpl, cfg)
        for key in ('mu', 'EBV', 'A'):
            assert np.isfinite(row[key])

    def test_too_few_epochs_reported_not_dropped(self):
        tmpl, lc = _lc_frame(n_per_band=2)
        cfg = DP2Config(run_period_search=False)
        row = fit_lightcurve(*clean_epochs(lc, cfg), tmpl, cfg)
        assert row['status'] == 'too_few_epochs'
        assert np.isnan(row['tf_period'])

    def test_too_few_bands_reported(self):
        tmpl, lc = _lc_frame(bands=('g',), n_per_band=30)
        cfg = DP2Config(run_period_search=False)
        row = fit_lightcurve(*clean_epochs(lc, cfg), tmpl, cfg)
        assert row['status'] == 'too_few_bands'

    def test_band_absent_from_template_reported_not_raised(self):
        """A band the template lacks must be a reported status, not a crash."""
        tmpl = _template(bands=('g', 'r'))
        cfg = DP2Config(run_period_search=False)
        _, lc = _lc_frame(bands=('g', 'r', 'i', 'z'))
        row = fit_lightcurve(*clean_epochs(lc, cfg), tmpl, cfg)
        assert row['status'] == 'band_mismatch'
        assert 'i' in row['error']

    def test_prefilter_skips_constant_star(self):
        tmpl, lc = _lc_frame(A=0.0, noise=0.01)  # no variability
        cfg = DP2Config(run_period_search=False, prefilter=True)
        row = fit_lightcurve(*clean_epochs(lc, cfg), tmpl, cfg)
        assert row['status'] == 'prefiltered'
        assert np.isnan(row['tf_period'])

    def test_prefilter_passes_variable_star(self):
        tmpl, lc = _lc_frame(A=0.6, noise=0.01)
        cfg = DP2Config(run_period_search=False, prefilter=True)
        row = fit_lightcurve(*clean_epochs(lc, cfg), tmpl, cfg)
        assert row['status'] == 'ok'

    def test_variability_features_always_recorded(self):
        tmpl, lc = _lc_frame()
        cfg = DP2Config(run_period_search=False)
        row = fit_lightcurve(*clean_epochs(lc, cfg), tmpl, cfg)
        assert np.isfinite(row['lchi_med']) and np.isfinite(row['sig_max'])

    def test_return_results_gives_plottable_objects(self):
        tmpl, lc = _lc_frame()
        cfg = DP2Config(run_period_search=True, n_thresh=0)
        row, ps, tf = fit_lightcurve(*clean_epochs(lc, cfg), tmpl, cfg,
                                     return_results=True)
        assert hasattr(ps, 'plot_phased') and hasattr(tf, 'plot_phased')
        assert tf.best_period == row['tf_period']

    def test_chi2_dof_is_reasonable_for_good_fit(self):
        tmpl, lc = _lc_frame(noise=0.01, n_per_band=30)
        cfg = DP2Config(run_period_search=False)
        row = fit_lightcurve(*clean_epochs(lc, cfg), tmpl, cfg)
        assert 0.0 < row['tf_chi2_dof'] < 10.0


class TestPeriodBoundFlag:
    def test_flagged_when_true_period_sits_on_the_bound(self):
        """A star at exactly pmin fits at the edge, which must be flagged."""
        tmpl, lc = _lc_frame(period=0.44, noise=0.01, n_per_band=30)
        cfg = DP2Config(run_period_search=False)  # RRab range starts at 0.44
        row = fit_lightcurve(*clean_epochs(lc, cfg), tmpl, cfg)
        assert row['tf_period'] == pytest.approx(0.44, abs=0.005)
        assert row['at_period_bound'] is True

    def test_refined_period_stays_inside_search_range(self):
        """Refinement must not escape [pmin, pmax] and break the RRab constraint."""
        tmpl, lc = _lc_frame(period=0.44, noise=0.01, n_per_band=30)
        cfg = DP2Config(run_period_search=False, refine=True)
        row = fit_lightcurve(*clean_epochs(lc, cfg), tmpl, cfg)
        assert cfg.pmin <= row['tf_period'] <= cfg.pmax

    def test_not_flagged_for_interior_period(self):
        tmpl, lc = _lc_frame(period=0.62, noise=0.01, n_per_band=30)
        cfg = DP2Config(run_period_search=False)
        row = fit_lightcurve(*clean_epochs(lc, cfg), tmpl, cfg)
        assert row['at_period_bound'] is False

    def test_flag_is_false_when_no_fit_ran(self):
        tmpl, lc = _lc_frame(n_per_band=2)
        cfg = DP2Config(run_period_search=False)
        row = fit_lightcurve(*clean_epochs(lc, cfg), tmpl, cfg)
        assert row['at_period_bound'] is False


class TestRefinement:
    """The coarse dphi grid is too sparse on a long baseline; refinement fixes it."""

    def _row(self, refine, period=0.55):
        tmpl, lc = _lc_frame(period=period, noise=0.01, n_per_band=30)
        cfg = DP2Config(run_period_search=False, refine=refine)
        return fit_lightcurve(*clean_epochs(lc, cfg), tmpl, cfg)

    def test_refinement_improves_period_accuracy(self):
        coarse = self._row(refine=False)
        fine = self._row(refine=True)
        err_coarse = abs(coarse['tf_period'] - 0.55)
        err_fine = abs(fine['tf_period'] - 0.55)
        assert err_fine < err_coarse / 5.0

    def test_refinement_gives_sane_chi2(self):
        assert self._row(refine=True)['tf_chi2_dof'] < 5.0
        assert self._row(refine=False)['tf_chi2_dof'] > 50.0

    def test_coarse_period_recorded_separately(self):
        row = self._row(refine=True)
        assert np.isfinite(row['tf_period_coarse'])
        assert row['tf_period'] != row['tf_period_coarse']

    def test_coarse_result_attached_for_plotting(self):
        tmpl, lc = _lc_frame(period=0.55, noise=0.01, n_per_band=30)
        cfg = DP2Config(run_period_search=False, refine=True)
        _, _, tf = fit_lightcurve(*clean_epochs(lc, cfg), tmpl, cfg,
                                  return_results=True)
        assert hasattr(tf, 'coarse')
        # the coarse grid spans the whole search range; the refined one does not
        assert len(tf.coarse.periods) > len(tf.periods)
        assert tf.coarse.periods.min() == pytest.approx(cfg.pmin, abs=1e-3)

    def test_refinement_never_worsens_rss(self):
        tmpl, lc = _lc_frame(period=0.62, noise=0.02, n_per_band=25)
        cfg = DP2Config(run_period_search=False, refine=True)
        t, m, me, f = clean_epochs(lc, cfg)
        _, _, tf = fit_lightcurve(t, m, me, f, tmpl, cfg, return_results=True)
        assert float(np.min(tf.rss)) <= float(np.min(tf.coarse.rss))

    def test_refine_disabled_leaves_no_coarse_attribute(self):
        tmpl, lc = _lc_frame(period=0.55, noise=0.01, n_per_band=30)
        cfg = DP2Config(run_period_search=False, refine=False)
        _, _, tf = fit_lightcurve(*clean_epochs(lc, cfg), tmpl, cfg,
                                  return_results=True)
        assert not hasattr(tf, 'coarse')


# ---------------------------------------------------------------------------
# meta / map_partitions
# ---------------------------------------------------------------------------

class TestFitMeta:
    def test_meta_matches_blank_row_keys_and_order(self):
        tmpl = _template()
        cfg = DP2Config()
        assert list(fit_meta(tmpl, cfg).columns) == list(_blank_row(tmpl, cfg).keys())

    def test_meta_is_empty(self):
        assert len(fit_meta(_template(), DP2Config())) == 0

    def test_multiband_template_gets_per_band_mu(self):
        phase = np.linspace(0.0, 1.0, 100, endpoint=False)
        gamma = np.array([np.sin(2 * np.pi * phase) for _ in range(2)])
        mb = RRTemplate(name='mb', bands=['g', 'r'], phase=phase, gamma=gamma,
                        dust=None, betas=None)
        cols = list(fit_meta(mb, DP2Config()).columns)
        assert 'mu_g' in cols and 'mu_r' in cols and 'mu' not in cols


class TestMakeDp2FitFn:
    def _fn(self, tmpl, **kw):
        cfg = DP2Config(run_period_search=False, **kw)
        return make_dp2_fit_fn(cfg, template=tmpl)

    def test_returns_callable_and_meta(self):
        fn, meta = self._fn(_template())
        assert callable(fn) and isinstance(meta, pd.DataFrame)

    def test_one_row_per_object(self):
        tmpl, lc = _lc_frame()
        fn, _ = self._fn(tmpl)
        out = fn(_partition([lc, lc.copy(), lc.copy()]))
        assert len(out) == 3

    def test_failures_produce_rows_not_drops(self):
        """The DP1 helper silently drops failures; this one must not."""
        tmpl, good = _lc_frame()
        _, short = _lc_frame(n_per_band=2, seed=3)
        fn, _ = self._fn(tmpl)
        out = fn(_partition([good, short]))
        assert len(out) == 2
        assert set(out['status']) == {'ok', 'too_few_epochs'}

    def test_objectid_taken_from_column_not_index(self):
        tmpl, lc = _lc_frame()
        fn, _ = self._fn(tmpl)
        part = _partition([lc, lc.copy()], start_id=735954534639105918)
        out = fn(part)
        assert out['objectId'].tolist() == part['objectId'].tolist()
        assert out['objectId'].iloc[0] != out.index[0]

    def test_healpix_index_preserved(self):
        tmpl, lc = _lc_frame()
        fn, _ = self._fn(tmpl)
        part = _partition([lc, lc.copy()])
        out = fn(part)
        assert out.index.name == '_healpix_29'
        assert out.index.tolist() == part.index.tolist()

    def test_coordinates_carried_through(self):
        tmpl, lc = _lc_frame()
        fn, _ = self._fn(tmpl)
        out = fn(_partition([lc]))
        assert out['coord_ra'].iloc[0] == pytest.approx(61.25)
        assert out['coord_dec'].iloc[0] == pytest.approx(-48.46)

    def test_dtypes_match_meta(self):
        tmpl, lc = _lc_frame()
        fn, meta = self._fn(tmpl)
        out = fn(_partition([lc, lc.copy()]))
        assert out.dtypes.to_dict() == meta.dtypes.to_dict()

    def test_empty_partition_returns_empty_meta_shaped_frame(self):
        tmpl, _ = _lc_frame()
        fn, meta = self._fn(tmpl)
        out = fn(_partition([]))
        assert len(out) == 0
        assert list(out.columns) == list(meta.columns)

    def test_malformed_lightcurve_reported_per_object(self):
        """One bad object must not take down the whole partition."""
        tmpl, good = _lc_frame()
        bad = good.drop(columns=['psfMag', 'psfMagErr_corrected'])
        fn, _ = self._fn(tmpl)
        out = fn(_partition([good, bad]))
        assert len(out) == 2
        assert out['status'].tolist() == ['ok', 'error']
        assert 'KeyError' in out['error'].iloc[1]

    def test_period_recovered_through_partition_path(self):
        tmpl, lc = _lc_frame(period=0.62, n_per_band=30, noise=0.01)
        fn, _ = self._fn(tmpl)
        out = fn(_partition([lc]))
        assert out['tf_period'].iloc[0] == pytest.approx(0.62, rel=0.02)
