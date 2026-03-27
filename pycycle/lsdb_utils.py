"""Utilities for running pycycle on LSST/Rubin catalogs via LSDB.

Provides helpers for:
- Flux → magnitude conversion (Rubin AB system)
- DES → LSST/Rubin filter corrections (RTN-099)
- Variability feature computation for pre-filtering
- Factory functions that return map_partitions-compatible callables

Example usage (full two-stage pipeline sketch)::

    import lsdb
    from dask.distributed import Client
    from pycycle.templates import load_rr_template
    from pycycle.lsdb_utils import (
        apply_des_to_lsst_correction,
        make_template_fit_fn,
    )

    client = Client(n_workers=8, threads_per_worker=1)

    objects = lsdb.open_catalog('https://data.lsdb.io/hats/dp01_object',
                                 columns=['id', 'ra', 'dec', 'r_psfMag',
                                          'extendedness'])
    sources = lsdb.open_catalog('https://data.lsdb.io/hats/dp01_forced_source',
                                 columns=['objectId', 'midpointMjdTai', 'band',
                                          'psfFlux', 'psfFluxErr'])

    # Point sources fainter than Gaia, within single-epoch depth
    stars = objects.query('extendedness == 0 and 21 < r_psfMag < 24.5')
    joined = stars.nest_sources(sources, source_id_col='objectId')

    template = load_rr_template('~/software/rr-templates/template_des', name='des')
    apply_des_to_lsst_correction(template)   # zero-cost distance correction

    fit_fn, meta = make_template_fit_fn(
        template,
        bands=['g', 'r', 'i', 'z', 'y'],
        pmin=0.44, dphi=0.02, pmax=0.89,
        n_newton=5, warm_start=True,
    )
    results = joined.map_partitions(fit_fn, meta=meta).compute()
"""
from __future__ import annotations

import numpy as np

__all__ = [
    'flux_to_mag',
    'apply_des_to_lsst_correction',
    'compute_variability_features',
    'make_template_fit_fn',
]

# ---------------------------------------------------------------------------
# DES → LSST photometric corrections (RTN-099, 2025-08-22 release)
# Applied to betas c0 term at template load time for a typical RRab
# (g-i ≈ 0.3, r-i ≈ 0.1, i-z ≈ 0.1).  Per-band offsets in magnitudes.
# For period finding these offsets are absorbed into µ; for distances they
# matter at the ~10–30 mmag level (< 1% distance error).
# ---------------------------------------------------------------------------
_DES_TO_LSST_C0 = {
    'g': +0.011,   # g_L = g_D + 0.016*(g-i) - 0.003*(g-i)^2 + 0.006
    'r': +0.028,   # r_L = r_D + 0.185*(r-i) - 0.015*(r-i)^2 + 0.010
    'i': +0.006,   # i_L = i_D + 0.150*(r-i) - 0.003*(r-i)^2 - 0.009
    'z': +0.024,   # z_L = z_D + 0.270*(i-z) + 0.036*(i-z)^2 - 0.003
    'Y': +0.000,   # Y-band very similar between DES and Rubin
    'y': +0.000,   # lower-case alias
}


def flux_to_mag(flux, flux_err, zpt: float = 31.4):
    """Convert Rubin nJy fluxes to AB magnitudes.

    Parameters
    ----------
    flux : array-like
        PSF flux in nJy (as returned by Rubin pipelines for DP1+).
    flux_err : array-like
        PSF flux uncertainty in nJy.
    zpt : float
        AB zero-point offset; default 31.4 (Rubin convention: m = -2.5*log10(f_nJy) + 31.4).

    Returns
    -------
    mag : ndarray
    magerr : ndarray
        Propagated magnitude uncertainty.  Observations with flux ≤ 0 are
        assigned ``magerr = 9.99`` so they are removed by the standard cut
        ``magerr <= 0.2``.
    """
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    pos = flux > 0
    mag = np.where(pos, -2.5 * np.log10(np.where(pos, flux, 1.0)) + zpt, np.nan)
    magerr = np.where(pos,
                      (2.5 / np.log(10)) * np.abs(flux_err) / np.where(pos, flux, 1.0),
                      9.99)
    return mag, magerr


def apply_des_to_lsst_correction(template, mean_colors: dict | None = None) -> None:
    """Adjust DES rr-template beta coefficients for LSST filter differences.

    Modifies ``template.betas[:, 0]`` (the c0 constant term) in-place so
    that distance moduli derived from Rubin photometry are on the correct
    absolute scale.  Has **zero runtime cost** during fitting — the correction
    is baked into the template at load time.

    The offsets are evaluated at the mean RRab color (g-i ≈ 0.3, r-i ≈ 0.1,
    i-z ≈ 0.1) from the RTN-099 synthetic transformations.  For individual
    stars the color-dependent residual is ≲5 mmag (< 0.2% distance error).

    Parameters
    ----------
    template : RRTemplate
        Template loaded via :func:`~pycycle.templates.load_rr_template`.
        Must have ``template.betas`` array (rr-templates mode).
        No-op for Multiband templates (``template.dust is None``).
    mean_colors : dict, optional
        Override the default mean RRab colors used to evaluate the correction.
        Keys: ``'g-i'``, ``'r-i'``, ``'i-z'``.  Values in magnitudes.
        If provided, the full polynomial from RTN-099 is re-evaluated.
    """
    if template.dust is None:
        return  # Multiband template — no betas to correct

    if mean_colors is not None:
        gi = mean_colors.get('g-i', 0.3)
        ri = mean_colors.get('r-i', 0.1)
        iz = mean_colors.get('i-z', 0.1)
        corrections = {
            'g': 0.016 * gi - 0.003 * gi ** 2 + 0.006,
            'r': 0.185 * ri - 0.015 * ri ** 2 + 0.010,
            'i': 0.150 * ri - 0.003 * ri ** 2 - 0.009,
            'z': 0.270 * iz + 0.036 * iz ** 2 - 0.003,
            'Y': 0.0,
            'y': 0.0,
        }
    else:
        corrections = _DES_TO_LSST_C0

    for bi, band in enumerate(template.bands):
        template.betas[bi, 0] += corrections.get(band, 0.0)


def compute_variability_features(lc_df, bands: list[str]) -> dict:
    """Compute variability statistics for RF pre-filtering (following S19 §3.2).

    Features follow Stringer et al. (2019) Table 4:

    - ``lchi_med``: median of log10(chi2_nu) across bands where chi2_nu =
      Var(mag) / mean(sigma^2).  Values >> 0 indicate variability.
    - ``sig_max``: maximum of log10(significance) across bands where
      significance = (max_mag - min_mag) / sqrt(sigma_max^2 + sigma_min^2).

    Pre-filtering thresholds from S19: keep objects with ``lchi_med >= 0.5``
    and ``sig_max >= 0`` (i.e., log10(significance) >= 0 → significance >= 1).

    Parameters
    ----------
    lc_df : pandas.DataFrame
        Light curve with columns ``band``, ``mag``, ``magerr``.
    bands : list of str
        Band names to consider.

    Returns
    -------
    dict with keys ``lchi_med``, ``sig_max``
    """
    chi2_list = []
    sig_list = []
    for band in bands:
        sub = lc_df[lc_df['band'] == band]
        m = sub['mag'].values
        e = sub['magerr'].values
        ok = (e > 0) & np.isfinite(m) & np.isfinite(e)
        m, e = m[ok], e[ok]
        if len(m) < 2:
            continue
        chi2_nu = np.var(m, ddof=1) / np.mean(e ** 2)
        chi2_list.append(np.log10(max(chi2_nu, 1e-10)))
        imin, imax = np.argmin(m), np.argmax(m)
        sig = (m[imax] - m[imin]) / np.sqrt(e[imax] ** 2 + e[imin] ** 2)
        sig_list.append(np.log10(max(sig, 1e-10)))

    return {
        'lchi_med': float(np.median(chi2_list)) if chi2_list else 0.0,
        'sig_max': float(max(sig_list)) if sig_list else 0.0,
    }


def make_template_fit_fn(template, bands: list[str], **fit_kwargs):
    """Return a ``map_partitions``-compatible function and its Dask meta DataFrame.

    The returned function processes one LSDB partition (a pandas DataFrame with
    a nested ``sources`` column) and returns a DataFrame with one row per object.

    Each object's ``sources`` column must be a DataFrame with columns:
    ``midpointMjdTai`` (or ``mjd``), ``band``, ``psfFlux``, ``psfFluxErr``.

    Parameters
    ----------
    template : RRTemplate
        Pre-loaded template (apply :func:`apply_des_to_lsst_correction` beforehand
        if needed).
    bands : list of str
        Band names matching the template (e.g. ``['g','r','i','z','y']``).
    **fit_kwargs
        Passed to :meth:`~pycycle.template_fit.TemplateFitter.fit`
        (e.g. ``pmin=0.44``, ``dphi=0.02``, ``pmax=0.89``,
        ``n_newton=5``, ``warm_start=True``).

    Returns
    -------
    fn : callable
        Function suitable for ``catalog.map_partitions(fn, meta=meta)``.
    meta : pandas.DataFrame
        Empty DataFrame with the correct output schema for Dask.

    Example
    -------
    ::

        fn, meta = make_template_fit_fn(
            template_des,
            bands=['g', 'r', 'i', 'z', 'y'],
            pmin=0.44, dphi=0.02, pmax=0.89,
            warm_start=True,
        )
        results = joined.map_partitions(fn, meta=meta).compute()
    """
    import pandas as pd
    from pycycle.template_fit import TemplateFitter

    n_newton = fit_kwargs.pop('n_newton', 5)
    n_start = fit_kwargs.pop('n_start', 4)
    warm_start = fit_kwargs.pop('warm_start', False)

    band_map = {b: i for i, b in enumerate(bands)}

    def _fit_partition(df):
        fitter = TemplateFitter(template, n_newton=n_newton, n_start=n_start,
                                warm_start=warm_start)
        rows = []
        for _, obj in df.iterrows():
            lc = obj['sources']
            # support both 'midpointMjdTai' (DP1) and 'mjd' column names
            time_col = 'midpointMjdTai' if 'midpointMjdTai' in lc.columns else 'mjd'
            mag, magerr = flux_to_mag(lc['psfFlux'].values, lc['psfFluxErr'].values)
            bidx_raw = lc['band'].map(band_map)
            valid = (magerr > 0) & (magerr <= 0.2) & bidx_raw.notna()
            if valid.sum() < 10:
                continue
            t = lc[time_col].values[valid]
            m = mag[valid]
            me = magerr[valid]
            bi = bidx_raw[valid].astype(int).values
            try:
                res = fitter.fit(t, m, me, bi, bands, **fit_kwargs)
            except Exception:
                continue
            row = {'id': obj.name,
                   'period': res.best_period,
                   'rss': float(res.rss.min())}
            row.update({k: float(v) for k, v in res.best_coeffs.items()})
            rows.append(row)
        return pd.DataFrame(rows)

    # build meta from an example with the standard rr-templates coefficients
    _has_dust = template.dust is not None
    if _has_dust:
        meta_cols = {'id': np.int64, 'period': np.float64, 'rss': np.float64,
                     'mu': np.float64, 'EBV': np.float64, 'A': np.float64}
    else:
        meta_cols = {'id': np.int64, 'period': np.float64, 'rss': np.float64,
                     'A': np.float64}
        for band in template.bands:
            meta_cols[f'mu_{band}'] = np.float64
    meta = pd.DataFrame({k: pd.Series(dtype=v) for k, v in meta_cols.items()})

    return _fit_partition, meta
