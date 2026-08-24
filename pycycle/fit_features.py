"""Post-fit diagnostics that judge whether a template fit is believable.

These exist because the natural quality metric -- ``|P_fit - P_true|/P_true``
-- needs the truth and so can never flag a candidate in real data.  Measured on
labelled mocks, the problem splits in two:

*Is it variable, or is it noise?*  ``r2 = 1 - RSS_best / RSS_flat`` separates
real variables from flat stars essentially perfectly (AUC 1.000, zero false
positives out of 1,000 flat stars at ``r2 >= 0.5``) while keeping 98% of correct
detections.  ``RSS_best`` is already computed by the fit; ``RSS_flat`` is one
extra pass over the data.

*Is the period right?*  Harder, and variance-explained statistics cannot do it:
a real RR Lyrae fitted at the wrong period genuinely is variable and the
template genuinely does explain much of its scatter.  The features here that
target period correctness specifically are

``peak_ratio``
    How much better the best period is than the best *well-separated*
    alternative.  A unique minimum scores high; one tooth of an alias comb
    scores near zero.
``phase_scatter``
    Fit the phase independently in each band at the fixed best period and
    measure the circular scatter.  A true period phases every band coherently;
    a spurious one does not.
``amp_ratio``
    Weakest-to-strongest per-band amplitude.  RRab amplitude falls
    monotonically from g to z, so a physical fit lands in a narrow range while
    noise scatters uniformly.
``win_power`` / ``win_pct``
    The spectral window power at the fitted frequency.  Ground-based cadence
    puts most of its window power on periods commensurate with a day, and a
    period sitting on such a peak is a property of the observing calendar
    rather than of the star.  Recorded per object so the alias structure can be
    modelled generally instead of vetoed field by field.

Everything here is recorded, never used to drop a row: the point is to let a
downstream classifier weigh these against each other on labelled data.
"""
from __future__ import annotations

import numpy as np

from .template_fit import _interp_template, fit_weights

__all__ = ['flat_rss', 'peak_features', 'window_features', 'band_fold_features',
           'fit_features']


# ---------------------------------------------------------------------------
# Variability: how much of the scatter does the fold explain?
# ---------------------------------------------------------------------------

def flat_rss(mag, magerr, filts, use_errors: bool = True) -> float:
    """Weighted RSS of the null model: one constant per band.

    This is the denominator of ``r2``.  It uses exactly the weights the
    template fit used, so the ratio of the two RSS values is meaningful.

    Returns
    -------
    float
        ``nan`` if there is nothing to fit.
    """
    mag = np.asarray(mag, dtype=float)
    if mag.size == 0:
        return float('nan')
    w = fit_weights(magerr, use_errors=use_errors)
    filts = np.asarray(filts).astype(str)

    rss = 0.0
    for b in np.unique(filts):
        s = filts == b
        wb = w[s]
        sw = wb.sum()
        if sw <= 0:
            continue
        mu = np.sum(wb * mag[s]) / sw
        rss += float(np.sum(wb * (mag[s] - mu) ** 2))
    return rss


def _r2(rss_best, rss_flat):
    """``1 - RSS_best / RSS_flat``, guarded against a degenerate denominator."""
    if not np.isfinite(rss_flat) or rss_flat <= 0:
        return float('nan')
    return float(1.0 - rss_best / rss_flat)


# ---------------------------------------------------------------------------
# Period uniqueness: is the minimum a peak, or one tooth of a comb?
# ---------------------------------------------------------------------------

def peak_features(periods, rss, rss_flat, sep_frac: float = 0.01) -> dict:
    """Compare the best period against the best well-separated alternative.

    Parameters
    ----------
    periods, rss : ndarray
        The fit's period grid and its weighted RSS, as returned by
        :class:`~pycycle.template_fit.TemplateFitResult`.
    rss_flat : float
        From :func:`flat_rss`, used to put the RSS gap on a scale-free footing.
    sep_frac : float
        Fractional period separation that counts as "a different period".  The
        default 1% matches the tolerance used to call a recovered period
        correct, so the runner-up is a genuine competitor rather than a
        neighbouring grid point on the same peak.

    Returns
    -------
    dict
        ``period_2nd``, ``rss_2nd``, ``peak_ratio``.

        ``peak_ratio = (RSS_2nd - RSS_best) / RSS_flat`` -- the extra variance
        the runner-up fails to explain, as a fraction of the total.  Zero means
        the two periods fit equally well and the choice between them is
        arbitrary.
    """
    periods = np.asarray(periods, dtype=float)
    rss = np.asarray(rss, dtype=float)
    out = {'period_2nd': float('nan'), 'rss_2nd': float('nan'),
           'peak_ratio': float('nan')}
    good = np.isfinite(rss) & np.isfinite(periods)
    if good.sum() < 2:
        return out

    p, r = periods[good], rss[good]
    k = int(np.argmin(r))
    p_best, r_best = p[k], r[k]

    far = np.abs(p / p_best - 1.0) > sep_frac
    if not far.any():
        return out
    j = int(np.argmin(r[far]))
    out['period_2nd'] = float(p[far][j])
    out['rss_2nd'] = float(r[far][j])
    if np.isfinite(rss_flat) and rss_flat > 0:
        out['peak_ratio'] = float((out['rss_2nd'] - r_best) / rss_flat)
    return out


# ---------------------------------------------------------------------------
# The observing window
# ---------------------------------------------------------------------------

def window_features(hjd, freq, freq_grid=None) -> dict:
    """Spectral window power at the fitted frequency.

    The spectral window ``W(f) = |sum_j exp(2*pi*i*f*t_j)|^2 / N^2`` is a
    property of the observing calendar alone -- it knows nothing about the star.
    It is 1 at ``f = 0`` and has strong secondary peaks wherever the cadence is
    periodic, which for any ground-based survey means frequencies commensurate
    with a solar or sidereal day.  A fitted period sitting on one of those peaks
    is the calendar talking.

    Parameters
    ----------
    hjd : ndarray
        Epoch times [days] actually used in the fit.
    freq : float
        Fitted frequency [cycles/day].
    freq_grid : ndarray, optional
        Frequencies to compare against, for the percentile.  Defaults to a
        uniform grid spanning the fit's plausible range.

    Returns
    -------
    dict
        ``win_power``  -- ``W(freq)``, in [0, 1].
        ``win_pct``    -- fraction of ``freq_grid`` with lower power, so ~1
                          means the fit sits on top of a window peak.
        ``win_max``    -- the largest ``W`` on the grid, excluding near-zero
                          frequency; how alias-prone this cadence is at all.
    """
    t = np.asarray(hjd, dtype=float)
    out = {'win_power': float('nan'), 'win_pct': float('nan'),
           'win_max': float('nan')}
    n = t.size
    if n < 2 or not np.isfinite(freq):
        return out

    t = t - t.mean()          # only the phase pattern matters

    def W(f):
        f = np.atleast_1d(np.asarray(f, dtype=float))
        ph = 2.0 * np.pi * np.outer(f, t)
        return ((np.cos(ph).sum(axis=1) ** 2 + np.sin(ph).sum(axis=1) ** 2)
                / float(n * n))

    out['win_power'] = float(W(freq)[0])

    if freq_grid is None:
        # span the RRab-ish range generously; the percentile only needs a fair
        # reference population, not the fit's exact grid, so keep it small --
        # this is an (n_freq x n_epoch) outer product per object
        freq_grid = np.linspace(0.5, 4.0, 512)
    fg = np.asarray(freq_grid, dtype=float)
    fg = fg[np.isfinite(fg) & (np.abs(fg) > 1e-6)]
    if fg.size == 0:
        return out
    wg = W(fg)
    out['win_pct'] = float(np.mean(wg < out['win_power']))
    out['win_max'] = float(np.max(wg))
    return out


# ---------------------------------------------------------------------------
# Per-band coherence at the fitted period
# ---------------------------------------------------------------------------

def band_fold_features(hjd, mag, magerr, filts, template, period, phi,
                       use_errors: bool = True, n_phase: int = 64) -> dict:
    """Refit phase and amplitude independently per band at a fixed period.

    The multiband solver shares one phase and one amplitude across all bands, so
    it cannot tell you whether the bands actually agree.  They do agree for a
    real variable at the right period, and they do not for a spurious one, which
    makes this the one cheap handle on period correctness that variance-explained
    statistics do not provide.

    Only bands with at least four epochs are used; fewer than two such bands
    leaves the scatter undefined rather than zero.

    Returns
    -------
    dict
        ``phase_scatter`` -- circular standard deviation of the per-band phase
        offsets, in cycles.  0 means perfect agreement, ~0.29 is the value for
        phases drawn uniformly at random.

        ``amp_ratio`` -- weakest over strongest per-band amplitude.  Negative
        when some band prefers an inverted template, which is unphysical.

        ``n_band_fit`` -- how many bands contributed.
    """
    out = {'phase_scatter': float('nan'), 'amp_ratio': float('nan'),
           'n_band_fit': 0}
    t = np.asarray(hjd, dtype=float)
    m = np.asarray(mag, dtype=float)
    filts = np.asarray(filts).astype(str)
    if t.size == 0 or not np.isfinite(period) or period <= 0:
        return out

    w = fit_weights(magerr, use_errors=use_errors)
    freq = 1.0 / period
    grid = (np.arange(n_phase) / n_phase) + float(phi)

    phases, amps = [], []
    for b in np.unique(filts):
        s = filts == b
        if s.sum() < 4:
            continue
        try:
            bi = template.band_index(b)
        except ValueError:
            continue
        tb, mb, wb = t[s], m[s], w[s]

        # All trial phases at once: G is (n_phase, n_epoch_in_band).  The
        # closed-form (mu_b, A_b) solution then reduces to five weighted sums
        # per row, so the whole per-band scan is a handful of matrix products
        # rather than a Python loop over phases -- this keeps the diagnostic at
        # a few percent of the fit it is describing.
        ph = (freq * tb[None, :] + grid[:, None]) % 1.0
        bidx = np.full(ph.size, bi, dtype=int)
        G = _interp_template(template.gamma, bidx, ph.ravel()).reshape(ph.shape)

        sw = float(wb.sum())
        swm = float(np.sum(wb * mb))
        swg = G @ wb
        swgg = (G * G) @ wb
        swgm = G @ (wb * mb)

        det = sw * swgg - swg * swg
        ok = np.abs(det) > 1e-20
        if not ok.any():
            continue
        A = np.where(ok, (sw * swgm - swg * swm) / np.where(ok, det, 1.0), np.nan)
        mu = np.where(ok, (swm - A * swg) / sw, np.nan)
        # RSS = sum w*(m - mu - A g)^2, expanded so G is touched only above
        rss = (np.sum(wb * mb * mb) + mu * mu * sw + A * A * swgg
               - 2.0 * mu * swm - 2.0 * A * swgm + 2.0 * mu * A * swg)
        rss = np.where(ok, rss, np.inf)

        k = int(np.argmin(rss))
        if np.isfinite(rss[k]):
            phases.append(float(grid[k] % 1.0))
            amps.append(float(A[k]))

    out['n_band_fit'] = len(phases)
    if len(phases) < 2:
        return out

    # circular scatter: |mean resultant| -> std, in cycles
    ang = 2.0 * np.pi * np.asarray(phases)
    R = np.hypot(np.cos(ang).mean(), np.sin(ang).mean())
    R = min(max(R, 1e-12), 1.0)
    out['phase_scatter'] = float(np.sqrt(-2.0 * np.log(R)) / (2.0 * np.pi))

    a = np.asarray(amps, dtype=float)
    amax = np.max(np.abs(a))
    if amax > 0:
        # signed: a negative ratio means a band flipped the template
        out['amp_ratio'] = float(a[np.argmin(np.abs(a))] / amax)
    return out


# ---------------------------------------------------------------------------
# Everything, in one call
# ---------------------------------------------------------------------------

def fit_features(tf_result, sep_frac: float = 0.01,
                 use_errors: bool = True, per_band: bool = True,
                 window: bool = True) -> dict:
    """All post-fit diagnostics for one :class:`TemplateFitResult`.

    Reads the light curve back off the result object, so it needs no arguments
    beyond the fit itself.  When the fit was refined on a narrow grid, the
    coarse periodogram attached as ``.coarse`` is used for the peak-uniqueness
    features -- the refined grid spans far less than ``sep_frac`` and would
    report no competing period at all.

    Parameters
    ----------
    tf_result : TemplateFitResult
    sep_frac : float
        Passed to :func:`peak_features`.
    use_errors : bool
        Must match the fit's own setting, or the RSS values are not comparable.
    per_band, window : bool
        Skip the per-band refit or the spectral window if you do not need them.
        Both are cheap next to the fit, but the per-band refit is the more
        expensive of the two.

    Returns
    -------
    dict
        ``rss_flat``, ``r2``, ``r2_2nd``, plus everything from
        :func:`peak_features`, :func:`window_features` and
        :func:`band_fold_features` when enabled.
    """
    hjd = tf_result._hjd
    mag = tf_result._mag
    magerr = tf_result._magerr
    filts = tf_result._filts

    rss_flat = flat_rss(mag, magerr, filts, use_errors=use_errors)
    rss_best = float(np.min(tf_result.rss))

    out = {'rss_flat': rss_flat, 'r2': _r2(rss_best, rss_flat)}

    # the coarse grid is the one that spans competing periods
    grid_src = getattr(tf_result, 'coarse', None) or tf_result
    pk = peak_features(grid_src.periods, grid_src.rss, rss_flat,
                       sep_frac=sep_frac)
    out.update(pk)
    out['r2_2nd'] = _r2(pk['rss_2nd'], rss_flat)

    if window:
        out.update(window_features(hjd, 1.0 / tf_result.best_period
                                   if tf_result.best_period else float('nan')))
    if per_band:
        out.update(band_fold_features(hjd, mag, magerr, filts,
                                      tf_result.template,
                                      tf_result.best_period,
                                      tf_result.best_phi,
                                      use_errors=use_errors))
    return out
