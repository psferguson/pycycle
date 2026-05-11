"""Search all Fink/LSST alert light curves for RR Lyrae using pycycle template fitting.

Fits each object against all 136 individual RRab templates from Baeza-Villagra et al. (2025)
and reports the best-fitting template.

Usage
-----
    conda run -n pycycle python search_rrl_fink.py [OPTIONS]

Options
-------
    --input        PATH   Input parquet (default: see INPUT below)
    --output       PATH   Output parquet (default: rrl_candidates.parquet)
    --workers      N      Parallel worker processes (default: cpu_count)
    --pmin         FLOAT  Minimum period in days (default: 0.44)
    --pmax         FLOAT  Maximum period in days (default: 0.89)
    --dphi         FLOAT  Phase resolution (default: 0.02)
    --min-obs      N      Minimum good observations (default: 10)
    --no-varfilter        Disable variability pre-filter

Pipeline
--------
1. Parse light curves from prvDiaSources (scienceFlux, bands g/r/i).
2. Quality cut: magerr in (0, 0.2].
3. Variability pre-filter (S19 §3.2): lchi_med ≥ 0.5 and sig_max ≥ 0.
4. Fit all 136 RRab templates; keep the result with the lowest RSS.
5. Save all fit results; add quality flags for post-hoc filtering.

Template library
----------------
Baeza-Villagra et al. (2025, A&A, arXiv:2501.03813) — 136 individual RRab DECam griz
templates normalised to [0, 1].  The gri subset is used here to match the Fink alert bands.

Clone: https://github.com/KarinaBaezaV/Multiband-templates

Quality notes
-------------
The Rubin AP pipeline reports very small scienceFluxErr (photon-noise only),
leading to chi²/dof >> 1 even for good template fits. We therefore report
two metrics that are robust to underestimated errors:

  rss_frac   = RSS_template / RSS_null   [0=perfect, 1=template doesn't help]
               where RSS_null = weighted variance of constant-magnitude model.
               Typical RRAB ~0.3–0.6; random objects vary widely.

  rss_depth  = 1 - RSS_min / median(RSS)   [0=no preferred period, 1=clear minimum]
               Measures how much the best period stands out.
               Typical RRAB ≥ 0.5; random or non-periodic objects < 0.5.

Output columns
--------------
    diaObjectId   int64   Object identifier
    period        float   Best-fit period (days)
    template_id   str     OGLE ID of the best-fitting template
    mu_g          float   Mean g-band magnitude from best fit
    mu_r          float   Mean r-band magnitude from best fit
    mu_i          float   Mean i-band magnitude from best fit
    A             float   Template amplitude (physical range: 0.3–1.5 for RRab)
    phi           float   Phase at first observation epoch
    n_good        int     Number of good observations used
    lchi_med      float   Variability: median log10(chi2_nu) across bands
    sig_max       float   Variability: max log10(significance) across bands
    rss_frac      float   RSS_template / RSS_null (lower = better)
    rss_depth     float   1 - RSS_min / median(RSS) (higher = clearer period)
    flag_amp      bool    True if |A| outside [0.3, 1.5] mag
    flag_depth    bool    True if rss_depth < 0.5 (no clear period minimum)
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT       = '/astro/store/shire/pferguso/alert_sprint/concatenated_catalog/latest_obs.parquet'
MULTIBAND_ZIP = os.path.expanduser('~/software/Multiband-templates/RRab_normalized.zip')
BANDS       = ['g', 'r', 'i']

# Quality thresholds
AMP_MIN   = 0.3    # minimum RRab amplitude (mag)
AMP_MAX   = 1.5    # maximum RRab amplitude (mag)
DEPTH_MIN = 0.5    # minimum RSS depth to consider period significant


# ---------------------------------------------------------------------------
# Worker initializer — load all templates once per process
# ---------------------------------------------------------------------------

def _init_worker(zip_path: str) -> None:
    """Load all 136 RRab templates per worker process."""
    global _FITTERS, _TMPL_BAND_SETS, _TMPL_NAMES

    from pycycle.templates import load_multiband_templates, RRTemplate
    from pycycle.template_fit import TemplateFitter

    all_tmpl = load_multiband_templates(zip_path)

    _FITTERS        = []
    _TMPL_BAND_SETS = []  # list of set(tmpl.bands) for fast membership tests
    _TMPL_NAMES     = []

    for tmpl in all_tmpl:
        # Only keep templates that share at least 2 bands with our BANDS
        shared = set(tmpl.bands) & set(BANDS)
        if len(shared) < 2:
            continue

        _FITTERS.append(TemplateFitter(tmpl, n_newton=5, warm_start=True))
        _TMPL_BAND_SETS.append(set(tmpl.bands))
        _TMPL_NAMES.append(tmpl.name)


def _fit_one(args: tuple) -> dict | None:
    """Fit a single object against all templates; called in a worker process."""
    (obj_id, t, m, me, bands_arr, n_good, lchi_med, sig_max, fit_kwargs) = args

    # Null (constant) model RSS — weighted variance
    w        = 1.0 / np.maximum(me ** 2, 1e-30)
    m_wmean  = np.dot(w, m) / w.sum()
    rss_null = float(np.dot(w, (m - m_wmean) ** 2))

    best_rss_min  = np.inf
    best_result   = None
    best_tmpl_idx = -1

    for tmpl_idx, (fitter, tmpl_bands) in enumerate(zip(_FITTERS, _TMPL_BAND_SETS)):
        valid = np.array([b in tmpl_bands for b in bands_arr])
        if valid.sum() < 5:
            continue

        try:
            result = fitter.fit(
                t[valid], m[valid], me[valid], bands_arr[valid],
                **fit_kwargs,
            )
        except Exception:
            continue

        rss_min = float(result.rss.min())
        if rss_min < best_rss_min:
            best_rss_min  = rss_min
            best_result   = result
            best_tmpl_idx = tmpl_idx

    if best_result is None:
        warnings.warn(f'diaObjectId={obj_id}: all template fits failed', stacklevel=2)
        return None

    rss_frac  = best_rss_min / max(rss_null, 1e-30)
    rss_depth = float(1.0 - best_rss_min / max(np.median(best_result.rss), 1e-30))

    coeffs = best_result.best_coeffs
    mu_g = float(coeffs.get('mu_g', np.nan))
    mu_r = float(coeffs.get('mu_r', np.nan))
    mu_i = float(coeffs.get('mu_i', np.nan))
    A    = float(coeffs.get('A', np.nan))
    phi  = float(best_result.best_phi)

    return {
        'diaObjectId': obj_id,
        'period':      float(best_result.best_period),
        'template_id': _TMPL_NAMES[best_tmpl_idx],
        'mu_g':        mu_g,
        'mu_r':        mu_r,
        'mu_i':        mu_i,
        'A':           A,
        'phi':         phi,
        'n_good':      n_good,
        'lchi_med':    lchi_med,
        'sig_max':     sig_max,
        'rss_frac':    rss_frac,
        'rss_depth':   rss_depth,
        'flag_amp':    not (AMP_MIN <= abs(A) <= AMP_MAX),
        'flag_depth':  rss_depth < DEPTH_MIN,
    }


# ---------------------------------------------------------------------------
# Light-curve parsing (main process)
# ---------------------------------------------------------------------------

def parse_light_curve(srcs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract (t, mag, magerr, band) from prvDiaSources array.

    Uses ``scienceFlux`` / ``scienceFluxErr`` (direct science-image fluxes).
    Keeps only observations in BANDS with magerr in (0, 0.2].
    """
    from pycycle.lsdb_utils import flux_to_mag

    n = len(srcs)
    if n == 0:
        return np.empty(0), np.empty(0), np.empty(0), np.empty(0, dtype='U2')

    t_all   = np.fromiter((s['midpointMjdTai'] for s in srcs), float, n)
    sf_all  = np.fromiter((s['scienceFlux']    for s in srcs), float, n)
    sfe_all = np.fromiter((s['scienceFluxErr'] for s in srcs), float, n)
    b_all   = np.array([s['band'] for s in srcs])

    mag, magerr = flux_to_mag(sf_all, sfe_all)

    in_band = np.array([b in BANDS for b in b_all])
    good = (magerr > 0) & (magerr <= 0.2) & in_band
    return t_all[good], mag[good], magerr[good], b_all[good]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_work_items(
    df: pd.DataFrame,
    min_obs: int,
    varfilter: bool,
    fit_kwargs: dict,
) -> list[tuple]:
    """Parse all rows into work items for the worker pool."""
    from pycycle.lsdb_utils import compute_variability_features

    items      = []
    n_skip_obs = 0
    n_skip_var = 0

    for _, row in df.iterrows():
        obj_id = int(row['diaObjectId'])
        srcs   = row['prvDiaSources']

        t, m, me, b = parse_light_curve(srcs)
        n_good = len(t)

        if n_good < min_obs:
            n_skip_obs += 1
            continue

        if varfilter:
            lc_df = pd.DataFrame({
                'band':   b,
                'mag':    m,
                'magerr': me,
            })
            feats    = compute_variability_features(lc_df, BANDS)
            lchi_med = feats['lchi_med']
            sig_max  = feats['sig_max']
            if lchi_med < 0.5 or sig_max < 0.0:
                n_skip_var += 1
                continue
        else:
            lchi_med = np.nan
            sig_max  = np.nan

        items.append((obj_id, t, m, me, b, n_good, lchi_med, sig_max, fit_kwargs))

    print(
        f'Objects: {len(df)} total | {n_skip_obs} skipped (few obs) '
        f'| {n_skip_var} skipped (not variable) | {len(items)} to fit'
    )
    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--input',        default=INPUT,                    help='Input parquet')
    parser.add_argument('--output',       default='rrl_candidates.parquet', help='Output parquet')
    parser.add_argument('--workers',      type=int,   default=None,         help='Worker processes')
    parser.add_argument('--pmin',         type=float, default=0.44,         help='Min period (days)')
    parser.add_argument('--pmax',         type=float, default=0.89,         help='Max period (days)')
    parser.add_argument('--dphi',         type=float, default=0.02,         help='Phase resolution')
    parser.add_argument('--min-obs',      type=int,   default=10,           help='Min good obs')
    parser.add_argument('--no-varfilter', action='store_true',              help='Skip variability filter')
    args = parser.parse_args()

    n_workers  = args.workers or mp.cpu_count()
    fit_kwargs = dict(pmin=args.pmin, pmax=args.pmax, dphi=args.dphi)

    print(f'Input    : {args.input}')
    print(f'Output   : {args.output}')
    print(f'Workers  : {n_workers}')
    print(f'Period   : [{args.pmin}, {args.pmax}] days  dphi={args.dphi}')
    print(f'Min obs  : {args.min_obs}')
    print(f'Var filt : {not args.no_varfilter}')
    print(f'Templates: {MULTIBAND_ZIP}')
    print()

    t0_total = time.perf_counter()
    df = pd.read_parquet(args.input)
    print(f'Loaded {len(df)} objects in {time.perf_counter()-t0_total:.1f}s')

    t0    = time.perf_counter()
    items = build_work_items(df, args.min_obs, not args.no_varfilter, fit_kwargs)
    print(f'Parsed light curves in {time.perf_counter()-t0:.1f}s')

    if not items:
        print('No objects passed filters — nothing to fit.')
        sys.exit(0)

    print(f'\nFitting {len(items)} objects × 136 templates on {n_workers} workers ...')
    t0 = time.perf_counter()

    with mp.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(MULTIBAND_ZIP,),
    ) as pool:
        rows = pool.map(_fit_one, items, chunksize=max(1, len(items) // (n_workers * 4)))

    elapsed = time.perf_counter() - t0
    rows    = [r for r in rows if r is not None]
    print(
        f'Fitted {len(rows)} objects in {elapsed:.1f}s  '
        f'({elapsed / max(len(items), 1) * 1000:.0f} ms/object)'
    )

    if not rows:
        print('All fits failed.')
        sys.exit(1)

    results = pd.DataFrame(rows).sort_values('rss_frac').reset_index(drop=True)

    out_path = Path(args.output)
    results.to_parquet(out_path, index=False)
    print(f'\nSaved {len(results)} results → {out_path}')

    # Summary
    clean = results[~results['flag_amp'] & ~results['flag_depth']]
    print(f'\n--- Summary ---')
    print(f'Total fits            : {len(results)}')
    print(f'Clean (no flags)      : {len(clean)}')
    print(f'\nTop 15 candidates (sorted by rss_frac, no flags):')
    cols = ['diaObjectId', 'period', 'template_id', 'rss_frac', 'rss_depth', 'mu_r', 'A', 'n_good']
    print(clean[cols].head(15).to_string(index=False))


if __name__ == '__main__':
    main()
