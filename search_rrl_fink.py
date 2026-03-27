"""Search all Fink/LSST alert light curves for RR Lyrae using pycycle template fitting.

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
    --use-sfd             Look up SFD E(B-V) and pre-subtract dust from magnitudes

Pipeline
--------
1. Parse light curves from prvDiaSources (scienceFlux, bands g/r/i).
2. Quality cut: magerr in (0, 0.2].
3. Variability pre-filter (S19 §3.2): lchi_med ≥ 0.5 and sig_max ≥ 0.
4. Optional SFD dust pre-subtraction (--use-sfd).
5. RRab template fit (DES gri + LSST correction, warm_start for speed).
6. Save all fit results; add quality flags for post-hoc filtering.

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

SFD dust note
-------------
With --use-sfd the SFD E(B-V) for each object is looked up from the Schlegel,
Finkbeiner & Davis (1998) dust map and pre-subtracted from the magnitudes before
fitting.  The fitted EBV in the output is then a residual δEBV = EBV_fit − EBV_SFD.
In low-dust fields (EBV_SFD < 0.05) this correction is negligible.  In high-dust
fields it removes the dominant degeneracy between μ and E(B-V).

Note: with gri-only data the template colour (g−i ≈ 0.85 from DES betas) can
differ systematically from observed LSST colours (~0.27 in this Virgo field).
The model compensates with EBV ≈ −0.30, which SFD cannot fix — it reflects a
template/calibration colour offset, not real dust.  Use EBV values with caution.

Output columns
--------------
    diaObjectId   int64   Object identifier
    period        float   Best-fit period (days)
    mu            float   Distance modulus
    EBV           float   Fitted E(B-V); residual δEBV if --use-sfd was set
    EBV_SFD       float   SFD E(B-V) from dust map (NaN if --use-sfd not used)
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
INPUT        = '/astro/store/shire/pferguso/alert_sprint/concatenated_catalog/latest_obs.parquet'
TEMPLATE_DIR = os.path.expanduser('~/software/rr-templates/template_des')
BANDS        = ['g', 'r', 'i']
BAND_MAP     = {b: i for i, b in enumerate(BANDS)}

# Quality thresholds
AMP_MIN   = 0.3    # minimum RRab amplitude (mag)
AMP_MAX   = 1.5    # maximum RRab amplitude (mag)
DEPTH_MIN = 0.5    # minimum RSS depth to consider period significant


# ---------------------------------------------------------------------------
# Worker initializer — load template once per process
# ---------------------------------------------------------------------------

def _init_worker(template_dir: str) -> None:
    """Load and cache the gri template in each worker process."""
    global _TEMPLATE
    from pycycle.templates import load_rr_template, RRTemplate
    from pycycle.lsdb_utils import apply_des_to_lsst_correction

    template_des = load_rr_template(template_dir, name='des')
    apply_des_to_lsst_correction(template_des)

    sub_idx = [template_des.bands.index(b) for b in BANDS]
    _TEMPLATE = RRTemplate(
        name='des_gri',
        bands=BANDS,
        phase=template_des.phase,
        gamma=template_des.gamma[sub_idx],
        dust=template_des.dust[sub_idx],
        betas=template_des.betas[sub_idx],
    )


def _fit_one(args: tuple) -> dict | None:
    """Fit a single object; called in a worker process."""
    from pycycle.template_fit import TemplateFitter

    (obj_id, t, m, me, bi, n_good, lchi_med, sig_max, ebv_sfd, fit_kwargs) = args

    # Optional SFD dust pre-subtraction: absorb known dust into magnitudes so
    # the fitted EBV is a small residual δEBV rather than the full extinction.
    if np.isfinite(ebv_sfd):
        m = m - ebv_sfd * _TEMPLATE.dust[bi]

    # Null (constant) model RSS — weighted variance of (dust-corrected) magnitudes
    w        = 1.0 / np.maximum(me ** 2, 1e-30)
    m_wmean  = np.dot(w, m) / w.sum()
    rss_null = float(np.dot(w, (m - m_wmean) ** 2))

    try:
        fitter = TemplateFitter(_TEMPLATE, n_newton=5, warm_start=True)
        result = fitter.fit(t, m, me, bi, BANDS, **fit_kwargs)
    except Exception as exc:
        warnings.warn(f'diaObjectId={obj_id}: fit failed ({exc})', stacklevel=2)
        return None

    rss_template = float(result.rss.min())
    rss_frac     = rss_template / max(rss_null, 1e-30)
    rss_depth    = float(1.0 - result.rss.min() / max(np.median(result.rss), 1e-30))

    mu  = float(result.best_coeffs.get('mu', np.nan))
    EBV = float(result.best_coeffs.get('EBV', np.nan))
    A   = float(result.best_coeffs.get('A', np.nan))
    phi = float(result.best_phi)

    return {
        'diaObjectId': obj_id,
        'period':      float(result.best_period),
        'mu':          mu,
        'EBV':         EBV,      # residual δEBV when --use-sfd; full EBV otherwise
        'EBV_SFD':     ebv_sfd,  # NaN when --use-sfd not set
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
    """Extract (t, mag, magerr, band_index) from prvDiaSources array.

    Uses ``scienceFlux`` / ``scienceFluxErr`` (direct science-image fluxes).
    Keeps only observations in BANDS with magerr in (0, 0.2].
    """
    from pycycle.lsdb_utils import flux_to_mag

    n = len(srcs)
    if n == 0:
        return np.empty(0), np.empty(0), np.empty(0), np.empty(0, dtype=int)

    t_all   = np.fromiter((s['midpointMjdTai'] for s in srcs), float, n)
    sf_all  = np.fromiter((s['scienceFlux']    for s in srcs), float, n)
    sfe_all = np.fromiter((s['scienceFluxErr'] for s in srcs), float, n)
    b_all   = np.array([s['band'] for s in srcs])

    mag, magerr = flux_to_mag(sf_all, sfe_all)

    bi_map = np.full(n, -1, dtype=int)
    for band, idx in BAND_MAP.items():
        bi_map[b_all == band] = idx

    good = (magerr > 0) & (magerr <= 0.2) & (bi_map >= 0)
    return t_all[good], mag[good], magerr[good], bi_map[good]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def lookup_sfd_ebv(df: pd.DataFrame) -> np.ndarray:
    """Vectorised SFD E(B-V) lookup for all objects in df.

    Reads ra/dec from diaObject['ra'] / diaObject['dec'].
    Returns an array of shape (len(df),) with NaN on failure.
    """
    from dustmaps.sfd import SFDQuery
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    ra  = np.array([row['diaObject']['ra']  for _, row in df.iterrows()], dtype=float)
    dec = np.array([row['diaObject']['dec'] for _, row in df.iterrows()], dtype=float)
    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    return np.asarray(SFDQuery()(coords), dtype=float)


def build_work_items(
    df: pd.DataFrame,
    min_obs: int,
    varfilter: bool,
    use_sfd: bool,
    fit_kwargs: dict,
) -> list[tuple]:
    """Parse all rows into work items for the worker pool."""
    from pycycle.lsdb_utils import compute_variability_features

    # SFD lookup — vectorised over the whole catalog (fast)
    if use_sfd:
        print('Looking up SFD E(B-V) for all objects ...')
        ebv_sfd_all = lookup_sfd_ebv(df)
        print(f'  SFD E(B-V): min={ebv_sfd_all.min():.4f}  max={ebv_sfd_all.max():.4f}'
              f'  median={np.median(ebv_sfd_all):.4f}')
    else:
        ebv_sfd_all = np.full(len(df), np.nan)

    items      = []
    n_skip_obs = 0
    n_skip_var = 0

    for row_idx, (_, row) in enumerate(df.iterrows()):
        obj_id  = int(row['diaObjectId'])
        srcs    = row['prvDiaSources']
        ebv_sfd = float(ebv_sfd_all[row_idx])

        t, m, me, bi = parse_light_curve(srcs)
        n_good = len(t)

        if n_good < min_obs:
            n_skip_obs += 1
            continue

        if varfilter:
            lc_df = pd.DataFrame({
                'band':   [BANDS[b] for b in bi],
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

        items.append((obj_id, t, m, me, bi, n_good, lchi_med, sig_max, ebv_sfd, fit_kwargs))

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
    parser.add_argument('--use-sfd',      action='store_true',              help='Look up SFD E(B-V) and pre-subtract dust')
    args = parser.parse_args()

    n_workers  = args.workers or mp.cpu_count()
    fit_kwargs = dict(pmin=args.pmin, pmax=args.pmax, dphi=args.dphi)

    print(f'Input    : {args.input}')
    print(f'Output   : {args.output}')
    print(f'Workers  : {n_workers}')
    print(f'Period   : [{args.pmin}, {args.pmax}] days  dphi={args.dphi}')
    print(f'Min obs  : {args.min_obs}')
    print(f'Var filt : {not args.no_varfilter}')
    print(f'SFD dust : {args.use_sfd}')
    print()

    t0_total = time.perf_counter()
    df = pd.read_parquet(args.input)
    print(f'Loaded {len(df)} objects in {time.perf_counter()-t0_total:.1f}s')

    t0    = time.perf_counter()
    items = build_work_items(df, args.min_obs, not args.no_varfilter, args.use_sfd, fit_kwargs)
    print(f'Parsed light curves in {time.perf_counter()-t0:.1f}s')

    if not items:
        print('No objects passed filters — nothing to fit.')
        sys.exit(0)

    print(f'\nFitting {len(items)} objects on {n_workers} workers ...')
    t0 = time.perf_counter()

    with mp.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(TEMPLATE_DIR,),
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
    cols = ['diaObjectId', 'period', 'rss_frac', 'rss_depth', 'mu', 'EBV', 'A', 'n_good']
    print(clean[cols].head(15).to_string(index=False))


if __name__ == '__main__':
    main()
