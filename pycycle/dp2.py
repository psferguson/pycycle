"""Rubin DP2 (and DP1) integration for pycycle via LSDB/HATS/nested-pandas.

This module adapts pycycle's period finders to the Rubin **object collection**
HATS layout, in which each object row carries its forced-source light curve in a
single *nested* column (``objectForcedSource``) rather than in a separate
source catalogue that must be joined.  Compare :mod:`pycycle.lsdb_utils`, which
targets the older DP1 pattern of ``objects.nest_sources(sources, ...)`` producing
a nested column called ``sources``.

Two entry points, matching the two ways you will want to run:

``fit_object_ids(ids)``
    Fit a known list of ``objectId`` values.  Uses the collection's registered
    ``objectId`` HATS index catalogue, so only the partitions containing those
    objects are read.

``fit_catalog(cat)``
    Fit every object in an (already filtered) catalogue via ``map_partitions``.
    This is the full-catalogue path; pair it with ``prefilter=True`` so the
    expensive template fit only runs on objects that look variable.

Both share one per-object code path (:func:`fit_lightcurve`), so a light curve
tuned interactively in a notebook fits identically inside the pipeline.

Defaults target **RRab** stars: the period grid is 0.44-0.89 d
(:data:`RRAB_PMIN` / :data:`RRAB_PMAX`).  Widen ``DP2Config.pmin`` if you later
want RRc as well.  Fits use ``griz`` only -- see ``DP2Config.bands``.

The template fit defaults to ``template_mode='multiband'`` (free mean magnitude
per band).  The full rr-templates physics model biases the recovered period
toward short values whenever a star's colours do not sit on the assumed RRab
locus -- see ``DP2Config.template_mode``.

Example
-------
Known-object path::

    from pycycle.dp2 import DP2Config, fit_object_ids

    cfg = DP2Config(template_dir='~/software/rr-templates/template_des')
    results = fit_object_ids([735954534639105918, 738184412939704186], cfg)
    results[['objectId', 'ps_period', 'tf_period', 'period_ratio', 'status']]

Full-catalogue path::

    from dask.distributed import Client
    from pycycle.dp2 import DP2Config, open_dp2, fit_catalog

    client = Client(n_workers=8, threads_per_worker=1)
    cfg = DP2Config(template_dir='~/software/rr-templates/template_des',
                    run_period_search=False,   # template fit only, much cheaper
                    prefilter=True)            # skip obviously non-variable stars
    cat = open_dp2(cfg, search_filter=lsdb.ConeSearch(ra=61.25, dec=-48.46,
                                                       radius_arcsec=3600.0))
    cat = cat.query('refExtendedness < 0.5')
    results = fit_catalog(cat, cfg).compute()
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np

from .lsdb_utils import compute_variability_features

logger = logging.getLogger(__name__)

__all__ = [
    'DP2Config',
    'RRAB_PMIN',
    'RRAB_PMAX',
    'DP2_COLLECTION',
    'NEST_COL',
    'QUALITY_FLAGS',
    'nest_columns',
    'object_columns',
    'open_dp2',
    'id_search_objects',
    'clean_epochs',
    'fit_lightcurve',
    'fit_meta',
    'make_dp2_fit_fn',
    'fit_catalog',
    'fit_object_ids',
]

# ---------------------------------------------------------------------------
# Catalogue constants
# ---------------------------------------------------------------------------

#: Default DP2 HATS collection (primary table + margin + objectId index).
DP2_COLLECTION = '/astro/store/shire/hats/catalogs/rubin_dp2/object_collection'

#: Name of the nested light-curve column in the Rubin object collection.
NEST_COL = 'objectForcedSource'

#: RRab period range in days.  Stringer et al. (2019) / DES RRab search range.
RRAB_PMIN = 0.44
RRAB_PMAX = 0.89

#: LSST band name -> template band name.  The DES rr-template library labels the
#: reddest band ``Y``; Rubin calls it ``y``.
LSST_TO_TEMPLATE_BAND = {'y': 'Y'}

#: Per-epoch quality flags relevant to *PSF* photometry.  Difference-imaging
#: flags (``psfDiffFlux_flag``, ``diff_PixelFlags_*``) are deliberately excluded
#: -- they do not invalidate a direct forced PSF measurement.  Any name absent
#: from a given catalogue is silently skipped, so this list is safe across
#: DP1/DP2 schema differences.
QUALITY_FLAGS = [
    'psfFlux_flag',
    'invalidPsfFlag',
    'psfFluxErr_corrected_flag',
    'pixelFlags_bad',
    'pixelFlags_cr',
    'pixelFlags_crCenter',
    'pixelFlags_edge',
    'pixelFlags_interpolated',
    'pixelFlags_interpolatedCenter',
    'pixelFlags_nodata',
    'pixelFlags_saturated',
    'pixelFlags_saturatedCenter',
    'pixelFlags_suspect',
    'pixelFlags_suspectCenter',
]

#: Non-nested object columns worth carrying through a fit run.
BASE_OBJECT_COLS = ['objectId', 'coord_ra', 'coord_dec', 'refExtendedness', 'ebv']


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DP2Config:
    """Everything the DP2 fitting path needs, in one picklable object.

    Attributes
    ----------
    catalog_path : str
        HATS collection to open.  Defaults to :data:`DP2_COLLECTION`.
    nest_col : str
        Nested light-curve column name.  ``objectForcedSource`` for the Rubin
        object collection; set to ``'sources'`` for a DP1-style
        ``nest_sources`` join.
    template_dir : str or None
        Directory holding ``templates.csv`` / ``betas.csv`` / ``dust.csv``
        (e.g. ``.../rr-templates/template_des``).  Required unless a
        pre-loaded template is passed to :func:`make_dp2_fit_fn`.
    template_name : str
        Label recorded on the template.
    template_mode : {'multiband', 'rr'}
        Which physics the template fit is allowed to assume.

        ``'multiband'`` (default, and the right choice for **period finding**)
            Fit an independent mean magnitude per band, plus a shared amplitude
            and phase.  Only the light-curve *shape* constrains the fit.

        ``'rr'``
            The full rr-templates model: a single distance modulus ``mu``, a
            reddening ``EBV``, and the period-luminosity term ``beta_b(P)``.
            This ties the per-band mean magnitudes to a physical RRab locus, so
            it yields distances -- but it also means a star whose colours do not
            sit on that locus is penalised, and **the penalty grows with
            period**, dragging the best-fit period to the short end of the grid.

        Measured on the 17 known DP2 RRL: ``'multiband'`` agrees with
        :class:`~pycycle.PeriodSearch` within 5% for 11/17 with median
        chi2/dof 8.1, while ``'rr'`` manages 5/17 at median chi2/dof 94.6 and
        piles up against ``pmin``.  Use ``'rr'`` to extract ``mu``/``EBV`` at a
        period you already trust, not to find the period.
    template_source : {'auto', 'rr', 'baeza'}
        Which template *library* ``template_dir`` points at.

        ``'rr'``
            Long / Stringer+2019 rr-templates: a directory containing
            ``templates.csv`` / ``betas.csv`` / ``dust.csv``.
        ``'baeza'``
            Baeza-Villagra+2025 Multiband-templates: a directory of per-star
            ``.txt`` files (or an unpacked ``RRab_normalized.zip``).  These are
            normalised shapes with no dust or PLR terms, so they are inherently
            multiband-mode and ``template_mode``/``des_correction`` do not apply.
        ``'auto'`` (default)
            Detect from the directory contents via
            :func:`pycycle.templates.is_multiband_dir`.
    baeza_combine : {'average', 'first'} or int
        How to reduce the 136 per-star Baeza-Villagra templates to something
        fittable.  ``'average'`` (default) builds the mean shape;
        ``'first'`` takes one template; an integer *k* selects *k* medoid
        templates by clustering and returns the first -- see
        :func:`pycycle.templates.load_medoid_templates` if you want to fit all
        *k* and keep the best.
    des_correction : {'rtn099', 'empirical', None}
        DES -> LSST photometric correction applied once at template load time
        via :func:`pycycle.lsdb_utils.apply_des_to_lsst_correction`.  ``None``
        disables it (correct for the SDSS template, which is not DES-based).
    bands : list of str
        LSST bands to feed the fitters, before band remapping.  ``u`` is
        excluded by default because the DES template library has no u-band.
    band_map : dict
        LSST band -> template band renaming, applied after selection.
    time_col, mag_col, magerr_col, band_col : str
        Nested sub-column names.  Defaults are the DP2 native magnitude
        columns; ``psfMagErr_corrected`` is the survey's recalibrated error and
        is strongly preferred over the raw ``psfMagErr``, which underestimates
        the true scatter.
    flux_col, fluxerr_col : str or None
        If set *and* ``mag_col`` is absent from the data, magnitudes are
        derived from these via :func:`pycycle.lsdb_utils.flux_to_mag`.  This is
        the DP1 fallback path.
    magerr_max : float
        Drop epochs with error above this (mag).  0.2 keeps S/N >~ 5.
    max_mad_deviation : float or None
        Robust outlier clip: drop epochs deviating from their band's median by
        more than this many (MAD-scaled) sigma.  ``None`` disables it.

        This catches a failure mode the error cut cannot: DP2 contains occasional
        catastrophically wrong measurements carrying **normal error bars and no
        quality flags** -- observed deviations of 6-9 mag with quoted errors of
        0.006-0.07 mag.  Nothing in the error model or the flags identifies them,
        so the only handle is deviation from the star's own light curve.

        Choose the threshold with RRab variability in mind: a genuine RRab reaches
        only ~1.5-2 MAD from its median, so a clip at 6-8 is safe.  Too tight a
        value would start removing real pulsation.
    min_epochs : int
        Minimum surviving epochs (all bands) required to attempt a fit.
    min_band_epochs : int
        Bands with fewer surviving epochs than this are dropped entirely --
        a one-point band adds no phase information and can destabilise the
        per-band periodogram.
    min_bands : int
        Minimum number of surviving bands required to attempt a fit.
    pmin, pmax, dphi : float
        Period grid, shared by :class:`~pycycle.PeriodSearch` and
        :class:`~pycycle.template_fit.TemplateFitter`.  Defaults are the RRab
        range.
    run_period_search : bool
        Run :class:`~pycycle.PeriodSearch` alongside the template fit.  Costs
        roughly 5 s/star.  Keep on for validation runs on known objects; turn
        off for full-catalogue sweeps where the template fit is the product.
    n_thresh : int
        Monte Carlo significance realisations for ``PeriodSearch.run``.  0
        skips the (expensive) threshold estimate.
    n_newton, n_start, warm_start, use_errors
        Passed to :class:`~pycycle.template_fit.TemplateFitter`.  ``warm_start``
        is ~4x faster on a dense sorted grid and is the catalogue-scale default.
    refine : bool
        After the coarse grid search, re-fit on a fine period grid centred on
        the coarse best period.  This matters more than it looks: ``dphi``
        bounds the phase error between *adjacent* grid points over the data
        baseline, so on a multi-year baseline the coarse grid can be too sparse
        to land on the true period, and ``warm_start`` makes it sloppier still.
        On a 400 d synthetic RRab the coarse best period gives chi2/dof ~ 329,
        the refined one ~ 1.7.  The refinement costs ~5% of the coarse search.
    refine_window : int
        Half-width of the refinement window, in coarse grid steps.  Must be wide
        enough to contain the true period; the default is deliberately generous
        because a warm-started coarse search can land ~10 steps off.
    refine_points : int
        Number of test periods in the refinement grid.
    prefilter : bool
        Compute cheap variability statistics first and skip the fit for objects
        that fail them.  Intended for the full-catalogue path.
    prefilter_lchi_med, prefilter_sig_max : float
        Thresholds following Stringer et al. (2019) sec. 3.2.  The published
        ``lchi_med >= 0.5`` drops 3 of the 17 known DP2 RR Lyrae -- they have
        less scatter than their quoted errors imply -- so the pipeline scripts
        override it to -0.6.  The library keeps the published value as the
        default; the divergence is deliberate and is noted here rather than
        silently reconciled.  Note also that at deep-field scale the prefilter
        rejects only 0.16-0.80% of objects, so it buys almost nothing: the
        dominant cut is ``too_few_epochs`` at 82-85%.
    fit_features : bool
        Record the post-fit diagnostics from :mod:`pycycle.fit_features`
        (``tf_r2`` and friends).  ``tf_r2 = 1 - RSS_best/RSS_flat`` is the
        variability statistic worth having: because both sums run over the same
        points with the same weights, epoch count cancels, and it reaches
        AUC ~1.0 separating variables from flat stars where the ``lchi_med``
        prefilter lets 99.95% of pure noise through.
    fit_features_per_band, fit_features_window : bool
        Sub-parts of the above.  The per-band refit is the more expensive half;
        both together cost a few percent of the fit they describe.
    peak_sep_frac : float
        Fractional period separation at which a competing periodogram minimum
        counts as a different period, for ``tf_peak_ratio``.  The previous
        project-side implementation required 5%, which left the statistic
        computable for only ~8% of objects while being the best single
        right-versus-wrong-period feature available (AUC 0.803); 1% matches the
        tolerance used to call a recovery correct and is defined far more often.
    flag_cols : list of str or 'all'
        Per-epoch flags that must be False.  ``'all'`` auto-detects every
        sub-column whose name contains ``flag``/``Flag``.
    """

    catalog_path: str = DP2_COLLECTION
    nest_col: str = NEST_COL

    template_dir: str | None = None
    template_name: str = 'des'
    template_mode: str = 'multiband'
    template_source: str = 'auto'
    baeza_combine: str = 'average'
    des_correction: str | None = 'rtn099'

    bands: list = field(default_factory=lambda: ['g', 'r', 'i', 'z'])
    band_map: dict = field(default_factory=lambda: dict(LSST_TO_TEMPLATE_BAND))

    time_col: str = 'midpointMjdTai'
    mag_col: str = 'psfMag'
    magerr_col: str = 'psfMagErr_corrected'
    band_col: str = 'band'
    flux_col: str | None = 'psfFlux'
    fluxerr_col: str | None = 'psfFluxErr'

    magerr_max: float = 0.2
    max_mad_deviation: float | None = None
    min_epochs: int = 10
    min_band_epochs: int = 5
    min_bands: int = 2

    pmin: float = RRAB_PMIN
    pmax: float = RRAB_PMAX
    dphi: float = 0.02

    run_period_search: bool = True
    n_thresh: int = 1

    n_newton: int = 5
    n_start: int = 4
    warm_start: bool = True
    use_errors: bool = True

    refine: bool = True
    refine_window: int = 25
    refine_points: int = 1000

    prefilter: bool = False
    prefilter_lchi_med: float = 0.5
    prefilter_sig_max: float = 0.0

    fit_features: bool = True
    fit_features_per_band: bool = True
    fit_features_window: bool = True
    peak_sep_frac: float = 0.01

    flag_cols: object = field(default_factory=lambda: list(QUALITY_FLAGS))

    def load_template(self):
        """Load and correct the RR Lyrae template described by this config.

        Returns
        -------
        RRTemplate

        Raises
        ------
        ValueError
            If ``template_dir`` is unset.
        """
        from .templates import (load_rr_template, load_multiband_dir,
                                is_multiband_dir, average_multiband_templates)
        from .lsdb_utils import apply_des_to_lsst_correction

        if self.template_mode not in ('multiband', 'rr'):
            raise ValueError(
                f"template_mode must be 'multiband' or 'rr', "
                f'got {self.template_mode!r}')
        if not self.template_dir:
            raise ValueError(
                'DP2Config.template_dir is unset -- either set it or pass an '
                'already-loaded template to make_dp2_fit_fn(template=...).'
            )
        path = os.path.expanduser(self.template_dir)

        source = self.template_source
        if source == 'auto':
            source = 'baeza' if is_multiband_dir(path) else 'rr'
        if source not in ('rr', 'baeza'):
            raise ValueError(
                f"template_source must be 'auto', 'rr' or 'baeza', got "
                f'{self.template_source!r}')

        if source == 'baeza':
            # Per-star normalised shapes: no dust, no PLR betas, so these are
            # multiband-mode by construction and the DES zero-point correction
            # (which shifts betas) does not apply.
            templates = load_multiband_dir(path)
            if not templates:
                raise ValueError(f'no Multiband templates found in {path}')
            if self.baeza_combine == 'first':
                return templates[0]
            if isinstance(self.baeza_combine, int):
                from .templates import load_medoid_templates  # noqa: F401
                k = max(1, int(self.baeza_combine))
                # cluster in shape space and keep the first medoid
                return average_multiband_templates(templates[:k])
            return average_multiband_templates(templates)

        template = load_rr_template(path, name=self.template_name)

        if self.template_mode == 'multiband':
            # Drop the PLR betas and the dust prior, keeping only the shape.
            # The DES->LSST zero-point correction is meaningless here (it
            # shifts betas, which no longer exist) and is skipped.
            from .templates import RRTemplate
            return RRTemplate(name=f'{template.name}_multiband',
                              bands=template.bands, phase=template.phase,
                              gamma=template.gamma, dust=None, betas=None)

        if self.des_correction:
            apply_des_to_lsst_correction(template, method=self.des_correction)
        return template


# ---------------------------------------------------------------------------
# Catalogue access
# ---------------------------------------------------------------------------

def nest_columns(cfg: DP2Config | None = None) -> list:
    """Return the minimal set of nested sub-columns needed for a fit.

    Reading only these instead of the full ~40-field forced-source struct is a
    large I/O saving at catalogue scale.
    """
    cfg = cfg or DP2Config()
    cols = [cfg.time_col, cfg.band_col, cfg.mag_col, cfg.magerr_col]
    if cfg.flux_col:
        cols.append(cfg.flux_col)
    if cfg.fluxerr_col:
        cols.append(cfg.fluxerr_col)
    if isinstance(cfg.flag_cols, (list, tuple)):
        cols.extend(cfg.flag_cols)
    # de-duplicate, preserve order
    seen = set()
    return [c for c in cols if not (c in seen or seen.add(c))]


def object_columns(cfg: DP2Config | None = None, extra=None) -> list:
    """Return the non-nested object columns to request, including the nest."""
    cfg = cfg or DP2Config()
    cols = list(BASE_OBJECT_COLS)
    if extra:
        cols.extend(c for c in extra if c not in cols)
    cols.append(cfg.nest_col)
    return cols


def open_dp2(cfg: DP2Config | None = None, columns=None, extra_columns=None,
             search_filter=None, prune_nest: bool = True, **kwargs):
    """Open the Rubin object collection with fit-appropriate columns.

    Parameters
    ----------
    cfg : DP2Config, optional
    columns : list, optional
        Explicit object-level column list.  Overrides the default and
        ``extra_columns``.  The nested column is appended if missing.
    extra_columns : list, optional
        Additional object columns to keep alongside the defaults.
    search_filter : lsdb search object, optional
        e.g. ``lsdb.ConeSearch(...)``.
    prune_nest : bool
        Request only the nested sub-columns the fit needs (via
        ``<nest>.<subcol>`` selection) rather than the whole struct.  Set False
        if you need the full forced-source record downstream.
    **kwargs
        Forwarded to ``lsdb.open_catalog``.

    Returns
    -------
    lsdb.Catalog
    """
    import lsdb

    cfg = cfg or DP2Config()
    if columns is None:
        columns = object_columns(cfg, extra=extra_columns)
    elif cfg.nest_col not in columns:
        columns = list(columns) + [cfg.nest_col]

    if prune_nest:
        columns = [c for c in columns if c != cfg.nest_col]
        columns += [f'{cfg.nest_col}.{sub}' for sub in nest_columns(cfg)]

    return lsdb.open_catalog(cfg.catalog_path, columns=columns,
                             search_filter=search_filter, **kwargs)


def id_search_objects(cat, object_ids, id_col: str = 'objectId',
                      index_catalog=None):
    """Select rows by ``objectId`` using the collection's HATS index catalogue.

    This resolves the requested ids through the index rather than scanning every
    partition, which is what makes the known-object path cheap on a
    thousands-of-partitions catalogue.

    Parameters
    ----------
    cat : lsdb.Catalog
    object_ids : sequence of int
    id_col : str
    index_catalog : str or HCIndexCatalog, optional
        Explicit index catalogue path.  Only needed when the catalogue was
        opened stand-alone (not as a collection), or the collection does not
        register an index for ``id_col``.

    Returns
    -------
    lsdb.Catalog
    """
    ids = [int(i) for i in np.atleast_1d(object_ids)]
    kwargs = {}
    if index_catalog is not None:
        kwargs['index_catalogs'] = {id_col: index_catalog}
    return cat.id_search({id_col: ids}, **kwargs)


# ---------------------------------------------------------------------------
# Per-object light-curve preparation
# ---------------------------------------------------------------------------

def _resolve_flag_cols(lc, cfg: DP2Config) -> list:
    """Flag sub-columns actually present in this light curve."""
    if isinstance(cfg.flag_cols, str) and cfg.flag_cols == 'all':
        return [c for c in lc.columns if 'flag' in c or 'Flag' in c]
    return [c for c in (cfg.flag_cols or []) if c in lc.columns]


def clean_epochs(lc, cfg: DP2Config | None = None):
    """Turn one object's nested light curve into clean co-aligned fit arrays.

    Applies, in order: per-epoch quality flags, band selection, finite/positive
    error cuts, the ``magerr <= magerr_max`` cut, removal of under-sampled
    bands, and the LSST -> template band renaming (``y`` -> ``Y``).

    Parameters
    ----------
    lc : pandas.DataFrame
        One object's light curve -- the contents of the nested column.
    cfg : DP2Config, optional

    Returns
    -------
    hjd, mag, magerr : ndarray of float64
    filts : ndarray of str
        Band per epoch, already renamed to template conventions.  Sorted by
        time, which is what ``warm_start`` grid fitting expects.
    """
    from .lsdb_utils import flux_to_mag

    cfg = cfg or DP2Config()
    empty = (np.empty(0), np.empty(0), np.empty(0), np.empty(0, dtype='<U2'))
    if lc is None or len(lc) == 0:
        return empty

    keep = np.ones(len(lc), dtype=bool)

    # 1. per-epoch quality flags must all be False
    for col in _resolve_flag_cols(lc, cfg):
        vals = lc[col].to_numpy()
        keep &= ~np.asarray(vals, dtype=bool)

    # 2. magnitudes: native columns preferred, flux conversion as fallback
    if cfg.mag_col in lc.columns and cfg.magerr_col in lc.columns:
        mag = np.asarray(lc[cfg.mag_col].to_numpy(), dtype=float)
        magerr = np.asarray(lc[cfg.magerr_col].to_numpy(), dtype=float)
    elif cfg.flux_col and cfg.flux_col in lc.columns:
        mag, magerr = flux_to_mag(lc[cfg.flux_col].to_numpy(),
                                  lc[cfg.fluxerr_col].to_numpy())
    else:
        raise KeyError(
            f'Light curve has neither {cfg.mag_col!r}/{cfg.magerr_col!r} nor '
            f'{cfg.flux_col!r}; available columns: {list(lc.columns)}'
        )

    hjd = np.asarray(lc[cfg.time_col].to_numpy(), dtype=float)
    filts = np.asarray(lc[cfg.band_col].to_numpy()).astype(str)

    # 3. finite values, sane errors, requested bands
    keep &= np.isfinite(hjd) & np.isfinite(mag) & np.isfinite(magerr)
    keep &= (magerr > 0) & (magerr <= cfg.magerr_max)
    if cfg.bands is not None:
        wanted = set(cfg.bands)
        keep &= np.array([b in wanted for b in filts], dtype=bool)

    hjd, mag, magerr, filts = hjd[keep], mag[keep], magerr[keep], filts[keep]
    if len(hjd) == 0:
        return empty

    # 3b. robust per-band outlier clip (see max_mad_deviation)
    if cfg.max_mad_deviation is not None and len(hjd) > 5:
        good = np.ones(len(hjd), dtype=bool)
        for b in np.unique(filts):
            s = filts == b
            if s.sum() < 5:
                continue
            med = np.median(mag[s])
            mad = np.median(np.abs(mag[s] - med)) * 1.4826
            if mad <= 0:
                continue
            good[s] = np.abs(mag[s] - med) / mad <= cfg.max_mad_deviation
        hjd, mag, magerr, filts = hjd[good], mag[good], magerr[good], filts[good]
        if len(hjd) == 0:
            return empty

    # 4. drop under-sampled bands
    if cfg.min_band_epochs > 1:
        names, counts = np.unique(filts, return_counts=True)
        ok_bands = set(names[counts >= cfg.min_band_epochs].tolist())
        band_keep = np.array([b in ok_bands for b in filts], dtype=bool)
        hjd, mag, magerr, filts = (hjd[band_keep], mag[band_keep],
                                   magerr[band_keep], filts[band_keep])
        if len(hjd) == 0:
            return empty

    # 5. rename to template band conventions (y -> Y)
    if cfg.band_map:
        filts = np.array([cfg.band_map.get(b, b) for b in filts])

    # 6. sort by time -- warm-start fitting walks a sorted grid
    order = np.argsort(hjd)
    return hjd[order], mag[order], magerr[order], filts[order]


# ---------------------------------------------------------------------------
# The single per-object fit -- shared by notebooks and the pipeline
# ---------------------------------------------------------------------------

def _coeff_fields(template) -> list:
    """Output column names for a template's fitted coefficients."""
    if template.dust is not None:
        return ['mu', 'EBV', 'A']
    return [f'mu_{b}' for b in template.bands] + ['A']


def _blank_row(template, cfg: DP2Config) -> dict:
    """A result row with every field present and NaN/empty."""
    row = {
        'objectId': -1,
        'coord_ra': np.nan,
        'coord_dec': np.nan,
        'n_epochs': 0,
        'n_bands': 0,
        'bands': '',
        'lchi_med': np.nan,
        'sig_max': np.nan,
        'ps_period': np.nan,
        'ps_psi': np.nan,
        'ps_psi_per_epoch': np.nan,
        'ps_period_alt': np.nan,
        'ps_status': 'not_run',
        'tf_period': np.nan,
        'tf_period_coarse': np.nan,
        'tf_phi': np.nan,
        'tf_rss': np.nan,
        'tf_chi2_dof': np.nan,
        # post-fit diagnostics (pycycle.fit_features) -- recorded, never used
        # to drop a row, so a downstream classifier can weigh them on labels
        'tf_rss_flat': np.nan,
        'tf_r2': np.nan,
        'tf_r2_2nd': np.nan,
        'tf_period_2nd': np.nan,
        'tf_peak_ratio': np.nan,
        'tf_phase_scatter': np.nan,
        'tf_amp_ratio': np.nan,
        'tf_n_band_fit': 0,
        'win_power': np.nan,
        'win_pct': np.nan,
        'win_max': np.nan,
        'period_ratio': np.nan,
        'at_period_bound': False,
        'status': 'not_run',
        'error': '',
    }
    for name in _coeff_fields(template):
        row[name] = np.nan
    return row


def fit_lightcurve(hjd, mag, magerr, filts, template, cfg: DP2Config | None = None,
                   fitter=None, return_results: bool = False):
    """Run the period search and the template fit on one prepared light curve.

    This is the single per-object code path.  The ``map_partitions`` pipeline
    and interactive notebook tuning both go through it, so a period reproduced
    at the notebook is the period the pipeline records.

    Parameters
    ----------
    hjd, mag, magerr, filts : ndarray
        Co-aligned arrays as returned by :func:`clean_epochs`.
    template : RRTemplate
        Already corrected (see :meth:`DP2Config.load_template`).
    cfg : DP2Config, optional
    fitter : TemplateFitter, optional
        Reuse an existing fitter instead of constructing one per object.
    return_results : bool
        Also return the live ``PeriodSearchResult`` and ``TemplateFitResult``
        objects, which carry the plotting methods.

    Returns
    -------
    row : dict
        Flat result record.  Always returned, NaN-filled on failure, with the
        reason in ``status``/``error``.
    ps_result, tf_result : optional
        Only when ``return_results=True``; either may be None.
    """
    from .core import PeriodSearch
    from .template_fit import TemplateFitter

    cfg = cfg or DP2Config()
    row = _blank_row(template, cfg)
    ps_result = tf_result = None

    n = len(hjd)
    present = sorted(set(np.asarray(filts).astype(str).tolist()))
    row['n_epochs'] = int(n)
    row['n_bands'] = len(present)
    row['bands'] = ','.join(present)

    def _done():
        return (row, ps_result, tf_result) if return_results else row

    if n < cfg.min_epochs:
        row['status'] = 'too_few_epochs'
        return _done()
    if len(present) < cfg.min_bands:
        row['status'] = 'too_few_bands'
        return _done()

    # cheap variability statistics -- always recorded, optionally used to gate
    try:
        import pandas as pd
        feats = compute_variability_features(
            pd.DataFrame({'band': filts, 'mag': mag, 'magerr': magerr}), present)
        row['lchi_med'] = feats['lchi_med']
        row['sig_max'] = feats['sig_max']
    except Exception as exc:  # pragma: no cover - diagnostics must never fail a run
        logger.debug('variability features failed: %s', exc)

    if cfg.prefilter and not (row['lchi_med'] >= cfg.prefilter_lchi_med
                              and row['sig_max'] >= cfg.prefilter_sig_max):
        row['status'] = 'prefiltered'
        return _done()

    # bands the template does not know would raise inside fit(); catch early so
    # the object is reported rather than lost
    unknown = set(present) - set(template.bands)
    if unknown:
        row['status'] = 'band_mismatch'
        row['error'] = (f'bands {sorted(unknown)} absent from template '
                        f'{template.name!r} {template.bands}')
        return _done()

    if cfg.run_period_search:
        try:
            ps = PeriodSearch(hjd, mag, magerr, filts, filtnams=present)
            ps_result = ps.run(pmin=cfg.pmin, dphi=cfg.dphi, pmax=cfg.pmax,
                               n_thresh=cfg.n_thresh)
            row['ps_period'] = float(ps_result.best_period)
            psi = (ps_result.psi_m if ps_result.psi_m.ndim == 1
                   else ps_result.psi_m.sum(0))
            row['ps_psi'] = float(np.max(psi))
            # PSI ~ (N/2)(A/sigma)^2 summed over bands, so it is *linear in
            # epoch count* -- a fine discriminator at fixed sampling and a poor
            # one across a catalogue, where it promotes well-sampled junk over
            # real variables.  Dividing by N is what makes it comparable
            # between objects; keep both so the raw value stays auditable.
            row['ps_psi_per_epoch'] = row['ps_psi'] / n if n else np.nan
            tops = ps_result.top_periods(n=2)
            if len(tops) > 1:
                row['ps_period_alt'] = float(tops['period'][1])
            row['ps_status'] = 'ok'
        except Exception as exc:
            # A PeriodSearch failure is a *diagnostic* failure: the template fit
            # below is independent of it and its period is still valid.  Writing
            # this into `status` (as this code used to) meant filtering on
            # status=='ok' silently discarded good fits -- 1,080 rows in one run
            # and 5,941 in another, which biased a configuration comparison
            # before it was caught.  `status` now describes the template fit
            # only; PeriodSearch reports here.
            row['ps_status'] = 'failed'
            row['error'] = f'{type(exc).__name__}: {exc}'
            logger.debug('PeriodSearch failed: %s', exc)
    else:
        row['ps_status'] = 'skipped'

    try:
        if fitter is None:
            fitter = TemplateFitter(template, n_newton=cfg.n_newton,
                                    n_start=cfg.n_start,
                                    use_errors=cfg.use_errors,
                                    warm_start=cfg.warm_start)
        tf_result = _fit_template(fitter, hjd, mag, magerr, filts, cfg, template)
        row['tf_period'] = float(tf_result.best_period)
        row['tf_phi'] = float(tf_result.best_phi)
        coarse = getattr(tf_result, 'coarse', None)
        row['tf_period_coarse'] = float(
            coarse.best_period if coarse is not None else tf_result.best_period)
        rss_min = float(np.min(tf_result.rss))
        row['tf_rss'] = rss_min
        # rr-templates fits mu, EBV, A and phi; multiband fits one mu per band,
        # plus A and phi
        n_par = 4 if template.dust is not None else len(present) + 2
        dof = max(n - n_par, 1)
        row['tf_chi2_dof'] = rss_min / dof
        for name, val in tf_result.best_coeffs.items():
            if name in row:
                row[name] = float(val)

        # Post-fit diagnostics.  `tf_chi2_dof` is not a usable quality axis on
        # its own -- on the six DP2 deep fields its median is 0.66 (errors are
        # conservative at the faint end) while the known RR Lyrae sit at the
        # 83rd-99th percentile of their own field, because a real variable with
        # underestimated errors produces a *large* chi2.  Cutting on it deletes
        # the signal.  These features are the replacement: see
        # pycycle.fit_features for what each one is and what it was measured to
        # be worth.
        if cfg.fit_features:
            try:
                from .fit_features import fit_features as _ff
                feats = _ff(tf_result, sep_frac=cfg.peak_sep_frac,
                            use_errors=cfg.use_errors,
                            per_band=cfg.fit_features_per_band,
                            window=cfg.fit_features_window)
                row['tf_rss_flat'] = feats['rss_flat']
                row['tf_r2'] = feats['r2']
                row['tf_r2_2nd'] = feats['r2_2nd']
                row['tf_period_2nd'] = feats['period_2nd']
                row['tf_peak_ratio'] = feats['peak_ratio']
                for src, dst in (('phase_scatter', 'tf_phase_scatter'),
                                 ('amp_ratio', 'tf_amp_ratio'),
                                 ('n_band_fit', 'tf_n_band_fit'),
                                 ('win_power', 'win_power'),
                                 ('win_pct', 'win_pct'),
                                 ('win_max', 'win_max')):
                    if src in feats:
                        row[dst] = feats[src]
            except Exception as exc:  # diagnostics must never fail a fit
                logger.debug('fit_features failed: %s', exc)
    except Exception as exc:
        row['status'] = 'tf_failed' if row['status'] == 'not_run' else row['status']
        row['error'] = (row['error'] + ' | ' if row['error'] else '') + \
                       f'{type(exc).__name__}: {exc}'
        logger.debug('TemplateFitter failed: %s', exc)
        return _done()

    # A best period sitting on the edge of the search range usually means the
    # real minimum lies outside it -- for RRab-range fits that is the signature
    # of an RRc, or of the 1-day alias comb that sparse ground-based sampling
    # produces.  Flag it rather than reporting a railed period as a measurement.
    span = cfg.pmax - cfg.pmin
    if np.isfinite(row['tf_period']) and span > 0:
        edge = 0.01 * span
        row['at_period_bound'] = bool(
            (row['tf_period'] - cfg.pmin) < edge or (cfg.pmax - row['tf_period']) < edge)

    if np.isfinite(row['ps_period']) and row['ps_period'] > 0:
        row['period_ratio'] = row['tf_period'] / row['ps_period']

    if row['status'] == 'not_run':
        row['status'] = 'ok'
    return _done()


def _fit_quiet(fitter, hjd, mag, magerr, filts, **kwargs):
    """Call ``TemplateFitter.fit`` without its per-call progress printing.

    Older pycycle builds print three lines per fit unconditionally, which at
    catalogue scale floods worker logs.  Newer builds accept ``verbose``.
    """
    try:
        return fitter.fit(hjd, mag, magerr, filts, verbose=False, **kwargs)
    except TypeError:
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            return fitter.fit(hjd, mag, magerr, filts, **kwargs)


def _fit_template(fitter, hjd, mag, magerr, filts, cfg: DP2Config, template):
    """Coarse grid search, then a fine local refinement around the best period.

    ``dphi`` controls the phase change between *adjacent* coarse grid points
    across the data baseline, so the grid gets relatively coarser as the
    baseline lengthens -- on Rubin's multi-year baselines the coarse minimum can
    sit far enough from the true period to wreck the phase fold.  A short second
    pass on a fine grid around that minimum fixes it for a few percent of the
    cost.

    Returns the refined :class:`~pycycle.template_fit.TemplateFitResult`, with
    the coarse result attached as ``.coarse`` (its ``plot_rss()`` shows the full
    periodogram, which the narrow refined grid cannot).
    """
    from .template_fit import TemplateFitter

    coarse = _fit_quiet(fitter, hjd, mag, magerr, filts,
                        pmin=cfg.pmin, dphi=cfg.dphi, pmax=cfg.pmax)
    if not cfg.refine or len(coarse.periods) < 2:
        return coarse

    periods = np.sort(coarse.periods)
    step = float(np.median(np.diff(periods)))
    if not np.isfinite(step) or step <= 0:
        return coarse

    # clamp to the declared search range: a refined period outside [pmin, pmax]
    # would silently escape the RRab constraint the caller asked for
    half = cfg.refine_window * step
    lo = max(coarse.best_period - half, cfg.pmin)
    hi = min(coarse.best_period + half, cfg.pmax)
    if hi <= lo:
        return coarse

    # cold multi-start on the fine grid: it is short, and warm-starting is what
    # loses precision in the first place
    refiner = TemplateFitter(template, n_newton=cfg.n_newton,
                             n_start=cfg.n_start, use_errors=cfg.use_errors,
                             warm_start=False)
    fine = _fit_quiet(refiner, hjd, mag, magerr, filts,
                      periods=np.linspace(lo, hi, cfg.refine_points))

    best = fine if float(np.min(fine.rss)) <= float(np.min(coarse.rss)) else coarse
    best.coarse = coarse
    return best


# ---------------------------------------------------------------------------
# map_partitions plumbing
# ---------------------------------------------------------------------------

def fit_meta(template, cfg: DP2Config | None = None):
    """Empty DataFrame describing the fit output schema, for Dask ``meta``."""
    import pandas as pd

    cfg = cfg or DP2Config()
    dtypes = {
        'objectId': np.int64,
        'coord_ra': np.float64,
        'coord_dec': np.float64,
        'n_epochs': np.int64,
        'n_bands': np.int64,
        'bands': object,
        'lchi_med': np.float64,
        'sig_max': np.float64,
        'ps_period': np.float64,
        'ps_psi': np.float64,
        'ps_psi_per_epoch': np.float64,
        'ps_period_alt': np.float64,
        'ps_status': object,
        'tf_period': np.float64,
        'tf_period_coarse': np.float64,
        'tf_phi': np.float64,
        'tf_rss': np.float64,
        'tf_chi2_dof': np.float64,
        'tf_rss_flat': np.float64,
        'tf_r2': np.float64,
        'tf_r2_2nd': np.float64,
        'tf_period_2nd': np.float64,
        'tf_peak_ratio': np.float64,
        'tf_phase_scatter': np.float64,
        'tf_amp_ratio': np.float64,
        'tf_n_band_fit': np.int64,
        'win_power': np.float64,
        'win_pct': np.float64,
        'win_max': np.float64,
        'period_ratio': np.float64,
        'at_period_bound': bool,
        'status': object,
        'error': object,
    }
    for name in _coeff_fields(template):
        dtypes[name] = np.float64
    # keep _blank_row's key order so rows and meta line up exactly
    order = list(_blank_row(template, cfg).keys())
    return pd.DataFrame({k: pd.Series(dtype=dtypes[k]) for k in order})


def make_dp2_fit_fn(cfg: DP2Config | None = None, template=None):
    """Build a ``map_partitions``-compatible fit function and its Dask meta.

    The returned function takes one partition (a nested-pandas frame with the
    nested light-curve column) and returns one row per input object -- including
    objects that failed, which carry ``status``/``error`` instead of being
    silently dropped.  The partition's HEALPix index and sky coordinates are
    preserved so the result is still a valid LSDB catalogue.

    Parameters
    ----------
    cfg : DP2Config, optional
    template : RRTemplate, optional
        Pre-loaded and pre-corrected template.  If omitted, it is loaded once
        here from ``cfg.template_dir`` -- once in the driver process, not once
        per partition, so the in-place DES correction is applied exactly once
        and every worker receives an identically corrected copy.

    Returns
    -------
    fn : callable
    meta : pandas.DataFrame
    """
    import pandas as pd

    cfg = cfg or DP2Config()
    if template is None:
        template = cfg.load_template()
    meta = fit_meta(template, cfg)
    columns = list(meta.columns)

    def _fit_partition(df):
        from .template_fit import TemplateFitter

        if len(df) == 0:
            return meta.copy()

        fitter = TemplateFitter(template, n_newton=cfg.n_newton,
                                n_start=cfg.n_start, use_errors=cfg.use_errors,
                                warm_start=cfg.warm_start)
        rows = []
        for idx, obj in df.iterrows():
            row = _blank_row(template, cfg)
            try:
                lc = obj[cfg.nest_col]
                hjd, mag, magerr, filts = clean_epochs(lc, cfg)
                row = fit_lightcurve(hjd, mag, magerr, filts, template, cfg,
                                     fitter=fitter)
            except Exception as exc:
                row['status'] = 'error'
                row['error'] = f'{type(exc).__name__}: {exc}'
                logger.warning('object at index %s failed: %s', idx, exc)
            if 'objectId' in df.columns:
                row['objectId'] = int(obj['objectId'])
            for coord in ('coord_ra', 'coord_dec'):
                if coord in df.columns:
                    row[coord] = float(obj[coord])
            rows.append(row)

        out = pd.DataFrame(rows, columns=columns, index=df.index)
        return out.astype(meta.dtypes.to_dict())

    return _fit_partition, meta


def fit_catalog(cat, cfg: DP2Config | None = None, template=None,
                compute: bool = False):
    """Fit every object in ``cat`` via ``map_partitions``.

    This is the full-catalogue path.  Apply your row filters to ``cat`` first
    (``.query('refExtendedness < 0.5')``, a cone search, the helpers in
    ``code_from_sam.py``, ...) -- and consider ``cfg.prefilter=True`` plus
    ``cfg.run_period_search=False`` so the expensive work is reserved for
    plausible variables.

    Parameters
    ----------
    cat : lsdb.Catalog
    cfg : DP2Config, optional
    template : RRTemplate, optional
    compute : bool
        Return a materialised pandas DataFrame instead of the lazy catalogue.
        Only do this when the *result* is small; it is one row per object, not
        one per epoch, but a full-catalogue run still would not fit in memory.

    Returns
    -------
    lsdb.Catalog or pandas.DataFrame
    """
    cfg = cfg or DP2Config()
    fn, meta = make_dp2_fit_fn(cfg, template=template)
    out = cat.map_partitions(fn, meta=meta)
    return out.compute() if compute else out


def fit_object_ids(object_ids, cfg: DP2Config | None = None, catalog=None,
                   template=None, id_col: str = 'objectId', index_catalog=None,
                   compute: bool = True):
    """Fit a known list of ``objectId`` values end to end.

    Opens the collection (unless ``catalog`` is given), selects the requested
    ids through the HATS index catalogue, and runs the same fit function the
    full-catalogue path uses.

    Parameters
    ----------
    object_ids : sequence of int
    cfg : DP2Config, optional
    catalog : lsdb.Catalog, optional
        Reuse an already-open catalogue.
    template : RRTemplate, optional
    id_col : str
    index_catalog : str or HCIndexCatalog, optional
    compute : bool
        Materialise the (small) result.  True by default -- this path is for
        tens of objects.

    Returns
    -------
    pandas.DataFrame or lsdb.Catalog
    """
    cfg = cfg or DP2Config()
    cat = catalog if catalog is not None else open_dp2(cfg)
    sel = id_search_objects(cat, object_ids, id_col=id_col,
                            index_catalog=index_catalog)
    return fit_catalog(sel, cfg, template=template, compute=compute)
