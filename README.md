# pycycle

**pycycle** is a hybrid Lomb-Scargle / Lafler-Kinman period finder for multiband
variable-star light curves, based on
[Saha & Vivas (2017, AJ 154, 231)](https://ui.adsabs.harvard.edu/abs/2017AJ....154..231S)
It also includes a fast Cython-accelerated RR Lyrae template fitter and utilities
for catalog-scale searches on Rubin LSST data via [LSDB](https://lsdb.io).

---

## Period search

```python
from pycycle import PeriodSearch
import numpy as np

hjd, mag, magerr, filts = np.loadtxt('data.tab', unpack=True)
ps = PeriodSearch(hjd, mag, magerr, filts, filtnams=['V'])
result = ps.run(pmin=0.2, dphi=0.02)
print(result.best_period)
result.plot_phased()
```

See `notebooks/tutorial.ipynb` for a full walkthrough.

---

## Template fitting

pycycle fits RR Lyrae light curves against two external template libraries.
Clone them separately:

### rr-templates (Long / Stringer et al.)

SDSS and DES averaged templates with full physics parameterisation
(distance modulus µ, dust E(B-V), amplitude A, phase φ).

```
git clone https://github.com/longjp/rr-templates
```

> Templates constructed for and used in
> [Stringer et al. 2019 (AJ 157, 187)](https://doi.org/10.3847/1538-3881/ab1f46).
> Please cite this work if you use these templates.

```python
from pycycle.templates import load_rr_template
from pycycle.template_fit import TemplateFitter

template = load_rr_template('/path/to/rr-templates/template_des', name='des')
fitter = TemplateFitter(template, n_newton=5, warm_start=True)
result = fitter.fit(hjd, mag, magerr, filts,
                    pmin=0.44, dphi=0.02, pmax=0.89)
print(result.best_period, result.best_coeffs)
result.plot_phased()
```

**`warm_start=True`** carries the solution `(φ, µ, E(B-V), A)` from each
frequency as the starting point for the next — ~4× faster than the default
multi-start mode, following the optimisation in Stringer et al. (2019).

**`verbose=True`** (default) on `fit()` prints three diagnostic lines per
call — backend, template name, and grid size. Pass `verbose=False` for
catalog-scale runs, where one fit per object would otherwise flood worker
logs.

### Multiband-templates (Baeza-Villagra et al.)

136 RRab and 144 RRc individual DECam griz templates, normalised to [0, 1].

```
git clone https://github.com/KarinaBaezaV/Multiband-templates
```

> [Baeza-Villagra et al. 2025](https://ui.adsabs.harvard.edu/abs/2025A%26A...694A..72B/abstract).
> Please cite this work if you use these templates.

```python
from pycycle.templates import load_multiband_templates, average_multiband_templates
from pycycle.template_fit import TemplateFitter

templates = load_multiband_templates('/path/to/Multiband-templates/RRab_normalized.zip')
avg = average_multiband_templates(templates)
fitter = TemplateFitter(avg)
result = fitter.fit(hjd, mag, magerr, filts, pmin=0.2, dphi=0.02)
```

See `notebooks/template_fitting.ipynb` for a complete worked example.

---

## When to use period search vs template fitting

| LSST phase | Obs/band | Recommended strategy |
|---|---|---|
| Year 1–3 | ~4–10 | **Template fitting directly** — period search is unreliable at this sparsity |
| Year 5–7 | ~15–30 | **Template fitting directly** |
| Year 10 | ~50–100 | **Period search → template fitting** on high-PSI candidates |
| Well-sampled (>30/band) | any | **Period search first** (~5 s/star), then template fit candidates |
| All variable types | any | **Period search** — template fitting is RRab-specific |

---

## LSST / Rubin utilities (DP1)

`pycycle.lsdb_utils` provides helpers for running the template fitter at
catalog scale via [LSDB](https://lsdb.io), targeting the **DP1** pattern where
a separate forced-source catalog is joined onto the objects with
`objects.nest_sources(sources, ...)`, producing a nested column called
`sources`, with magnitudes derived from `psfFlux`. For **DP2** and later, where
the object collection already carries the light curve in a nested
`objectForcedSource` column with native `psfMag`, use `pycycle.dp2` instead
(below).

```python
import lsdb
from pycycle.templates import load_rr_template
from pycycle.lsdb_utils import apply_des_to_lsst_correction, make_template_fit_fn

# Apply DES → LSST filter corrections (RTN-099) once at load time
template = load_rr_template('/path/to/rr-templates/template_des', name='des')
apply_des_to_lsst_correction(template)   # zero runtime cost during fitting

# Build a map_partitions-compatible function
fit_fn, meta = make_template_fit_fn(
    template,
    bands=['g', 'r', 'i', 'z', 'y'],
    nest_col='sources',   # 'objectForcedSource' for a Rubin object collection
    pmin=0.44, dphi=0.02, pmax=0.89,
    n_newton=5, warm_start=True,
)

# Run on a joined object+source catalog
objects = lsdb.open_catalog('...dp01_object...',  columns=[...])
sources = lsdb.open_catalog('...dp01_forced_source...', columns=[...])
stars   = objects.query('extendedness == 0 and 21 < r_psfMag < 24.5')
joined  = stars.nest_sources(sources, source_id_col='objectId')
results = joined.map_partitions(fit_fn, meta=meta).compute()
```

Objects that fail the fit, or have fewer than 10 usable epochs, are **omitted**
from the output — see `pycycle.dp2.make_dp2_fit_fn` below for a one-row-per-object
variant.

`apply_des_to_lsst_correction` mutates the template's betas in place and
records the applied method as `template.lsst_correction`; calling it a second
time on the same template raises `ValueError` rather than silently
double-counting the offset.

Also provided: `flux_to_mag` (Rubin nJy → AB magnitude) and
`compute_variability_features` (χ²_ν and significance for pre-filtering
following Stringer et al. 2019 §3.2).

See `notebooks/lsdb_lsst_pipeline.ipynb` for the full two-stage pipeline.

---

## Rubin DP2 (HATS/LSDB) pipeline

`pycycle.dp2` adapts pycycle to the Rubin **object collection** HATS layout,
where each object row already carries its forced-source light curve in a
nested `objectForcedSource` column, rather than requiring a separate join
(compare `pycycle.lsdb_utils` above, for DP1). Two entry points cover the two
ways you'll want to run it, and both go through the same per-object function,
`fit_lightcurve`, so a period tuned interactively in a notebook is the period
the pipeline records.

Defaults target **RRab**: period grid 0.44–0.89 d, and fit bands `griz` (`u`
is excluded — no u-band in the DES template library and shallow DP2 u-band
depth; `y` is excluded by default too). Magnitudes come from DP2's native
`psfMag` / `psfMagErr_corrected` columns (the survey's recalibrated errors —
raw `psfMagErr` underestimates the true scatter), falling back to
`psfFlux`/`psfFluxErr` → mag conversion only if the mag columns are absent.

**Known object IDs** — resolved through the collection's registered HATS
`objectId` index catalog, so only the relevant partitions are read:

```python
from pycycle.dp2 import DP2Config, fit_object_ids

cfg = DP2Config(template_dir='~/software/rr-templates/template_des')
results = fit_object_ids([735954534639105918, 738184412939704186], cfg)
results[['objectId', 'ps_period', 'tf_period', 'period_ratio', 'status']]
```

**Full catalog sweep** — a `map_partitions` path; pair it with
`prefilter=True` and `run_period_search=False` for a cheap large sweep that
only spends the template-fit budget on plausible variables:

```python
from dask.distributed import Client
from pycycle.dp2 import DP2Config, open_dp2, fit_catalog
import lsdb

client = Client(n_workers=8, threads_per_worker=1)
cfg = DP2Config(template_dir='~/software/rr-templates/template_des',
                run_period_search=False,   # template fit only, much cheaper
                prefilter=True)            # skip obviously non-variable stars
cat = open_dp2(cfg, search_filter=lsdb.ConeSearch(ra=61.25, dec=-48.46,
                                                   radius_arcsec=3600.0))
cat = cat.query('refExtendedness < 0.5')
results = fit_catalog(cat, cfg).compute()
```

Both paths return **one row per object, including failures** — unlike
`lsdb_utils.make_template_fit_fn`, which drops them — tagged by a `status`
column: `ok`, `too_few_epochs`, `too_few_bands`, `band_mismatch`, `prefiltered`,
`ps_failed`, `tf_failed`, or `error`. The partition's HEALPix index and
`coord_ra`/`coord_dec` are preserved, so the result is still a valid LSDB
catalog; `objectId` is taken from its column, not the index (LSDB indexes on
`_healpix_29`).

Output columns include both `ps_period` (from `PeriodSearch`, if
`run_period_search=True`) and `tf_period` (from the template fit), plus
`period_ratio = tf_period / ps_period` as an alias diagnostic (0.5 or 2.0
signals a period alias), `tf_period_coarse`, `tf_chi2_dof`, `at_period_bound`
(the fitted period landed within 1% of `pmin`/`pmax` — usually a railed fit
rather than a real measurement), and the `lchi_med`/`sig_max` variability
statistics.

**Template mode** (`DP2Config.template_mode`, default `'multiband'`): controls
what physics the fit may assume.

* `'multiband'` — a free mean magnitude per band, plus a shared amplitude and
  phase. Only light-curve *shape* constrains the fit. **Use this to find
  periods.**
* `'rr'` — the full rr-templates model: one distance modulus `mu`, a reddening
  `EBV`, and the period-luminosity term `beta_b(P)`. This ties the per-band mean
  magnitudes to a physical RRab locus, so it yields distances — but a star whose
  colours do not sit on that locus is penalised, and because `beta_b` depends on
  period **the penalty grows with period**, sloping the RSS curve upward across
  the whole grid and dragging the best-fit period toward `pmin`.

Measured on the 17 known DP2 RR Lyrae:

| mode | agrees with `PeriodSearch` (<5%) | median chi2/dof | railed at a period bound |
|---|---|---|---|
| `'rr'` | 5/17 | 94.6 | 4/17 |
| `'multiband'` | **11/17** | **8.1** | **0/17** |

Use `'rr'` to extract `mu`/`EBV` at a period you already trust, not to find the
period. The DES→LSST zero-point correction is skipped in multiband mode, since
it shifts `betas`, which that mode does not use.

**Period refinement** (`DP2Config.refine`, default `True`): `dphi` bounds the
phase error between *adjacent* points on the coarse grid across the data
baseline, so on Rubin's multi-year baselines the coarse grid is too sparse and
`warm_start` makes it worse. A second, cold-started fine-grid pass around the
coarse minimum fixes it, for a few percent of the coarse search's cost. On a
400-day synthetic RRab this took chi2/dof from ~329 (coarse) to ~1.7
(refined). The coarse result is attached to the returned fit as `.coarse`
(`tf.coarse.plot_rss()` shows the full periodogram; the refined grid is too
narrow for that).

`DP2Config` holds every other knob (epoch/band quality cuts, period grid,
fitter settings, prefilter thresholds) as one picklable dataclass — see its
docstring in `pycycle/dp2.py` for the full field list.

`DP2Config`, `open_dp2`, `clean_epochs`, `fit_lightcurve`, `make_dp2_fit_fn`,
`fit_catalog`, and `fit_object_ids` are re-exported from the top-level
`pycycle` package; less commonly used helpers (`id_search_objects`,
`fit_meta`, `RRAB_PMIN`/`RRAB_PMAX`, `nest_columns`, `object_columns`, and the
catalog constants) are only available via `pycycle.dp2`.

---

## Installation

Requires Python ≥ 3.9. The C/Cython extensions are optional but give a
significant speed boost.

```bash
pip install -e .
# or with Cython extensions:
pip install -e ".[dev]"
```

The extensions are built automatically if Cython and a C compiler are available.

---

## Notebooks

| Notebook | Description |
|---|---|
| `notebooks/tutorial.ipynb` | Period finding with `PeriodSearch` on the bundled B1392 dataset |
| `notebooks/template_fitting.ipynb` | Template fitting with both rr-templates and Multiband-templates |
| `notebooks/lsdb_lsst_pipeline.ipynb` | Catalog-scale RR Lyrae search on Rubin LSST data via LSDB |

---

## Credits

- **Period search algorithm:** Saha & Vivas (2017, AJ 154, 231)
- **rr-templates:** Long/Stringer et al. (2019, AJ 157, 187)
- **Multiband-templates:** Baeza-Villagra et al. (2025, A&A, arXiv:2501.03813)
- **LSST pipeline strategy:** Stringer et al. (2019), Stringer & Drlica-Wagner et al. (2021, arXiv:2011.13930)
