# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -e ".[dev]"        # install with Cython + pytest
pytest                          # run all tests
pytest tests/test_core.py -v    # single test file
pytest tests/test_core.py::test_best_period_close_to_gold   # one test
```

The `[dev]` extra brings in Cython + pytest. The `setup.py` invokes `cythonize` on `pycycle/_ext/*.pyx` automatically when a C compiler is available; if the build fails, `ext_modules` silently becomes empty and the package falls back to pure-Python / numba paths.

After editing `.pyx` files you must `pip install -e .` again — `*.so` artefacts under `pycycle/_ext/` are gitignored but are required for the fast path.

## Architecture

### Two pipelines, one package

1. **Period search** (`PeriodSearch` in `core.py`) — implements the Saha & Vivas (2017) hybrid PSI statistic `PSI = 2·fy/theta`, combining Lomb-Scargle power (`scargle.py`) with the Lafler-Kinman phase-dispersion statistic (`lafler_kinman.py`). `periodogram.compute_periodogram` is run once per band; results are stacked in `PeriodSearchResult.psi_m` (shape `(nfilts, N)`, flattened to `(N,)` for single-band).
2. **Template fitting** (`TemplateFitter` in `template_fit.py`) — fits an RR Lyrae template over a period grid. Two physics modes are selected automatically by `template.dust is None`:
   - **rr-templates mode** (has `dust` + `betas`): full model `mag = beta_b(P) + mu + EBV·dust_b + A·gamma_b(phase)`, fit with linear-solve for `(mu, EBV, A)` and Newton step on phase.
   - **Multiband mode** (no `dust`): simpler `mag = mu_b + A·gamma_b(phase)` with per-band offsets.

The README's "When to use period search vs template fitting" table is the load-bearing decision rule: at LSST-sparse sampling (≲30 pts/band) skip period search and template-fit directly.

### Backend dispatch pattern

`scargle.py` and `lafler_kinman.py` follow the same triple-fallback pattern:

```python
try:    from ._ext._pycycle_c import scargle_fast as _c    # compiled
except: ...
try:    import numba                                        # JIT
except: ...
# else pure-Python loops
```

`template_fit.py` only has a two-way fallback (`._ext.template_fit_c` compiled
or pure-NumPy `_rss_grid_*_py`) — there is no numba path for the template-fit
inner loop.

When modifying numerics, change the pure-Python reference first (it's readable), then mirror the change into the `.pyx` source under `pycycle/_ext/`. The `_USE_C` flag in each module decides which path runs; tests don't pin it, so both paths must agree.

### Template loading

External templates are **not bundled** — `templates.py` reads them from a user-specified directory or zip:
- `load_rr_template(dir)` — Long/Stringer CSV format (`templates.csv`, `betas.csv`, `dust.csv`).
- `load_multiband_templates(zip)` / `load_multiband_template(zip, star_id)` — Baeza-Villagra normalised txt files inside `RRab_normalized.zip` / `RRc_normalized.zip`.
- `average_multiband_templates(list)` produces an `RRTemplate` with `dust=None`, which downstream switches the fitter to Multiband mode.

`RRTemplate.dgamma()` returns the circular central-difference derivative used by the Newton phase update — note the wrap-around boundary at indices 0 and -1.

### LSST/Rubin integration (`lsdb_utils.py`, DP1) and (`dp2.py`, DP2+)

`lsdb_utils.make_template_fit_fn(template, bands, nest_col='sources', **fit_kwargs)` returns a `(fn, meta)` pair designed for `lsdb_catalog.map_partitions(fn, meta=meta)`. The function expects a nested column (default name `sources`, from `objects.nest_sources(sources, ...)`; pass `nest_col='objectForcedSource'` for a Rubin object collection, though `dp2.py` below is preferred for that) with either `midpointMjdTai` (DP1) or `mjd` time columns, and Rubin nJy fluxes which it converts via `flux_to_mag` (zpt = 31.4). Objects that fail the fit or have <10 usable epochs are silently dropped from the output.

`apply_des_to_lsst_correction(template, method='rtn099')` mutates `template.betas[:, 0]` in place at load time so the DES→LSST color terms (RTN-099) cost nothing during fitting, and records `template.lsst_correction = method`. Two correction sets exist: `'rtn099'` (synthetic, default) and `'empirical'` (residuals from 26 M49 RRab calibrators); the empirical set absorbs PLR zero-point errors that RTN-099 doesn't capture but is only calibrated for g/r/i. Calling it a second time on the same template (`template.lsst_correction` already set) raises `ValueError` — the mutation is not idempotent and a second application would double the offset.

`dp2.py` targets the newer Rubin **object collection** HATS layout, where each object row carries its light curve directly in a nested `objectForcedSource` column — no join needed. Contrast with `lsdb_utils.py` above, which targets DP1's `objects.nest_sources(sources, ...)` producing a `sources` column.

Everything the DP2 path needs is one picklable `DP2Config` dataclass (period grid, quality cuts, fitter knobs, prefilter thresholds, template-load params). `DP2Config.load_template()` loads + DES-corrects the template once.

Two entry points share one per-object function, `fit_lightcurve(hjd, mag, magerr, filts, template, cfg)`, so notebook tuning and the pipeline agree exactly:
- `fit_object_ids(ids, cfg)` — resolves `objectId`s through the collection's registered HATS index catalog (`id_search_objects`/`cat.id_search`) so only relevant partitions are read; calls `fit_catalog` under the hood.
- `fit_catalog(cat, cfg)` — `map_partitions` over an already-filtered catalog via `make_dp2_fit_fn`. Pair with `cfg.prefilter=True` and `cfg.run_period_search=False` for cheap large sweeps.

Per-object flow inside `fit_lightcurve`: `clean_epochs` (flags → mag/magerr resolution, native `psfMag`/`psfMagErr_corrected` preferred over `psfFlux`→mag fallback → finite/error/band cuts → drop under-sampled bands → `y`→`Y` band rename → sort by time) feeds an optional `PeriodSearch` (`cfg.run_period_search`) and `_fit_template` (coarse-then-refine template fit, see below). Every object emits exactly one row, even on failure, via `_blank_row`/`status` (`ok`, `too_few_epochs`, `too_few_bands`, `band_mismatch`, `prefiltered`, `ps_failed`, `tf_failed`, `error`) — this is the behavioral contrast with `lsdb_utils.make_template_fit_fn`'s silent-drop.

`_fit_template` runs the coarse grid (`cfg.pmin/pmax/dphi`, `warm_start`), then — if `cfg.refine` (default `True`) — a cold multi-start fine grid of `cfg.refine_points` periods spanning `± cfg.refine_window` coarse-grid steps around the coarse best period, keeping whichever of the two has lower RSS. The coarse result is attached to whichever result is returned as `.coarse`. Rationale: `dphi` bounds phase error between *adjacent* coarse grid points, not absolute period error, so long Rubin baselines make the coarse grid too sparse, and `warm_start` compounds it. Measured on a 400 d synthetic RRab: chi2/dof ~329 (coarse) → ~1.7 (refined), at ~5% extra cost.

Output row also carries `at_period_bound` (bool): `tf_period` within 1% of `cfg.pmin`/`cfg.pmax` of the search span — usually means the true minimum lies outside the searched range (RRc star searched with RRab bounds, or a 1-day alias), so treat it as "not a reliable measurement" rather than trusting the railed value. `period_ratio = tf_period / ps_period` (0.5 or 2.0 flags an alias between the two independent period estimates).

`pycycle/__init__.py` re-exports only `DP2Config`, `open_dp2`, `clean_epochs`, `fit_lightcurve`, `make_dp2_fit_fn`, `fit_catalog`, `fit_object_ids` at top level. `id_search_objects`, `fit_meta`, `RRAB_PMIN`/`RRAB_PMAX`, `nest_columns`, `object_columns`, `DP2_COLLECTION`, `NEST_COL`, `QUALITY_FLAGS` are in `dp2.__all__` but must be imported from `pycycle.dp2` directly.

### Result objects

Both `PeriodSearchResult` and `TemplateFitResult` hold the original `(hjd, mag, magerr, filts, filtnams)` arrays for plotting — they're heavy. `PeriodSearchResult.top_periods()` returns an `astropy.table.Table`; `TemplateFitResult.top_periods()` returns a list of dicts. Don't conflate them.
