# pycycle: outstanding work

Updated 2026-08-21, after the DP2 deep-field campaign. Every item below is backed by
a measurement made during that work; the evidence is quoted so priorities can be
re-argued rather than taken on trust.

Branch `dp2-lsdb`, 5 commits ahead of `main`, 107 tests passing.

## Done (this branch)

| | evidence |
|---|---|
| DP2 support: `objectId` index path and full-catalogue `map_partitions` | 2.5M-object fields fitted end to end |
| `template_mode='multiband'` as default | rr-templates PLR drags periods to `pmin`; agreement with PeriodSearch 5/17 → 11/17, median chi2/dof 94.6 → 8.1, railed 4/17 → 0/17 |
| Period refinement after the coarse grid | chi2/dof 329 → 1.7 on a 400 d synthetic |
| `TemplateFitter.fit(verbose=)` | printed 3 lines per fit, flooding worker logs |
| `plot_phased(ax=)` plots every band | previously drew only the first and silently dropped the rest |
| `plotting.py` accepts `axes=` | helpers always built and closed their own figure |
| `apply_des_to_lsst_correction` refuses double application | it mutates betas in place; twice doubled the offset silently |
| `load_multiband_dir` + `template_source` | Baeza-Villagra templates loadable; Stringer still the default |
| `max_mad_deviation` outlier clip | DP2 has 6-9 mag bad epochs with normal errors and no flags |

## 1. Fix `PeriodSearch.best_period` band combination — highest value

`core.py:54` sums **raw** PSI across bands with no per-band normalisation, so a band
with larger raw PSI dominates. On a worked mock (true P = 0.6231): single-band g gives
0.62332 and i gives 0.62247, but the raw sum follows r and z to 0.4753, a 23.7% error.
Grid density is irrelevant (`dphi` 0.02 → 0.002 changes nothing) and the true period is
in the top 5 about **60%** of the time — the search works, the *pick* fails.

Rank-summing the per-band periodograms instead lifts bright-mock recovery from
**23% → 37%**. Expose the per-band periodograms so callers can combine deliberately.

Overall on mocks, PeriodSearch has Spearman **0.109** with truth versus 0.472 for the
template fit, so nothing that depends on its period should be trusted until this is fixed.

## 2. Normalise PSI, and record `ps_psi` and `tf_r2` as first-class outputs

PSI ≈ `(N/2)·(A/σ)²` summed over bands — **linear in epoch count**. It is therefore an
excellent discriminator at fixed sampling and a poor one across a catalogue: in M49 the
visually-good objects had a median 336 epochs and the rejects 527, so raw `ps_psi`
actively promoted well-sampled junk. Dividing by N moved a *known* RR Lyrae from rank
162 to 34, and the two Pai stars from #29/#162 to #18/#34.

`tf_r2 = 1 - RSS_best/RSS_flat` is the complementary statistic: because both sums run
over the same points with the same weights, **N cancels**. It reaches AUC **0.9997**
separating variables from noise (`tf_r2 >= 0.5` gave zero false positives on 1,000 flat
stars) but only 0.78 for right-period versus wrong-period.

Both currently live in project scripts. They should be computed in `fit_lightcurve` and
returned in the row: `tf_r2`, `ps_psi_per_epoch`. See `docs/statistics_explained.md`.

Related: every script runs `n_thresh=0`, so the Monte-Carlo significance path in
`periodogram.py:130-153` never executes and `ps_psi` is the raw uncalibrated statistic.
Either use it or document that it is unused.

## 3. Separate "the fit failed" from "a diagnostic failed"

`fit_lightcurve` sets `status='ps_failed'` when *PeriodSearch* raises even though the
template fit succeeded, so those rows carry valid periods. Filtering on `status=='ok'`
silently discards them — 1,080 rows in one run, 5,941 in another, and it biased a
configuration comparison before it was caught. Add a separate `ps_status`, or make
`status` describe only the template fit.

## 4. Defaults that the measurements argue for

* **`magerr_max = 0.5`** rather than 0.2: faint recovery (g 23-24.5) **39% → 57%**, no
  gain beyond 0.5. One line, largest single win available.
* **Do not gate on `period_ratio`.** P(correct | agree) = 68.2% versus
  P(correct | disagree) = 60.4% — eight points of discrimination, while discarding
  **71%** of correctly recovered RRab. Keep it as a recorded flag.
* **`prefilter_lchi_med`**: the library default is the published Stringer +0.5, which
  drops 3 of 17 known DP2 RRL. The pipeline scripts use −0.6. Either align them or
  document the divergence in `DP2Config`.

## 5. Fix the error-calibration claim in the docs

`docs/flux_fitting_plan.md` states DP2 errors are "~1.4x conservative (median chi2_nu
0.505)". **That is wrong** — it came from i-band only, with `magerr <= 0.2` applied and
no quality flags. Measured properly on 261,918 light curves with flags applied, the
calibration is band- and magnitude-dependent: roughly correct at 22-24, conservative
(0.5-0.8) fainter than 23, and *under*estimated by 2-5x between 19 and 21.5. Raw
`psfMagErr`/`psfFluxErr` give chi2_nu of 20-1200 and are unusable; the `_corrected`
columns are validated as the right choice. The plan's argument is unaffected — only the
stated figure. `artifacts/dp2_error_model.json` in the rrl project has the numbers.

## 6. Metallicity term and the i-band PLR zero point

Validated against 24 Gaia RRL cross-matched in M49: our distances are **+11.8%** high
(+0.24 mag in mu) with 0.127 mag rms once that offset is removed. Two causes, both
diagnosable:

* **The i band drives it.** Per-band distance ratios versus Gaia are g **0.995**,
  r **1.030**, i **1.376**. The g and r scales are essentially correct; the i-band PLR
  zero point (the DES→LSST correction) is wrong.
* **The residual correlates with [Fe/H] at rho = +0.93.** The rr-template PLR has no
  metallicity term, and metal-poor RRab are intrinsically brighter at fixed period.

Encouragingly, periods agree with Gaia to a median **0.18%**, 22 of 23 within 1%.

## 7. Flux-space fitting

The real fix for faint work: every epoch dropped for a NaN magnitude or `magerr > 0.2`
is a valid flux measurement with a finite error. See `docs/flux_fitting_plan.md` for the
solver derivation (F0_b stays closed form; A and phi become Gauss-Newton), the file-by-
file plan, phasing and acceptance criteria. Phase 1 is a NumPy solver plus tests, 1-2 days.

## 8. Architecture: keep DP2 in pycycle, but fix the seam

Agreed direction, not started. `dp2.py` is 975 lines and `lsdb_utils.py` 355; the core
is already survey-agnostic and lsdb is imported lazily.

1. Delete the hardcoded site path at `dp2.py:94` — machine-specific, not survey-specific.
2. Extract the column-name schema out of `DP2Config` into a reusable object, so DP2
   becomes *data* and DP1/DP3/ZTF/mocks are other instances. This is 80% done already:
   `DP2Config` parameterises every column name, which is how the mock harnesses run
   LightCurveLynx data through the identical code path.
3. Make lsdb/hats an optional extra.

Split into `pycycle_lsst` only when a second survey or consumer actually needs it; doing
(2) first makes that split nearly free.

## 9. Smaller items

* The averaged Stringer template **overshoots light-curve minima**, visible on good
  M49 candidates, inflating chi2/dof for genuine RRab. Plausibly why chi2 discriminated
  so weakly (AUC 0.635) against visual labels. Argues for per-star Baeza-Villagra
  templates — but validate on the known RRL first: on one poor-fit object the averaged
  Stringer template still beat the best of 136 BV templates (chi2/dof 4.36 vs 6.92).
* `rss_margin` is the best single feature for right-versus-wrong period (AUC 0.803) but
  is computable for only ~8% of objects. Loosening the ">5% away from the best period"
  criterion could unlock real signal.
* An `HistGradientBoostingClassifier` over the fit-time features reaches CV AUC **0.894**
  for right-versus-wrong period, against 0.792 for the best single feature. Worth
  shipping as an optional scorer once the features above are first-class outputs.
* Environment: matplotlib in the `pycycle` env now resolves the *system* `libstdc++`
  (no `CXXABI_1.3.15`) and fails to import. Workaround
  `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"`; fix with
  `conda install -n pycycle libstdcxx-ng`.
