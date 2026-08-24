# Fitting faint RR Lyrae: a plan for flux-space template fitting in pycycle

Status: proposal. Nothing here is implemented.

## 1. The problem

RRab period recovery in DP2 collapses below g ≈ 23.5. The cause is not the fitter —
it is that we convert forced photometry to magnitudes before fitting, and magnitudes
cannot represent a non-detection.

Two mechanisms discard epochs in `pycycle.dp2.clean_epochs`:

1. `psfFlux <= 0` — a perfectly ordinary noise-dominated measurement of a faint
   source — makes `psfMag` **NaN**, so the epoch is dropped before any cut applies.
2. Everything with `psfMagErr_corrected > 0.2` is cut.

Measured on 4,000 real ECDFS objects, the fraction of griz epochs surviving those
cuts is:

| median mag | epochs kept |
|---|---|
| 20–21 | 86% |
| 21–22 | 64% |
| 22–23 | 42% |
| 23–24 | 11% |
| 24–25 | ~0% |

For a constant star that costs precision. For an RR Lyrae, which swings ~1 mag, the
loss is **phase-dependent**: stacking 150 faint mocks, retention falls monotonically
from **87% near maximum light to 32% near minimum**. We delete the faint half of the
cycle — the descending branch and minimum — which is where the template gets much of
its phase leverage and what sets the measured amplitude.

The cut is also demonstrably costing science. On mocks at g = 23.0–24.5, relaxing
`magerr_max` alone moves period recovery:

| `magerr_max` | median epochs | recovery |
|---|---|---|
| 0.2 (current) | 90 | 39.2% |
| 0.3 | 122 | 45.0% |
| **0.5** | 144 | **56.7%** |
| 1.0 | 155 | 56.7% |

That plateau at 0.5 is the signature of a *representation* limit, not a noise limit:
past that point the extra epochs are the ones magnitudes cannot express at all.

## 2. Two options, and why flux wins

### Option A — censored likelihood on "upper limit" magnitudes

Treat non-detections as upper limits and use survival analysis: detections contribute
a Gaussian term, non-detections contribute `-2 ln Φ((m_lim - m_model)/σ)`.

This is the wrong abstraction for forced photometry, for three reasons:

* **There is no censoring.** DP2 measures a flux and an uncertainty at every visit.
  We know the value. Censored regression exists for when you genuinely only know
  "fainter than X" — replacing a measured `-212 ± 180 nJy` with "fainter than the
  3σ limit" throws information away rather than recovering it.
* **The threshold is arbitrary.** A limiting magnitude per epoch has to be invented,
  and the answer depends on the choice.
* **It does not fix the detections.** Magnitude errors are badly non-Gaussian near
  the limit, so even the epochs we keep are mis-weighted. Censoring leaves that
  untouched.

It is also the more invasive change: a general likelihood destroys the closed-form
and Newton structure the current solver depends on.

### Option B — fit in flux space (recommended)

In flux, the errors are approximately Gaussian by construction, negative values are
legal, and a non-detection is just a low-S/N measurement. **No censoring machinery is
needed and no error cut is needed**, because a `0 ± 200 nJy` epoch contributes
correctly-weighted information automatically.

Every epoch currently dropped for `NaN` magnitude or `magerr > 0.2` becomes usable.
At g = 23.6 that is ~160 epochs instead of ~90.

## 3. The maths

### Current multiband model (magnitude space)

`pycycle/template_fit.py::_rss_grid_mb_py`, mirrored in
`pycycle/_ext/template_fit_c.pyx::rss_grid_mb`:

```
m_i = mu_b(i) + A · γ_b(i)(φ_i)          minimise  Σ w_i (m_i − model)²,  w = 1/σ_m²
```

solved by coordinate descent per trial frequency:

| block | update | form |
|---|---|---|
| `mu_b` | weighted mean of `mag − A·γ` per band | closed form |
| `A` | `Σ w·resid·γ / Σ w·γ²` | closed form |
| `φ` | Newton step using `dγ` | Newton |

### Proposed flux model

```
F_i = F0_b(i) · 10^(−0.4 · A · γ_b(i)(φ_i))
```

Write `c = 0.4·ln10 ≈ 0.921` and `h_i = exp(−c·A·γ_i)`, so `M_i = F0_b·h_i` and the
objective is `Σ W_i (F_i − M_i)²` with `W = 1/σ_F²`.

The same three-block structure survives; two blocks change from closed form to
Gauss-Newton:

| block | update | form |
|---|---|---|
| `F0_b` | `Σ_b W·h·F / Σ_b W·h²` | **still closed form** |
| `A` | `∂M/∂A = −c·γ·M` → Gauss-Newton step | changed |
| `φ` | `∂M/∂φ = −c·A·dγ·M` → Gauss-Newton step | changed |

Notes:

* `A` keeps its magnitude-space meaning (it lives inside the exponent), so amplitudes
  remain comparable with existing results and with the literature.
* Natural initialisation: `A = 0` gives `h = 1`, so `F0_b` starts as the weighted mean
  flux per band — the exact solution of the constant model. Derivatives are non-zero
  there, so it is a usable start, not a stationary point.
* Guards needed: clamp `|A·γ|` (say to 20 mag) against overflow, and handle
  `F0_b ≤ 0` for a band with no signal — allow it but flag rather than clamp, since a
  negative mean flux is evidence, not an error.
* The `rr` (PLR + dust) mode is harder: `mu` and `EBV` also move into the exponent, so
  all three become nonlinear. **Defer it.** Multiband is already the DP2 default and
  is where the faint science is.

## 4. Changes to pycycle, by file

### `pycycle/template_fit.py` — the core work

* Add `TemplateFitter(..., space='mag')`, with `'flux'` as the new option. Default
  stays `'mag'` so nothing existing changes behaviour.
* Add `fit_flux(hjd, flux, fluxerr, filts, pmin=, dphi=, pmax=, periods=, verbose=)`
  as a sibling of `fit()`, sharing grid construction and result assembly.
* New solver `_rss_grid_mb_flux_py(t, flux, w, bidx, gamma, dgamma, freqs, n_bands,
  n_newton, n_start)`, same signature shape as `_rss_grid_mb_py`.
* `TemplateFitResult` gains `space`. In flux mode `best_coeffs` carries
  `f0_<band>` plus `A`; also expose derived `mu_<band> = −2.5·log10(f0_b) + zp` when
  `f0_b > 0`, so downstream code and plots keep working.
* `predict()` must return flux in flux mode.
* `plot_phased()` should plot flux in flux mode — and should show the negative-flux
  points rather than silently dropping them, since displaying them is half the point.

### `pycycle/_ext/template_fit_c.pyx` — performance, phase 2

* New `rss_grid_mb_flux` kernel mirroring `rss_grid_mb` (~150 lines, same memoryview
  and `nogil` patterns). Needed only for catalogue-scale runs; the NumPy path is
  adequate to prove correctness first.
* Requires a rebuild step; note the `.so` files are gitignored and built in place.

### `pycycle/dp2.py` — pipeline wiring

* `DP2Config` gains `space='mag'|'flux'`. In flux mode:
  * read `psfFlux` / `psfFluxErr_corrected` (both exist in the DP2 nested schema —
    `psfFluxErr_corrected` is the flux analogue of `psfMagErr_corrected` and should
    be preferred over the raw `psfFluxErr` for the same reason);
  * `magerr_max` becomes inapplicable — replace with an optional very loose S/N floor
    or nothing at all;
  * do not drop non-finite magnitudes.
* `clean_epochs` in flux mode keeps every epoch with finite flux and `fluxerr > 0`,
  **regardless of sign**, while still applying the per-epoch quality flags
  (`pixelFlags_*`, `psfFlux_flag`, `invalidPsfFlag`). This is where the epochs come
  back.
* `fit_lightcurve` dispatches to `fit_flux`; χ²/dof computed from flux residuals.
* `fit_meta` schema changes (`f0_<band>` alongside or instead of `mu_<band>`) —
  keep both to avoid breaking the existing candidate notebooks.

### `pycycle/lsdb_utils.py` — the prefilter

* `compute_variability_features` computes `chi2_nu` and a significance in
  magnitudes. A flux version is needed (χ² about a constant *flux*, significance in
  flux units).
* **The prefilter thresholds must be recalibrated.** The current default
  (`lchi_med >= -0.6`) was tuned against magnitude-space scatter and against an
  error calibration that is not, as previously stated, a flat ~1.4× conservative
  factor. Measured on 261,918 light curves with quality flags applied
  (`artifacts/dp2_error_model.json` in the rrl project), it is band- and
  magnitude-dependent: roughly correct at 22-24, conservative (0.5-0.8×) fainter
  than 23, and elevated 2-5× between 19 and 21.5 — including an unexplained
  chi2_nu ~180 spike at g~20.5 (173 objects) that is not yet investigated. The
  flux-space distribution will sit somewhere else entirely regardless. Re-derive
  it against the known RRL, as was done for the magnitude version.

### `pycycle/core.py` — PeriodSearch, optional

Lomb-Scargle and Lafler-Kinman are scale-free enough to run on flux, but PSI's
behaviour would change and need re-validation. Lower priority: separate work already
showed the more urgent PeriodSearch problem is that `best_period` sums raw PSI across
bands with no per-band normalisation, which is a distinct bug.

## 5. Phasing

1. **Correctness** — NumPy solver, `fit_flux`, unit tests. No pipeline changes.
2. **Validation** — run the injection-recovery harness in both modes and compare.
   Gate the rest of the work on the result.
3. **Pipeline** — `dp2.py` wiring, flux variability features, prefilter recalibration.
4. **Performance** — Cython kernel, once the science case is proven.

## 6. How we will know it worked

The project already has the harness to answer this, so the acceptance criteria are
quantitative rather than a judgement call:

* `scripts/mock_rrl_recovery.py` — recovery vs magnitude. Flux mode should **meet or
  beat the 57% that `magerr_max=0.5` achieves at g = 23.0–24.5**, and ideally push the
  g ≈ 23.5 cliff fainter.
* `scripts/mock_distance_grid.py` — the (epochs × distance) response surface. Success
  is the *reachable* region expanding: cells currently marked unreachable, where a
  faint star cannot retain N usable epochs, should become populated, since flux mode
  imposes no error cut.
* `notebooks/validate_known_rrl.ipynb` — the 17 known RRL must not regress.
* Existing pytest suite must pass unchanged, since `space='mag'` stays the default.

New unit tests to add:

* round-trip: generate from the flux model, recover `F0_b`, `A`, `φ`, period;
* equivalence: at high S/N, flux and magnitude fits agree on period and amplitude;
* negative-flux handling: a light curve containing negative epochs fits without error
  and uses them;
* guards: `A·γ` clamping, `F0_b ≤ 0`.

## 7. Risks

* **Robustness.** Two closed-form blocks become Gauss-Newton, so more local minima are
  possible. Mitigations: keep multi-start, and optionally seed from the magnitude-space
  solution on the subset of epochs that have valid magnitudes.
* **Speed.** Exponentials per iteration are dearer than the current arithmetic. The
  refinement pass already dominates the cost, so expect a real slowdown until the
  Cython kernel lands. Do not run a full-field sweep in flux mode before phase 4.
* **Schema churn.** `f0_<band>` vs `mu_<band>` ripples into `fit_meta`, the parquet
  outputs and the candidate notebooks. Carrying both is cheap insurance.
* **The prefilter is a hidden dependency.** If its thresholds are not recalibrated,
  flux mode could silently reject the faint objects it was built to recover — which
  would look like the change failing.

## 8. Effort

Rough, assuming familiarity with the existing solver:

| phase | estimate |
|---|---|
| 1. NumPy solver + `fit_flux` + tests | 1–2 days |
| 2. Validation runs and analysis | half a day, mostly compute |
| 3. `dp2.py` wiring + prefilter recalibration | 1 day |
| 4. Cython kernel + rebuild | 1 day |

The cheap interim move, available today and independent of all of the above: set
`DP2Config.magerr_max = 0.5`, which is a one-line change worth ~18 points of faint
recovery on its own.
