"""Two-stage RRL identification + period-refinement test.

Single-process per object (no multiprocessing — assumed to be parallelised at
the object level via Dask/LSDB).

Stage 1 (triage): fit a TINY library against each simulated LC over a COARSE
period grid. Goal: classify RRab vs RRc and get an approximate period.

Stage 2 (refine): fit the FULL RRab library on a NARROW period range around
the Stage 1 period at a FINE dphi. Goal: pin down the period.

We measure per-object wall time and period accuracy. Compares three Stage 1
options:
  (A) 2 average templates: avg RRab + avg RRc
  (B) k=5 RRab medoids + k=5 RRc medoids (10 templates total)
  (C) k=10 RRab medoids + k=10 RRc medoids (20 templates total)

Run with the pycycle conda env.
"""
from __future__ import annotations

import contextlib
import io
import os
import pickle
import sys
import time

import numpy as np

# pycycle is editable-installed in the pycycle conda env; no path hack needed.
from pycycle.templates import (
    load_multiband_templates,
    average_multiband_templates,
)
from pycycle.template_fit import TemplateFitter

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RRAB_ZIP = "/astro/store/shire/pferguso/software/Multiband-templates/RRab_normalized.zip"
RRC_ZIP = "/astro/store/shire/pferguso/software/Multiband-templates/RRc_normalized.zip"

# Period ranges per class (days)
P_RANGE = {"RRab": (0.4, 0.9), "RRc": (0.25, 0.45)}


# ---------------------------------------------------------------------------
# Light-curve simulator (same as recovery_parallel.simulate_lc)
# ---------------------------------------------------------------------------

def simulate_lc(template, period_days, n_obs=30, noise=0.05, t_span=180.0,
                seed=1, mag_base=20.0, amplitude=1.0):
    rng = np.random.default_rng(seed)
    bands = template.bands
    per_band = n_obs // len(bands)
    extras = n_obs - per_band * len(bands)
    counts = [per_band + (1 if i < extras else 0) for i in range(len(bands))]
    t_all, mag_all, err_all, nam_all = [], [], [], []
    for i, b in enumerate(bands):
        t = rng.uniform(0, t_span, counts[i])
        ph = (t / period_days) % 1.0
        idx = (ph * template.n_phase).astype(int) % template.n_phase
        gamma_b = template.gamma[i, idx]
        mag = mag_base + amplitude * gamma_b + rng.normal(0, noise, size=counts[i])
        t_all.append(t); mag_all.append(mag)
        err_all.append(np.full(counts[i], noise))
        nam_all.extend([b] * counts[i])
    t = np.concatenate(t_all); mag = np.concatenate(mag_all)
    err = np.concatenate(err_all); filts = np.array(nam_all)
    order = np.argsort(t)
    return t[order], mag[order], err[order], filts[order]


def fit_one(template, hjd, mag, err, filts, *, pmin, pmax, dphi,
            n_newton=3, n_start=4):
    """Single template fit; returns (rss_min, period_at_min, fit_result)."""
    fitter = TemplateFitter(template, n_newton=n_newton, n_start=n_start)
    with contextlib.redirect_stdout(io.StringIO()):
        res = fitter.fit(hjd, mag, err, filts,
                         pmin=pmin, pmax=pmax, dphi=dphi)
    k = int(np.argmin(res.rss))
    return float(res.rss[k]), float(res.periods[k]), res


def fit_library(templates, hjd, mag, err, filts, **kw):
    """Loop over a list of templates; return best (rss, period, template_idx)."""
    best = (np.inf, np.nan, -1)
    for k, tpl in enumerate(templates):
        try:
            rss, p, _ = fit_one(tpl, hjd, mag, err, filts, **kw)
        except Exception:
            continue
        if rss < best[0]:
            best = (rss, p, k)
    return best


# ---------------------------------------------------------------------------
# Build stage-1 libraries
# ---------------------------------------------------------------------------

def build_libraries():
    rrab_all = load_multiband_templates(RRAB_ZIP)
    rrc_all = load_multiband_templates(RRC_ZIP)

    avg_ab = average_multiband_templates(rrab_all)
    avg_ab.name = "avg_RRab"
    avg_c = average_multiband_templates(rrc_all)
    avg_c.name = "avg_RRc"

    with open(os.path.join(OUT_DIR, "medoid_indices.pkl"), "rb") as fh:
        med = pickle.load(fh)
    med_ab_k5 = [rrab_all[i] for i in med["RRab"]["medoids_by_k"][5]]
    med_c_k5 = [rrc_all[i] for i in med["RRc"]["medoids_by_k"][5]]

    stage1 = {
        "A_averages":       {"templates": [avg_ab, avg_c],
                             "labels":    ["RRab", "RRc"]},
        "B_k5medoids":      {"templates": med_ab_k5 + med_c_k5,
                             "labels":    ["RRab"] * 5 + ["RRc"] * 5},
    }
    # Variant C (k=10 per class) dropped from this test — its per-trial cost
    # was ~3× higher and the conclusion is dominated by A and B. Keep the
    # medoid_indices around so callers can build it if they want.
    return rrab_all, rrc_all, stage1


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main(n_trials_per_class=20, seed=2027):
    rrab_all, rrc_all, stage1 = build_libraries()
    print(f"loaded {len(rrab_all)} RRab + {len(rrc_all)} RRc templates")

    rng = np.random.default_rng(seed)
    truth_pool = []
    for _ in range(n_trials_per_class):
        i = int(rng.integers(len(rrab_all)))
        p = float(rng.uniform(*P_RANGE["RRab"]))
        truth_pool.append(("RRab", i, p, rrab_all[i]))
    for _ in range(n_trials_per_class):
        i = int(rng.integers(len(rrc_all)))
        p = float(rng.uniform(*P_RANGE["RRc"]))
        truth_pool.append(("RRc", i, p, rrc_all[i]))

    results = []
    for trial, (cls_true, idx_true, p_true, tpl_true) in enumerate(truth_pool):
        hjd, mag, err, filts = simulate_lc(
            tpl_true, period_days=p_true, n_obs=30, noise=0.05,
            seed=trial + 200)

        row = {"trial": trial, "cls_true": cls_true, "idx_true": idx_true,
               "p_true": p_true}

        # ---- Stage 1 variants ----
        # Coarse search over the wide RRab+RRc period range
        pmin_s1, pmax_s1, dphi_s1 = 0.2, 1.0, 0.05
        kbest_by_variant = {}
        for name, lib in stage1.items():
            t0 = time.time()
            rss_per, p_per = [], []
            for tpl in lib["templates"]:
                try:
                    rss, p, _ = fit_one(tpl, hjd, mag, err, filts,
                                        pmin=pmin_s1, pmax=pmax_s1,
                                        dphi=dphi_s1, n_newton=3, n_start=4)
                except Exception:
                    rss, p = np.inf, np.nan
                rss_per.append(rss); p_per.append(p)
            dt = time.time() - t0
            rss_per = np.array(rss_per); p_per = np.array(p_per)
            kbest = int(np.argmin(rss_per))
            kbest_by_variant[name] = kbest
            row[f"s1_{name}_dt"] = dt
            row[f"s1_{name}_class"] = lib["labels"][kbest]
            row[f"s1_{name}_period"] = float(p_per[kbest])
            row[f"s1_{name}_rss"] = float(rss_per[kbest])
            row[f"s1_{name}_correct_class"] = (lib["labels"][kbest] == cls_true)

        # ---- Stage 2 (lite): refine period using the Stage-1 best template ----
        # Narrow window around Stage 1 (B) period, fine dphi, more Newton iters.
        # This is the production workflow: one template, narrow range, accurate.
        p_seed = row["s1_B_k5medoids_period"]
        best_lib = stage1["B_k5medoids"]["templates"]
        kbest_global = kbest_by_variant["B_k5medoids"]
        pmin_s2 = max(0.15, p_seed - 0.05)
        pmax_s2 = p_seed + 0.05
        t0 = time.time()
        rss_lite, p_lite, _ = fit_one(best_lib[kbest_global], hjd, mag, err, filts,
                                      pmin=pmin_s2, pmax=pmax_s2,
                                      dphi=0.005, n_newton=5, n_start=4)
        dt_lite = time.time() - t0
        row["s2_lite_dt"] = dt_lite
        row["s2_lite_period"] = p_lite
        row["s2_lite_rss"] = rss_lite

        results.append(row)
        print(f"  trial {trial:>2} [{cls_true}] P_true={p_true:.4f}  "
              f"s1A={row['s1_A_averages_class']:>4}/{row['s1_A_averages_period']:.3f}  "
              f"s1B={row['s1_B_k5medoids_class']:>4}/{row['s1_B_k5medoids_period']:.3f}  "
              f"s2_lite={p_lite:.5f} (dt={dt_lite:.2f}s)  "
              f"|dP|={abs(p_lite-p_true):.5f}",
              flush=True)

    with open(os.path.join(OUT_DIR, "two_stage_results.pkl"), "wb") as fh:
        pickle.dump(results, fh)
    print(f"\nwrote two_stage_results.pkl ({len(results)} trials)")

    summarise(results)


def summarise(results):
    import pandas as pd
    rows = []
    by_cls = {"RRab": [], "RRc": []}
    for r in results:
        by_cls[r["cls_true"]].append(r)

    for cls, trials in by_cls.items():
        # Classification accuracy per stage-1 variant
        for name in ["A_averages", "B_k5medoids"]:
            correct = np.array([t[f"s1_{name}_correct_class"] for t in trials])
            dt = np.array([t[f"s1_{name}_dt"] for t in trials])
            p_err = np.array([t[f"s1_{name}_period"] - t["p_true"] for t in trials])
            rows.append({
                "class": cls, "variant": name,
                "N": len(trials),
                "class_acc": f"{correct.sum()}/{len(trials)}",
                "med_dt_s": np.median(dt),
                "med_|dP|": np.median(np.abs(p_err)),
                "p90_|dP|": np.quantile(np.abs(p_err), 0.9),
            })
        # Stage 2 (lite only)
        for tag in ["s2_lite"]:
            dt = np.array([t[f"{tag}_dt"] for t in trials])
            p_err = np.array([t[f"{tag}_period"] - t["p_true"] for t in trials])
            rows.append({
                "class": cls, "variant": tag,
                "N": len(trials),
                "class_acc": "-",
                "med_dt_s": np.median(dt),
                "med_|dP|": np.median(np.abs(p_err)),
                "p90_|dP|": np.quantile(np.abs(p_err), 0.9),
            })
    df = pd.DataFrame(rows)
    print("\n", df.to_string(index=False))


if __name__ == "__main__":
    main(n_trials_per_class=15)
