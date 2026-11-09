"""Serving-path benchmark: single-window latency, micro-batch throughput,
per-batch elapsed time, and a fleet-concurrency fill analysis.

Timing protocol follows Section 6.4 of the paper:
  - numeric libraries pinned to one thread, and the classifier scored with
    n_jobs=1, so the reported figures are genuinely single-core
  - training time measured and excluded from all latency figures
  - 200 warm-up predictions discarded before measurement
  - 2,000 isolated single-window predict calls timed individually
  - throughput measured over 5 independent repeats per batch size, so the
    reported figure carries an observed spread rather than a single reading

Thread pinning must happen before numpy or scikit-learn are imported.
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import json
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from common import make_rf, load_windows, ensure_dirs, RESULTS, WINDOW_S, STEP_S

# WINDOW_S and STEP_S come from common.py (Section 3.2). Each active vehicle
# emits one window every STEP_S seconds, not every WINDOW_S; importing them
# rather than redeclaring keeps this benchmark in step with the feature builder.
N_WARMUP = 200            # discarded before measurement
N_SINGLE = 2000           # timed single-window calls
N_REPEATS = 5             # independent repeats per batch size
N_BOOT = 10000            # bootstrap resamples for the latency mean
BATCHES = [1, 32, 128, 512, 2048]
FLEETS = [10, 100, 1_000, 10_000, 100_000]

ensure_dirs()
_df, X, y = load_windows()

# n_jobs=1: scoring a single window across many threads is dominated by thread
# dispatch, and the paper reports a single-core serving condition.
clf = make_rf(n_jobs=1)
t0 = time.perf_counter()
clf.fit(X, y)
train_s = time.perf_counter() - t0
print(f"model trained in {train_s:.2f} s (excluded from all figures below)")

# --- warm-up, discarded
for i in range(N_WARMUP):
    clf.predict(X[i:i + 1])

# --- single-window latency
t = np.empty(N_SINGLE)
for i in range(N_SINGLE):
    j = i % len(X)
    row = X[j:j + 1]
    t0 = time.perf_counter()
    clf.predict(row)
    t[i] = (time.perf_counter() - t0) * 1e3
lat = {k: float(v) for k, v in zip(
    ("mean", "p50", "p95", "p99", "max"),
    (t.mean(), *np.percentile(t, [50, 95, 99]), t.max()))}

# bootstrap a 95% interval on the mean, so the latency figure is reported with
# its sampling uncertainty rather than as a bare point estimate
_rng = np.random.default_rng(42)
_means = np.array([_rng.choice(t, len(t), replace=True).mean() for _ in range(N_BOOT)])
lat["mean_ci95"] = [float(v) for v in np.percentile(_means, [2.5, 97.5])]
lat["sd"] = float(t.std(ddof=1))

print("single-window latency ms: "
      + "  ".join(f"{k}={v:.2f}" for k, v in lat.items() if isinstance(v, float)))
print(f"  mean 95% CI [{lat['mean_ci95'][0]:.2f}, {lat['mean_ci95'][1]:.2f}] ms "
      f"over {N_SINGLE:,} calls")

# --- throughput and per-batch elapsed time
print(f"\n{'batch':>6s} {'thr win/s':>11s} {'spread':>17s} "
      f"{'per-batch ms':>13s} {'per-win ms':>11s}")
batch_stats = {}
for B in BATCHES:
    nb = max(1, min(len(X) // B, 200))
    thr_runs, batch_ms_runs = [], []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        for b in range(nb):
            clf.predict(X[b * B:(b + 1) * B])
        el = time.perf_counter() - t0
        thr_runs.append(nb * B / el)
        batch_ms_runs.append(el / nb * 1e3)
    thr = float(np.median(thr_runs))
    per_batch_ms = float(np.median(batch_ms_runs))
    batch_stats[B] = {"throughput_windows_per_s": thr,
                      "throughput_min": float(min(thr_runs)),
                      "throughput_max": float(max(thr_runs)),
                      "throughput_sd": float(np.std(thr_runs, ddof=1)),
                      "elapsed_ms_per_batch": per_batch_ms,
                      "elapsed_ms_per_batch_min": float(min(batch_ms_runs)),
                      "elapsed_ms_per_batch_max": float(max(batch_ms_runs)),
                      "inference_ms_per_window": per_batch_ms / B,
                      "n_batches_timed": nb,
                      "n_repeats": N_REPEATS}
    print(f"{B:>6d} {thr:>11,.0f} "
          f"{f'{min(thr_runs):,.0f}-{max(thr_runs):,.0f}':>17s} "
          f"{per_batch_ms:>13.2f} {per_batch_ms/B:>11.4f}")
print(f"  throughput is the median of {N_REPEATS} repeats; spread is min-max")

# --- fleet concurrency: how long a batch takes to fill
print(f"\ntime to fill one batch, seconds "
      f"(one window per active vehicle every {STEP_S:.1f}s)")
print(f"{'batch':>6s} " + " ".join(f"{n:>10,}" for n in FLEETS))
fill = {}
for B in BATCHES:
    fill[B] = {N: B * STEP_S / N for N in FLEETS}
    print(f"{B:>6d} " + " ".join(
        f"{fill[B][N]:>10.2f}" if fill[B][N] >= 0.01 else f"{'<0.01':>10s}"
        for N in FLEETS))

print("\nvehicles needed to fill a batch within a target wait")
need = {}
for target in (1.0, 0.5, 0.1):
    need[target] = {B: B * STEP_S / target for B in BATCHES}
    print(f"  <= {target:>4.1f}s : " +
          "  ".join(f"B={B}->{int(v):,}" for B, v in need[target].items()))

json.dump({"inference_latency_ms": lat,
           "batch_stats": batch_stats,
           "window_seconds": WINDOW_S,
           "step_seconds": STEP_S,
           "fill_time_seconds": {str(k): {str(n): v for n, v in d.items()}
                                 for k, d in fill.items()},
           "vehicles_needed_for_target_wait": {str(k): {str(b): v for b, v in d.items()}
                                               for k, d in need.items()},
           "n_single_calls": N_SINGLE,
           "n_warmup_calls": N_WARMUP,
           "n_repeats_per_batch": N_REPEATS,
           "n_bootstrap": N_BOOT,
           "train_seconds": train_s,
           "n_jobs": 1,
           "threads_pinned": True},
          open(RESULTS/'latency_measured.json', 'w'), indent=2)
print("\nSaved results/uah/latency_measured.json")
