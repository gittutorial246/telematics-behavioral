"""Bootstrap confidence intervals over folds for each validation protocol,
and a side-by-side comparison of the three.

Six drivers and five trip folds are small samples, so the paper reports
intervals rather than point estimates alone.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from common import RESULTS

SEED = 42
N_RESAMPLES = 20_000

lodo = json.load(open(RESULTS/'results.json'))
trip = json.load(open(RESULTS/'results_trip_grouped.json'))
rand = json.load(open(RESULTS/'results_driver_dependent.json'))

def ci(values, n=N_RESAMPLES):
    """Percentile bootstrap on the mean of the fold scores.

    Each call gets its own generator seeded identically, so an interval
    depends only on the values passed in. A single shared generator would
    make every interval depend on how many draws preceded it, which means
    reordering the keys below would silently change the published numbers.
    """
    rng = np.random.default_rng(SEED)
    means = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(n)])
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]

lodo_a = np.array([f["accuracy"] for f in lodo["fold_reports"]])
lodo_f = np.array([f["macro_f1"] for f in lodo["fold_reports"]])
trip_a = np.array([f["accuracy"] for f in trip["fold_reports"]])
trip_f = np.array([f["macro_f1"] for f in trip["fold_reports"]])

out = {
    "n_resamples": N_RESAMPLES, "seed": SEED,
    "random_acc": rand["driver_dependent_accuracy"],
    "random_f1": rand["driver_dependent_macro_f1"],
    "trip_grouped_acc": trip["overall_accuracy"], "trip_grouped_ci": ci(trip_a),
    "trip_grouped_f1": trip["overall_macro_f1"], "trip_grouped_f1_ci": ci(trip_f),
    "lodo_acc": lodo["overall_accuracy"], "lodo_ci": ci(lodo_a),
    "lodo_f1": lodo["overall_macro_f1"], "lodo_f1_ci": ci(lodo_f),
    "leakage_points": 100 * (rand["driver_dependent_accuracy"] - trip["overall_accuracy"]),
    "driver_effect_points": 100 * (lodo["overall_accuracy"] - trip["overall_accuracy"]),
    "per_driver_acc_sd": float(lodo_a.std(ddof=1)),
    "per_driver_f1_sd": float(lodo_f.std(ddof=1)),
}
print(f"{'protocol':32s} {'acc':>7s} {'95% CI':>16s} {'F1':>7s}")
print("-" * 66)
print(f"{'Random 5-fold (window level)':32s} {out['random_acc']:7.4f} {'--':>16s} {out['random_f1']:7.4f}")
tg=f"[{out['trip_grouped_ci'][0]:.3f}, {out['trip_grouped_ci'][1]:.3f}]"
lo=f"[{out['lodo_ci'][0]:.3f}, {out['lodo_ci'][1]:.3f}]"
print(f"{'Trip-grouped 5-fold':32s} {out['trip_grouped_acc']:7.4f} {tg:>16s} {out['trip_grouped_f1']:7.4f}")
print(f"{'Leave-one-driver-out 6-fold':32s} {out['lodo_acc']:7.4f} {lo:>16s} {out['lodo_f1']:7.4f}")
print(f"\nrandom -> trip-grouped : {out['leakage_points']:+.1f} points  (within-trip leakage)")
print(f"trip-grouped -> LODO   : {out['driver_effect_points']:+.1f} points  (unseen driver)")

json.dump(out, open(RESULTS/'validation_protocol_comparison.json', 'w'), indent=2)
print("\nSaved results/uah/validation_protocol_comparison.json")
