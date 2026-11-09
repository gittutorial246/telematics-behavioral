"""Run the whole pipeline end to end and verify the deterministic results.

    python run_all.py

Runs the UAH stage then the LEVIN stage, writes a timestamped log to
results/run_log.txt, and checks every value that must reproduce exactly against
the committed figures. Stops at the first failure.

Optional flags:
    --skip-levin    UAH only; skips the ~8 minute ingestion stage
    --skip-uah      LEVIN only
"""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.resolve()

UAH = [
    ("01_build_features.py",        "extract 5 s windows from 40 trips"),
    ("02_eval_random_split.py",     "random 5-fold, driver-dependent"),
    ("03_eval_trip_grouped.py",     "trip-grouped 5-fold"),
    ("04_eval_lodo.py",             "leave-one-driver-out 6-fold"),
    ("05_baseline_comparison.py",   "five model families, both protocols"),
    ("06_bootstrap_intervals.py",   "bootstrap confidence intervals"),
    ("07_timing_benchmark.py",      "serving-path latency and throughput"),
    ("08_make_figures.py",          "UAH result figures"),
    ("09_make_arch_diagram.py",     "architecture diagram"),
    ("10_make_model_comparison.py", "model comparison figure"),
]

LEVIN = [
    ("01_inventory.py",   "inventory 29 device files"),
    ("02_build_windows.py", "decode accData into 5 s windows"),
    ("03_make_figures.py", "LEVIN ingestion figures"),
]

# Values that must reproduce exactly. Timing is excluded: it is hardware- and
# load-dependent by nature.
EXPECTED = [
    ("results/uah/results.json",                    "n_windows",              12310),
    ("results/uah/results.json",                    "n_trips",                40),
    ("results/uah/results.json",                    "n_drivers",              6),
    ("results/uah/results.json",                    "overall_accuracy",       0.5662063363119415),
    ("results/uah/results.json",                    "overall_macro_f1",       0.5724432481677334),
    ("results/uah/results_driver_dependent.json",   "driver_dependent_accuracy", 0.6633631194151096),
    ("results/uah/results_driver_dependent.json",   "driver_dependent_macro_f1", 0.6664577472397751),
    ("results/uah/results_trip_grouped.json",       "overall_accuracy",       0.5051990251827783),
    ("results/uah/results_trip_grouped.json",       "overall_macro_f1",       0.5144028899519447),
    ("results/levin/levin_pipeline_metrics.json",   "total_rows_read",        7214695),
    ("results/levin/levin_pipeline_metrics.json",   "decoded_acc_samples",    147307675),
    ("results/levin/levin_pipeline_metrics.json",   "windows_produced",       1186159),
    ("results/levin/levin_pipeline_metrics.json",   "devices_decoded",        22),
    ("results/levin/levin_pipeline_metrics.json",   "trips_decoded",          5333),
    ("results/levin/levin_dataset_summary.json",    "file_count",             29),
    ("results/levin/levin_dataset_summary.json",    "total_trip_ids_across_files", 6553),
]

# Artifacts the paper's tables are built from that EXPECTED does not otherwise
# touch. Rather than pin more magic constants, these are checked for internal
# agreement: each derived value must match what its own sources imply.
CONSISTENCY = [
    ("results/uah/validation_protocol_comparison.json", "leakage_points",
     ("results/uah/results_driver_dependent.json", "driver_dependent_accuracy",
      "results/uah/results_trip_grouped.json", "overall_accuracy")),
    ("results/uah/validation_protocol_comparison.json", "driver_effect_points",
     ("results/uah/results.json", "overall_accuracy",
      "results/uah/results_trip_grouped.json", "overall_accuracy")),
]

# Every model family the paper's Table 2 reports. A silently missing XGBoost
# would otherwise leave a blank row rather than an error.
EXPECTED_MODELS = [
    "Majority-class baseline",
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting (XGBoost)",
]

log_lines = []


def say(msg=""):
    print(msg, flush=True)
    log_lines.append(msg)


def run_stage(folder, scripts):
    for name, desc in scripts:
        path = Path("scripts") / folder / name
        if not (ROOT / path).exists():
            say(f"  MISSING  {path}")
            return False
        say(f"  running  {path}  ({desc})")
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, str(path)], cwd=ROOT,
                              capture_output=True, text=True)
        el = time.perf_counter() - t0
        log_lines.extend("      " + ln for ln in proc.stdout.rstrip().splitlines())
        if proc.returncode != 0:
            say(f"  FAILED   {path}  after {el:.1f}s")
            say("")
            say(proc.stderr.rstrip())
            return False
        say(f"  ok       {path}  {el:.1f}s")
    return True


def verify():
    say()
    say("Verifying deterministic results")
    say("-" * 72)
    ok = True
    for rel, key, want in EXPECTED:
        p = ROOT / rel
        if not p.exists():
            say(f"  MISSING FILE  {rel}")
            ok = False
            continue
        got = json.loads(p.read_text()).get(key)
        if got is None:
            say(f"  MISSING KEY   {rel}:{key}")
            ok = False
        elif isinstance(want, float):
            match = abs(got - want) < 1e-12
            say(f"  {'ok  ' if match else 'DIFF'}  {key:<28s} {got!r}"
                + ("" if match else f"   expected {want!r}"))
            ok &= match
        else:
            match = got == want
            say(f"  {'ok  ' if match else 'DIFF'}  {key:<28s} {got:,}"
                + ("" if match else f"   expected {want:,}"))
            ok &= match

    if not any("uah" in rel for rel, _, _ in EXPECTED):
        return ok

    for rel, key, (a_rel, a_key, b_rel, b_key) in CONSISTENCY:
        paths = [ROOT / rel, ROOT / a_rel, ROOT / b_rel]
        if not all(p.exists() for p in paths):
            say(f"  MISSING FILE  {rel} or one of its sources")
            ok = False
            continue
        got = json.loads(paths[0].read_text()).get(key)
        a = json.loads(paths[1].read_text()).get(a_key)
        b = json.loads(paths[2].read_text()).get(b_key)
        if got is None or a is None or b is None:
            say(f"  MISSING KEY   {rel}:{key} or a source key")
            ok = False
            continue
        want = 100 * (a - b)
        match = abs(got - want) < 1e-9
        say(f"  {'ok  ' if match else 'DIFF'}  {key:<28s} {got:.6f}"
            + ("" if match else f"   sources imply {want:.6f}"))
        ok &= match

    bc = ROOT / "results/uah/baseline_comparison.json"
    if not bc.exists():
        say("  MISSING FILE  results/uah/baseline_comparison.json")
        ok = False
    else:
        got_models = json.loads(bc.read_text()).get("order", [])
        missing = [m for m in EXPECTED_MODELS if m not in got_models]
        match = not missing
        say(f"  {'ok  ' if match else 'DIFF'}  {'model families':<28s} {len(got_models)}"
            + ("" if match else f"   missing {missing}"))
        ok &= match

    return ok


def main():
    args = sys.argv[1:]
    do_uah = "--skip-uah" not in args
    do_levin = "--skip-levin" not in args

    started = datetime.now()
    say(f"Pipeline run started {started:%Y-%m-%d %H:%M:%S}")
    say(f"Repository: {ROOT}")
    say(f"Python: {sys.version.split()[0]}")
    try:
        import sklearn, xgboost
        say(f"scikit-learn {sklearn.__version__}, xgboost {xgboost.__version__}")
    except ImportError as e:
        say(f"Dependency missing: {e}. Run: pip install -r requirements.txt")
        return 1
    say()

    t0 = time.perf_counter()
    if do_uah:
        say("UAH-DriveSet stage")
        say("-" * 72)
        if not run_stage("uah", UAH):
            say("\nStopped. Fix the failure above and rerun.")
            (ROOT / "results").mkdir(exist_ok=True)
            (ROOT / "results/run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
            return 1
        say()

    if do_levin:
        say("LEVIN stage")
        say("-" * 72)
        if not run_stage("levin", LEVIN):
            say("\nStopped. Fix the failure above and rerun.")
            (ROOT / "results").mkdir(exist_ok=True)
            (ROOT / "results/run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
            return 1

    total = time.perf_counter() - t0
    checks = [e for e in EXPECTED
              if (do_uah or "levin" in e[0]) and (do_levin or "uah" in e[0])]
    globals()["EXPECTED"] = checks
    ok = verify()

    say()
    say("=" * 72)
    say(f"Total run time: {total/60:.1f} minutes")
    if ok:
        say("All deterministic results reproduced exactly.")
        say("Timing figures are hardware-dependent and are expected to differ.")
    else:
        say("MISMATCH. Do not edit the scripts to force agreement; report the")
        say("difference, since it usually means a library version changed.")
    say("=" * 72)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    say(f"\nFull log written to results/run_log.txt")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
