import json, os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RESULTS, FIGURES, ensure_dirs

with open(RESULTS/'results.json') as f:
    res = json.load(f)
with open(RESULTS/'latency_measured.json') as f:
    lat = json.load(f)

ensure_dirs()
FIGDIR = str(FIGURES)

# --- Confusion matrix (LODO) ---
cm = np.array(res["confusion_matrix"])
labels = res["labels"]
cm_norm = cm / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(5, 4.2))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
# title removed: the caption belongs in the article text (IOP guidelines)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]*100:.0f}%)", ha="center", va="center",
                color="white" if cm_norm[i,j] > 0.5 else "black", fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/confusion_matrix.png", dpi=200)
plt.close()

# --- Feature importance ---
imp = res["feature_importances"]
names = [d["feature"] for d in imp][::-1]
vals = [d["importance"] for d in imp][::-1]
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.barh(names, vals, color="#3B6EA5")
ax.set_xlabel("Random Forest feature importance (Gini)")
# title removed: the caption belongs in the article text (IOP guidelines)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/feature_importance.png", dpi=200)
plt.close()

# --- Protocol comparison: all three validation protocols ---
# The paper's central result is the three-way decomposition, so this figure
# shows the intermediate trip-grouped protocol rather than only the endpoints.
with open(RESULTS/'results_driver_dependent.json') as f:
    dd = json.load(f)
with open(RESULTS/'results_trip_grouped.json') as f:
    tg = json.load(f)

fig, ax = plt.subplots(figsize=(6.5, 4.2))
groups = ["Random split\n(driver-dependent)", "Trip-grouped\n(GroupKFold on trip)",
          "Leave-one-driver-out\n(driver-independent)"]
acc_vals = [dd["driver_dependent_accuracy"], tg["overall_accuracy"], res["overall_accuracy"]]
f1_vals = [dd["driver_dependent_macro_f1"], tg["overall_macro_f1"], res["overall_macro_f1"]]
x = np.arange(3)
w = 0.35
ax.bar(x - w/2, acc_vals, width=w, label="Accuracy", color="#3B6EA5")
ax.bar(x + w/2, f1_vals, width=w, label="Macro F1", color="#E07B39")
ax.axhline(1/3, color="gray", linestyle="--", linewidth=1, label="Uniform random-guess accuracy (33.3%)")
ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=8.5)
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
# title removed: the caption belongs in the article text (IOP guidelines)
ax.legend(fontsize=8)
for i, v in enumerate(acc_vals):
    ax.text(i - w/2, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
for i, v in enumerate(f1_vals):
    ax.text(i + w/2, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
# Annotate each change as a span between the two accuracy bars it relates,
# rather than as a floating label. Causes are discussed in Section 6.3; the
# figure reports the sizes only.
GREY = "#555555"
a0, a1, a2 = acc_vals            # random split, trip-grouped, LODO

def delta(x_at, lo, hi, text, line_from, line_to, side="right", dy=0.030):
    """Vertical span between two bar heights, with guide lines to each bar."""
    ax.plot([line_from, x_at + 0.06], [hi, hi], color=GREY, lw=0.7, ls=":", zorder=1)
    ax.plot([line_to, x_at - 0.06], [lo, lo], color=GREY, lw=0.7, ls=":", zorder=1)
    ax.annotate("", xy=(x_at, hi), xytext=(x_at, lo),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.0))
    off, ha = (0.07, "left") if side == "right" else (-0.07, "right")
    ax.text(x_at + off, hi + dy, text, fontsize=8.5, color=GREY,
            ha=ha, va="bottom", fontweight="bold")

delta(0.46, a1, a0, f"-{100*(a0-a1):.1f} pp accuracy", -0.175, 0.825,
      side="right", dy=-0.055)
delta(1.38, a1, a2, f"+{100*(a2-a1):.1f} pp accuracy", 1.825, 0.825,
      side="right", dy=0.052)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/generalization_gap.png", dpi=200)
plt.close()

# --- Throughput scaling by batch size (measured) ---
# Batch sizes come from whatever the benchmark actually measured, so this
# figure cannot fall out of step with BATCHES in 07_timing_benchmark.py.
batches = sorted(int(b) for b in lat["batch_stats"])
throughputs = [lat["batch_stats"][str(b)]["throughput_windows_per_s"] for b in batches]
# min-max over the five repeats, so the figure shows the spread the JSON records
# rather than implying the median is a precise measurement
lo = [t - lat["batch_stats"][str(b)]["throughput_min"] for t, b in zip(throughputs, batches)]
hi = [lat["batch_stats"][str(b)]["throughput_max"] - t for t, b in zip(throughputs, batches)]
fig, ax = plt.subplots(figsize=(5.5, 4))
mins = [lat["batch_stats"][str(b)]["throughput_min"] for b in batches]
maxs = [lat["batch_stats"][str(b)]["throughput_max"] for b in batches]
ax.fill_between(batches, mins, maxs, color="#3B6EA5", alpha=0.25, linewidth=0,
                label="min-max over 5 repeats")
ax.plot(batches, throughputs, marker="o", color="#3B6EA5", linewidth=1.5,
        label="median")
ax.legend(fontsize=8, loc="upper left", frameon=False)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("Micro-batch size (windows)")
ax.set_ylabel("Throughput (windows/sec, single-threaded)")
# title removed: the caption belongs in the article text (IOP guidelines)
ax.grid(True, which="both", linestyle=":", linewidth=0.5)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/throughput_scaling.png", dpi=200)
plt.close()

# --- Latency percentiles bar ---
li = lat["inference_latency_ms"]
fig, ax = plt.subplots(figsize=(5, 4))
keys = ["mean", "p50", "p95", "p99", "max"]
vals = [li[k] for k in keys]
ax.bar(keys, vals, color="#3B6EA5")
ax.set_ylabel("Latency (ms)")
# title removed: the caption belongs in the article text (IOP guidelines)
for i, v in enumerate(vals):
    ax.text(i, v + 0.2, f"{v:.1f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/latency_percentiles.png", dpi=200)
plt.close()

print("Figures generated:")
for f in os.listdir(FIGDIR):
    print(" ", f)
