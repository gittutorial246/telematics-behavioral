"""Figures for the LEVIN fleet-scale ingestion stage.

Reads results/levin/levin_pipeline_metrics.json only; runs no measurement, so it
can be regenerated without repeating the ingestion sweep.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

m = json.load(open("results/levin/levin_pipeline_metrics.json"))
FIGDIR = "figures/levin"
os.makedirs(FIGDIR, exist_ok=True)

dec = [f for f in m["files"] if f["windows_produced"] > 0]
skip = [f for f in m["files"] if f["windows_produced"] == 0]

# Device counts per payload format, derived from the metrics rather than typed
# in, so the labels cannot disagree with the bars they sit above if the LEVIN
# snapshot changes. A device is counted once, by the format it actually emits.
dev_150 = [f for f in m["files"] if f["valid_acc_rows"] > 0]
dev_12 = [f for f in m["files"] if f["valid_acc_rows"] == 0 and f["short_acc_rows"] > 0]
dev_none = [f for f in m["files"] if f["valid_acc_rows"] == 0 and f["short_acc_rows"] == 0]
n_dev = [len(dev_150), len(dev_12), len(dev_none)]
if sum(n_dev) != m["file_count"]:
    raise SystemExit(
        f"device counts {n_dev} sum to {sum(n_dev)}, expected {m['file_count']}"
    )
print(f"payload formats by device: 150-hex={n_dev[0]} 12-hex={n_dev[1]} none={n_dev[2]}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
ax.scatter([f["row_count"] for f in dec], [f["rows_per_second"] for f in dec],
           s=45, color="#3B6EA5", label=f"decoded ({len(dec)} devices)", zorder=3)
ax.scatter([f["row_count"] for f in skip], [f["rows_per_second"] for f in skip],
           s=45, marker="^", color="#E07B39", label=f"parse only ({len(skip)} devices)", zorder=3)
ax.set_xscale("log")
ax.set_xlabel("Records per device file")
ax.set_ylabel("Ingestion throughput (records/sec)")
ax.grid(True, which="both", linestyle=":", linewidth=0.5)
ax.legend(fontsize=8)

ax = axes[1]
cats = ["150-hex payload\n(25 samples/record)", "12-hex payload\n(2 samples/record)", "no payload"]
n_rows = [m["valid_acc_rows"], m["short_acc_rows"], m["missing_acc_rows"]]
x = range(3)
b = ax.bar(x, n_rows, color=["#3B6EA5", "#E07B39", "#B0B0B0"], width=0.6)
ax.set_xticks(list(x)); ax.set_xticklabels(cats, fontsize=8)
ax.set_ylabel("OBD records")
for rect, nr, nd in zip(b, n_rows, n_dev):
    ax.text(rect.get_x() + rect.get_width()/2, nr + max(n_rows) * 0.02,
            f"{nr:,}\n{nd} device{'s' if nd > 1 else ''}", ha="center", fontsize=8)
ax.set_ylim(0, max(n_rows) * 1.22)
ax.grid(True, axis="y", linestyle=":", linewidth=0.5)

fig.tight_layout()
fig.savefig(f"{FIGDIR}/ingestion_throughput.png", dpi=200)
plt.close(fig)
print(f"Saved {FIGDIR}/ingestion_throughput.png")
