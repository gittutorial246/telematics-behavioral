"""Model comparison figure: five model families under the random-split and
leave-one-driver-out protocols.

Reads results/uah/baseline_comparison.json; runs no models itself, so it can be
regenerated without repeating the sweep. Trip-grouped validation is a third
protocol in the paper but is measured for Random Forest only, so it is not
plotted here.
"""
import json, os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RESULTS, FIGURES, ensure_dirs

b = json.load(open(RESULTS/'baseline_comparison.json'))
ensure_dirs()
FIGDIR = str(FIGURES)

order = b["order"]
short = {"Majority-class baseline": "Majority\nclass",
         "Logistic Regression": "Logistic\nRegression",
         "Decision Tree": "Decision\nTree",
         "Random Forest": "Random\nForest",
         "Gradient Boosting (XGBoost)": "XGBoost"}
rand = [b["results"][k]["random_acc"] for k in order]
lodo = [b["results"][k]["lodo_acc"] for k in order]

x = np.arange(len(order)); w = 0.36
fig, ax = plt.subplots(figsize=(7, 4.2))
r1 = ax.bar(x - w/2, rand, w, label="Random split (driver-dependent)", color="#3B6EA5")
r2 = ax.bar(x + w/2, lodo, w, label="Leave-one-driver-out", color="#E07B39")
ax.axhline(1/3, color="gray", linestyle="--", linewidth=1,
           label="Uniform random-guess accuracy (33.3%)")
for rects, vals in ((r1, rand), (r2, lodo)):
    for rect, v in zip(rects, vals):
        ax.text(rect.get_x() + rect.get_width()/2, v + 0.012, f"{v:.3f}",
                ha="center", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels([short[k] for k in order], fontsize=8.5)
ax.set_ylabel("Accuracy"); ax.set_ylim(0, 0.80)
# title removed: the caption belongs in the article text (IOP guidelines)
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, axis="y", linestyle=":", linewidth=0.5)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/model_comparison.png", dpi=200)
plt.close(fig)
print(f"Saved {FIGDIR}/model_comparison.png")
