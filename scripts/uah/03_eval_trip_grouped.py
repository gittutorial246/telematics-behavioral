"""Trip-grouped validation, an intermediate protocol between the random
window split and leave-one-driver-out.

Windows overlap by 50 percent, so a random split places neighbouring windows
from the same trip in both train and test. Grouping by trip removes that
leakage while still allowing the model to see every driver during training,
which isolates window-level leakage from driver-level generalisation.
"""
import json
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import FEATURES, make_rf, load_windows, ensure_dirs, RESULTS
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ensure_dirs()
df, X, y = load_windows()
g = df["trip_id"].values

gkf = GroupKFold(n_splits=5)
yt, yp, folds = [], [], []
for k, (tr, te) in enumerate(gkf.split(X, y, groups=g), 1):
    clf = make_rf()
    clf.fit(X[tr], y[tr]); p = clf.predict(X[te])
    a, f = accuracy_score(y[te], p), f1_score(y[te], p, average="macro")
    folds.append({"fold": k, "n_test_trips": int(len(set(g[te]))),
                  "n_test_windows": int(len(te)), "accuracy": a, "macro_f1": f})
    print(f"  fold {k}: {len(set(g[te])):2d} trips, {len(te):5,} windows, acc={a:.3f} f1={f:.3f}")
    yt.extend(y[te]); yp.extend(p)

labels = sorted(set(y))
acc, f1 = accuracy_score(yt, yp), f1_score(yt, yp, average="macro")
print(f"\nTrip-grouped pooled: accuracy={acc:.4f} macro_f1={f1:.4f}")
print(classification_report(yt, yp, labels=labels))

json.dump({"protocol": "trip-grouped 5-fold (GroupKFold on trip_id)",
           "overall_accuracy": acc, "overall_macro_f1": f1,
           "n_windows": int(len(df)), "n_trips": int(df.trip_id.nunique()),
           "fold_reports": folds,
           "confusion_matrix": confusion_matrix(yt, yp, labels=labels).tolist(),
           "labels": labels},
          open(RESULTS/'results_trip_grouped.json','w'), indent=2)
print("Saved results/uah/results_trip_grouped.json")
