import json
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import FEATURES, make_rf, load_windows, ensure_dirs, RESULTS, WINDOW_S
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

ensure_dirs()
df, X, y = load_windows()

groups = df["driver"].values  # leave-one-driver-out to test true generalization

gkf = GroupKFold(n_splits=6)
all_true, all_pred = [], []
fold_reports = []

for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups)):
    clf = make_rf()
    clf.fit(X[tr_idx], y[tr_idx])
    pred = clf.predict(X[te_idx])
    all_true.extend(y[te_idx])
    all_pred.extend(pred)
    held_out_driver = np.unique(groups[te_idx])
    acc = accuracy_score(y[te_idx], pred)
    f1 = f1_score(y[te_idx], pred, average="macro")
    fold_reports.append({"held_out_driver": held_out_driver.tolist(), "accuracy": acc, "macro_f1": f1})
    print(f"Fold {fold} (held out {held_out_driver}): acc={acc:.3f} macro_f1={f1:.3f}")

labels = sorted(set(y))
cm = confusion_matrix(all_true, all_pred, labels=labels)
report = classification_report(all_true, all_pred, labels=labels, output_dict=True)
overall_acc = accuracy_score(all_true, all_pred)
overall_f1 = f1_score(all_true, all_pred, average="macro")

print("\n=== Leave-One-Driver-Out pooled results ===")
print("Labels order:", labels)
print("Confusion matrix:\n", cm)
print("Overall accuracy:", overall_acc)
print("Overall macro F1:", overall_f1)
print(classification_report(all_true, all_pred, labels=labels))

# Feature importance from a model trained on all data (for reporting which
# signals matter most -- informs the "behavioral feature layer" design section)
clf_full = make_rf()
clf_full.fit(X, y)
importances = sorted(zip(FEATURES, clf_full.feature_importances_), key=lambda x: -x[1])
print("\nFeature importances:")
for f, imp in importances:
    print(f"  {f}: {imp:.4f}")

results = {
    "labels": labels,
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "overall_accuracy": overall_acc,
    "overall_macro_f1": overall_f1,
    "fold_reports": fold_reports,
    "feature_importances": [{"feature": f, "importance": float(imp)} for f, imp in importances],
    "n_windows": len(df),
    "n_trips": int(df["trip_id"].nunique()),
    "n_drivers": int(df["driver"].nunique()),
    "window_s": WINDOW_S,
}
with open(RESULTS/'results.json', 'w') as fh:
    json.dump(results, fh, indent=2)
print("\nSaved results.json")
