import json
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import FEATURES, make_rf, load_windows, ensure_dirs, RESULTS
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import platform, sklearn

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception as e:
    HAVE_XGB = False
    print("xgboost unavailable:", e)

ensure_dirs()
df, X, y_str = load_windows()
groups = df["driver"].values

# encode labels for xgboost
classes = sorted(set(y_str))
cls_to_int = {c:i for i,c in enumerate(classes)}
y = np.array([cls_to_int[c] for c in y_str])

def eval_random(make_model):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    yt, yp = [], []
    for tr, te in skf.split(X, y):
        m = make_model()
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        yt.extend(y[te]); yp.extend(pred)
    return accuracy_score(yt, yp), f1_score(yt, yp, average="macro")

def eval_lodo(make_model):
    gkf = GroupKFold(n_splits=6)
    yt, yp = [], []
    for tr, te in gkf.split(X, y, groups):
        m = make_model()
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        yt.extend(y[te]); yp.extend(pred)
    return accuracy_score(yt, yp), f1_score(yt, yp, average="macro")

models = {
    "Majority-class baseline": lambda: DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    "Decision Tree": lambda: DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, class_weight="balanced", random_state=42),
    "Random Forest": make_rf,
}
if HAVE_XGB:
    # n_jobs=1: with tree_method="hist" XGBoost is not deterministic across
    # different thread counts, so pinning to one thread keeps the reported
    # figures reproducible on any machine.
    models["Gradient Boosting (XGBoost)"] = lambda: XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.9,
        colsample_bytree=0.9, random_state=42, n_jobs=1, eval_metric="mlogloss",
        tree_method="hist")

results = {}
order = ["Majority-class baseline","Logistic Regression","Decision Tree","Random Forest"]
if HAVE_XGB: order.append("Gradient Boosting (XGBoost)")

for name in order:
    mk = models[name]
    racc, rf1 = eval_random(mk)
    lacc, lf1 = eval_lodo(mk)
    results[name] = {"random_acc": racc, "random_f1": rf1, "lodo_acc": lacc, "lodo_f1": lf1}
    print(f"{name:32s} | random acc={racc:.3f} f1={rf1:.3f} | LODO acc={lacc:.3f} f1={lf1:.3f}")

env = {
    "python_version": sys.version.split()[0],
    "platform": platform.platform(),
    "processor": platform.processor() or platform.machine(),
    "sklearn_version": sklearn.__version__,
    "have_xgboost": HAVE_XGB,
}
try:
    import xgboost
    env["xgboost_version"] = xgboost.__version__
except Exception:
    pass

with open(RESULTS/'baseline_comparison.json', 'w') as f:
    json.dump({"results": results, "order": order, "env": env}, f, indent=2)
print("\nEnvironment:", env)
print("Saved baseline_comparison.json")
