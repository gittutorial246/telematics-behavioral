"""Driver-dependent (random 5-fold stratified) baseline, for comparison
against the trip-grouped and leave-one-driver-out protocols."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import FEATURES, make_rf, load_windows, ensure_dirs, RESULTS
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ensure_dirs()
_df, X, y = load_windows()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
yt, yp = [], []
for tr, te in skf.split(X, y):
    clf = make_rf()
    clf.fit(X[tr], y[tr]); p = clf.predict(X[te])
    yt.extend(y[te]); yp.extend(p)
labels = sorted(set(y))
acc, f1 = accuracy_score(yt, yp), f1_score(yt, yp, average='macro')
print('Driver-dependent (random split) pooled results:')
print('accuracy:', acc, 'macro_f1:', f1)
print(classification_report(yt, yp, labels=labels))
print(confusion_matrix(yt, yp, labels=labels))
json.dump({'driver_dependent_accuracy': acc, 'driver_dependent_macro_f1': f1},
          open(RESULTS/'results_driver_dependent.json','w'), indent=2)
