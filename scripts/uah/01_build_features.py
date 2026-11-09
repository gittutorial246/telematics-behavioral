"""
Feature extraction over the UAH-DriveSet (Romera et al., 2016).
Parses RAW_ACCELEROMETERS.txt (inertial, ~high-rate) and RAW_GPS.txt (1Hz)
for each of the 40 trips, segments each trip into fixed-length
sliding windows (simulating what a real-time streaming feature layer would
compute), and produces a window-level feature table with the trip-level
behavior label (normal / aggressive / drowsy) attached.

Column layout (from Eromera/uah_driveset_reader GUI source + Romera et al. 2016):
RAW_ACCELEROMETERS.txt:
  0 timestamp(s) | 1 activation_bool(speed>50kmh) | 2 X_raw(g) | 3 Y_raw(g) | 4 Z_raw(g)
  5 X_KF(g) | 6 Y_KF(g, lateral) | 7 Z_KF(g, longitudinal) | 8 roll | 9 pitch | 10 yaw

RAW_GPS.txt:
  0 timestamp(s) | 1 speed(km/h) | 2 lat | 3 lon | 4 altitude
  5 vertical_accuracy | 6 horizontal_accuracy | 7 course | 8 difcourse | ...

No raw dataset content is retained in outputs -- only derived, aggregated
features and metrics, consistent with the UAH-DriveSet license terms.
"""
import os, re, glob, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import WINDOW_S, STEP_S, WINDOWS, ensure_dirs

ROOT = "data/raw/uah-driveset"
HARSH_LON_G = 0.30   # harsh accel/brake threshold (g)
HARSH_LAT_G = 0.20   # harsh turn threshold (g)

ensure_dirs()

trip_dirs = sorted(glob.glob(os.path.join(ROOT, "D*", "*")))
trip_dirs = [d for d in trip_dirs if os.path.isdir(d)]
print(f"Found {len(trip_dirs)} trip folders")

name_re = re.compile(r"(?P<date>\d+)-(?P<dist>[\d.]+km)-(?P<driver>D\d+)-(?P<behavior>[A-Z0-9]+)-(?P<road>MOTORWAY|SECONDARY)")

def normalize_behavior(b):
    b = b.upper()
    if b.startswith("NORMAL"):
        return "normal"
    if b.startswith("AGGRESSIVE"):
        return "aggressive"
    if b.startswith("DROWSY"):
        return "drowsy"
    return b.lower()

rows = []
skipped = []
trip_index = 0
for d in trip_dirs:
    base = os.path.basename(d)
    m = name_re.match(base)
    if not m:
        skipped.append(base)
        continue
    driver = m.group("driver")
    behavior = normalize_behavior(m.group("behavior"))
    road = m.group("road").lower()

    acc_path = os.path.join(d, "RAW_ACCELEROMETERS.txt")
    gps_path = os.path.join(d, "RAW_GPS.txt")
    if not (os.path.exists(acc_path) and os.path.exists(gps_path)):
        skipped.append(base)
        continue

    acc = np.loadtxt(acc_path)
    gps = np.loadtxt(gps_path)
    if acc.ndim != 2 or acc.shape[1] < 8 or gps.ndim != 2 or gps.shape[1] < 2:
        skipped.append(base)
        continue

    trip_index += 1
    t_acc = acc[:, 0]
    y_kf = acc[:, 6]   # lateral
    z_kf = acc[:, 7]   # longitudinal

    t_gps = gps[:, 0]
    speed = gps[:, 1]

    t_end = min(t_acc[-1], t_gps[-1])
    t0 = max(t_acc[0], t_gps[0])

    w_start = t0
    while w_start + WINDOW_S <= t_end:
        w_end = w_start + WINDOW_S
        acc_mask = (t_acc >= w_start) & (t_acc < w_end)
        gps_mask = (t_gps >= w_start) & (t_gps < w_end)

        if acc_mask.sum() < 5 or gps_mask.sum() < 1:
            w_start += STEP_S
            continue

        z_win = z_kf[acc_mask]
        y_win = y_kf[acc_mask]
        spd_win = speed[gps_mask]

        # jerk = derivative of longitudinal accel
        jerk = np.diff(z_win)

        feat = {
            "trip_id": trip_index,
            "driver": driver,
            "road": road,
            "behavior": behavior,
            "w_start": w_start,
            "speed_mean": np.mean(spd_win),
            "speed_std": np.std(spd_win),
            "speed_max": np.max(spd_win),
            "lon_accel_mean": np.mean(z_win),
            "lon_accel_std": np.std(z_win),
            "lon_accel_absmax": np.max(np.abs(z_win)),
            "lat_accel_mean": np.mean(y_win),
            "lat_accel_std": np.std(y_win),
            "lat_accel_absmax": np.max(np.abs(y_win)),
            "harsh_lon_rate": np.mean(np.abs(z_win) > HARSH_LON_G),
            "harsh_lat_rate": np.mean(np.abs(y_win) > HARSH_LAT_G),
            "jerk_std": np.std(jerk) if len(jerk) > 1 else 0.0,
        }
        rows.append(feat)
        w_start += STEP_S

if skipped:
    print(f"Skipped {len(skipped)} folders (malformed/missing files): {skipped[:5]}...")

df = pd.DataFrame(rows)
print("Total trips used:", df["trip_id"].nunique())
print("Total windows:", len(df))
print(df["behavior"].value_counts())
print(df.groupby("driver")["trip_id"].nunique())

df.to_parquet(WINDOWS)
print(f"Saved {WINDOWS}")
