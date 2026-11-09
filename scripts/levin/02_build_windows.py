from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


CHUNK_SIZE = 100_000
ACC_HEX_LENGTH = 150
ACC_SAMPLES_PER_ROW = 25
ACC_SCALE = 0.01536
# Window length, declared once. The pandas floor alias is derived from it so the
# figure reported in the metrics cannot drift from the windowing actually done.
WINDOW_SECONDS = 5
WINDOW_FLOOR_ALIAS = f"{WINDOW_SECONDS}s"
OBD_FIELDS = [
    "gps_speed",
    "battery",
    "cTemp",
    "dtc",
    "eLoad",
    "iat",
    "imap",
    "kpl",
    "maf",
    "rpm",
    "speed",
    "tAdv",
    "tPos",
]
READ_COLUMNS = ["timeStamp", "tripID", "accData", *OBD_FIELDS]
ACC_AXES = ["x", "y", "z", "mag"]


def decode_acc_payloads(payloads: pd.Series) -> dict[str, np.ndarray]:
    hex_blob = "".join(payloads.tolist())
    raw = np.frombuffer(bytes.fromhex(hex_blob), dtype=np.uint8)
    raw = raw.reshape(len(payloads), ACC_SAMPLES_PER_ROW, 3)

    signed = raw.astype(np.int16)
    signed[signed > 127] -= 256
    acc = signed.astype(np.float32) * ACC_SCALE
    mag = np.sqrt(np.sum(acc * acc, axis=2))

    return {
        "x_sum": acc[:, :, 0].sum(axis=1),
        "x_sumsq": (acc[:, :, 0] ** 2).sum(axis=1),
        "x_min": acc[:, :, 0].min(axis=1),
        "x_max": acc[:, :, 0].max(axis=1),
        "y_sum": acc[:, :, 1].sum(axis=1),
        "y_sumsq": (acc[:, :, 1] ** 2).sum(axis=1),
        "y_min": acc[:, :, 1].min(axis=1),
        "y_max": acc[:, :, 1].max(axis=1),
        "z_sum": acc[:, :, 2].sum(axis=1),
        "z_sumsq": (acc[:, :, 2] ** 2).sum(axis=1),
        "z_min": acc[:, :, 2].min(axis=1),
        "z_max": acc[:, :, 2].max(axis=1),
        "mag_sum": mag.sum(axis=1),
        "mag_sumsq": (mag**2).sum(axis=1),
        "mag_min": mag.min(axis=1),
        "mag_max": mag.max(axis=1),
    }


def aggregate_chunk(chunk: pd.DataFrame, device_id: str) -> pd.DataFrame:
    timestamps = pd.to_datetime(chunk["timeStamp"], errors="coerce")
    valid = (
        timestamps.notna()
        & chunk["tripID"].ne("")
        & chunk["accData"].str.len().eq(ACC_HEX_LENGTH)
    )
    valid_chunk = chunk.loc[valid].copy()
    if valid_chunk.empty:
        return pd.DataFrame()

    timestamps = timestamps.loc[valid]
    valid_chunk["device_id"] = device_id
    valid_chunk["window_start"] = timestamps.dt.floor(WINDOW_FLOOR_ALIAS).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    valid_chunk["acc_sample_count"] = ACC_SAMPLES_PER_ROW

    decoded = decode_acc_payloads(valid_chunk["accData"])
    for name, values in decoded.items():
        valid_chunk[name] = values

    for field in OBD_FIELDS:
        valid_chunk[field] = pd.to_numeric(valid_chunk[field], errors="coerce")
        valid_chunk[f"{field}_count"] = valid_chunk[field].notna().astype(np.int32)

    group_keys = ["device_id", "tripID", "window_start"]
    aggregations: dict[str, str | list[str]] = {
        "acc_sample_count": "sum",
        "timeStamp": "count",
    }

    for field in OBD_FIELDS:
        aggregations[field] = ["sum", "min", "max"]
        aggregations[f"{field}_count"] = "sum"

    for axis in ACC_AXES:
        aggregations[f"{axis}_sum"] = "sum"
        aggregations[f"{axis}_sumsq"] = "sum"
        aggregations[f"{axis}_min"] = "min"
        aggregations[f"{axis}_max"] = "max"

    grouped = valid_chunk.groupby(group_keys, sort=False).agg(aggregations)
    grouped.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in grouped.columns
    ]
    rename_map = {
        "acc_sample_count_sum": "acc_sample_count",
        "timeStamp_count": "obd_row_count",
    }
    for field in OBD_FIELDS:
        rename_map[f"{field}_count_sum"] = f"{field}_count"
    for axis in ACC_AXES:
        rename_map[f"{axis}_sum_sum"] = f"{axis}_sum"
        rename_map[f"{axis}_sumsq_sum"] = f"{axis}_sumsq"
        rename_map[f"{axis}_min_min"] = f"{axis}_min"
        rename_map[f"{axis}_max_max"] = f"{axis}_max"
    grouped = grouped.rename(columns=rename_map)
    return grouped.reset_index()


def combine_partials(partials: list[pd.DataFrame]) -> pd.DataFrame:
    if not partials:
        return pd.DataFrame()

    partial = pd.concat(partials, ignore_index=True)
    group_keys = ["device_id", "tripID", "window_start"]
    aggregations: dict[str, str] = {
        "acc_sample_count": "sum",
        "obd_row_count": "sum",
    }

    for field in OBD_FIELDS:
        aggregations[f"{field}_sum"] = "sum"
        aggregations[f"{field}_min"] = "min"
        aggregations[f"{field}_max"] = "max"
        aggregations[f"{field}_count"] = "sum"

    for axis in ACC_AXES:
        aggregations[f"{axis}_sum"] = "sum"
        aggregations[f"{axis}_sumsq"] = "sum"
        aggregations[f"{axis}_min"] = "min"
        aggregations[f"{axis}_max"] = "max"

    combined = partial.groupby(group_keys, sort=False).agg(aggregations).reset_index()
    return finalize_windows(combined)


def finalize_windows(windows: pd.DataFrame) -> pd.DataFrame:
    for field in OBD_FIELDS:
        count = windows[f"{field}_count"].replace(0, np.nan)
        windows[f"{field}_mean"] = windows[f"{field}_sum"] / count
        windows = windows.drop(columns=[f"{field}_sum"])

    sample_count = windows["acc_sample_count"].replace(0, np.nan)
    for axis in ACC_AXES:
        mean = windows[f"{axis}_sum"] / sample_count
        variance = windows[f"{axis}_sumsq"] / sample_count - mean * mean
        variance = variance.clip(lower=0)
        windows[f"acc_{axis}_mean"] = mean
        windows[f"acc_{axis}_std"] = np.sqrt(variance)
        windows[f"acc_{axis}_min"] = windows[f"{axis}_min"]
        windows[f"acc_{axis}_max"] = windows[f"{axis}_max"]
        windows = windows.drop(
            columns=[
                f"{axis}_sum",
                f"{axis}_sumsq",
                f"{axis}_min",
                f"{axis}_max",
            ]
        )

    ordered = [
        "device_id",
        "tripID",
        "window_start",
        "obd_row_count",
        "acc_sample_count",
    ]
    for field in OBD_FIELDS:
        ordered.extend(
            [
                f"{field}_count",
                f"{field}_mean",
                f"{field}_min",
                f"{field}_max",
            ]
        )
    for axis in ACC_AXES:
        ordered.extend(
            [
                f"acc_{axis}_mean",
                f"acc_{axis}_std",
                f"acc_{axis}_min",
                f"acc_{axis}_max",
            ]
        )
    return windows[ordered]


def process_file(path: Path, output_path: Path, write_header: bool) -> dict:
    started = perf_counter()
    device_id = path.stem
    partials: list[pd.DataFrame] = []
    row_count = 0
    valid_acc_rows = 0
    short_acc_rows = 0
    missing_acc_rows = 0

    columns = list(pd.read_csv(path, nrows=0).columns)
    usecols = [column for column in READ_COLUMNS if column in columns]

    for chunk in pd.read_csv(
        path,
        chunksize=CHUNK_SIZE,
        dtype=str,
        keep_default_na=False,
        usecols=usecols,
    ):
        row_count += len(chunk)
        acc_lengths = chunk["accData"].str.len() if "accData" in chunk.columns else pd.Series(dtype=int)
        valid_acc_rows += int(acc_lengths.eq(ACC_HEX_LENGTH).sum())
        short_acc_rows += int((acc_lengths.ne(0) & acc_lengths.ne(ACC_HEX_LENGTH)).sum())
        missing_acc_rows += int(acc_lengths.eq(0).sum())
        partial = aggregate_chunk(chunk, device_id)
        if not partial.empty:
            partials.append(partial)

    windows = combine_partials(partials)
    windows_produced = len(windows)
    # Trips surviving decode, per device. The paper reports the total, so it
    # needs an artifact behind it rather than a figure derived by hand.
    trips_decoded = int(windows["tripID"].nunique()) if windows_produced else 0
    if windows_produced:
        windows.to_csv(output_path, mode="a", header=write_header, index=False)
        write_header = False

    elapsed = perf_counter() - started
    return {
        "file_name": path.name,
        "device_id": device_id,
        "row_count": row_count,
        "valid_acc_rows": valid_acc_rows,
        "short_acc_rows": short_acc_rows,
        "missing_acc_rows": missing_acc_rows,
        "decoded_acc_samples": valid_acc_rows * ACC_SAMPLES_PER_ROW,
        "windows_produced": windows_produced,
        "trips_decoded": trips_decoded,
        "elapsed_seconds": elapsed,
        "rows_per_second": row_count / elapsed if elapsed else 0.0,
        "decoded_samples_per_second": (
            valid_acc_rows * ACC_SAMPLES_PER_ROW / elapsed if elapsed else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode LEVIN accData and build 5-second feature windows."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/levin/open_obd_data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/levin"),
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("results/levin").mkdir(parents=True, exist_ok=True)
    window_path = output_dir / "levin_window_features.csv"
    metrics_path = Path("results/levin") / "levin_pipeline_metrics.json"

    if window_path.exists():
        window_path.unlink()

    csv_paths = sorted(args.data_dir.glob("*.csv"), key=lambda item: item.name)
    if not csv_paths:
        raise SystemExit(f"No CSV files found in {args.data_dir}")

    total_started = perf_counter()
    write_header = True
    file_metrics = []
    for path in csv_paths:
        metrics = process_file(path, window_path, write_header)
        write_header = False if metrics["windows_produced"] else write_header
        file_metrics.append(metrics)
        print(
            f"{path.name}: rows={metrics['row_count']} "
            f"valid_acc_rows={metrics['valid_acc_rows']} "
            f"windows={metrics['windows_produced']}"
        )

    elapsed = perf_counter() - total_started
    total_rows = sum(item["row_count"] for item in file_metrics)
    total_valid_acc_rows = sum(item["valid_acc_rows"] for item in file_metrics)
    total_decoded_samples = sum(item["decoded_acc_samples"] for item in file_metrics)
    total_windows = sum(item["windows_produced"] for item in file_metrics)
    total_trips_decoded = sum(item["trips_decoded"] for item in file_metrics)
    devices_decoded = sum(1 for item in file_metrics if item["windows_produced"])
    total_short_acc_rows = sum(item["short_acc_rows"] for item in file_metrics)
    total_missing_acc_rows = sum(item["missing_acc_rows"] for item in file_metrics)

    metrics = {
        "data_dir": str(args.data_dir),
        "window_features_path": str(window_path),
        "file_count": len(csv_paths),
        "total_rows_read": total_rows,
        "valid_acc_rows": total_valid_acc_rows,
        "short_acc_rows": total_short_acc_rows,
        "missing_acc_rows": total_missing_acc_rows,
        "decoded_acc_samples": total_decoded_samples,
        "windows_produced": total_windows,
        "trips_decoded": total_trips_decoded,
        "devices_decoded": devices_decoded,
        "window_seconds": WINDOW_SECONDS,
        "acc_samples_per_valid_row": ACC_SAMPLES_PER_ROW,
        "elapsed_seconds": elapsed,
        "rows_per_second": total_rows / elapsed if elapsed else 0.0,
        "decoded_samples_per_second": (
            total_decoded_samples / elapsed if elapsed else 0.0
        ),
        "windows_per_second": total_windows / elapsed if elapsed else 0.0,
        "files": file_metrics,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Wrote {window_path}")
    print(f"Wrote {metrics_path}")
    print(f"Rows read: {total_rows}")
    print(f"Decoded samples: {total_decoded_samples}")
    print(f"Windows produced: {total_windows}")
    print(f"Decoded from: {devices_decoded} devices, {total_trips_decoded} trips")
    print(f"Elapsed seconds: {elapsed:.2f}")
    print(f"Rows/sec: {metrics['rows_per_second']:.2f}")
    print(f"Decoded samples/sec: {metrics['decoded_samples_per_second']:.2f}")
    print(f"Windows/sec: {metrics['windows_per_second']:.2f}")


if __name__ == "__main__":
    main()
