from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import pandas as pd


KEY_FIELDS = [
    "timeStamp",
    "tripID",
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
    "accData",
    "mx",
    "my",
    "mz",
]

TIMESTAMP_PATTERN = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?$"
CHUNK_SIZE = 200_000


def inventory_file(path: Path) -> dict:
    started = perf_counter()
    row_count = 0
    bad_timestamp_count = 0
    non_monotonic_timestamp_count = 0
    min_timestamp: str | None = None
    max_timestamp: str | None = None
    previous_timestamp: str | None = None
    trip_ids: set[str] = set()
    missingness = Counter()
    acc_lengths = Counter()

    columns = list(pd.read_csv(path, nrows=0).columns)
    usecols = [column for column in columns if column in KEY_FIELDS]

    chunks = pd.read_csv(
        path,
        chunksize=CHUNK_SIZE,
        dtype=str,
        keep_default_na=False,
        usecols=usecols,
    )

    for chunk in chunks:
        chunk_rows = len(chunk)
        row_count += chunk_rows

        if "timeStamp" in chunk.columns:
            timestamps = chunk["timeStamp"].str.strip()
            valid_timestamps = timestamps[timestamps.str.match(TIMESTAMP_PATTERN)]
            bad_timestamp_count += chunk_rows - len(valid_timestamps)

            if not valid_timestamps.empty:
                chunk_min = valid_timestamps.min()
                chunk_max = valid_timestamps.max()
                if min_timestamp is None or chunk_min < min_timestamp:
                    min_timestamp = chunk_min
                if max_timestamp is None or chunk_max > max_timestamp:
                    max_timestamp = chunk_max

                values = valid_timestamps.to_numpy()
                if previous_timestamp is not None and values[0] < previous_timestamp:
                    non_monotonic_timestamp_count += 1
                if len(values) > 1:
                    non_monotonic_timestamp_count += int((values[1:] < values[:-1]).sum())
                previous_timestamp = values[-1]

        if "tripID" in chunk.columns:
            trip_ids.update(value for value in chunk["tripID"].unique() if value != "")

        for field in KEY_FIELDS:
            if field in chunk.columns:
                missingness[field] += int((chunk[field] == "").sum())

        if "accData" in chunk.columns:
            lengths = chunk.loc[chunk["accData"] != "", "accData"].str.len().value_counts()
            for length, count in lengths.items():
                acc_lengths[int(length)] += int(count)

    elapsed_seconds = perf_counter() - started
    file_size_bytes = path.stat().st_size

    return {
        "file_name": path.name,
        "device_id_inferred": path.stem,
        "file_size_bytes": file_size_bytes,
        "row_count": row_count,
        "column_count": len(columns),
        "columns": columns,
        "trip_count": len(trip_ids),
        "min_timestamp": min_timestamp or "",
        "max_timestamp": max_timestamp or "",
        "bad_timestamp_count": bad_timestamp_count,
        "non_monotonic_timestamp_count": non_monotonic_timestamp_count,
        "missingness": dict(missingness),
        "acc_length_patterns": dict(sorted(acc_lengths.items())),
        "elapsed_seconds": elapsed_seconds,
        "rows_per_second": row_count / elapsed_seconds if elapsed_seconds else 0.0,
    }


def write_inventory_csv(rows: list[dict], output_path: Path) -> None:
    fieldnames = [
        "file_name",
        "device_id_inferred",
        "file_size_bytes",
        "row_count",
        "column_count",
        "trip_count",
        "min_timestamp",
        "max_timestamp",
        "bad_timestamp_count",
        "non_monotonic_timestamp_count",
        "elapsed_seconds",
        "rows_per_second",
        "columns",
        "missingness",
        "acc_length_patterns",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "columns": json.dumps(row["columns"], sort_keys=True),
                    "missingness": json.dumps(row["missingness"], sort_keys=True),
                    "acc_length_patterns": json.dumps(
                        row["acc_length_patterns"], sort_keys=True
                    ),
                }
            )


def build_summary(rows: list[dict], dataset_dir: Path, elapsed_seconds: float) -> dict:
    all_columns = sorted({column for row in rows for column in row["columns"]})
    schema_patterns = Counter(tuple(row["columns"]) for row in rows)
    timestamp_mins = [row["min_timestamp"] for row in rows if row["min_timestamp"]]
    timestamp_maxes = [row["max_timestamp"] for row in rows if row["max_timestamp"]]
    total_rows = sum(row["row_count"] for row in rows)
    total_size = sum(row["file_size_bytes"] for row in rows)

    return {
        "dataset_dir": str(dataset_dir),
        "file_count": len(rows),
        "total_size_bytes": total_size,
        "total_size_gb": total_size / 1_000_000_000,
        "total_rows": total_rows,
        "inferred_device_count": len({row["device_id_inferred"] for row in rows}),
        "total_trip_ids_across_files": sum(row["trip_count"] for row in rows),
        "overall_min_timestamp": min(timestamp_mins) if timestamp_mins else "",
        "overall_max_timestamp": max(timestamp_maxes) if timestamp_maxes else "",
        "all_columns": all_columns,
        "schema_patterns": [
            {"columns": list(columns), "file_count": count}
            for columns, count in schema_patterns.items()
        ],
        "bad_timestamp_count": sum(row["bad_timestamp_count"] for row in rows),
        "non_monotonic_timestamp_count": sum(
            row["non_monotonic_timestamp_count"] for row in rows
        ),
        "elapsed_seconds": elapsed_seconds,
        "rows_per_second": total_rows / elapsed_seconds if elapsed_seconds else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory extracted LEVIN CSV files.")
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

    data_dir = args.data_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("results/levin").mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(data_dir.glob("*.csv"), key=lambda item: item.name)
    if not csv_paths:
        raise SystemExit(f"No CSV files found in {data_dir}")

    started = perf_counter()
    rows = [inventory_file(path) for path in csv_paths]
    elapsed_seconds = perf_counter() - started
    summary = build_summary(rows, data_dir, elapsed_seconds)

    inventory_path = output_dir / "levin_file_inventory.csv"
    summary_path = Path("results/levin") / "levin_dataset_summary.json"
    write_inventory_csv(rows, inventory_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Inventoried {summary['file_count']} files")
    print(f"Rows: {summary['total_rows']}")
    print(f"Time span: {summary['overall_min_timestamp']} to {summary['overall_max_timestamp']}")
    print(f"Wrote {inventory_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
