# Telematics Behavioral Intelligence

Code and derived results for *From Telematics Data to Behavioral Intelligence:
A Scalable Analytics Architecture for Real-Time Predictive Decision Systems*.

Two independent experiments share this repository:

- **UAH** builds a windowed feature pipeline over naturalistic driving data,
  trains a driver-behavior classifier, evaluates it under three validation
  protocols, and benchmarks the serving path.
- **LEVIN** exercises the multi-rate ingestion stage at fleet scale on 7.2
  million OBD records.

They share no data and no code. LEVIN produces the ingestion and throughput
figures; UAH produces everything else. `scripts/`, `results/`, `data/processed/`
and `figures/` are each split accordingly.

## Repository layout

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── run_all.py                            <- runs everything and verifies it
├── .gitignore
├── data/
│   ├── raw/                              <- you download this; not in git
│   │   ├── uah-driveset/                 <- D1/ ... D6/ trip folders
│   │   └── levin/open_obd_data/          <- 29 per-vehicle CSV files
│   └── processed/                        <- generated; not in git
│       ├── uah/window_features.parquet             (~3 MB)
│       └── levin/levin_window_features.csv         (~490 MB)
│           levin/levin_file_inventory.csv
├── scripts/
│   ├── uah/                              <- driver-behavior classification
│   │   ├── common.py                     <- shared paths, constants, model
│   │   ├── 01_build_features.py
│   │   ├── 02_eval_random_split.py
│   │   ├── 03_eval_trip_grouped.py
│   │   ├── 04_eval_lodo.py
│   │   ├── 05_baseline_comparison.py
│   │   ├── 06_bootstrap_intervals.py
│   │   ├── 07_timing_benchmark.py
│   │   ├── 08_make_figures.py
│   │   ├── 09_make_arch_diagram.py
│   │   └── 10_make_model_comparison.py
│   └── levin/                            <- fleet-scale ingestion
│       ├── 01_inventory.py
│       ├── 02_build_windows.py
│       └── 03_make_figures.py
├── results/                              <- committed
│   ├── run_log.txt                       (written by run_all.py)
│   ├── uah/
│   │   ├── results.json                            (leave-one-driver-out)
│   │   ├── results_driver_dependent.json           (random split)
│   │   ├── results_trip_grouped.json               (trip-grouped)
│   │   ├── validation_protocol_comparison.json
│   │   ├── baseline_comparison.json
│   │   └── latency_measured.json
│   └── levin/
│       ├── levin_dataset_summary.json
│       └── levin_pipeline_metrics.json
└── figures/                              <- committed
    ├── architecture_diagram.png          (whole system)
    ├── uah/
    │   ├── confusion_matrix.png
    │   ├── feature_importance.png
    │   ├── generalization_gap.png
    │   ├── latency_percentiles.png
    │   ├── model_comparison.png
    │   └── throughput_scaling.png
    └── levin/
        └── ingestion_throughput.png
```

Raw datasets and large intermediates are excluded by `.gitignore`. Everything in
`results/` and `figures/` is committed, so every number in the paper can be
checked against its artifact without re-running anything.

Shared configuration lives in `scripts/uah/common.py`: the window length and
step, the twelve-feature list, the Random Forest parameters, and the output
paths. Nothing else declares them, so a change in one place cannot silently
disagree with another.

## Getting the data

### UAH-DriveSet

Released under Creative Commons Attribution-NonCommercial 4.0, so it cannot be
redistributed here.

1. Go to <http://www.robesafe.uah.es/personal/eduardo.romera/uah-driveset/>
2. Download **UAH-DRIVESET-v1**
3. Unzip and place the six driver folders directly under
   `data/raw/uah-driveset/`, so the tree reads:

```
data/raw/uah-driveset/
├── D1/
│   ├── 20151110175712-16km-D1-NORMAL1-SECONDARY/
│   │   ├── RAW_ACCELEROMETERS.txt
│   │   ├── RAW_GPS.txt
│   │   └── ...
│   └── ...
└── D6/
```

The pipeline reads only `RAW_ACCELEROMETERS.txt` and `RAW_GPS.txt`. Behavior
labels are parsed from the trip folder names, which encode driver, distance,
behavior and road type, for example `20151110175712-16km-D1-NORMAL1-SECONDARY`.

Verify the download before running:

```bash
find data/raw/uah-driveset -maxdepth 1 -name 'D*' -type d | wc -l   # expect 6
find data/raw/uah-driveset -name RAW_ACCELEROMETERS.txt | wc -l     # expect 40
```

The version used here is UAH-DRIVESET-v1, downloaded July 2026: 6 drivers,
40 trips, 3 behaviors (normal, drowsy, aggressive), 2 road types. If
`01_build_features.py` reports anything other than 40 trips and 12,310 windows,
the download differs from the one behind the committed results.

Cite: E. Romera, L. M. Bergasa, and R. Arroyo, "Need Data for Driver Behavior
Analysis? Presenting the Public UAH-DriveSet," in *Proc. IEEE 19th Int. Conf.
Intelligent Transportation Systems (ITSC)*, 2016, pp. 387-392.

### LEVIN

The GitHub repository holds the data description, the license and an extraction
notebook, but not the bulk data. The data itself is linked from that README and
hosted externally.

1. Go to <https://github.com/YunSolutions/levin-openData>
2. In its README, under "The full data set is available as", follow download
   option 3, "A Single Compressed Zip with individual CSVs"
3. Extract it so the per-vehicle CSV files sit directly under
   `data/raw/levin/open_obd_data/`

The same data is mirrored on Kaggle at
<https://www.kaggle.com/datasets/yunlevin/levin-vehicle-telematics>, which is
easier to script against if the Drive link has moved. The dataset is not
versioned by commit, so the row counts below are the check that you have the
same snapshot.

Verify the extraction:

```bash
ls data/raw/levin/open_obd_data/*.csv | wc -l    # expect 29
du -sh data/raw/levin/open_obd_data              # expect ~1.6 GB
```

The snapshot behind the committed results, downloaded July 2026, holds
7,214,695 OBD records across 29 files
(1,590,657,875 bytes), spanning
2017-08-30 to 2017-12-31, with
6,553 trip identifiers. `01_inventory.py` prints
the row count and time span, so compare them against
`results/levin/levin_dataset_summary.json` before trusting a re-run.

Two accelerometer payload formats appear in the archive: a 150-hex payload
(25 samples per record) on 22 devices and a 12-hex payload (2 samples) on 6, with
1 device emitting none. `02_build_windows.py` decodes only the 150-hex format, so
the windows it produces come from 22 of the 29 devices and 5,333 of the 6,553
trips. `03_make_figures.py` derives those device counts from the metrics and
stops if they do not sum to the file count.

The LEVIN README identifies CC BY-NC-SA as the license for the dataset, while
GitHub's repository metadata lists MIT. This work follows the license stated in
the LEVIN README: non-commercial use, no redistribution of the raw archive. Only
derived aggregates are committed here.

The LEVIN stage dominates the run time and writes a 490 MB intermediate to
`data/processed/levin/`.

## Running

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py
```

`run_all.py` runs the UAH stage then the LEVIN stage, stops at the first
failure, writes a full log to `results/run_log.txt`, and verifies every value
that must reproduce against the committed figures. It reports its own
total run time. `--skip-levin` runs UAH only; `--skip-uah` runs LEVIN only.

The verification covers window, trip and driver counts, the pooled accuracy and
macro F1 under all three protocols, the LEVIN record, sample and window totals,
the agreement of the protocol-gap figures with the results they are derived
from, and the presence of all five model families. Timing is deliberately
excluded, being hardware-dependent.

To run stages individually instead, from the repository root:

```bash
python scripts/uah/01_build_features.py        # 40 trips -> 12,310 windows
python scripts/uah/02_eval_random_split.py     # random 5-fold
python scripts/uah/03_eval_trip_grouped.py     # trip-grouped 5-fold
python scripts/uah/04_eval_lodo.py             # leave-one-driver-out
python scripts/uah/05_baseline_comparison.py   # five model families
python scripts/uah/06_bootstrap_intervals.py   # confidence intervals
python scripts/uah/07_timing_benchmark.py      # serving-path benchmark
python scripts/uah/08_make_figures.py
python scripts/uah/09_make_arch_diagram.py
python scripts/uah/10_make_model_comparison.py

python scripts/levin/01_inventory.py           # 29 files -> inventory + summary
python scripts/levin/02_build_windows.py       # decode accData -> 5 s windows
python scripts/levin/03_make_figures.py        # ingestion throughput figure
```

Scripts are numbered in execution order within each experiment; later stages
read the artifacts written by earlier ones. Run every script from the repository
root; paths are relative to it.

## Reproducibility

Random seeds are fixed in `scripts/uah/common.py` (`SEED = 42`), which supplies
`random_state` to the model and to every fold split. Feature extraction,
classification and storage results are deterministic given that seed and the
library versions pinned in `requirements.txt`. Re-running the pipeline in the
same environment reproduces every value in `results/uah/`, and `run_all.py`
checks each one to within 1e-12. Minor
version changes in scikit-learn can shift classification metrics in the third
decimal place, so the pinned versions should be used for exact comparison.

The single-threaded timing condition is enforced rather than assumed. The benchmark
sets every numeric library to one thread before importing numpy or scikit-learn,
and scores the classifier with `n_jobs=1`. XGBoost is likewise pinned to one
thread, because `tree_method="hist"` is not deterministic across differing
thread counts.

Wall-clock timings in `results/uah/latency_measured.json` are hardware-dependent
and will differ between machines and between runs on the same machine. That
variation is not small: the same benchmark on this host returned means between
21 and 46 ms depending on CPU power state, so the figures above were taken with
the machine on mains power at its full sustained clock. The mean carries a
bootstrapped 95 percent interval over the timed calls, and throughput is the
median of five repeats with the observed spread recorded alongside it.

Environment used for the committed results:

| | |
|---|---|
| CPU | Intel Core i7-8550U @ 1.80 GHz, 4 physical / 8 logical cores |
| RAM | 15.86 GiB |
| OS | Windows-11-10.0.26200-SP0 |
| Python | 3.12.6 |
| scikit-learn | 1.9.0 |
| xgboost | 3.3.0 |

## License

Code in `scripts/` and `run_all.py` is released under the MIT License; see
`LICENSE`. The datasets keep their own licenses and are not redistributed here;
see the links above. Derived artifacts in `results/` and `figures/` are
aggregate statistics containing no raw records.
