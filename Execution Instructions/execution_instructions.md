# Execution Instructions

How to run the project from a clean checkout. See the root `README.md` for project context and results.

## 1. Prerequisites

- Python 3.8 or newer
- Git

Clone the repository and install the pinned dependencies:

```bash
git clone https://github.com/HendrickDang/PRT661-DATA-SCIENCE-PRACTICE.git
cd PRT661-DATA-SCIENCE-PRACTICE
pip install -r "Environment Setup Instructions/requirements.txt"
```

Versions are pinned deliberately. XGBoost results are not reproducible across versions, so an unpinned install will produce different figures from those in the report. Linear models are unaffected.

## 2. Check the source data is present

The pipeline reads from `dataset/source/`, which is committed to the repository. Six files should be there:

```
nt_crime_statistics_2020-2023.csv
nt_crime_statistics_latest.csv
nt-population-regions_1986-to-2025.xlsx
wholesale-alcohol-supply-by-quarter-2023.xlsx
wholesale-alcohol-supply-by-quarter-2024.xlsx
wholesale-alcohol-supply-by-quarter-2025.xlsx
```

Filenames are matched with fuzzy logic, so minor differences (punctuation, a trailing `(2)`) are tolerated. Never edit these files by hand; all corrections happen in code.

## 3. Run the pipeline

```bash
python Updated_End_to_End_pipeline.py
```

Runs five stages in one pass. Takes a few minutes.

| Stage | Writes to |
| --- | --- |
| 1. Ingest | `dataset/raw/` plus `manifest.json` |
| 2. Build panel | `dataset/processed/nt_crime_merged_2015_2025.csv` |
| 3. EDA | `eda_plots/` (13 figures) |
| 4. PCA | console output only |
| 5. Regression | `regression_plots/` (4 figures), console tables |

To capture the console output for reference:

```bash
python Updated_End_to_End_pipeline.py > run_log.txt 2>&1
```

`run_log.txt` is a generated file and should not be committed.

**Run `Updated_End_to_End_pipeline.py`, not `End to End pipeline.py`.** The latter is retained for contribution history and contains three data leakage issues corrected in the nominated file. It also fails on Python 3.8.

## 4. Run the evaluation module

```bash
python evaluation.py
```

Independent of the pipeline; it builds its own panel from the processed dataset. Writes four files to `Outputs/`:

| File | Contents |
| --- | --- |
| `E2_variant_selection_cv.csv` | Feature variant comparison on training-period CV |
| `E4_backtest_metrics.csv` | Forecast accuracy by model and horizon (Table 13) |
| `E4_backtest_predictions.csv` | Per-origin backtest predictions |
| `E5_prediction_intervals.csv` | Empirical prediction intervals and coverage |

Slower than the pipeline, because the rolling-origin backtest refits every model at each origin for four horizons.

## 5. Verify the output

The pipeline should end with `PIPELINE COMPLETE`. Key figures to check against the report:

- Panel: 13,629 rows, 696 usable region-month observations after lag warm-up
- Selected feature variant: V4 on log-scale CV
- Best XGBoost parameters: `max_depth=3, learning_rate=0.05, reg_alpha=1.0, reg_lambda=3.0`
- XGBoost test RMSE: 84.5 per 100,000

The evaluation module should report XGBoost best at h=1 (RMSE 90.7) and the 12-month moving average best at h=3, 6 and 12.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `ModuleNotFoundError: openpyxl` | Dependencies not installed; `openpyxl` is needed to read the `.xlsx` sources |
| `TypeError: 'type' object is not subscriptable` | Running `End to End pipeline.py` on Python 3.8; use the nominated file instead |
| `FileNotFoundError` in Stage 1 | Not running from the repository root, or `dataset/source/` is incomplete |
| XGBoost figures differ from the report | `xgboost` version differs from the pin; check with `pip show xgboost` |
