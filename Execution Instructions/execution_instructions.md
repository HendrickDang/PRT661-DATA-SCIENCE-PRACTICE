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

The reported figures were produced on Linux x86_64 with Python 3.11. With the same pinned versions, XGBoost on Windows differs by a few RMSE points. See "Reference environment" in the root `README.md`.

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
| 5. Regression | `regression_plots/` (6 figures), result tables in `Outputs/` (R1, R4, R9, R10, `pipeline_run_summary.json`) |

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

Independent of the pipeline; it builds its own panel from the processed dataset. Writes six files to `Outputs/` and `R8_backtest_rmse_by_horizon.png` to `regression_plots/`:

| File | Contents |
| --- | --- |
| `E2_variant_selection_cv.csv` | Feature variant comparison on training-period CV |
| `E4_backtest_metrics.csv` | Forecast accuracy by model and horizon |
| `E4_backtest_metrics_common_window.csv` | The same table on the target months scored at every horizon |
| `E4_backtest_paired_bootstrap.csv` | Best fitted model against the moving average, with a 95% interval |
| `E4_backtest_predictions.csv` | Every backtest forecast, by origin, target, horizon, model and region |
| `E5_prediction_intervals.csv` | 80% prediction intervals and out-of-sample coverage |

Takes well under a minute.

## 5. Run the forecast stage

```bash
python forecast.py
```

Run after steps 3 and 4: it reads `Outputs/E4_backtest_metrics.csv` and `Outputs/E5_prediction_intervals.csv` to choose the method and interval for each horizon. Writes:

| File | Contents |
| --- | --- |
| `Outputs/F1_forecasts.csv` | Forecasts from December 2025 at 1, 3, 6 and 12 months for every region and model, with the 80% interval on the recommended method |
| `Outputs/E6_holdout_2026_metrics.csv` | Frozen models scored on January to April 2026 (provisional actuals), with bias and interval coverage |
| `Outputs/E6_holdout_2026_predictions.csv` | Every forecast behind that table |
| `Outputs/forecast_run_summary.json` | Origin, data As At date, set-up and package versions |
| `Models/forecast_h1.joblib` ... `forecast_h12.joblib` | One fitted model bundle per horizon |
| `regression_plots/R9_forecasts_2026.png` | Forecasts with intervals against the 2026 provisional actuals |

The pipeline (step 3) also writes `Models/model_a.joblib`. To load a saved bundle:

```python
import joblib, numpy as np
bundle = joblib.load("Models/forecast_h1.joblib")
model = bundle["models"]["XGBoost"]          # a fitted scikit-learn Pipeline
rate = np.expm1(model.predict(X[bundle["features"]]))   # X: rows built like evaluation.horizon_frame
```

Saved models load only with the pinned versions in `requirements.txt` (including `joblib==1.4.2`).

## 6. Run the release checks

```bash
python monitoring.py --as-of 2026-10-18
```

Run after step 5, and after every new crime statistics release (without `--as-of` the check date is today). Takes a few seconds and refits nothing. Writes:

| File | Contents |
| --- | --- |
| `Outputs/M1_monitoring_checks.csv` | One row per check: area, value, rule, PASS / WARN / FAIL and a note |
| `Outputs/M2_monitoring_scored.csv` | Every scored forecast: month, region, horizon, method, actual, forecast, error, inside the 80% interval or not, settled or provisional |
| `Outputs/M3_monitoring_summary.json` | Overall status, counts, and the decisions to promote the forecasts and whether to retrain |

The console ends with the overall result, `Promote forecasts to the dashboard: yes/no` and the retraining decision.

## 7. Run the dashboard

```bash
streamlit run app.py
```

Opens at http://localhost:8501 (on Windows, `run_dashboard.bat` does the same). The Administrator Portal's **Forecasts & Planning Ranges** tab and the Workbench's **Backtest, Forecasts & Monitoring** tab read the files from steps 4-6. To host it, see "Hosted version" in the root `README.md` (Streamlit Community Cloud, Python 3.11).

## 8. Verify the output

The pipeline should end with `PIPELINE COMPLETE`. Key figures to check against the report:

- Panel: 13,629 rows, 696 usable region-month observations after lag warm-up
- Selected feature variant: V4 (CV RMSE 0.2088 on the log scale)
- Best XGBoost parameters: `max_depth=3, learning_rate=0.05, reg_alpha=1.0, reg_lambda=3.0`
- Linear regression test RMSE: 125.2 per 100,000 (identical on every platform)
- XGBoost test RMSE: 117.7 per 100,000 in the reference environment

The evaluation module should report XGBoost best at h=1 (RMSE 99.8), the 12-month moving average best at h=3 and h=6 (116.3 and 130.4), and the seasonal naive benchmark best at h=12 (145.5). The moving-average and seasonal-naive figures are identical on every platform.

The forecast stage should print that 2025 rebuilt from the raw file matches the panel exactly, find 24 region-months after the panel (2026-01 to 2026-04), and report on those months an h=1 RMSE of 80.5 for linear regression and 103.8 for XGBoost.

The release checks with `--as-of 2026-10-18` should end with `Monitoring result: WARN (19 PASS, 6 WARN, 0 FAIL)` and `Promote forecasts to the dashboard: yes`.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `ModuleNotFoundError: openpyxl` | Dependencies not installed; `openpyxl` is needed to read the `.xlsx` sources |
| `TypeError: 'type' object is not subscriptable` | Running `End to End pipeline.py` on Python 3.8; use the nominated file instead |
| `FileNotFoundError` in Stage 1 | Not running from the repository root, or `dataset/source/` is incomplete |
| `SCHEMA ERROR` in Stage 1, then `STOPPED` | A source file has a missing or renamed column or unexpected values (the line names them). Nothing was written; fix or re-download the file. Run `python -m unittest discover tests` to check the ingest checks themselves |
| XGBoost figures differ from the report | `xgboost` version differs from the pin (check with `pip show xgboost`), or the run is not on Linux x86_64 |
