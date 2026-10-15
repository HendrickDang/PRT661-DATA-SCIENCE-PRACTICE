# Change Log

Project changes since Assessment 1, with justification. Referenced from Section 8 (Risk analysis) and Section 11 (Other technical developments) of the Assessment 2 report.

---

## Assessment 4 (October 2026)

### Forecast stage (`forecast.py`, new)

- **Forecasts with intervals (F1).** Every region is forecast 1, 3, 6 and 12 months ahead of December 2025, using the method with the lowest backtest RMSE at each horizon and its empirical 80% interval. `Outputs/F1_forecasts.csv` is in a long format the dashboard can read directly. Justification: the project goal is a forecast PFES can plan with; until now the repository held only evaluations of forecasts.
- **First out-of-sample check (E6).** The latest crime file already held January to April 2026, which the panel omitted because population ends in 2025. The saved models were kept frozen and scored on these 24 region-months at one month ahead and 12 at three months. Justification: the backtest re-uses the history it was tuned on; these months are the first data no decision in the project has seen. Result: the linear models (80.5 per 100,000) beat XGBoost (103.8) and both benchmarks at one month, XGBoost's 80% interval covered 62.5% of outcomes, and every method over-forecast on average, as expected for provisional counts [A23]. The sample is too small to change the recommendation, but it is the first entry for monitoring.
- **Rebuild check.** Before using the 2026 rows, `forecast.py` rebuilds 2025 from the raw file with the pipeline's own crosswalk and confirms it matches the processed panel exactly, so the new months are built the same way as the old ones.
- **Leakage check.** For all 16 forecast origin and horizon combinations, the features and benchmarks were rebuilt with every value after the origin removed; none changed.

### Saved models

- **[A21] `Models/model_a.joblib`** written by the pipeline, and **`Models/forecast_h1/3/6/12.joblib`** by the forecast stage, each with the feature order and package versions. Closes the Assessment 3 open item "Fitted models are not saved". `joblib==1.4.2` is pinned, the newest version that still supports Python 3.8.

### Code structure

- `evaluation.choose_setup()` now holds the variant and alpha selection, so `forecast.py` reuses it instead of copying it. `evaluation.horizon_frame()` gains `require_target=False` to build rows for months not yet observed. Neither change alters any evaluation output (checked byte for byte).

### Reproducibility

- Pipeline, evaluation and forecast outputs re-run in the reference environment matched the Assessment 3 files byte for byte. The evaluation and forecast stages were also run on Python 3.8 with the same pins and matched byte for byte.

### Known open items

- **Monitoring thresholds.** The monitoring design draft quotes XGBoost skill of 29.7% and the moving average as the 12-month method. The current figures are 27.4% and the seasonal naive value. The E6 coverage of 62.5% would already be a WARN under the draft's 70-90% band, but over four months, not the rolling year the rule specifies.
- **Re-score E6 when the 2026 months are final**, and extend the panel to 2026 once 2026 population estimates are published.
- **Region-specific intervals.** The 80% interval is pooled across regions, so it is too wide for Greater Darwin and too narrow for Barkly.
- **Dashboard multi-horizon view** to read `Outputs/F1_forecasts.csv`, and the Model A tab to load `Models/model_a.joblib` instead of retraining.
- Carried over from Assessment 3: one shared module for panel building, the policy simulator's assumed factors, and the stale `Visualizations/` folder.

---

## Assessment 3 (October 2026)

### Feature-timing corrections

**[A18] Crime-context features lagged one month.**
`Alcohol_offences` and `DV_offences` were counted from the assault rows of the same region-month as the target. They are 49% and 64% of the number being predicted and are not known when a forecast is made. Variants V3 and V4 now use the previous month's counts (`Alcohol_offences_lag1`, `DV_offences_lag1`). Found by the Assessment 3 data audit (`audit/README.md`, Finding 3). Applied in `Updated_End_to_End_pipeline.py`, `evaluation.py` and `core_data.py`.

**[A19] Model B tested on identical rows, with supply that precedes the target.**
Model B previously used the wholesale supply of the quarter being predicted, which is published about three months later, and was compared with a Model A fitted on a different period. It is now fitted three ways on the same 102 training and 72 test rows: no alcohol feature, same-quarter supply (association only) and previous-quarter supply (the forecast test).

**Backtest horizons made genuine (`evaluation.py`, E9).**
The Assessment 2 backtest refitted the models at each origin but built the features at the target month. A "3-month-ahead" forecast therefore used the assault rate from one month before the target, and the moving-average benchmark used the twelve months ending one month before the target. In addition, h = 1 was scored on Jan/Apr/Jul/Oct targets and h = 3, 6 and 12 on Mar/Jun/Sep/Dec targets. The backtest now forecasts directly: at horizon h the lags are h, h+2 and 12 months, the moving average ends h months before the target, one model is fitted per horizon, and every month from the first origin is scored. Training windows still expand from a 60-month minimum and are refitted every 3 months.

**Consequence.** Figures quoted in the Assessment 2 report (Tables 11 to 13) are superseded.

| Result (RMSE per 100,000) | Assessment 2 | Assessment 3 |
| --- | ---: | ---: |
| Model A, XGBoost (single split) | 84.5 | 117.7 |
| Model A, linear regression | 106.0 | 125.2 |
| Model A, Ridge | 105.6 | 125.2 |
| Model A, Lasso | 140.0 | 184.0 |
| V4 cross-validated RMSE (log scale) | 0.1794 | 0.2088 |
| Backtest h = 1, XGBoost | 90.7 | 99.8 |
| Backtest h = 3, moving average | 104.5 | 116.3 |
| Backtest h = 6, moving average | 105.7 | 130.4 |
| Backtest h = 12, best entry | 110.5 (moving average) | 145.5 (seasonal naive) |

The conclusions that survive: XGBoost is the best model on the single split and at one month ahead, and simple benchmarks win beyond one month. What changes: the errors are larger, XGBoost's lead over the linear models on the single split falls from about 20% to about 6%, at 12 months the seasonal naive value is the best entry and every fitted model is worse than both benchmarks, the selected Ridge alpha moves from 0.1 to 0.01, and the pipeline and the evaluation module now select the same feature variant (V4).

### Evaluation additions

- **Out-of-sample interval coverage (E5).** Coverage was previously computed on the same residuals that set the interval, which returns about 80% by construction. Each target month is now checked against an interval built only from earlier months. Coverage is 75-79% against a nominal 80%.
- **Paired bootstrap (E10).** The gap between the best fitted model and the moving average is resampled by target month. At one month XGBoost leads by 7.6 RMSE points, ahead in 93% of resamples, with a 95% interval of -17.7 to +2.3.
- **Common-window check.** `E4_backtest_metrics_common_window.csv` repeats the backtest table on the target months scored at every horizon. The ranking is unchanged.
- **Alphas tuned inside `evaluation.py`** on training-period cross-validation, replacing a hard-coded value copied from the pipeline.

### Reproducibility

- **[A20] Result tables written to `Outputs/`.** The regression stage writes R1, R4, R9 and R10 as CSV, plus `pipeline_run_summary.json` with the selected features, tuned parameters, platform and package versions.
- **Three figures added to the code** (`R3_model_comparison.png`, `R7_per_region_rmse.png`, `R8_backtest_rmse_by_horizon.png`). The equivalent charts in earlier slides were not produced by the nominated scripts.
- **Reference environment declared** (Linux x86_64, Python 3.11, pinned versions). XGBoost differs by a few RMSE points between Windows and Linux with identical versions (audit Finding 2), so the platform is now recorded with every run.
- **`evaluation.py` writes to `Outputs/`.** It previously wrote to `outputs/`, which is a different folder on Linux, including Streamlit Community Cloud (audit Finding 4).

### Dashboard

- `core_data.py` uses the lagged crime-context features, and tunes on the training period only. It had cross-validated over the full panel, the issue [A17] removed from the pipeline.
- Model B in the Data Science Workbench uses the previous quarter's supply [A19].
- The policy scenario simulator scales the previous month's counts and is labelled as illustrative. Its 0.4 and 0.2 pass-through factors are assumptions with no estimate behind them.
- The figure viewer reads `eda_plots/` and `regression_plots/` first. It had preferred the older copies under `Visualizations/`.

### Known open items

- **Assessment 2 report.** Section 4.5 and 4.6 quote the superseded figures above. The final report must use the Assessment 3 figures and state the correction.
- **`Visualizations/`** holds figures from early September, including regression charts from before the leakage fixes. Remove it or regenerate it.
- **Three copies of the panel-building code** (pipeline, `evaluation.py`, `core_data.py`) had to be corrected separately. Move the shared feature construction into one module.
- **Policy scenario simulator.** Replace the assumed pass-through factors with an estimate, or remove the harm-reduction figure.
- **Fitted models are not saved.** `joblib` serialisation is still planned (see `Models/README.md`); it would also remove the Windows/Linux difference between the dashboard and the report. *Closed in Assessment 4.*
- **Dashboard does not yet show the backtest forecasts or prediction intervals.**

---

## Assessment 2 (September 2026)

### Pipeline structure

**Merged `ingest.py` and `data_processing.py` into `End to End pipeline.py`.**
Simplifies execution to a single entry point and a single dependency/environment setup. Justified in Section 3.1 of the report.

**Added `Updated_End_to_End_pipeline.py` as the nominated code to run.**
A copy of the pipeline carrying three data leakage corrections, kept as a separate file so previously published results were not disturbed while the group reviewed the changes. The original `End to End pipeline.py` is retained for contribution history.

### Modelling

**Added XGBoost to the candidate model set.**
Captures potential non-linear patterns in the lag features that the linear models cannot. Tuned over a four-point grid; selected parameters `max_depth=3, learning_rate=0.05, reg_alpha=1.0, reg_lambda=3.0`.

**Added `evaluation.py`, a standalone forecast evaluation module.**
Provides seasonal naive and 12-month moving average benchmarks, rolling-origin backtesting at 1, 3, 6 and 12 month horizons, and empirical prediction intervals. Responds directly to the Assessment 1 feedback that the modelling plan lacked baseline comparison and a defined validation approach.

### Data leakage corrections

**[A15] Scaler moved inside the cross-validation pipeline.**
`cv_rmse_for_features` previously called `StandardScaler().fit_transform()` on the whole frame before `TimeSeriesSplit`, so every fold was standardised using statistics that included later folds. The scaler is now wrapped in a `sklearn.pipeline.Pipeline` and refits on each fold's training rows only.

**[A16] Feature-variant selection moved off the test set.**
`evaluate_feature_variants` previously scored each variant on the 2023-2025 test period and selected the minimum, after which the final models were evaluated on that same period. Reported performance was therefore optimistic. Selection now uses TimeSeriesSplit cross-validation over the training period, leaving the test set untouched until final evaluation.

**[A17] Cross-validation frame restricted to the training period.**
`prepare_panel` previously built `cv_frame` from the full panel, so alpha tuning and XGBoost tuning cross-validated over folds containing test rows. `cv_frame` is now built from `train` only. This changed the selected Ridge alpha from 1.0 to 0.1.

**Consequence.** Tables 11 and 12 in Section 4.5 were regenerated after these corrections. XGBoost moved from RMSE 0.1477 log / 80.3 per 100k to 0.1517 log / 84.5 per 100k.

### Code defects fixed

- Two undeclared dependencies (`xgboost`, `openpyxl`) that broke the pipeline on any machine other than the original developer's. Now pinned in `requirements.txt`.
- A Windows-only Unicode crash inside a `print()` statement.
- A duplicated function definition left over from a Git merge, with a missing argument in the abandoned copy.
- A biased VIF calculation, missing its constant term.
- An unsafe `eval()` call, replaced with index-based lookup into the parameter grid.
- Missing `from __future__ import annotations`, which caused the pipeline to fail on Python 3.8 due to `list[str]` type hints.

### Data issues resolved

**November 2023 gap.**
Absent from both crime files as a result of the PROMIS-to-SerPro systems transition. Retained as NaN and never imputed, to avoid introducing artificial values into the time series. Tagged [A3].

**ANZSOC category-label mismatch.**
The two crime files use different offence category labels. A manual translation table was added to ensure consistent classification across the full 2015-2025 period. Documented in Table 6 of the report.

**Region boundary mismatch.**
Crime data uses 7 NT Police reporting regions; population and alcohol data use the 6 NTG statistics regions. A two-stage remapping was added, including SA2-level lookup for NT Balance rows. Without this, Greater Darwin population would be double-counted and per-capita rates roughly halved.

### Repository structure

**Source datasets moved to `dataset/source/` (19-20 August).**
Raw files were originally committed to `dataset/` directly. They were relocated to a dedicated `source/` folder to establish the three-stage convention `dataset/source/` to `dataset/raw/` to `dataset/processed/`, so that original files are never overwritten by pipeline output.

**`Execution Instructions` folder renamed (26 August).**
The original folder name contained a typo (`Execution Instrctions`) and was recreated correctly.

**Working document consolidated to `documents/Assignment 2.docx` (9 September).**
Two parallel copies of the report existed briefly (`PRT661_Assignment 2_Group_2_Theme_2.docx` and `Assignment 2.docx`). The longer-named copy was deleted to prevent the team editing different files. Because a `.docx` cannot be merged by Git, the group now coordinates edits in the team chat before opening the file.

**Visualisation outputs reorganised into `Visualizations/` (10 September).**
Plots previously sat at the repository root alongside code.

### Code structure

**`run_regression` decomposed into separate functions (2 September).**
The regression stage was originally a single monolithic function. It was split into discrete functions for feature-variant comparison, VIF screening, alpha tuning, XGBoost tuning and final model training, to make the stage easier to debug and to allow individual steps to be modified without touching the rest.

### Version control practice

**Branch and pull request workflow adopted.**
Changes now go through a named branch and a pull request rather than direct commits to `main`, in line with the working agreements below. Pull requests #5 to #7 record this. Earlier work in July and August was committed directly to `main`.

### Architecture

**AWS-equivalent mapping corrected.**
Athena is a query and cataloguing tool rather than a transformation engine, so AWS Glue ETL replaces it as the conceptual equivalent for the cleaning and feature-engineering stage in Table 2.

### Governance

**Working agreements adopted.**
Code changes go through a pull request reviewed by at least one other member before merging. Jira tickets move to Done only when the associated code has been merged and the relevant pipeline stage runs without errors. Blockers unresolved within two working days are escalated at the next check-in. Scope and dataset changes are logged here with justification before work begins.

**Output-artefact policy.**
Pipeline outputs (`eda_plots/`, `regression_plots/`, `dataset/raw/`, `dataset/processed/`, `Outputs/`) are committed rather than gitignored, so markers can inspect results without re-running the pipeline.

### Known open items at Assessment 2

*Superseded by the Assessment 3 section above.*

- **Figure 10** (Model B console output) shows pre-[A17] values: Ridge a=1 at 79.8 and an alcohol coefficient of +0.1027. Current values are a=0.1, 60.2 and +0.0577. The figure is an image and needs re-capturing.
- **Dashboard, deployment, monitoring and incremental-learning stages** are not yet started. Scheduled for Weeks 8 and 10.
