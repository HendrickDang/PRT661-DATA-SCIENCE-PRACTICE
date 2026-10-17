# PRT661 Data Science Practice - Dan2 Theme 2

**Forecasting Assault Rates Across NT Regions to Support Public Safety Resource Planning**

Charles Darwin University, Semester 2 2026. Unit coordinator Dr Reem Sherif, group lecturer Dr Thuseethan Selvarajah. Theme 2: Predictive Analytics and Forecasting. End user: NT Police, Fire and Emergency Services (PFES).

## Team

| Member | Student ID | Area |
| --- | --- | --- |
| Thi Xuan Thanh Tran (Emma) | S389244 | Data cleaning and preprocessing |
| Ngoc Anh Nguyen (Will) | S386805 | Data engineering |
| Van Hoi Dang (Hendrick) | S395598 | Analytics and machine learning |
| Le Nhat Minh Tran (Thomas) | S390789 | Project governance |

Project board: https://hendrickdang.atlassian.net/jira/software/projects/SCRUM/boards/1/timeline?rangeMode=WEEKS

## Problem

Assault rates vary substantially across the six NT Government service regions. PFES needs month-ahead forecasts at regional granularity to inform resource allocation. The project produces forecasts at 1, 3, 6 and 12 month horizons, with prediction intervals, from 2015-2025 crime data joined to regional population and wholesale alcohol supply.

Regions covered: Barkly, Big Rivers, Central Australia, East Arnhem, Greater Darwin, Top End.

## Which script do I run?

| File | Status | Purpose |
| --- | --- | --- |
| `Updated_End_to_End_pipeline.py` | **Final code to run** | Full pipeline: ingest, panel build, EDA, PCA and regression. Produces the figures in Sections 4.1-4.5 of the report. |
| `evaluation.py` | **Final** | Standalone forecast evaluation and benchmarking. Produces Section 4.6. Runs independently of the pipeline. |
| `forecast.py` | **Final (Assessment 4)** | Forecasts for the next 1, 3, 6 and 12 months with 80% intervals, one saved model bundle per horizon, and a first out-of-sample check on January to April 2026. Run after `evaluation.py`. |
| `monitoring.py` | **Final (Assessment 4)** | Release checks from the monitoring design: data quality, forecast accuracy and drift, each PASS, WARN or FAIL, plus the decisions to promote the forecasts and whether to retrain. Run after `forecast.py`. |
| `app.py` (+ `pages/`, `core_data.py`) | **Final** | Streamlit dual-role dashboard for PFES-facing briefing and technical model review. Shows the forecasts with 80% ranges and the latest release check. Requires the processed panel from the pipeline. |
| `run_dashboard.bat` | Helper (Windows) | Launches `streamlit run app.py` using a local Anaconda install when available. |
| `End to End pipeline.py` | Superseded | The original pipeline. Retained for contribution history. It contains three data leakage issues corrected in the file above, and does not run on Python 3.8. |
| `End to End pipeline_Will fix.py` | Superseded | Earlier debugging copy. Not part of the current workflow. |

## Requirements

```bash
pip install -r "Environment Setup Instructions/requirements.txt"
```

Python 3.8 or newer for the pipeline and evaluation modules. The Streamlit dashboard packages (`streamlit`, `plotly`) need **Python 3.9+** (local demos use Anaconda 3.13).

### Reference environment

Every figure quoted in this README, in `Outputs/` and in the Assessment 3 slides was produced in one environment: **Linux x86_64, Python 3.11, with the pinned versions in `requirements.txt`**. This is the environment of the Assessment 3 data audit (`audit/README.md`) and of Streamlit Community Cloud, where the dashboard is to be deployed. `Outputs/pipeline_run_summary.json` and `Outputs/forecast_run_summary.json` record the platform and package versions of the runs that wrote the tables. The evaluation and forecast outputs were also re-run on Python 3.8 with the same pins and matched byte for byte.

Linear models reproduce exactly on any machine. XGBoost does not: with identical pinned versions it gives slightly different trees on Windows and on Linux (the audit measured a gap of up to 4 RMSE points), and different versions differ again. If your XGBoost figures are a few points away from the ones below, check the platform before suspecting the code.

## Running the pipeline

```bash
python Updated_End_to_End_pipeline.py
```

Runs five stages in a single pass: ingest to `dataset/raw/`, panel build to `dataset/processed/`, EDA to `eda_plots/`, PCA, and the regression stage to `regression_plots/`.

The regression stage also writes its tables to `Outputs/` (`R1_feature_variants_cv.csv`, `R4_model_a_metrics.csv`, `R9_per_region_accuracy.csv`, `R10_model_b_metrics.csv`) so reports and slides quote files instead of console output.

Headline result (single train/test split, train 2016-2022, test 2023-2025): XGBoost is the strongest model at RMSE 117.7 per 100,000, against 125.2 for linear regression and Ridge and 184.0 for Lasso. The gap between XGBoost and the linear models is about 6%. Feature variant V4 is selected on cross-validation, and the tuned XGBoost parameters are `max_depth=3, learning_rate=0.05, reg_alpha=1.0, reg_lambda=3.0`.

Alcohol supply (Model B): on identical rows (train 2023-2024, 102 rows; test 2025, 72 rows), adding the previous quarter's wholesale supply per capita changes test RMSE by less than 1 per 100,000 for linear regression, Ridge and XGBoost. The coefficient is positive in every linear model (+0.06 to +0.14, standardised). The data show a small positive association and no measurable forecast gain.

## Running the evaluation module

```bash
python evaluation.py
```

Writes its results to `Outputs/` and one figure to `regression_plots/`. Independent of the pipeline.

What it does:

- Seasonal naive and 12-month moving average baselines, so model performance is judged against a benchmark rather than only against other models
- Rolling-origin backtesting at h = 1, 3, 6 and 12 months: an expanding window of at least 60 months, refitted every 3 months, with every month from June 2021 to June 2025 scored
- Genuine horizons: a forecast made h months ahead uses no feature and no baseline value more recent than h months before the target
- Empirical 80% prediction intervals from residual quantiles, with coverage measured out of sample (75-79%)
- A paired bootstrap on the gap between the best model and the moving average
- Metrics on the assault-rate scale, with provisional months excluded

Headline result: no single approach dominates.

| Horizon | Lowest RMSE | RMSE per 100k | Next best |
| --- | --- | ---: | --- |
| 1 month | XGBoost | 99.8 | Moving average 107.3 |
| 3 months | 12-month moving average | 116.3 | XGBoost 124.5 |
| 6 months | 12-month moving average | 130.4 | XGBoost 139.3 |
| 12 months | Seasonal naive | 145.5 | Moving average 150.8 |

XGBoost's skill against the seasonal naive benchmark at one month is 27.4%. Its lead over the moving average at one month is 7.6 RMSE points; the bootstrap puts XGBoost ahead in 93% of resamples, and the 95% interval (-17.7 to +2.3) includes zero, so the lead is likely but not established. At 12 months every fitted model is worse than both naive benchmarks. The operational recommendation is a split specification: XGBoost at one month, the 12-month moving average at 3 and 6 months, and the seasonal naive value at 12 months.

The same ranking holds when all horizons are scored on the same target months (`E4_backtest_metrics_common_window.csv`).

## Running the forecast stage

```bash
python forecast.py
```

Run after the pipeline and `evaluation.py`: it reads the backtest tables to pick the method for each horizon. Takes about 30 seconds.

What it does:

- **F1 forecasts.** From the last month of the processed panel (December 2025) it forecasts every region 1, 3, 6 and 12 months ahead (January, March, June and December 2026). Each horizon uses the method with the lowest backtest RMSE and its empirical 80% interval: XGBoost at one month, the 12-month moving average at three and six months, the seasonal naive value at twelve. The other models' forecasts are written alongside for comparison. Saved to `Outputs/F1_forecasts.csv` and plotted in `regression_plots/R9_forecasts_2026.png`.
- **Saved models.** One bundle per horizon in `Models/` (`forecast_h1.joblib` to `forecast_h12.joblib`), trained on every target month to December 2025 with the backtested set-up. The pipeline also saves Model A as `Models/model_a.joblib` [A21].
- **E6 out-of-sample check.** The latest crime file (As At 1 June 2026) already contains January to April 2026, which the processed panel stops short of because population figures end in 2025. No model has seen these months. The saved models are kept frozen and scored on them: four target months at one month ahead, two at three months, six regions each. Rates use the 2025 population [A22].

Headline result on the 2026 months (RMSE per 100,000, provisional actuals):

| Horizon | Linear / Ridge | XGBoost | Moving average | Seasonal naive | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 month | 80.5 | 103.8 | 135.0 | 166.7 | 24 |
| 3 months | 123.6 | 171.0 | 176.6 | 211.5 | 12 |

All four fitted models beat both benchmarks on these months, and the linear models beat XGBoost at both horizons. XGBoost's one-month error (103.8) is close to its backtest error (99.8), but its 80% interval covered only 15 of 24 outcomes (62.5%). Every method forecasts above the outcome on average, which is expected: all 2026 months are within six months of the As At date, so the counts are provisional and will rise [A23]. With four months and six regions, this check cannot overturn the four-year backtest. It is the first data point for the monitoring stage, which should re-score these months once they are final.

`F1_forecasts.csv` has one row per origin, horizon, region and model, with `Recommended = True` on the method to show and `Lower_80` / `Upper_80` filled for that method. The dashboard reads it directly for the Forecasts & Planning Ranges tab.

## Running the release checks (monitoring)

```bash
python monitoring.py                 # check date defaults to today
python monitoring.py --as-of 2026-10-18
```

Run after `forecast.py`, and again after every new NT crime statistics release. It implements the team's monitoring design (`documents/Monitoring_Design_DAN2_Theme2.docx`) and takes a few seconds. Nothing is refitted.

| Area | Checks | Rule |
| --- | --- | --- |
| Data (D1-D10) | Source files, schema and header drift, category map, region crosswalk, calendar gaps, raw-to-panel rebuild, provisional window, release age, alcohol supply lag, population coverage | FAIL blocks the release; WARN is logged |
| Performance (P, C) | Recommended method per horizon on the latest settled months: skill against the seasonal naive benchmark, RMSE against the backtest, 80% interval coverage | h = 1: WARN if skill < 10%, FAIL if < 0%. h = 3-12: WARN if RMSE > 1.25 x backtest, FAIL if skill < 0%. Coverage: WARN outside 70-90% |
| Newest months (N, NC) | The same rules on the 2026 months scored by `forecast.py` | Provisional actuals never FAIL; a FAIL is shown as WARN [A23] |
| Drift (R1-R3) | Regional rate against the same quarter of the previous three years; alcohol- and DV-involved shares | WARN for human review only [A24]; never an automatic retrain |

It writes `Outputs/M1_monitoring_checks.csv` (one row per check), `Outputs/M2_monitoring_scored.csv` (every scored forecast with its error and whether it fell inside the 80% interval) and `Outputs/M3_monitoring_summary.json` (overall status and two decisions: promote the forecasts to the dashboard, and retrain or not). A FAIL on settled months stops the forecasts being promoted and opens an investigation; retraining follows only if the FAIL repeats or is explained by lasting drift. Planned refreshes happen when the panel can be extended, which needs the 2026 population estimates [A1].

Result of the check on 18 October 2026: **WARN, 19 PASS, 6 WARN, 0 FAIL**, forecasts promoted, no retraining needed. On the latest settled months XGBoost keeps 28.4% skill at one month and every interval covers 74-81% of outcomes. The warnings are the age of the crime file (As At 1 June 2026), 2026 rates using 2025 population, the low interval coverage and three-month error on the provisional 2026 months, and a drift flag: Central Australia's October-December 2025 rate is 32% below the same quarter of 2022-2024, a sustained fall the review should confirm is real rather than a recording change.

## Running the dashboard

Prerequisite: run the pipeline at least once so `dataset/processed/nt_crime_merged_2015_2025.csv` exists.

```bash
streamlit run app.py
```

On Windows you can also double-click `run_dashboard.bat` after installing dependencies.

The app opens at http://localhost:8501 with two roles:

| Page | Audience | Contents |
| --- | --- | --- |
| **Administrator Portal** (`pages/1_Administrator_Portal.py`) | Policy / PFES / health and justice planners | KPIs, **forecasts and planning ranges** (1-12 months, 80% ranges, expected assaults per month, release check status), spatial demand map, staffing priority matrix, seasonal calendar, policy scenario simulator, briefing export |
| **Data Science Workbench** (`pages/2_Data_Science_Workbench.py`) | Technical reviewers | Assumptions [A1-A24], EDA, PCA, feature selection and VIF, hyperparameter curves, Model A vs B, residual diagnostics, pipeline figure viewer, **backtest, 2026 check and release monitoring** |

Shared loading and model training live in `core_data.py` (cached). Do **not** run files under `pages/` with `python` directly — Streamlit must start from `app.py` at the repo root so imports resolve.

The forecast tab and the Workbench's backtest tab read `Outputs/` (F1, E4-E6, M1-M3) and refit nothing, so they always show the figures in the report. The Model A/B tabs still retrain the suite from the processed panel for interactive inspection, with the same features and tuning rules as the pipeline. The policy scenario simulator is illustrative: its pass-through factors are assumptions, not estimates. Priority tiers use neutral wording (High need, Elevated, Baseline): they describe a need for services, not a judgement about a community.

### Hosted version (Streamlit Community Cloud)

The dashboard is deployed from this repository's `main` branch: share.streamlit.io, New app, repository `HendrickDang/PRT661-DATA-SCIENCE-PRACTICE`, branch `main`, main file `app.py`, and under Advanced settings **Python 3.11** (the pinned NumPy 1.24.4 has no wheels for Python 3.12 or later). Streamlit Cloud installs from the root `requirements.txt`, a copy of `Environment Setup Instructions/requirements.txt`; keep the two identical. Every push to `main` redeploys the app. The cloud host is Linux x86_64 with Python 3.11, the reference environment, so the hosted numbers match the report.

## Feature selection in the pipeline and in the evaluation module

The pipeline computes cross-validated RMSE on the log scale, which is the objective the models are fitted against. The evaluation module back-transforms and computes it on the assault rate per 100,000. Log-scale error measures proportional accuracy, so Barkly at roughly 850 per 100,000 and Greater Darwin at roughly 210 carry equal weight. Rate-scale error measures absolute accuracy, so high-rate regions dominate.

Both scales now select **V4** (temporal features, region effects and the previous month's alcohol- and DV-flagged assault counts): 0.2088 against 0.2146 for V2 on the log scale, and 81.6 against 85.1 on the rate scale.

In Assessment 2 the two scales disagreed (V4 on the log scale, V2 on the rate scale). That disagreement came from the crime-context features being taken from the same month as the target, which is corrected under [A18]. See `CHANGELOG.md`.

## Repository layout

| Path | Contents |
| --- | --- |
| `dataset/source/` | Raw source files, stored untouched. All corrections happen in code. |
| `dataset/raw/` | Ingested copies plus `manifest.json` |
| `dataset/processed/` | Merged panel, `nt_crime_merged_2015_2025.csv` (13,629 rows x 44 cols) |
| `Outputs/` | Result tables: pipeline (R1, R4, R9, R10, run summary), evaluation module (E2, E4, E5), forecast stage (F1, E6, run summary) and release checks (M1-M3) |
| `monitoring.py` | Release checks after each data release (PASS / WARN / FAIL) |
| `requirements.txt` | Copy of the pinned environment at the root, read by Streamlit Community Cloud |
| `Models/` | Saved model bundles: `model_a.joblib` (pipeline) and `forecast_h1/3/6/12.joblib` (forecast stage). See `Models/README.md` |
| `eda_plots/` | 13 exploratory figures |
| `regression_plots/` | 8 regression and forecast figures (R1-R9) |
| `app.py` | Streamlit landing page (role chooser + territory KPI banner) |
| `pages/` | Administrator Portal and Data Science Workbench |
| `core_data.py` | Shared data load, regression panel build, and cached model training for the dashboard |
| `run_dashboard.bat` | Windows launcher for the Streamlit app |
| `documents/` | Working documents and meeting minutes |
| `diagrams/` | Architecture and workflow diagrams |
| `Reports/` | Assessment reports |
| `audit/` | Assessment 3 read-only data lineage, reproducibility and leakage audit |
| `Environment Setup Instructions/` | Environment setup, including `requirements.txt` |
| `Execution Instructions/` | Run instructions |
| `CHANGELOG.md` | Change log: all changes since Assessment 1, with justification |

## Data sources

- **NT crime statistics.** Pre-SerPro (Jan 2008 - Nov 2023): https://data.nt.gov.au/dataset/current-nt-crime-statistics-november-2023 . Post-SerPro: https://data.nt.gov.au/dataset/current-nt-crime-statistics-april-2026
- **NT Government regional population estimates**, 1986-2025
- **Wholesale alcohol supply**, quarterly, 2023-2025 only

## Known data limitations

- **PROMIS to SerPro transition (Nov/Dec 2023).** Recording system change; series either side is not strictly comparable.
- **ANZSOC reclassification (April 2025).** Offence categories were redefined.
- **November 2023 missing** from both crime files - a genuine gap at the systems changeover.
- **Provisional months.** The most recent months are understated because incidents continue to be recorded after publication. The six trailing months are flagged. They are excluded from the rolling-origin backtest; they remain inside the test period of the single train/test split, where `R4_model_a_metrics.csv` reports the error with and without them.
- **Alcohol supply data** is quarterly, starts in 2023, and is published roughly three months in arrears. Any specification using it is fitted on a much shorter window than the main models, and the supply of the quarter being forecast is not available in time, which is why Model B is tested with the previous quarter's figure [A19].
- **Single-split figures are one-step-ahead.** The 2023-2025 test in the pipeline uses the actual assault rate of the previous month as a feature at every test month. It measures one-month-ahead accuracy with parameters frozen at the end of 2022. Longer horizons are measured only by `evaluation.py`.
- **2026 months are provisional.** The January to April 2026 counts used by `forecast.py` are all within six months of the As At date and will be revised upward. Rates for 2026 use the 2025 population [A22].
- **Prediction intervals are pooled across regions.** One residual distribution per horizon is applied to every region, so the interval is wide for Greater Darwin (rate near 200) and narrow for Barkly (near 850). A region- or scale-specific interval is listed as future work.
- **Six-region aggregation** was necessary to join the three sources onto a common geography, but it removes sub-regional variation that would be more useful operationally.

## Assumptions

The pipeline carries a numbered assumption log ([A1] to [A21]) in its module docstring, `forecast.py` adds [A22] and [A23] in its own, and `monitoring.py` adds [A24]. Every non-obvious modelling or data decision is tagged there and referenced at the point in the code where it applies. [A15] to [A17] cover the three data leakage corrections made for Assessment 2. [A18] and [A19] cover the two feature-timing corrections made for Assessment 3, and [A20] the result tables written to `Outputs/`. Assessment 4 adds:

- [A21] Model A is saved to `Models/model_a.joblib` with its scaler and feature order. It loads only under the pinned package versions.
- [A22] 2026 assault rates use each region's 2025 population, the latest published.
- [A23] The 2026 months are provisional, so `forecast.py` reports bias alongside error and treats the check as an early warning, not a final score.
- [A24] Initial drift thresholds in `monitoring.py`: WARN if a region's quarterly rate differs from the same quarter of the previous three years by more than 25%, or if the alcohol- or DV-involved share of assaults moves by more than 5 percentage points. Drift prompts human review and never triggers retraining by itself.
