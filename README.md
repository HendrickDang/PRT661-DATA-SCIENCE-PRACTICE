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
| `End to End pipeline.py` | Superseded | The original pipeline. Retained for contribution history. It contains three data leakage issues corrected in the file above, and does not run on Python 3.8. |
| `End to End pipeline_Will fix.py` | Superseded | Earlier debugging copy. Not part of the current workflow. |

## Requirements

```bash
pip install -r "Environment Setup Instructions/requirements.txt"
```

Python 3.8 or newer. Versions are pinned in `requirements.txt` because XGBoost results are not reproducible across versions: the same `random_state` produces different trees, and team members on different versions obtained RMSE(/100k) of 80.3 and 84.5 for the same model. The figures in the report were produced with the pinned versions. Linear models are unaffected.

## Running the pipeline

```bash
python Updated_End_to_End_pipeline.py
```

Runs five stages in a single pass: ingest to `dataset/raw/`, panel build to `dataset/processed/`, EDA to `eda_plots/`, PCA, and the regression stage to `regression_plots/`.

Headline result: XGBoost is the strongest model on the 2023-2025 held-out split, at RMSE 84.5 per 100,000 against 105.6 for Ridge and 106.0 for linear regression. Feature variant V4 is selected on log-scale cross-validation, and the tuned parameters are `max_depth=3, learning_rate=0.05, reg_alpha=1.0, reg_lambda=3.0`.

## Running the evaluation module

```bash
python evaluation.py
```

Writes its results to `Outputs/`. Independent of the pipeline.

What it does:

- Seasonal naive and 12-month moving average baselines, so model performance is judged against a benchmark rather than only against other models
- Rolling-origin backtesting with an expanding window at h = 1, 3, 6 and 12 months
- Empirical prediction intervals from residual quantiles (realised coverage 78-80% against a nominal 80%)
- Structural-break indicators for the SerPro transition and the ANZSOC reclassification
- Metrics on the assault-rate scale, with provisional months excluded

Headline result: no single model dominates. XGBoost is most accurate at one month ahead (RMSE 90.7, skill 29.7% against the seasonal naive benchmark), while the 12-month moving average leads at 3, 6 and 12 months (RMSE 104.5, 105.7 and 110.5). Lasso performs worse than the seasonal naive benchmark at 12 months (skill -9.6%). The operational recommendation is a split specification: XGBoost at one month, moving average beyond.

## Why the pipeline selects V4 and the evaluation module selects V2

This is a deliberate methodological difference, not an inconsistency.

The pipeline computes cross-validated RMSE on the log scale, which is the objective the models are fitted against. On that scale V4 wins (0.1794 against 0.2146). The evaluation module back-transforms predictions and computes the same statistic on the assault rate per 100,000, and on that scale V2 wins (85.1 against 107.6).

The ranking is stable under both configurations: V4 leads on the log scale and V2 leads on the rate scale, whether cross-validation is restricted to the training period or run over the full panel. The divergence is driven by the choice of scale alone.

Log-scale error measures proportional accuracy, so Barkly at roughly 850 per 100,000 and Greater Darwin at roughly 210 carry equal weight. Rate-scale error measures absolute accuracy, so high-rate regions dominate. Both are legitimate. The project keeps both, on the principle that a model should be selected on the same metric its results are reported in. See Section 4.6 of the report.

## Repository layout

| Path | Contents |
| --- | --- |
| `dataset/source/` | Raw source files, stored untouched. All corrections happen in code. |
| `dataset/raw/` | Ingested copies plus `manifest.json` |
| `dataset/processed/` | Merged panel, `nt_crime_merged_2015_2025.csv` (13,629 rows x 44 cols) |
| `Outputs/` | Evaluation module results (E2, E4, E5) |
| `eda_plots/` | 13 exploratory figures |
| `regression_plots/` | 4 regression figures |
| `documents/` | Working documents and meeting minutes |
| `diagrams/` | Architecture and workflow diagrams |
| `Reports/` | Assessment reports |
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
- **Provisional months.** The most recent months are understated because incidents continue to be recorded after publication. Six trailing months are excluded from evaluation.
- **Alcohol supply data** is quarterly, starts in 2023, and is published roughly three months in arrears, so any specification using it is fitted on a much shorter window than the main models.
- **Six-region aggregation** was necessary to join the three sources onto a common geography, but it removes sub-regional variation that would be more useful operationally.

## Assumptions

The pipeline carries a numbered assumption log ([A1] to [A17]) in its module docstring. Every non-obvious modelling or data decision is tagged there and referenced at the point in the code where it applies. [A15] to [A17] cover the three data leakage corrections made since Assessment 1.
