# Change Log

Project changes since Assessment 1, with justification. Referenced from Section 8 (Risk analysis) and Section 11 (Other technical developments) of the Assessment 2 report.

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

### Known open items

- **Figure 10** (Model B console output) shows pre-[A17] values: Ridge a=1 at 79.8 and an alcohol coefficient of +0.1027. Current values are a=0.1, 60.2 and +0.0577. The figure is an image and needs re-capturing.
- **Dashboard, deployment, monitoring and incremental-learning stages** are not yet started. Scheduled for Weeks 8 and 10.
