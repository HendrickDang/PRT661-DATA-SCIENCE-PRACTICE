# Data Lineage, Reproducibility and Leakage Audit (Assessment 3)

**Author:** Ngoc Anh Nguyen (Will), data engineering
**Date:** 2 October 2026
**Repository state audited:** commit `f71de48` (`main`, 1 October 2026)

**Verification log**

| Date | Check | Result |
|---|---|---|
| 2 Oct 2026 | Fresh clone of `main`; full pipeline, evaluation and audit re-run | `main` is still at `f71de48` (no new commits). All three audit CSVs are byte-identical; backtest outputs agree to within 1e-11. |
| 2 Oct 2026 | SHA-256 of the six data files in the project workspace compared with `dataset/source/` | All six are identical |

## Purpose and scope

This audit checks three things the group relies on in the Assessment 3 presentation:

1. **Lineage.** Every row count we quote can be traced back to the source files, with a stated reason for every row that is dropped.
2. **Reproducibility.** A fresh clone, installed exactly as the README instructs, reproduces the reported results.
3. **Feature availability.** Every model feature would actually be known at the time a forecast is made.

The audit is **read-only**. `data_audit.py` does not import or edit the pipeline or the evaluation module, and it writes only to `audit/`. This was verified by checksumming every file outside `audit/` before and after a run: nothing changed.

## How to reproduce this audit

```bash
git clone https://github.com/HendrickDang/PRT661-DATA-SCIENCE-PRACTICE.git
cd PRT661-DATA-SCIENCE-PRACTICE
uv venv -p 3.11 .venv && source .venv/bin/activate      # any Python 3.8-3.11 works
uv pip install -r "Environment Setup Instructions/requirements.txt"
python Updated_End_to_End_pipeline.py                    # builds dataset/processed/
python evaluation.py                                     # rolling-origin backtest
python audit/data_audit.py                               # this audit
```

Test environment: Linux x86_64, Python 3.11.15, pandas 2.0.3, numpy 1.24.4, scikit-learn 1.3.2, xgboost 2.1.4 (all as pinned).

## Finding 1: Row-count lineage

All figures below come from `A3_data_lineage.csv`.

| Stage | Rows | Explanation |
|---|---:|---|
| Raw crime rows (both files) | 55,146 | 46,036 (2008–2023) + 9,110 (Dec 2023–Apr 2026) |
| Inside the 2015–2025 window | 34,306 | 20,840 rows outside the window are not used |
| Processed offence-level panel | 13,629 | Aggregated by region, month, offence type and flags |
| Region-month assault cells | 786 | 131 observed months × 6 regions |
| Complete region-month grid | 792 | 132 months × 6 regions (Jan 2015–Dec 2025) |
| − lag-12 warm-up | −72 | Jan–Dec 2015 has no 12-month history |
| − Nov 2023 target missing | −6 | PROMIS→SerPro systems gap |
| − lag1 / lag3 / lag12 hit the gap | −18 | Dec 2023, Feb 2024 and Nov 2024 |
| **Usable model panel** | **696** | 116 usable months × 6 regions |
| Train (≤ 2022) | 504 | Jan 2016–Dec 2022, 84 months |
| Test (≥ 2023) | 192 | Jan 2023–Dec 2025, 32 months |
| Provisional flag | 36 | Last 6 months; inside the test split, flagged rather than dropped |

**Wording corrections this supports:**

* "55,146 crime records merged" → *55,146 raw rows ingested; 34,306 inside the 2015–2025 analysis window.* Rows are not offences (one row can represent many offences).
* "Train 2015–2022" → *Train Jan 2016–Dec 2022 (2015 is used only as lag-12 history).*
* "132 months" and "~116 months" are both correct, but they describe different stages: the full grid and the usable panel.
* "Last 6 months excluded from training" → *Last 6 months are flagged as provisional. They sit inside the test split of the single train/test evaluation and are excluded only from the rolling-origin backtest (`evaluation.py`).*

## Finding 2: Reproducibility

| Result | Reported | Clean clone | Status |
|---|---:|---:|---|
| Linear / Ridge / Lasso, single split (RMSE per 100k) | 106.0 / 105.6 / 140.0 | 106.0 / 105.6 / 140.0 | ✅ Exact |
| XGBoost, single split (RMSE per 100k) | 84.5 | **80.7** | ⚠️ Differs |
| XGBoost, single split (RMSE, log) | 0.1517 | 0.1449 | ⚠️ Differs |
| Rolling-origin backtest, all linear models and baselines | `Outputs/E4_*` | regenerated | ✅ Exact |
| Rolling-origin backtest, XGBoost (h = 1 / 3 / 6 / 12) | 90.7 / 115.1 / 115.4 / 121.2 | 90.9 / 111.7 / 117.0 / 118.1 | ⚠️ Differs by 0.2–3.4 |
| Prediction intervals (`E5`) | committed | regenerated | ✅ Identical |
| Model B alcohol coefficient (Linear / Ridge / Lasso) | +0.04 to +0.06 | +0.041 / +0.058 / +0.060 | ✅ Matches |

**Interpretation.** Linear models reproduce exactly. XGBoost is deterministic on a given machine: repeated runs, and runs with 1 or 2 threads, give the same result. It still differs from the committed figures even with identical pinned versions, which points to platform-dependent floating-point behaviour (operating system or CPU) rather than randomness.

**The conclusions do not change.** XGBoost is still the best model on the single split. In the backtest, XGBoost still wins at h = 1 (90.9 vs 96.6 for the 12-month moving average), and the moving average still wins at h ≥ 3.

**Recommendations:**

* Report XGBoost as platform-dependent (for example, "RMSE ≈ 81–85 per 100k depending on platform") rather than as a single exact value.
* Save the fitted model with `joblib`, as already planned in `Models/README.md`, so the dashboard and the report use one identical artefact.

**Stale figure in the slide deck.** The terminal screenshot on slide 14 shows a Ridge alcohol coefficient of +0.1027. That value comes from an earlier run with α = 1. The current code (α = 0.1, changed by fix [A17]) gives +0.058, which matches the "+0.04 to +0.06" text. The screenshot should be replaced.

## Finding 3: Same-month feature overlap (target leakage risk)

`Alcohol_offences` and `DV_offences` are aggregated from the **assault rows of the same region-month as the target** (`Updated_End_to_End_pipeline.py`, monthly aggregation feeding `build_regression_panel`). They are used unlagged in variants V3 and V4.

| Feature | Share of target offences | Correlation with target (same month) | Correlation if lagged 1 month | Known when forecasting? |
|---|---:|---:|---:|---|
| `Alcohol_offences` | 49.3% | 0.969 | 0.940 | No |
| `DV_offences` | 63.5% | 0.948 | 0.913 | No |

**Which results are affected:**

* **Affected:** feature-variant selection (slide 11; V4 is selected), the single-split results (slide 12, including XGBoost 84.5 / 80.7) and Model B (slide 14). Model B reuses the selected features and adds `alcohol_per_capita`.
* **Not affected:** the rolling-origin backtest (slide 13). `evaluation.py` selects V2 (temporal features plus region effects), which uses only lagged information.

**This is a different issue from the leakage fixes already in the pipeline.** Fixes [A15] (scaler fitted inside a `Pipeline`), [A16] (variant selection on training-period CV) and [A17] (CV frame restricted to the training period) all address *when* data is used for fitting. None of them changes *how* these two features are built, and no document in the repository discusses their same-month construction.

**Recommendation for the modelling owner:** re-run V3 and V4 with the two features lagged by one month. The lagged correlations (0.94 and 0.91) suggest much of their value would remain, so the fix is likely to change the numbers more than the story. This audit does not change any modelling code.

## Finding 4: Output folder name differs by case

`evaluation.py` writes to `outputs/` (lowercase), but the repository commits `Outputs/` and the README refers to `Outputs/`.

* On Windows and macOS (case-insensitive file systems) these are the same folder.
* On Linux, including Streamlit Community Cloud where deployment is planned for Week 10, they are **two different folders**. A fresh run would never update the committed results.

**Recommendation:** use a single folder name in both the code and the README.

## Finding 5: Header drift between crime releases

The newer crime file ships `'Offence type '` (with a trailing space) and `'Reporting Region'`; the older file uses `'Offence type'` and `'Reporting region'`. The pipeline handles both cases by hand (`str.strip()` and an explicit rename). `data_audit.py` now detects this kind of drift automatically. It is the starting point for the schema check planned for the ingest stage before Assessment 4, so that the next release cannot break the panel silently.

## Suggested actions

| # | Action | Suggested owner | Effort |
|---|---|---|---|
| 1 | Correct the wording on slides 3, 4 and 9 using Finding 1 | Presenters of those slides | 15 min |
| 2 | Replace the stale coefficient screenshot on slide 14 | Modelling | 5 min |
| 3 | Re-run V3/V4 with lagged `Alcohol_offences` / `DV_offences` | Modelling | 1–2 h |
| 4 | State that XGBoost results are platform-dependent; save the fitted model | Modelling / dashboard | 30 min |
| 5 | Use one output folder name in `evaluation.py` and the README | Evaluation code owner | 5 min |
| 6 | Add an automated schema check to ingest (Finding 5) | Data engineering | Before A4 |

## Files in this folder

| File | Contents |
|---|---|
| `data_audit.py` | The read-only audit script |
| `A3_source_fingerprints.csv` | Rows, year span, header drift and SHA-256 hash for each source file |
| `A3_data_lineage.csv` | Row counts at every stage, with the reason for each drop |
| `A3_leakage_overlap.csv` | Same-month overlap between candidate features and the target |
