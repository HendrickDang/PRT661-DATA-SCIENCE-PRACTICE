# Models

## Current status

No serialised model artefacts are stored at this milestone. Both `Updated_End_to_End_pipeline.py` and `evaluation.py` fit their models at run time and report metrics directly, rather than persisting fitted objects.

This is deliberate for Assessment 2. The panel is small (696 usable region-month observations), the models train in seconds, and the rolling-origin backtest refits at every origin by design, so a saved model would not be reusable across horizons. Reproducibility is provided instead by pinned dependencies and committed outputs: anyone can regenerate every reported figure by running the two scripts.

## Models used

| Model | Configuration | Where fitted |
| --- | --- | --- |
| Linear Regression | no hyperparameters | pipeline Stage 5, `evaluation.py` |
| Ridge | `alpha=0.1`, tuned over {0.01, 0.1, 1, 10, 100} on training-period CV | pipeline Stage 5, `evaluation.py` |
| Lasso | `alpha=0.01`, same grid | pipeline Stage 5, `evaluation.py` |
| XGBoost | `max_depth=3, learning_rate=0.05, reg_alpha=1.0, reg_lambda=3.0, n_estimators=800, subsample=0.9, colsample_bytree=0.9, random_state=42` | pipeline Stage 5, `evaluation.py` |
| Seasonal naive (t-12) | benchmark, no fitting | `evaluation.py` |
| Moving average (12m) | benchmark, no fitting | `evaluation.py` |

Feature variant: V4 in the pipeline (log-scale CV), V2 in the evaluation module (rate-scale CV). The reason for the difference is documented in the root `README.md` and Section 4.6 of the report.

## Planned for Assessment 3

The dashboard will need a fitted model available without re-running the pipeline. Serialisation with `joblib.dump()` will be added to the end of the regression stage at that point, writing the tuned XGBoost and Ridge models to this folder along with the feature list used to fit them.
