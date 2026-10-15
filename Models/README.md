# Models

## Saved model bundles (Assessment 4)

| File | Written by | Contents |
| --- | --- | --- |
| `model_a.joblib` | `Updated_End_to_End_pipeline.py`, regression stage [A21] | The four fitted Model A estimators (Linear Regression, Ridge, Lasso, XGBoost), the scaler the linear models were fitted with, the feature order, and the package versions. Trained on 2016-2022. |
| `forecast_h1.joblib`, `forecast_h3.joblib`, `forecast_h6.joblib`, `forecast_h12.joblib` | `forecast.py` | One bundle per horizon: the four fitted models as scikit-learn Pipelines (scaler inside), the features for that horizon, and the training window (target months 2016-01 to 2025-12). |

Each file is a dictionary. Example:

```python
import joblib
import numpy as np

bundle = joblib.load("Models/forecast_h1.joblib")
bundle["features"]          # column order the model expects
bundle["versions"]          # python, scikit-learn and xgboost used to fit it
pipe = bundle["models"]["XGBoost"]
rate_per_100k = np.expm1(pipe.predict(X[bundle["features"]].astype(float)))
```

For `model_a.joblib` the linear models expect features transformed by `bundle["scaler_for_linear_models"]`; XGBoost takes the raw features. All models predict `log1p` of the assault rate, so back-transform with `np.expm1`.

The bundles load only under the pinned versions in `Environment Setup Instructions/requirements.txt` (`scikit-learn==1.3.2`, `xgboost==2.1.4`, `joblib==1.4.2`). They were written on Linux x86_64 with Python 3.11 and load and reproduce the same forecasts on Python 3.8.

The rolling-origin backtest in `evaluation.py` still refits at every origin by design and does not use these files. They are for the dashboard and for forecasting without re-running the pipeline.

## Models used

| Model | Configuration | Where fitted |
| --- | --- | --- |
| Linear Regression | no hyperparameters | pipeline Stage 5, `evaluation.py`, `forecast.py` |
| Ridge | `alpha=0.01`, tuned over {0.01, 0.1, 1, 10, 100} on training-period CV | pipeline Stage 5, `evaluation.py`, `forecast.py` |
| Lasso | `alpha=0.01`, same grid | pipeline Stage 5, `evaluation.py`, `forecast.py` |
| XGBoost | `max_depth=3, learning_rate=0.05, reg_alpha=1.0, reg_lambda=3.0, n_estimators=800, subsample=0.9, colsample_bytree=0.9, random_state=42` | pipeline Stage 5, `evaluation.py`, `forecast.py` |
| Seasonal naive (t-12) | benchmark, no fitting | `evaluation.py`, `forecast.py` |
| Moving average (12m) | benchmark, no fitting | `evaluation.py`, `forecast.py` |

Feature variant: V4 in the pipeline (log-scale CV), the evaluation module (rate-scale CV) and the forecast stage, which reuses the evaluation set-up. The alcohol- and DV-flagged assault counts enter lagged to the forecast origin ([A18]). One model is fitted per horizon, with every lag pushed back to the origin.

Recommended method by horizon, from the backtest: XGBoost at 1 month, the 12-month moving average at 3 and 6 months, the seasonal naive value at 12 months (`Outputs/forecast_run_summary.json`).
