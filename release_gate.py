"""
release_gate.py - Publish or reject a candidate forecast (Assessment 4)

forecast.py writes each new forecast to Outputs/staging/F1_forecasts.csv.
That file is a candidate: the dashboard never reads it. monitoring.py runs
the release checks and then calls publish(), which

    * replaces Outputs/F1_forecasts.csv (the approved file the dashboard
      reads) with the candidate only when the checks allow promotion and the
      candidate passes the integrity checks below, using a temporary file
      and an atomic rename so a reader never sees a half-written file;
    * otherwise leaves the approved file untouched, so the dashboard keeps
      showing the last approved forecast, and keeps the rejected candidate
      in staging for investigation.

Integrity checks on the candidate (a candidate that fails is never published,
whatever the release checks say):
    - the expected columns are present;
    - one recommended forecast per region and horizon, for every region;
    - a single forecast origin;
    - forecasts are finite and not negative;
    - each recommended forecast has an 80% range with
      Lower_80 <= forecast <= Upper_80.

The decision is returned as a dictionary that monitoring.py writes into
Outputs/M3_monitoring_summary.json. tests/test_release_gate.py forces a FAIL
and checks that the approved file, and so the dashboard data, stay unchanged.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

APPROVED_NAME = "F1_forecasts.csv"
STAGING_DIRNAME = "staging"
REQUIRED_COLUMNS = ["Origin", "Horizon_months", "Target_month", "Region", "Model",
                    "Recommended", "Forecast_rate_100k", "Lower_80", "Upper_80"]


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_problems(path: Path) -> list[str]:
    """Integrity problems with a candidate forecast file; empty if it is valid."""
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # unreadable file
        return [f"cannot read candidate: {exc}"]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return [f"missing columns: {', '.join(missing)}"]

    problems = []
    if df["Origin"].nunique() != 1:
        problems.append("more than one forecast origin")
    rec = df[df["Recommended"].astype(str) == "True"]
    if rec.empty:
        return problems + ["no recommended forecasts"]
    per_cell = rec.groupby(["Region", "Horizon_months"]).size()
    if (per_cell != 1).any():
        problems.append("not exactly one recommended forecast per region and horizon")
    regions = set(df["Region"])
    for h, grp in rec.groupby("Horizon_months"):
        if set(grp["Region"]) != regions:
            problems.append(f"h = {h}: recommended forecast missing for some regions")
    values = pd.to_numeric(df["Forecast_rate_100k"], errors="coerce")
    if not np.isfinite(values).all() or (values < 0).any():
        problems.append("forecasts must be finite and not negative")
    lo = pd.to_numeric(rec["Lower_80"], errors="coerce")
    hi = pd.to_numeric(rec["Upper_80"], errors="coerce")
    fc = pd.to_numeric(rec["Forecast_rate_100k"], errors="coerce")
    if lo.isna().any() or hi.isna().any():
        problems.append("recommended forecasts need an 80% range")
    elif ((lo > fc) | (fc > hi)).any():
        problems.append("80% range does not contain the forecast")
    return problems


def publish(promote: bool, out_dir: Path) -> dict:
    """Publish the staged candidate if allowed; otherwise keep the approved file."""
    out_dir = Path(out_dir)
    staged = out_dir / STAGING_DIRNAME / APPROVED_NAME
    approved = out_dir / APPROVED_NAME
    result = {
        "candidate": f"{out_dir.name}/{STAGING_DIRNAME}/{APPROVED_NAME}",
        "approved": f"{out_dir.name}/{APPROVED_NAME}",
        "candidate_sha256": sha256(staged),
        "approved_sha256_before": sha256(approved),
    }

    if not staged.exists():
        result.update(action="none", published=False,
                      reason="no staged candidate; the approved forecasts are unchanged")
    else:
        problems = candidate_problems(staged)
        if problems:
            result.update(action="rejected", published=False,
                          reason="candidate failed integrity checks: " + "; ".join(problems))
        elif not promote:
            result.update(action="rejected", published=False,
                          reason="a release check failed; the last approved forecasts stay on the dashboard")
        else:
            tmp = out_dir / f".{APPROVED_NAME}.tmp"
            shutil.copyfile(staged, tmp)
            os.replace(tmp, approved)   # atomic on the same file system
            result.update(action="published", published=True,
                          reason="release checks allow promotion and the candidate is valid")

    result["approved_sha256_after"] = sha256(approved)
    if approved.exists():
        result["approved_origin"] = str(pd.read_csv(approved, usecols=["Origin"])["Origin"].iloc[0])
    return result
