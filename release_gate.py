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
whatever the release checks say). The expected set is fixed here, not read
from the candidate, so a file missing a whole region or horizon is caught:
    - the expected columns are present;
    - a single forecast origin;
    - exactly the six NT regions and the horizons 1, 3, 6 and 12 months;
    - exactly one recommended forecast for every region-horizon pair;
    - each target month is the origin plus the horizon;
    - forecasts are finite and not negative;
    - each recommended forecast has an 80% range with
      Lower_80 <= forecast <= Upper_80.

dashboard_status() turns the release record into the message the
Administrator Portal shows, so the page reports what was actually published,
not only what the checks allowed.

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
EXPECTED_REGIONS = ("Barkly", "Big Rivers", "Central Australia", "East Arnhem",
                    "Greater Darwin", "Top End")
EXPECTED_HORIZONS = (1, 3, 6, 12)
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
    regions, horizons = set(df["Region"]), set(pd.to_numeric(df["Horizon_months"], errors="coerce"))
    if regions != set(EXPECTED_REGIONS):
        missing_r = sorted(set(EXPECTED_REGIONS) - regions)
        extra_r = sorted(map(str, regions - set(EXPECTED_REGIONS)))
        problems.append("regions do not match the six NT regions"
                        + (f" (missing: {', '.join(missing_r)})" if missing_r else "")
                        + (f" (unexpected: {', '.join(extra_r)})" if extra_r else ""))
    if horizons != set(EXPECTED_HORIZONS):
        problems.append(f"horizons must be exactly {', '.join(map(str, EXPECTED_HORIZONS))} months")

    rec = df[df["Recommended"].astype(str) == "True"]
    if rec.empty:
        return problems + ["no recommended forecasts"]
    per_cell = rec.groupby(["Region", "Horizon_months"]).size()
    expected_cells = pd.MultiIndex.from_product([EXPECTED_REGIONS, EXPECTED_HORIZONS],
                                                names=["Region", "Horizon_months"])
    per_cell = per_cell.reindex(expected_cells, fill_value=0)
    if (per_cell != 1).any():
        problems.append("not exactly one recommended forecast for every region and horizon")

    if df["Origin"].nunique() == 1:
        try:
            origin = pd.Period(str(df["Origin"].iloc[0]), freq="M")
            expect = df["Horizon_months"].astype(int).map(lambda h: str(origin + int(h)))
            if (df["Target_month"].astype(str) != expect).any():
                problems.append("target months do not equal origin plus horizon")
        except (ValueError, TypeError):
            problems.append("origin or target month is not a valid YYYY-MM month")

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


def dashboard_status(summary: dict | None) -> dict:
    """What the Administrator Portal should say about the forecasts it shows.

    The source of truth is the release record (forecast_release), not the
    preliminary promotion flag: the checks can allow promotion while the gate
    still rejects an invalid candidate. Returns level (PASS / WARN / FAIL for
    the banner colour), cleared (did the shown forecasts pass the internal
    checks and get published; this is not operational approval, which needs
    PFES and community review) and a plain-language verdict.
    """
    if not summary:
        return {"level": "WARN", "cleared": False,
                "verdict": "Release checks have not been run yet (python monitoring.py)."}
    release = summary.get("forecast_release") or {}
    origin = release.get("approved_origin", "unknown")
    action = release.get("action")
    if action == "published":
        level = summary.get("overall_status", "PASS")
        return {"level": "WARN" if level == "FAIL" else level, "cleared": True,
                "verdict": ("Internal release checks passed and these forecasts were published. This is a "
                            "prototype demonstration only: it has not been approved for operational PFES use. "
                            "Warnings are reviewed at the next release.")}
    if action == "rejected" and summary.get("promote_forecasts_to_dashboard"):
        return {"level": "FAIL", "cleared": False,
                "verdict": ("The release checks passed, but the new forecast file failed its integrity "
                            "checks and was not published. The ranges below are the last approved "
                            f"forecasts (origin {origin}); do not treat them as an update.")}
    if action == "rejected":
        return {"level": "FAIL", "cleared": False,
                "verdict": ("A release check failed, so the new forecasts were not published. The ranges "
                            f"below are the last approved forecasts (origin {origin}).")}
    if action == "none" and summary.get("promote_forecasts_to_dashboard"):
        return {"level": summary.get("overall_status", "PASS"), "cleared": True,
                "verdict": (f"No new forecast was waiting for release; the published forecasts "
                            f"(origin {origin}) passed the latest internal checks. Prototype demonstration only.")}
    if summary.get("promote_forecasts_to_dashboard") and not release:
        # record written before the release gate existed
        return {"level": summary.get("overall_status", "WARN"), "cleared": False,
                "verdict": "The checks allowed promotion, but no publication record exists; re-run monitoring.py."}
    return {"level": "FAIL", "cleared": False,
            "verdict": "A release check failed: do not use these forecasts in briefings until reviewed."}
