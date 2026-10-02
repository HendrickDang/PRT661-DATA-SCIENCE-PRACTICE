"""
data_audit.py - Read-only data lineage, fingerprint and leakage audit (Assessment 3)
===================================================================================

Purpose
-------
Reconciles every row count the team reports (slides and reports) back to the
source files, fingerprints the inputs, and checks whether any model feature
overlaps with the target in the same month.

This script is READ-ONLY with respect to the pipeline:
  * it does not import or modify Updated_End_to_End_pipeline.py or evaluation.py
  * it does not write to dataset/, Outputs/, outputs/ or any plot folder
  * it writes only to audit/

Run from the repository root AFTER the pipeline has produced the processed panel:
    python Updated_End_to_End_pipeline.py
    python audit/data_audit.py

Outputs (all in audit/):
    A3_source_fingerprints.csv   one row per source file: rows, years, SHA-256
    A3_data_lineage.csv          row counts at each stage, with the reason for every drop
    A3_leakage_overlap.csv       same-month overlap between candidate features and the target

Author: Ngoc Anh Nguyen (Will) - data engineering
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = BASE_DIR / "dataset" / "source"
PROCESSED_CSV = BASE_DIR / "dataset" / "processed" / "nt_crime_merged_2015_2025.csv"
AUDIT_DIR = BASE_DIR / "audit"

# Mirrors the constants used in the pipeline. Read here, never changed there.
YEAR_MIN, YEAR_MAX = 2015, 2025
TRAIN_END_YEAR = 2022
LAGS = (1, 3, 12)
PROVISIONAL_MONTHS = 6
ASSAULT_CATEGORY = "02 Assault"

CRIME_FILES = ["nt_crime_statistics_2020-2023.csv", "nt_crime_statistics_latest.csv"]


# -----------------------------------------------------------------------------
# 1. Source fingerprints
# -----------------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_source(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        sheets = pd.read_excel(path, sheet_name=None)
        # Population ships as Sheet1, alcohol as Data; take the data-bearing sheet.
        df = sheets.get("Data", sheets.get("Sheet1", next(iter(sheets.values()))))
    return df


def crime_header_drift() -> dict:
    """Columns that differ between the two crime files only by case or spacing.
    The pipeline handles today's two cases by hand; a schema check should catch
    the next one automatically."""
    old, new = (list(pd.read_csv(SOURCE_DIR / f, nrows=0).columns) for f in CRIME_FILES)
    key = lambda c: str(c).strip().lower()
    old_by_key = {key(c): c for c in old}
    drift = [f"'{c}' vs '{old_by_key[key(c)]}' in older file"
             for c in new if key(c) in old_by_key and c != old_by_key[key(c)]]
    return {CRIME_FILES[1]: drift}


def source_fingerprints() -> pd.DataFrame:
    drift = crime_header_drift()
    records = []
    for path in sorted(SOURCE_DIR.iterdir()):
        if path.suffix.lower() not in (".csv", ".xlsx"):
            continue
        df = read_source(path)
        cols = [c for c in df.columns]
        year_col = next((c for c in cols if str(c).strip().lower() == "year"), None)
        span = (f"{int(df[year_col].min())}-{int(df[year_col].max())}"
                if year_col is not None else "quarterly (see Quarter Ending)")
        records.append({
            "file": path.name,
            "rows": len(df),
            "columns": len(cols),
            "year_span": span,
            "header_issues": "; ".join(drift.get(path.name, [])) or "none",
            "sha256": sha256(path),
        })
    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# 2. Row-count lineage: source -> window -> processed -> region-month -> model
# -----------------------------------------------------------------------------

def crime_window_counts() -> list[dict]:
    rows = []
    total_raw = total_window = 0
    for name in CRIME_FILES:
        df = pd.read_csv(SOURCE_DIR / name)
        df.columns = df.columns.str.strip()
        in_window = df["Year"].between(YEAR_MIN, YEAR_MAX)
        total_raw += len(df)
        total_window += int(in_window.sum())
        rows.append({"stage": "S1 raw crime rows", "item": name, "rows": len(df),
                     "note": f"years {df['Year'].min()}-{df['Year'].max()}"})
        rows.append({"stage": "S2 inside 2015-2025 window", "item": name,
                     "rows": int(in_window.sum()),
                     "note": f"dropped {int((~in_window).sum()):,} rows outside the window"})
    rows.append({"stage": "S1 raw crime rows", "item": "TOTAL", "rows": total_raw,
                 "note": "figure quoted as '55,146 crime records' in the deck"})
    rows.append({"stage": "S2 inside 2015-2025 window", "item": "TOTAL", "rows": total_window,
                 "note": "rows actually used by the pipeline"})
    return rows


def monthly_assault(panel: pd.DataFrame) -> pd.DataFrame:
    assault = panel[panel["Offence category"] == ASSAULT_CATEGORY]
    m = (assault.groupby(["Year", "Month number", "Region"])
         .agg(Assault_offences=("Number of offences", "sum"),
              Alcohol_offences=("Alcohol_offences", "sum"),
              DV_offences=("DV_offences", "sum"),
              Total_population=("Total_population", "first"))
         .reset_index())
    m["date"] = pd.to_datetime(dict(year=m["Year"], month=m["Month number"], day=1))
    m["rate"] = m["Assault_offences"] / m["Total_population"] * 100_000
    return m


def model_panel_lineage(m: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    regions = sorted(m["Region"].unique())
    dates = pd.date_range(m["date"].min(), m["date"].max(), freq="MS")
    grid = (pd.MultiIndex.from_product([regions, dates], names=["Region", "date"])
            .to_frame(index=False)
            .merge(m[["Region", "date", "rate"]], on=["Region", "date"], how="left")
            .sort_values(["Region", "date"]).reset_index(drop=True))
    for lag in LAGS:
        grid[f"lag{lag}"] = grid.groupby("Region")["rate"].shift(lag)

    first_month = grid["date"].min()

    def reason(r) -> str:
        if pd.isna(r["rate"]):
            return "target missing (Nov 2023 systems gap)"
        if r["date"] < first_month + pd.DateOffset(months=max(LAGS)):
            return "lag-12 warm-up (first 12 months)"
        for lag in LAGS:
            if pd.isna(r[f"lag{lag}"]):
                return f"lag{lag} points at the Nov 2023 gap"
        return "kept"

    grid["reason"] = grid.apply(reason, axis=1)
    kept = grid[grid["reason"] == "kept"]
    last = grid.loc[grid["rate"].notna(), "date"].max()
    provisional = kept["date"] > (last - pd.DateOffset(months=PROVISIONAL_MONTHS))
    train = kept[kept["date"].dt.year <= TRAIN_END_YEAR]
    test = kept[kept["date"].dt.year > TRAIN_END_YEAR]

    rows = [
        {"stage": "S4 region-month observed cells", "item": "assault aggregate",
         "rows": len(m), "note": f"{m['date'].nunique()} months x {len(regions)} regions"},
        {"stage": "S5 complete region-month grid", "item": "regression grid",
         "rows": len(grid), "note": f"{len(dates)} months x {len(regions)} regions "
                                    f"({dates.min():%b %Y}-{dates.max():%b %Y})"},
    ]
    for why, n in grid["reason"].value_counts().items():
        if why != "kept":
            d = grid.loc[grid["reason"] == why, "date"]
            label = (f"{d.min():%b %Y}" if d.nunique() == 1
                     else f"{d.min():%b %Y} to {d.max():%b %Y}")
            rows.append({"stage": "S6 dropped", "item": why, "rows": int(n), "note": label})
    rows += [
        {"stage": "S7 usable model panel", "item": "after drops", "rows": len(kept),
         "note": f"{kept['date'].nunique()} usable months x {len(regions)} regions"},
        {"stage": "S8 train split", "item": f"<= {TRAIN_END_YEAR}", "rows": len(train),
         "note": f"{train['date'].min():%b %Y}-{train['date'].max():%b %Y} "
                 f"({train['date'].nunique()} months; 2015 is consumed by the lag-12 warm-up)"},
        {"stage": "S8 test split", "item": f">= {TRAIN_END_YEAR + 1}", "rows": len(test),
         "note": f"{test['date'].min():%b %Y}-{test['date'].max():%b %Y} "
                 f"({test['date'].nunique()} months; 4 months lost to the Nov 2023 gap)"},
        {"stage": "S8 provisional flag", "item": "last 6 months", "rows": int(provisional.sum()),
         "note": "inside the test split; flagged, not dropped"},
    ]
    return rows, grid


# -----------------------------------------------------------------------------
# 3. Same-month overlap between features and the target
# -----------------------------------------------------------------------------

def leakage_overlap(m: pd.DataFrame) -> pd.DataFrame:
    """Alcohol_offences and DV_offences are built from the assault rows of the
    target month itself. If used unlagged, the model sees part of the answer."""
    records = []
    total = m["Assault_offences"].sum()
    for feat in ["Alcohol_offences", "DV_offences"]:
        lagged = m.sort_values("date").groupby("Region")[feat].shift(1)
        records.append({
            "feature": feat,
            "built_from": "assault rows of the SAME region-month as the target",
            "share_of_target_offences_%": round(100 * m[feat].sum() / total, 1),
            "corr_with_target_same_month": round(m[feat].corr(m["Assault_offences"]), 3),
            "corr_with_target_if_lagged_1m": round(lagged.corr(m["Assault_offences"]), 3),
            "known_at_forecast_time": "No - only after the target month is recorded",
        })
    return pd.DataFrame(records)


# -----------------------------------------------------------------------------

def main() -> None:
    if not PROCESSED_CSV.exists():
        raise SystemExit(f"{PROCESSED_CSV} not found - run Updated_End_to_End_pipeline.py first.")
    AUDIT_DIR.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 80)

    fp = source_fingerprints()
    fp.to_csv(AUDIT_DIR / "A3_source_fingerprints.csv", index=False)
    print("\n== Source fingerprints ==")
    print(fp.drop(columns="sha256").to_string(index=False))

    panel = pd.read_csv(PROCESSED_CSV)
    m = monthly_assault(panel)
    lineage = crime_window_counts()
    lineage.append({"stage": "S3 processed offence-level panel", "item": PROCESSED_CSV.name,
                    "rows": len(panel), "note": "aggregated by region-month-offence type-flags"})
    grid_rows, _ = model_panel_lineage(m)
    lineage += grid_rows
    lineage = pd.DataFrame(lineage)
    lineage.to_csv(AUDIT_DIR / "A3_data_lineage.csv", index=False)
    print("\n== Row-count lineage ==")
    print(lineage.to_string(index=False))

    leak = leakage_overlap(m)
    leak.to_csv(AUDIT_DIR / "A3_leakage_overlap.csv", index=False)
    print("\n== Same-month feature/target overlap ==")
    print(leak.to_string(index=False))
    print(f"\n[audit] wrote 3 files to {AUDIT_DIR.relative_to(BASE_DIR)}/")


if __name__ == "__main__":
    main()
