"""
ingest.py
=========
INGEST stage of the PRT661 forecasting pipeline (Group DAN2 - Theme 2).

Role: this script is owned by the "Ingest & Data Store" person (Table 4 of the
Group Project Plan). Its ONE job is to take the raw files exactly as they were
downloaded from the NT Government Open Data Portal, and load them into pandas
DataFrames with basic validation - it does NOT clean, merge, or engineer
features. That belongs in data_processing.py. Keeping the two separate is
what lets each teammate work independently and lets the GitHub commit history
show who did what (a marking requirement in Table 4).

Folder convention - matches the group's GitHub repo layout at the top level
(dataset/, diagrams/, documents/, README), with dataset/ split into three
subfolders so raw downloads, cleaned files, and the final model-ready table
can never be confused with each other:

    <repo root>/
      dataset/
        source/       <- the 6 raw files, exactly as downloaded - never edited by hand
        raw/          <- output of THIS script: same data, standard column
                         names, one file per dataset, plus manifest.json
        processed/    <- output of data_processing.py (created later)
      diagrams/
      documents/
      README
      ingest.py            <- this file
      data_processing.py

ingest.py and data_processing.py sit directly at the repo root - no separate
scripts/ folder - so the project root is simply "the folder this file is in".

Run:
    python ingest.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "dataset" / "source"
RAW_DIR = BASE_DIR / "dataset" / "raw"

# Every file we expect to find in dataset/source/, and how to read it.
# "filename" is the name as currently downloaded from data.nt.gov.au. Add new
# quarters/years here as the datasets are updated (e.g. wholesale alcohol
# 2026) - that is the only line that should ever need to change.
#
# NOTE: file names on the NT Open Data Portal (and Windows' own "(2)" suffix
# on a re-downloaded duplicate) are not perfectly stable - different team
# members downloading the "same" dataset have ended up with slightly
# different names. load_one() below does not fail immediately if the exact
# name isn't found; it first tries a fuzzy match against every file actually
# sitting in dataset/source/, so small naming differences between teammates'
# downloads don't break the pipeline.
EXPECTED_FILES = {
    "crime_2020_2023": {
        "filename": "nt_crime_statistics_2020-2023.csv",
        "reader": "csv",
    },
    "crime_latest": {
        "filename": "nt_crime_statistics_latest.csv",
        "reader": "csv",
    },
    "population": {
        "filename": "nt-population-regions_1986-to-2025.xlsx",
        "reader": "excel",
        "sheet_name": "Sheet1",
    },
    "alcohol_2023": {
        "filename": "wholesale-alcohol-supply-by-quarter-2023.xlsx",
        "reader": "excel",
        "sheet_name": "Data",
    },
    "alcohol_2024": {
        "filename": "wholesale-alcohol-supply-by-quarter-2024.xlsx",
        "reader": "excel",
        "sheet_name": "Data",
    },
    "alcohol_2025": {
        "filename": "wholesale-alcohol-supply-by-quarter-2025.xlsx",
        "reader": "excel",
        "sheet_name": "Data",
    },
}


def _normalise(name: str) -> str:
    """Strip everything but letters and digits, lowercase. Used only to
    compare file names loosely (e.g. 'wholesale-alcohol-supply-by-quarter
    -2024.xlsx' vs 'wholesalealcoholsupplybyquarter2024.xlsx' normalise to
    the same string, and 'wholesale-alcohol-supply-by-quarter-2025 (2).xlsx'
    - Windows' auto-renamed duplicate download - still matches)."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def find_source_file(expected_filename: str) -> Path:
    """Return the path to use for expected_filename. Tries the exact name
    first; if that's missing, scans dataset/source/ for a file whose
    normalised name matches or contains the expected one (handles
    dash/underscore differences and Windows " (2)" duplicate suffixes) and
    uses that instead (printing a note), so small naming differences don't
    stop the whole pipeline."""
    exact = SOURCE_DIR / expected_filename
    if exact.exists():
        return exact

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"dataset/source/ does not exist yet: {SOURCE_DIR}")

    target = _normalise(Path(expected_filename).stem)
    for candidate in SOURCE_DIR.iterdir():
        if not candidate.is_file():
            continue
        normalised_candidate = _normalise(candidate.stem)
        if normalised_candidate == target or target in normalised_candidate:
            print(f"  NOTE: expected '{expected_filename}' but found '{candidate.name}' "
                  f"(names differ, contents assumed the same) - using it.")
            return candidate

    raise FileNotFoundError(
        f"Expected source file not found: {exact}\n"
        f"Place the raw download in dataset/source/ before running ingest.py "
        f"(exact name doesn't have to match '{expected_filename}', but the file must be there)."
    )


def load_one(spec: dict) -> pd.DataFrame:
    path = find_source_file(spec["filename"])
    if spec["reader"] == "csv":
        df = pd.read_csv(path)
    elif spec["reader"] == "excel":
        df = pd.read_excel(path, sheet_name=spec["sheet_name"])
    else:
        raise ValueError(f"Unknown reader type: {spec['reader']}")
    return df


def validate(name: str, df: pd.DataFrame) -> list:
    """Cheap sanity checks - not full cleaning. If one of these fires, stop
    and look at the file before trusting anything downstream."""
    issues = []
    if df.empty:
        issues.append("DataFrame is empty")
    if df.columns.duplicated().any():
        issues.append(f"Duplicate column names: {list(df.columns[df.columns.duplicated()])}")
    fully_null_cols = [c for c in df.columns if df[c].isna().all()]
    if fully_null_cols:
        issues.append(f"Fully-null columns: {fully_null_cols}")
    return issues


def main():
    print(f"[ingest] project root (this script's folder): {BASE_DIR}")
    print(f"[ingest] looking for source files in: {SOURCE_DIR}")
    if not SOURCE_DIR.exists():
        print(f"[ingest] WARNING: that folder does not exist. Create dataset/source/ "
              f"next to this script and put the 6 raw files in it.\n")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"ingested_at_utc": datetime.now(timezone.utc).isoformat(), "datasets": {}}
    had_error = False

    for name, spec in EXPECTED_FILES.items():
        print(f"[ingest] loading {name} <- dataset/source/{spec['filename']}")
        try:
            df = load_one(spec)
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            had_error = True
            continue

        issues = validate(name, df)
        for issue in issues:
            print(f"  WARNING: {issue}")

        out_path = RAW_DIR / f"{name}.csv"
        df.to_csv(out_path, index=False)

        manifest["datasets"][name] = {
            "source_file": spec["filename"],
            "rows": len(df),
            "columns": list(df.columns),
            "saved_to": str(out_path.relative_to(BASE_DIR)),
            "issues": issues,
        }
        print(f"  OK: {len(df)} rows, {len(df.columns)} columns -> {out_path.name}")

    manifest_path = RAW_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n[ingest] manifest written to {manifest_path}")

    if had_error:
        print("\n[ingest] finished WITH missing files - see SKIPPED lines above.")
        sys.exit(1)
    print("[ingest] finished OK.")


if __name__ == "__main__":
    main()