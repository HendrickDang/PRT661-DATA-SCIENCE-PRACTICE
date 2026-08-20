"""
pipeline.py
===========
Combined INGEST + DATA PROCESSING stages of the PRT661 forecasting pipeline
(Group DAN2 - Theme 2).

This file is a straight merge of ingest.py and data_processing.py so the
whole pipeline can be run with a single command instead of two. NOTHING in
either script's logic was changed - every function body, every comment,
every [A1]-[A5] assumption, the fuzzy filename matching, the rename_columns
workaround, all of it is copied over exactly as it was. The only changes
are the mechanical ones needed to make two separate scripts coexist in one
file without clashing:

  - BASE_DIR / RAW_DIR are defined once instead of twice (both scripts
    computed the exact same paths independently, so sharing them changes
    nothing).
  - ingest.py's main() is renamed ingest_main() and data_processing.py's
    main() is renamed processing_main(), purely so Python doesn't see two
    functions named "main" in the same file. Their bodies are untouched.
  - At the very bottom, ingest_main() is called first and processing_main()
    second - the same order you'd run the two original scripts in.

If ingest_main() hits a missing file it calls sys.exit(1) exactly as
before, which stops the whole pipeline.py run before processing_main() ever
starts - same behaviour as if you'd stopped after a failed `python
ingest.py` and never run `python data_processing.py`.

Run:
    python pipeline.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "dataset" / "source"
RAW_DIR = BASE_DIR / "dataset" / "raw"
PROCESSED_DIR = BASE_DIR / "dataset" / "processed"


# =============================================================================
# SECTION 1 - INGEST (verbatim from ingest.py)
# =============================================================================

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


def ingest_main():
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
        try:
            df.to_csv(out_path, index=False)
        except PermissionError as e:
            raise PermissionError(
                f"Cannot write to {out_path} - Windows says the file is in use. "
                f"This almost always means it's currently open in Excel (or another "
                f"program) on your computer. Close that file and run this script again."
            ) from e

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


# =============================================================================
# SECTION 2 - DATA PROCESSING (verbatim from data_processing.py)
# =============================================================================

def rename_columns(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Drop-in replacement for df.rename(columns=mapping).

    Some pandas/numpy version combinations (seen with the group's newer
    prt564env conda environment) hit an internal pandas bug inside
    DataFrame.rename() itself - it fails deep inside pandas' own indexer
    machinery (Index.get_indexer_for -> ... -> Index.hasnans -> np.isnan),
    not because of anything wrong with our data or column names. Rather
    than depend on every teammate's machine having a pandas/numpy pairing
    that avoids that internal path, we just reassign df.columns directly -
    a plain Python list comprehension has no way to hit that bug, and the
    result is identical to .rename(columns=mapping) in every case we use
    it for here (no MultiIndex columns, no regex)."""
    df = df.copy()
    df.columns = [mapping.get(c, c) for c in df.columns]
    return df

# ---------------------------------------------------------------------------
# ASSUMPTION LOG - decisions a marker or teammate needs to be able to see.
# Every non-obvious choice below is numbered here and referenced in comments
# with its [A#] tag. Copy this block into the report's data processing
# section - assessors expect exactly this kind of documented judgement call.
# ---------------------------------------------------------------------------
# [A1] The two crime files use different classification schemes (PROMIS-era
#      categories pre-Nov-2023 vs the ANZSOC-aligned scheme from Dec-2023,
#      per Risk 1 in the project plan). We map both to a single "Assault"
#      series: old file -> Offence type == 'Assault'; new file -> Offence
#      type in {021 Serious Assault, 022 Assault of a prescribed officer,
#      023 Common Assault}. Sexual assault (031/032) is a different offence
#      category and is excluded on purpose.
# [A2] November 2023 has NO data in either file (old file ends Oct-2023, new
#      file starts Dec-2023) - a genuine one-month gap from the systems
#      transition. We do not fabricate a value; the row is left as NaN so it
#      is visible and can be interpolated or excluded deliberately later,
#      never silently averaged away.
# [A3] Per Risk 2 in the project plan, the most recent 6 months of crime data
#      are provisional (late-reported offences). This script flags them with
#      is_provisional=True rather than dropping them, so the modelling stage
#      can decide whether to exclude them from train/test.
# [A4] Wholesale alcohol PAC is only published quarterly. We assign the same
#      quarterly Total PAC value to each of the 3 months in that quarter
#      (not divided by 3) - it represents "supply during this quarter", so a
#      monthly average would understate it. Document this if the model later
#      normalises it.
# [A5] Population data uses 6 NTG service-delivery regions (Barkly, Big
#      Rivers, Central Australia, East Arnhem, Greater Darwin, Top End)
#      while crime/alcohol data uses 7 police reporting regions (Alice
#      Springs, Darwin, Katherine, Nhulunbuy, NT Balance, Palmerston,
#      Tennant Creek) - Risk 4 in the project plan. The crosswalk below is
#      an approximation based on which town sits in which service region,
#      NOT an official NTG concordance. "NT Balance" is the weakest link -
#      it is a catch-all for everywhere outside the seven towns, so it does
#      not map cleanly onto a single population region. BEFORE this goes in
#      the final report, verify this crosswalk against the region
#      boundary maps on data.nt.gov.au, or rebuild it bottom-up from the
#      Statistical Area 2 column, which uses recognisable ABS geography.
REGION_CROSSWALK_POLICE_TO_POPULATION = {
    "Darwin": "Greater Darwin",
    "Palmerston": "Greater Darwin",
    "Alice Springs": "Central Australia",
    "Katherine": "Big Rivers",
    "Tennant Creek": "Barkly",
    "Nhulunbuy": "East Arnhem",
    "NT Balance": "Top End",  # [A5] weakest link - approximation, verify
}

ASSAULT_TYPES_OLD = {"Assault"}
ASSAULT_TYPES_NEW = {
    "021 Serious Assault",
    "022 Assault of a prescribed officer",
    "023 Common Assault",
}


def process_crime() -> pd.DataFrame:
    old = pd.read_csv(RAW_DIR / "crime_2020_2023.csv")
    new = pd.read_csv(RAW_DIR / "crime_latest.csv")

    # standardise column names (new file has 'Offence type ' with a
    # trailing space and 'Reporting Region' with a capital R)
    old = rename_columns(old, {"Reporting region": "region"})
    new = rename_columns(new, {"Offence type ": "Offence type", "Reporting Region": "region"})

    old_assault = old[old["Offence type"].isin(ASSAULT_TYPES_OLD)].copy()
    new_assault = new[new["Offence type"].isin(ASSAULT_TYPES_NEW)].copy()
    new_assault = new_assault[new_assault["region"] != "Unknown"]

    keep_cols = ["Year", "Month number", "region", "Alcohol involvement", "DV involvement", "Number of offences"]
    combined = pd.concat([old_assault[keep_cols], new_assault[keep_cols]], ignore_index=True)
    combined = rename_columns(combined, {"Year": "year", "Month number": "month", "Number of offences": "offences"})

    # aggregate sub-rows (by Statistical Area 2 / alcohol / DV flags) up to
    # one assault_count per region-year-month, while also keeping the
    # alcohol/DV involvement proportions as separate engineered features
    # (the problem statement calls out both as demand drivers)
    total = combined.groupby(["region", "year", "month"], as_index=False)["offences"].sum()
    total = rename_columns(total, {"offences": "assault_count"})

    alcohol_yes = (
        combined[combined["Alcohol involvement"] == "Yes"]
        .groupby(["region", "year", "month"])["offences"].sum()
    )
    dv_yes = (
        combined[combined["DV involvement"] == "Yes"]
        .groupby(["region", "year", "month"])["offences"].sum()
    )
    total = total.set_index(["region", "year", "month"])
    total["assault_alcohol_involved"] = alcohol_yes
    total["assault_dv_involved"] = dv_yes
    total = total.fillna({"assault_alcohol_involved": 0, "assault_dv_involved": 0}).reset_index()
    total["pct_alcohol_involved"] = (total["assault_alcohol_involved"] / total["assault_count"]).round(3)
    total["pct_dv_involved"] = (total["assault_dv_involved"] / total["assault_count"]).round(3)

    # build a complete region x year x month grid so the Nov-2023 gap ([A2])
    # and any other silent gaps show up as explicit NaN rows, not missing rows
    all_regions = sorted(total["region"].unique())
    full_index = pd.MultiIndex.from_product(
        [
            all_regions,
            range(total["year"].min(), total["year"].max() + 1),
            range(1, 13),
        ],
        names=["region", "year", "month"],
    )
    full = pd.DataFrame(index=full_index).reset_index()
    full = full.merge(total, on=["region", "year", "month"], how="left")
    # trim to the actual observed date range so we don't invent 13 extra
    # months before the data starts / after it ends
    full["date"] = pd.to_datetime(dict(year=full["year"], month=full["month"], day=1))
    obs_min = pd.to_datetime(dict(year=[total["year"].min()], month=[total.loc[total["year"] == total["year"].min(), "month"].min()], day=[1])).iloc[0]
    obs_max = pd.to_datetime(dict(year=[total["year"].max()], month=[total.loc[total["year"] == total["year"].max(), "month"].max()], day=[1])).iloc[0]
    full = full[(full["date"] >= obs_min) & (full["date"] <= obs_max)].drop(columns="date")

    # [A3] flag provisional months: the 6 most recent calendar months in the data
    latest_year, latest_month = total["year"].max(), total.loc[total["year"] == total["year"].max(), "month"].max()
    cutoff = pd.Period(year=latest_year, month=int(latest_month), freq="M") - 5
    full["is_provisional"] = full.apply(
        lambda r: pd.Period(year=int(r["year"]), month=int(r["month"]), freq="M") >= cutoff, axis=1
    )

    print(f"[crime] {len(full)} region-month rows | {full['assault_count'].isna().sum()} missing "
          f"(expect 7, one per region, for the Nov-2023 gap)")
    return full


def process_alcohol() -> pd.DataFrame:
    frames = []
    for year in (2023, 2024, 2025):
        df = pd.read_csv(RAW_DIR / f"alcohol_{year}.csv")
        frames.append(df)
    alcohol = pd.concat(frames, ignore_index=True)
    alcohol["Quarter Ending"] = pd.to_datetime(alcohol["Quarter Ending"])
    alcohol["year"] = alcohol["Quarter Ending"].dt.year
    alcohol["quarter"] = alcohol["Quarter Ending"].dt.quarter
    alcohol = rename_columns(alcohol, {"Region": "region", "Total PAC": "total_pac"})

    # [A4] broadcast each quarterly value to its 3 months
    rows = []
    for _, r in alcohol.iterrows():
        start_month = (r["quarter"] - 1) * 3 + 1
        for m in range(start_month, start_month + 3):
            rows.append({"region": r["region"], "year": r["year"], "month": m, "total_pac": r["total_pac"]})
    monthly_alcohol = pd.DataFrame(rows)

    print(f"[alcohol] {len(alcohol)} quarterly rows -> {len(monthly_alcohol)} monthly rows")
    return monthly_alcohol


def process_population() -> pd.DataFrame:
    pop = pd.read_csv(RAW_DIR / "population.csv")
    # one figure per population-region/year: sum across sex, age group,
    # Aboriginal status. Prefer 'Final' status where more than one status
    # exists for the same year (shouldn't happen, but be defensive)
    status_priority = {"Final": 0, "Revised": 1, "Preliminary": 2}
    pop["status_rank"] = pop["Status"].map(status_priority)
    best_status = pop.groupby("Year")["status_rank"].transform("min")
    pop = pop[pop["status_rank"] == best_status]

    pop_totals = pop.groupby(["Region", "Year"], as_index=False)["Population"].sum()
    pop_totals = rename_columns(pop_totals, {"Region": "pop_region", "Year": "year", "Population": "population"})

    # [A5] map the 7 police regions onto the 6 population regions
    crosswalk = pd.DataFrame(
        [{"region": k, "pop_region": v} for k, v in REGION_CROSSWALK_POLICE_TO_POPULATION.items()]
    )
    mapped = crosswalk.merge(pop_totals, on="pop_region", how="left")
    print(f"[population] {len(pop_totals)} population-region/year rows mapped to "
          f"{mapped['region'].nunique()} police regions via the [A5] crosswalk")
    return mapped[["region", "year", "population"]]


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["region", "year", "month"]).reset_index(drop=True)

    # per-capita normalisation - the plan's stated reason for including
    # population data at all ("standardise crime rates per 100,000 population")
    panel["assault_rate_per_100k"] = (panel["assault_count"] / panel["population"] * 100_000).round(2)

    # cyclical month encoding - captures dry/wet season seasonality without
    # treating December and January as 11 months apart
    panel["month_sin"] = np.sin(2 * np.pi * panel["month"] / 12)
    panel["month_cos"] = np.cos(2 * np.pi * panel["month"] / 12)

    # lagged indicators, computed per region so region A's history never
    # leaks into region B's lag features
    for lag in (1, 2, 3, 12):
        panel[f"assault_count_lag{lag}"] = panel.groupby("region")["assault_count"].shift(lag)

    panel["total_pac_lag1"] = panel.groupby("region")["total_pac"].shift(1)

    return panel


def processing_main():
    print(f"[data_processing] reading cleaned files from: {RAW_DIR}")
    missing = [f"{n}.csv" for n in
               ("crime_2020_2023", "crime_latest", "population", "alcohol_2023", "alcohol_2024", "alcohol_2025")
               if not (RAW_DIR / f"{n}.csv").exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {RAW_DIR} - run ingest.py first, it creates these."
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    crime = process_crime()
    alcohol = process_alcohol()
    population = process_population()

    panel = crime.merge(alcohol, on=["region", "year", "month"], how="left")
    panel = panel.merge(population, on=["region", "year"], how="left")
    panel = add_features(panel)

    out_path = PROCESSED_DIR / "nt_assault_panel.csv"
    try:
        panel.to_csv(out_path, index=False)
    except PermissionError as e:
        raise PermissionError(
            f"Cannot write to {out_path} - Windows says the file is in use. "
            f"This almost always means it's currently open in Excel (or another "
            f"program) on your computer. Close that file and run this script again."
        ) from e
    print(f"\n[data_processing] final panel: {panel.shape[0]} rows x {panel.shape[1]} columns -> {out_path}")
    print(f"[data_processing] regions: {sorted(panel['region'].unique())}")
    print(f"[data_processing] date range: {panel['year'].min()}-{panel[panel['year']==panel['year'].min()]['month'].min():02d}"
          f" to {panel['year'].max()}-{panel[panel['year']==panel['year'].max()]['month'].max():02d}")
    print(f"[data_processing] rows missing population (check [A5] crosswalk): {panel['population'].isna().sum()}")
    print(f"[data_processing] rows missing alcohol PAC (expected before 2023): {panel['total_pac'].isna().sum()}")


# =============================================================================
# ENTRY POINT - run ingest, then processing, same order as the two scripts
# =============================================================================

if __name__ == "__main__":
    ingest_main()
    print()
    processing_main()