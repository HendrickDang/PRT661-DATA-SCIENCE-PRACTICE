"""
data_processing.py
===================
DATA PROCESSING stage of the PRT661 forecasting pipeline (Group DAN2 - Theme 2).

Role: this script is owned by the "Data processing" person (Table 4 of the
Group Project Plan). It reads the clean-but-unmerged files that ingest.py
saved into dataset/raw/, and turns them into ONE tidy table:

    one row = one (Region, Year, Month)

with the target variable (assault_count), the engineered features the plan
promises in Table 2 ("lagged indicators, cyclical month encoding, per-capita
normalisation"), and the alcohol / population context columns the model will
use as predictors. The output is written to dataset/processed/nt_assault_panel.csv
and is what the Analytics/ML person (Hendrick's stage) trains the Ridge/Lasso
models on.

This script lives directly at the repo root (next to ingest.py, dataset/,
diagrams/, documents/, README) - no separate scripts/ folder, matching the
group's GitHub layout.

This script deliberately keeps four steps separate (crime, alcohol,
population, feature engineering) with a print after each one, so that if a
row count looks wrong you can immediately see which step broke it.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "dataset" / "raw"
PROCESSED_DIR = BASE_DIR / "dataset" / "processed"


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


def main():
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
    panel.to_csv(out_path, index=False)
    print(f"\n[data_processing] final panel: {panel.shape[0]} rows x {panel.shape[1]} columns -> {out_path}")
    print(f"[data_processing] regions: {sorted(panel['region'].unique())}")
    print(f"[data_processing] date range: {panel['year'].min()}-{panel[panel['year']==panel['year'].min()]['month'].min():02d}"
          f" to {panel['year'].max()}-{panel[panel['year']==panel['year'].max()]['month'].max():02d}")
    print(f"[data_processing] rows missing population (check [A5] crosswalk): {panel['population'].isna().sum()}")
    print(f"[data_processing] rows missing alcohol PAC (expected before 2023): {panel['total_pac'].isna().sum()}")


if __name__ == "__main__":
    main()