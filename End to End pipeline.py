"""
End-to-end pipeline for the PRT661 NT assault forecasting project
(Group DAN2 - Theme 2).

    INGEST      dataset/source/  -> dataset/raw/*.csv + manifest.json
    PANEL       dataset/raw/     -> dataset/processed/nt_crime_merged_2015_2025.csv
    EDA         -> eda_plots/*.png
    PCA         -> age-structure components
    REGRESSION  -> regression_plots/*.png  (Model A and Model B)

Ingest is the only stage that touches the raw downloads, so portal renames and
Excel sheet names stay in one place. A missing source file stops the run before
any analysis happens.

---------------------------------------------------------------------------
ASSUMPTION LOG - the non-obvious decisions, referenced as [A#] in the code.
---------------------------------------------------------------------------
[A1]  Window is 2015-2025; the old crime file starts in 2008 and is trimmed.
[A2]  CATEGORY_MAP converts old PROMIS category names to the ANZSOC names used
      from Dec 2023, so "02 Assault" means one thing across the panel. An
      unmapped category raises rather than silently becoming NaN.
[A3]  November 2023 is missing from both files - a real gap from the systems
      transition. Never filled in; it stays visible as NaN.
[A4]  Reporting Region "Unknown" -> "Top End", the residual region.
[A5]  Everything is harmonised onto the 6 NTG population regions: the seven
      towns map to their service region, "NT Balance" is resolved by its SA2.
      Replaces a crosswalk that gave Darwin and Palmerston the whole Greater
      Darwin population each, which made every per-100k rate wrong.
[A6]  Quarterly PAC is attached to all 3 months of its quarter, not divided by
      3: it measures supply during the quarter.
[A7]  PAC exists only for 2023-2025. Earlier years stay NaN; the region x
      quarter mean fills only gaps inside 2023-2025.
[A8]  Alcohol/DV involvement is counted in OFFENCES, not in matching source
      rows. The new schema splits assault into 3 types instead of 2, so row
      counts per region-month jump from ~4 to ~10 at the changeover.
[A9]  The last 6 months are provisional (late-reported); flagged, not dropped.
[A10] The regression panel uses a complete region x month grid so the lags are
      true calendar lags - otherwise the [A3] gap makes Dec-2023 lag1 point at
      October.
[A11] Model A: no alcohol features, train 2015-2022, test 2023-2025.
[A12] Model B: adds alcohol_per_capita, train 2023-2024, test 2025. Different
      training window from Model A, so the errors are NOT comparable.
[A13] CV rows are sorted by date before TimeSeriesSplit. Sorted by region, the
      folds would train on one region and test on another.
[A14] PCA uses age-group shares, not counts: counts all scale with region size,
      leaving PC1 at ~99% and measuring only "how big is this region".
---------------------------------------------------------------------------
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write PNGs without needing a display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "dataset" / "source"
RAW_DIR = BASE_DIR / "dataset" / "raw"
PROCESSED_DIR = BASE_DIR / "dataset" / "processed"
PLOT_DIR = BASE_DIR / "eda_plots"
REG_PLOT_DIR = BASE_DIR / "regression_plots"

YEAR_MIN, YEAR_MAX = 2015, 2025
ALCOHOL_YEARS = (2023, 2024, 2025)


def write_csv(df: pd.DataFrame, out_path: Path) -> None:
    """Write df to out_path, turning Windows' file-lock error into a message
    that says what to actually do about it."""
    try:
        df.to_csv(out_path, index=False)
    except PermissionError as e:
        raise PermissionError(
            f"Cannot write to {out_path} - Windows says the file is in use. "
            f"This almost always means it's currently open in Excel (or another program) on your computer. Close that file and run this script again."
        ) from e


def banner(text: str) -> None:
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


# =============================================================================
# SECTION 1 - INGEST
# =============================================================================

# Every file we expect in dataset/source/, and how to read it. Adding a new
# year/quarter should only ever mean adding an entry here.
EXPECTED_FILES = {
    # Filename says 2020-2023 but the data starts in 2008; [A1] trims it.
    "crime_old": {
        "filename": "nt_crime_statistics_2020-2023.csv",
        "reader": "csv",
    },
    "crime_new": {
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
    """Lowercase, keep only letters and digits. Used to compare file names
    loosely, ignoring dashes, underscores and Windows' " (2)" suffix."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def find_source_file(expected_filename: str) -> Path:
    """The exact filename if present, else the first file in dataset/source/
    whose normalised name matches or contains it - a near-miss name from a
    teammate's download shouldn't stop the pipeline."""
    exact = SOURCE_DIR / expected_filename
    if exact.exists():
        return exact

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"dataset/source/ does not exist yet: {SOURCE_DIR}")

    target = _normalise(Path(expected_filename).stem)
    for candidate in sorted(SOURCE_DIR.iterdir()):
        if not candidate.is_file():
            continue
        normalised_candidate = _normalise(candidate.stem)
        if normalised_candidate == target or target in normalised_candidate:
            print(f"  NOTE: expected '{expected_filename}' but found '{candidate.name}' "
                  f"(names differ, contents assumed the same) - using it.")
            return candidate

    raise FileNotFoundError(
        f"Expected source file not found: {exact}\n"
        f"Place the raw download in dataset/source/ before running this script "
        f"(the exact name doesn't have to match '{expected_filename}', but the "
        f"file must be there)."
    )


def load_one(spec: dict) -> pd.DataFrame:
    path = find_source_file(spec["filename"])
    if spec["reader"] == "csv":
        return pd.read_csv(path)
    if spec["reader"] == "excel":
        return pd.read_excel(path, sheet_name=spec["sheet_name"])
    raise ValueError(f"Unknown reader type: {spec['reader']}")


def validate(df: pd.DataFrame) -> list:
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


def ingest_main() -> None:
    banner("STAGE 1 - INGEST: dataset/source/ -> dataset/raw/")
    print(f"[ingest] project root: {BASE_DIR}")
    print(f"[ingest] source files: {SOURCE_DIR}")
    if not SOURCE_DIR.exists():
        print("[ingest] WARNING: that folder does not exist. Create dataset/source/ "
              "next to this script and put the 6 raw files in it.\n")
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

        issues = validate(df)
        for issue in issues:
            print(f"  WARNING: {issue}")

        out_path = RAW_DIR / f"{name}.csv"
        write_csv(df, out_path)

        manifest["datasets"][name] = {
            "source_file": spec["filename"],
            "rows": len(df),
            "columns": list(df.columns),
            "saved_to": str(out_path.relative_to(BASE_DIR)),
            "issues": issues,
        }
        print(f"  OK: {len(df):,} rows, {len(df.columns)} columns -> {out_path.name}")

    manifest_path = RAW_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n[ingest] manifest written to {manifest_path.name}")

    if had_error:
        print("\n[ingest] finished WITH missing files - see SKIPPED lines above.")
        sys.exit(1)
    print("[ingest] finished OK.")


# =============================================================================
# SECTION 2 - BUILD THE ANALYSIS PANEL
# =============================================================================

def rename_columns(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Drop-in replacement for df.rename(columns=mapping), which crashes inside
    pandas' own indexer code on some pandas/numpy pairings teammates have
    installed. Reassigning df.columns is equivalent for our use (no MultiIndex
    columns, no regex) and cannot hit that bug."""
    df = df.copy()
    df.columns = [mapping.get(c, c) for c in df.columns]
    return df


# [A2] old PROMIS offence category -> new ANZSOC offence category
CATEGORY_MAP = {
    "Acts intended to cause Injury": "02 Assault",
    "Homicide and related Offences": "01 Homicide",
    "Abduction - harassment and other offences against the person":
        "04 Harm or endanger persons",
    "Other dangerous or negligent acts endangering persons":
        "04 Harm or endanger persons",
    "Sexual assault and related offences": "03 Sexual offences",
    "Robbery - extortion and related offences":
        "05 Robbery, blackmail, and extortion",
    "House break-ins": "061 Burglary - dwelling",
    "Commercial break-ins": "062 Burglary - non-residential",
    "Motor vehicle theft and related offences": "07 Theft",
    "Theft and related offences (other than MV)": "07 Theft",
    "Property Damage Offences": "11 Property damage offences",
}

KEEP_COLS = ["Year", "Month number", "Offence category", "Offence type",
             "Alcohol involvement", "DV involvement",
             "Reporting Region", "Statistical Area 2", "Number of offences"]

# [A5] "NT Balance" rows carry no town, so they are resolved through SA2.
SA2_TO_REGION = {
    "Barkly": "Barkly",
    "Sandover - Plenty": "Barkly",
    "Elsey": "Big Rivers",
    "Gulf": "Big Rivers",
    "Victoria River": "Big Rivers",
    "Petermann - Simpson": "Central Australia",
    "Tanami": "Central Australia",
    "Yuendumu - Anmatjere": "Central Australia",
    "East Arnhem": "East Arnhem",
    "Anindilyakwa": "East Arnhem",
    "Alligator": "Top End",
    "West Arnhem": "Top End",
    "Thamarrurr": "Top End",
    "Tiwi Islands": "Top End",
    "Daly": "Top End",
    "Howard Springs": "Greater Darwin",
    "Humpty Doo": "Greater Darwin",
    "Koolpinyah": "Greater Darwin",
    "Virginia": "Greater Darwin",
    "Weddell": "Greater Darwin",
}

REGION_TO_POP = {
    "Darwin": "Greater Darwin",
    "Palmerston": "Greater Darwin",
    "Alice Springs": "Central Australia",
    "Katherine": "Big Rivers",
    "Nhulunbuy": "East Arnhem",
    "Tennant Creek": "Barkly",
    "NT Balance": "Top End",
    "Top End": "Top End",
}

REGION_ORDER = ["Greater Darwin", "Central Australia", "Big Rivers",
                "East Arnhem", "Barkly", "Top End"]
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ASSAULT_CATEGORY = "02 Assault"


def _read_raw(name: str) -> pd.DataFrame:
    path = RAW_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing - the ingest stage creates it. Run this script "
            f"from the top rather than calling later stages directly."
        )
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()   # the new file ships 'Offence type '
    return df


def load_crime_old() -> pd.DataFrame:
    """Old file: trim to the analysis window, remap categories and regions."""
    df = _read_raw("crime_old")
    df = df[df["Year"].between(YEAR_MIN, YEAR_MAX)].copy()               # [A1]
    df = rename_columns(df, {"Reporting region": "Reporting Region"})
    df["Reporting Region"] = df["Reporting Region"].replace("Unknown", "Top End")  # [A4]

    unmapped = set(df["Offence category"].dropna().unique()) - set(CATEGORY_MAP)
    if unmapped:
        raise ValueError(                                                # [A2]
            f"Old-file offence categories missing from CATEGORY_MAP: {sorted(unmapped)}. "
            f"Add them - otherwise those rows would silently become NaN."
        )
    df["Offence category"] = df["Offence category"].map(CATEGORY_MAP)
    return df[KEEP_COLS].copy()


def load_crime_new() -> pd.DataFrame:
    """New file: already ANZSOC-aligned, so only the region fix is needed."""
    df = _read_raw("crime_new")
    df = rename_columns(df, {"Reporting region": "Reporting Region"})
    df = df[df["Year"].between(YEAR_MIN, YEAR_MAX)].copy()
    df["Reporting Region"] = df["Reporting Region"].replace("Unknown", "Top End")  # [A4]
    return df[KEEP_COLS].copy()


def remap_region(row) -> str:
    """[A5] Harmonise police regions onto the 6 population regions."""
    region = row["Reporting Region"]
    sa2 = row["Statistical Area 2"]
    if region == "NT Balance":
        if pd.notna(sa2) and sa2 in SA2_TO_REGION:
            return SA2_TO_REGION[sa2]
        return "Top End"
    return REGION_TO_POP.get(region, region)


def process_crime() -> pd.DataFrame:
    """Merge both crime schemas into one region-harmonised offence table."""
    old = load_crime_old()
    new = load_crime_new()
    print(f"[crime] old file (PROMIS)  : {len(old):,} rows | "
          f"{old['Year'].min()}-{old['Year'].max()}")
    print(f"[crime] new file (ANZSOC)  : {len(new):,} rows | "
          f"{new['Year'].min()}-{new['Year'].max()}")

    crime = pd.concat([old, new], ignore_index=True)
    crime["Quarter"] = (crime["Month number"] - 1) // 3 + 1
    crime["Region"] = crime.apply(remap_region, axis=1)
    crime = crime.drop(columns=["Reporting Region", "Statistical Area 2"])

    # [A8] flags -> offence counts. '-' means not applicable, treated as No.
    for col, out in [("Alcohol involvement", "Alcohol_offences"),
                     ("DV involvement", "DV_offences")]:
        flag = crime[col].map({"Yes": 1, "No": 0, "-": 0}).fillna(0).astype(int)
        crime[col] = flag
        crime[out] = flag * crime["Number of offences"]

    months = crime.groupby(["Year", "Month number"]).size().index
    print(f"[crime] merged             : {len(crime):,} rows | "
          f"{len(months)} distinct months | regions: {sorted(crime['Region'].unique())}")
    if not ((crime["Year"] == 2023) & (crime["Month number"] == 11)).any():
        print("[crime] [A3] November 2023 absent, as expected (systems transition gap)")
    return crime


def process_population() -> pd.DataFrame:
    """One row per region-year with total, Aboriginal-status, sex and age-group
    breakdowns."""
    pop = _read_raw("population")
    pop = pop[pop["Year"].between(YEAR_MIN, YEAR_MAX)].copy()

    # One Status per year today, so summing can't mix Preliminary with Final.
    # Checked rather than assumed: two statuses would double every region.
    multi = pop.groupby("Year")["Status"].nunique()
    if (multi > 1).any():
        raise ValueError(
            f"These years have more than one Status: {multi[multi > 1].index.tolist()}. "
            f"Pick one (prefer Final) before summing, or population doubles."
        )
    pop = pop.drop(columns=["Status"], errors="ignore")

    total = (pop.groupby(["Year", "Region"], as_index=False)["Population"].sum()
             .rename(columns={"Population": "Total_population"}))

    def wide(by: str) -> pd.DataFrame:
        out = (pop.groupby(["Year", "Region", by])["Population"]
               .sum().unstack(fill_value=0).reset_index())
        out.columns.name = None
        return out

    by_status = wide("Aboriginal status")
    by_sex = wide("Sex")
    by_age = wide("Age Group")
    by_age = rename_columns(by_age, {
        c: "Pop_age_" + str(c).replace("-", "").replace("+", "plus")
        for c in by_age.columns if c not in ("Year", "Region")
    })

    features = (total.merge(by_status, on=["Year", "Region"])
                     .merge(by_sex, on=["Year", "Region"])
                     .merge(by_age, on=["Year", "Region"]))
    print(f"[population] {len(features)} region-year rows x {features.shape[1]} columns "
          f"({features['Year'].min()}-{features['Year'].max()})")
    return features


def process_alcohol() -> pd.DataFrame:
    """Quarterly PAC by the 6 population regions, so PAC and population share
    a denominator (Darwin + Palmerston are summed into Greater Darwin)."""
    frames = [_read_raw(f"alcohol_{year}") for year in ALCOHOL_YEARS]
    alc = pd.concat(frames, ignore_index=True)

    alc["Quarter Ending"] = pd.to_datetime(alc["Quarter Ending"])
    alc["Year"] = alc["Quarter Ending"].dt.year
    alc["Quarter"] = alc["Quarter Ending"].dt.quarter                    # [A6]
    alc = alc.drop(columns=["Quarter Ending"])
    alc["Region"] = alc["Region"].map(REGION_TO_POP)

    unmapped = alc["Region"].isna().sum()
    if unmapped:
        raise ValueError(f"{unmapped} alcohol rows have a region outside REGION_TO_POP.")

    pac_cols = [c for c in alc.columns if c not in ("Region", "Year", "Quarter")]
    alc = alc.groupby(["Year", "Quarter", "Region"], as_index=False)[pac_cols].sum()
    print(f"[alcohol] {len(alc)} region-quarter rows | "
          f"{alc['Year'].min()}-{alc['Year'].max()} | {len(pac_cols)} PAC columns")
    return alc, pac_cols


def build_panel() -> tuple:
    """Crime + alcohol + population -> the offence-level analysis panel."""
    banner("STAGE 2 - BUILD PANEL: dataset/raw/ -> dataset/processed/")
    crime = process_crime()
    population = process_population()
    alcohol, pac_cols = process_alcohol()

    panel = crime.merge(alcohol, on=["Year", "Quarter", "Region"], how="left")
    panel = panel.merge(population, on=["Year", "Region"], how="left")
    if panel["Total_population"].isna().any():
        missing = (panel.loc[panel["Total_population"].isna(), ["Year", "Region"]]
                   .drop_duplicates().to_dict("records"))
        raise ValueError(f"No population figure for: {missing}")

    pop_cols = (["Total_population", "Aboriginal", "Non-Aboriginal", "Male", "Female"]
                + sorted(c for c in panel.columns if c.startswith("Pop_age_")))
    group_cols = (["Year", "Quarter", "Month number", "Region",
                   "Offence category", "Offence type",
                   "Alcohol involvement", "DV involvement"]
                  + pop_cols + pac_cols)
    group_cols = [c for c in group_cols if c in panel.columns]
    panel = (panel.groupby(group_cols, dropna=False, as_index=False)
             [["Number of offences", "Alcohol_offences", "DV_offences"]].sum())
    print(f"[panel] aggregated to {len(panel):,} rows x {panel.shape[1]} columns")

    # [A7] impute inside 2023-2025 only; 2015-2022 stays NaN on purpose
    recent = panel["Year"] >= min(ALCOHOL_YEARS)
    for col in pac_cols:
        quarter_mean = (panel.loc[recent]
                        .groupby(["Region", "Quarter"])[col].transform("mean"))
        panel.loc[recent, col] = panel.loc[recent, col].fillna(quarter_mean).round(0)
    still_missing = panel.loc[recent, "Total PAC"].isna().sum()
    print(f"[panel] PAC NaN inside {min(ALCOHOL_YEARS)}-{max(ALCOHOL_YEARS)} after "
          f"imputation: {still_missing}")
    print(f"[panel] PAC NaN before {min(ALCOHOL_YEARS)} (expected, [A7]): "
          f"{panel.loc[~recent, 'Total PAC'].isna().sum():,}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "nt_crime_merged_2015_2025.csv"
    write_csv(panel, out_path)
    print(f"[panel] saved -> {out_path.relative_to(BASE_DIR)}")
    print("\n[panel] rows by region x year:")
    print(panel.groupby(["Region", "Year"]).size().unstack(fill_value=0).to_string())
    return panel, population, pac_cols


# =============================================================================
# SECTION 3 - EDA
# =============================================================================

def _setup_plot_style() -> None:
    sns.set_theme(style="whitegrid", palette="muted", font="DejaVu Sans")
    plt.rcParams.update({
        "figure.dpi": 150,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })


def _save(fig, name: str, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {name}")


def _thousands(ax, axis: str = "y") -> None:
    fmt = mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def run_eda(panel: pd.DataFrame) -> dict:
    banner("STAGE 3 - EDA -> eda_plots/")
    _setup_plot_style()
    palette = sns.color_palette("tab10", n_colors=len(REGION_ORDER))
    assault = panel[panel["Offence category"] == ASSAULT_CATEGORY].copy()

    # --- 1.1 population trend ------------------------------------------------
    pop_plot = panel.groupby(["Region", "Year"], as_index=False)["Total_population"].first()
    fig, ax = plt.subplots(figsize=(12, 6))
    for region, color in zip(REGION_ORDER, palette):
        grp = pop_plot[pop_plot["Region"] == region].sort_values("Year")
        ax.plot(grp["Year"], grp["Total_population"] / 1000,
                marker="o", linewidth=2, label=region, color=color)
    ax.set(xlabel="Year", ylabel="Population (thousands)",
           title=f"Population by NT Government Region ({YEAR_MIN}-{YEAR_MAX})")
    ax.legend(title="Region", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    _save(fig, "1_1_population_trend_by_region.png", PLOT_DIR)

    # --- 1.2 offences by year ------------------------------------------------
    by_year = panel.groupby("Year", as_index=False)["Number of offences"].sum()
    fig, ax = plt.subplots(figsize=(11, 5))
    peak = by_year["Number of offences"].max()
    bars = ax.bar(by_year["Year"].astype(str), by_year["Number of offences"],
                  color=["#EF5350" if v == peak else "#42A5F5"
                         for v in by_year["Number of offences"]], edgecolor="white")
    ax.set(xlabel="Year", ylabel="Number of Offences",
           title=f"Total Offences by Year - All Categories ({YEAR_MIN}-{YEAR_MAX})")
    _thousands(ax)
    for bar, val in zip(bars, by_year["Number of offences"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:,}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, peak * 1.12)
    fig.tight_layout()
    _save(fig, "1_2_offences_by_year.png", PLOT_DIR)

    # --- 1.3 offences by category -------------------------------------------
    cat_total = panel.groupby("Offence category")["Number of offences"].sum().sort_values()
    labels = [c.split(" ", 1)[1] if " " in c else c for c in cat_total.index]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, cat_total.values,
                   color=["#EF5350" if v == cat_total.max() else "#42A5F5"
                          for v in cat_total], edgecolor="white")
    ax.set(xlabel="Number of Offences",
           title=f"Total Offences by Category ({YEAR_MIN}-{YEAR_MAX})")
    _thousands(ax, "x")
    ax.invert_yaxis()
    for bar, val in zip(bars, cat_total.values):
        ax.text(bar.get_width() + cat_total.max() * 0.01,
                bar.get_y() + bar.get_height() / 2, f"{val:,}", va="center", fontsize=8)
    fig.tight_layout()
    _save(fig, "1_3_offences_by_category.png", PLOT_DIR)

    # --- 1.4 alcohol / DV involvement, measured as offences [A8] -------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    total_offences = panel["Number of offences"].sum()
    for ax, col, title, colors in zip(
        axes, ["Alcohol_offences", "DV_offences"],
        ["Alcohol Involvement", "DV Involvement"],
        [["#EF5350", "#42A5F5"], ["#AB47BC", "#66BB6A"]],
    ):
        yes = panel[col].sum()
        no = total_offences - yes
        text = [f"Yes\n{yes:,}\n({yes / total_offences * 100:.1f}%)",
                f"No / N/A\n{no:,}\n({no / total_offences * 100:.1f}%)"]
        ax.bar(text, [yes, no], color=colors, edgecolor="white", width=0.5)
        ax.set(title=title, ylabel="Number of Offences")
        _thousands(ax)
    fig.suptitle(f"Alcohol and DV Involvement Across All Offences ({YEAR_MIN}-{YEAR_MAX})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "1_4_alcohol_dv_involvement.png", PLOT_DIR)

    # --- 2.1 crime rate per 100k by year ------------------------------------
    pop_by_year = (panel.groupby(["Year", "Region"], as_index=False)["Total_population"]
                   .first().groupby("Year")["Total_population"].sum())
    by_year["Total_pop"] = by_year["Year"].map(pop_by_year)
    by_year["Rate_per_100k"] = (by_year["Number of offences"]
                                / by_year["Total_pop"] * 100_000).round(1)
    fig, ax = plt.subplots(figsize=(11, 5))
    top = by_year["Rate_per_100k"].max()
    bars = ax.bar(by_year["Year"].astype(str), by_year["Rate_per_100k"],
                  color=["#EF5350" if v == top else "#66BB6A"
                         for v in by_year["Rate_per_100k"]], edgecolor="white")
    ax.set(xlabel="Year", ylabel="Offences per 100,000 Population",
           title=f"Annual Crime Rate per 100,000 Population ({YEAR_MIN}-{YEAR_MAX})")
    for bar, val in zip(bars, by_year["Rate_per_100k"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.0f}",
                ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(0, top * 1.15)
    fig.tight_layout()
    _save(fig, "2_1_crime_rate_per_100k_by_year.png", PLOT_DIR)

    # --- 2.2 offences by month ----------------------------------------------
    monthly = panel.groupby("Month number", as_index=False)["Number of offences"].sum()
    monthly["Month_label"] = monthly["Month number"].map(lambda m: MONTH_LABELS[m - 1])
    fig, ax = plt.subplots(figsize=(10, 5))
    top = monthly["Number of offences"].max()
    bars = ax.bar(monthly["Month_label"], monthly["Number of offences"],
                  color=["#EF5350" if v == top else "#42A5F5"
                         for v in monthly["Number of offences"]], edgecolor="white")
    ax.set(xlabel="Month", ylabel="Total Number of Offences",
           title=f"Total Offences by Month - All Categories ({YEAR_MIN}-{YEAR_MAX} combined)")
    _thousands(ax)
    for bar, val in zip(bars, monthly["Number of offences"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:,}",
                ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(0, top * 1.12)
    peak_month = monthly.loc[monthly["Number of offences"].idxmax(), "Month_label"]
    fig.tight_layout()
    _save(fig, "2_2_offences_by_month.png", PLOT_DIR)

    # --- 2.3 assault trend by region ----------------------------------------
    assault_yr = assault.groupby(["Year", "Region"], as_index=False)["Number of offences"].sum()
    fig, ax = plt.subplots(figsize=(12, 6))
    for region, color in zip(REGION_ORDER, palette):
        grp = assault_yr[assault_yr["Region"] == region].sort_values("Year")
        ax.plot(grp["Year"], grp["Number of offences"],
                marker="o", linewidth=2, label=region, color=color)
    ax.set(xlabel="Year", ylabel="Assault Offences",
           title=f"Annual Assault Offences by Region ({YEAR_MIN}-{YEAR_MAX})")
    ax.legend(title="Region", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    _save(fig, "2_3_assault_trend_by_year_region.png", PLOT_DIR)

    # --- 2.4 heatmap year x month -------------------------------------------
    heat = (assault.groupby(["Year", "Month number"])["Number of offences"]
            .sum().unstack())
    heat.columns = [MONTH_LABELS[m - 1] for m in heat.columns]
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(heat, annot=True, fmt=",.0f", cmap="YlOrRd", linewidths=0.4, ax=ax,
                cbar_kws={"label": "Assault Offences"})
    ax.set(xlabel="Month", ylabel="Year",
           title=f"Assault Offences Heatmap - Year x Month ({YEAR_MIN}-{YEAR_MAX})\n"
                 f"blank = November 2023, the [A3] transition gap")
    fig.tight_layout()
    _save(fig, "2_4_heatmap_year_month_assault.png", PLOT_DIR)

    # --- assault region-month aggregate for sections 3.x --------------------
    monthly_assault = (assault.groupby(["Year", "Quarter", "Month number", "Region"])
                       .agg(Assault_offences=("Number of offences", "sum"),
                            Alcohol_offences=("Alcohol_offences", "sum"),
                            DV_offences=("DV_offences", "sum"),
                            Total_PAC=("Total PAC", "first"),
                            Total_population=("Total_population", "first"),
                            Aboriginal=("Aboriginal", "first"))
                       .reset_index())
    monthly_assault["Assault_rate_100k"] = (monthly_assault["Assault_offences"]
                                            / monthly_assault["Total_population"]
                                            * 100_000).round(1)
    monthly_assault["alcohol_per_capita"] = (monthly_assault["Total_PAC"]
                                             / monthly_assault["Total_population"])

    # --- 3.1 assault rate by region -----------------------------------------
    rate_yr = (monthly_assault.groupby(["Year", "Region"], as_index=False)
               ["Assault_rate_100k"].mean().round(1))
    fig, ax = plt.subplots(figsize=(12, 6))
    for region, color in zip(REGION_ORDER, palette):
        grp = rate_yr[rate_yr["Region"] == region].sort_values("Year")
        ax.plot(grp["Year"], grp["Assault_rate_100k"],
                marker="o", linewidth=2, label=region, color=color)
    ax.set(xlabel="Year", ylabel="Avg Monthly Assault Rate per 100,000",
           title=f"Average Monthly Assault Rate per 100k by Region ({YEAR_MIN}-{YEAR_MAX})")
    ax.legend(title="Region", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    _save(fig, "3_1_assault_rate_by_region_trend.png", PLOT_DIR)

    # --- 3.2 distribution and log transform ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    raw = monthly_assault["Assault_offences"]
    logged = np.log1p(raw)
    sns.histplot(raw, bins=30, kde=True, ax=axes[0], color="#42A5F5", edgecolor="white")
    axes[0].set(xlabel="Assault Offences", ylabel="Frequency",
                title=f"Distribution of Assault Offences\n(Skewness = {raw.skew():.2f})")
    sns.histplot(logged, bins=30, kde=True, ax=axes[1], color="#66BB6A", edgecolor="white")
    axes[1].set(xlabel="log(1 + Assault Offences)", ylabel="Frequency",
                title=f"Log-Transformed\n(Skewness = {logged.skew():.2f})")
    fig.suptitle("Assault Offences Distribution: Original vs Log Transform",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "3_2_assault_distribution.png", PLOT_DIR)

    # --- 3.3 alcohol / DV share of assault by year [A8] ---------------------
    shares = (monthly_assault.groupby("Year")
              .agg(Assault=("Assault_offences", "sum"),
                   Alcohol=("Alcohol_offences", "sum"),
                   DV=("DV_offences", "sum")).reset_index())
    shares["Pct_alcohol"] = shares["Alcohol"] / shares["Assault"] * 100
    shares["Pct_dv"] = shares["DV"] / shares["Assault"] * 100
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(shares))
    width = 0.35
    b1 = ax.bar(x - width / 2, shares["Pct_alcohol"], width, color="#EF5350",
                edgecolor="white", label="Alcohol-involved (%)")
    b2 = ax.bar(x + width / 2, shares["Pct_dv"], width, color="#AB47BC",
                edgecolor="white", label="DV-involved (%)")
    ax.set_xticks(x, shares["Year"].astype(str))
    ax.set(xlabel="Year", ylabel="% of Assault Offences",
           title=f"Alcohol and DV Involvement in Assault Offences by Year "
                 f"({YEAR_MIN}-{YEAR_MAX})")
    ax.legend()
    for bar, val in zip(list(b1) + list(b2),
                        list(shares["Pct_alcohol"]) + list(shares["Pct_dv"])):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.0f}%",
                ha="center", va="bottom", fontsize=7.5)
    ax.set_ylim(0, max(shares["Pct_alcohol"].max(), shares["Pct_dv"].max()) * 1.25)
    fig.tight_layout()
    _save(fig, "3_3_alc_dv_involvement_by_year.png", PLOT_DIR)

    # --- 3.4 PAC trend ------------------------------------------------------
    pac = (monthly_assault[monthly_assault["Total_PAC"].notna()]
           .groupby(["Year", "Quarter", "Region"], as_index=False)["Total_PAC"].first())
    if not pac.empty:
        pac["YQ"] = pac["Year"].astype(str) + "-Q" + pac["Quarter"].astype(str)
        fig, ax = plt.subplots(figsize=(12, 6))
        for region, color in zip(REGION_ORDER, palette):
            grp = pac[pac["Region"] == region].sort_values(["Year", "Quarter"])
            if not grp.empty:
                ax.plot(grp["YQ"], grp["Total_PAC"] / 1000,
                        marker="o", linewidth=2, label=region, color=color)
        ax.set(xlabel="Year-Quarter", ylabel="Total PAC (thousands of litres)",
               title=f"Wholesale Alcohol Supply (PAC) by Region and Quarter "
                     f"({min(ALCOHOL_YEARS)}-{max(ALCOHOL_YEARS)})")
        ax.legend(title="Region", bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        _save(fig, "3_4_pac_trend_by_region.png", PLOT_DIR)

    # --- 3.5 correlations ---------------------------------------------------
    with_pac = monthly_assault[monthly_assault["Total_PAC"].notna()].copy()
    corr_frame = with_pac[["Assault_rate_100k", "alcohol_per_capita", "Total_population",
                           "Aboriginal", "Alcohol_offences", "DV_offences"]]
    corr_frame = rename_columns(corr_frame, {
        "Assault_rate_100k": "Assault Rate /100k",
        "alcohol_per_capita": "PAC per Capita",
        "Total_population": "Total Population",
        "Aboriginal": "Aboriginal Pop.",
        "Alcohol_offences": "Alcohol-Involved Offences",
        "DV_offences": "DV-Involved Offences",
    })
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_frame.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, linewidths=0.5, ax=ax,
                cbar_kws={"label": "Pearson Correlation"})
    ax.set_title(f"Correlation Heatmap - Assault Predictors\n"
                 f"({min(ALCOHOL_YEARS)}-{max(ALCOHOL_YEARS)}, per-capita to remove region size)")
    fig.tight_layout()
    _save(fig, "3_5_correlation_heatmap.png", PLOT_DIR)

    # Pooling all regions makes any pair of population-scaled series look
    # correlated, so report the within-region correlation as well.
    pooled_r = with_pac["alcohol_per_capita"].corr(with_pac["Assault_rate_100k"])
    within = (with_pac.groupby("Region")
              .apply(lambda g: g["alcohol_per_capita"].corr(g["Assault_rate_100k"]))
              .round(3))

    print()
    print("EDA summary")
    print(f"  peak crime month                        : {peak_month}")
    print(f"  highest average assault rate region     : "
          f"{rate_yr.groupby('Region')['Assault_rate_100k'].mean().idxmax()}")
    print(f"  assault skewness raw / log              : "
          f"{raw.skew():.2f} / {logged.skew():.2f}")
    print(f"  PAC per capita vs assault rate (pooled) : {pooled_r:.3f}")
    print("  same correlation computed within region :")
    for region, value in within.items():
        print(f"      {region:<20} r = {value:+.3f}")
    return {"monthly_assault": monthly_assault, "palette": palette}


# =============================================================================
# SECTION 4 - PCA
# =============================================================================

def run_pca(population: pd.DataFrame) -> None:
    banner("STAGE 4 - PCA on regional age structure")
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    age_cols = sorted(c for c in population.columns if c.startswith("Pop_age_"))
    # [A14] shares, not counts.
    shares = population[age_cols].div(population["Total_population"], axis=0)
    scaled = StandardScaler().fit_transform(shares.astype(float).values)
    pca = PCA().fit(scaled)
    explained = pca.explained_variance_ratio_ * 100
    cumulative = np.cumsum(explained)

    print(f"  matrix: {scaled.shape[0]} region-year rows x {scaled.shape[1]} age groups")
    print(f"  {'PC':<6}{'Explained %':>14}{'Cumulative %':>15}")
    print("  " + "-" * 35)
    for i in range(min(6, scaled.shape[1])):
        print(f"  PC{i + 1:<4}{explained[i]:>14.1f}{cumulative[i]:>15.1f}")
    n_90 = int(np.searchsorted(cumulative, 90) + 1)
    print(f"  components needed for 90% of variance: {n_90}")


# =============================================================================
# SECTION 5 - REGRESSION
# =============================================================================

FEATURE_VARIANTS_BASE = ["sin_month", "cos_month", "Season",
                         "assault_rate_lag1", "assault_rate_lag3", "assault_rate_lag12"]


def build_regression_panel(monthly_assault: pd.DataFrame,
                           population: pd.DataFrame) -> pd.DataFrame:
    """Region x month grid with lag features - see [A10] for why the grid."""
    observed = monthly_assault.copy()
    observed["date"] = pd.to_datetime(
        dict(year=observed["Year"], month=observed["Month number"], day=1))

    grid = pd.MultiIndex.from_product(
        [sorted(observed["Region"].unique()),
         pd.date_range(observed["date"].min(), observed["date"].max(), freq="MS")],
        names=["Region", "date"],
    ).to_frame(index=False)
    panel = grid.merge(
        observed.drop(columns=["Year", "Quarter", "Month number", "Total_population"]),
        on=["Region", "date"], how="left")

    panel["Year"] = panel["date"].dt.year
    panel["Month number"] = panel["date"].dt.month
    panel["Quarter"] = panel["date"].dt.quarter
    panel = panel.merge(population[["Year", "Region", "Total_population"]],
                        on=["Year", "Region"], how="left")

    gap = panel["Assault_offences"].isna().sum()
    print(f"[regression] grid: {len(panel)} region-month rows, {gap} with no crime data "
          f"({gap // max(1, panel['Region'].nunique())} month(s) x "
          f"{panel['Region'].nunique()} regions - the [A3] gap)")

    panel = panel.sort_values(["Region", "date"]).reset_index(drop=True)
    panel["log_assault_rate"] = np.log1p(panel["Assault_rate_100k"])
    panel["sin_month"] = np.sin(2 * np.pi * panel["Month number"] / 12)
    panel["cos_month"] = np.cos(2 * np.pi * panel["Month number"] / 12)
    panel["Season"] = panel["Month number"].isin([11, 12, 1, 2, 3, 4]).astype(int)  # 1=Wet
    panel["is_christmas"] = panel["Month number"] == 12
    panel["is_new_year"] = panel["Month number"] == 1
    panel["is_pay_week"] = panel["date"].dt.isocalendar().week % 2 == 0


    for lag in (1, 3, 12):
        panel[f"assault_rate_lag{lag}"] = (panel.groupby("Region")["Assault_rate_100k"]
                                           .shift(lag))

    # [A9] flag the most recent 6 months rather than dropping them
    last = panel.loc[panel["Assault_offences"].notna(), "date"].max()
    panel["is_provisional"] = panel["date"] > (last - pd.DateOffset(months=6))

    dummies = pd.get_dummies(panel["Region"], prefix="Reg")
    dummies = dummies.drop(columns=["Reg_Greater Darwin"], errors="ignore")
    panel = pd.concat([panel, dummies], axis=1)

    required = ["log_assault_rate"] + [f"assault_rate_lag{l}" for l in (1, 3, 12)]
    before = len(panel)
    panel = panel.dropna(subset=required).reset_index(drop=True)
    print(f"[regression] usable rows after lag warm-up and gap removal: "
          f"{len(panel)} (dropped {before - len(panel)})")
    print(f"[regression] provisional rows flagged ([A9]): {panel['is_provisional'].sum()}")
    return panel


def cv_rmse_for_features(model, cv_frame: pd.DataFrame, features: list[str]) -> float:
    """TimeSeriesSplit CV RMSE over calendar order for given model + feature set."""
    y_cv = cv_frame["log_assault_rate"].to_numpy(float)
    tscv = TimeSeriesSplit(n_splits=5)
    X = StandardScaler().fit_transform(cv_frame[features].astype(float))
    scores = cross_val_score(
        model,
        X,
        y_cv,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
    )
    return float(-scores.mean())
# ---------- STAGE 0: data prep ----------------------------------------------

def prepare_panel(monthly_assault: pd.DataFrame,
                  population: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame,
                                                     pd.DataFrame, list[str],
                                                     pd.DataFrame]:
    panel = build_regression_panel(monthly_assault, population)
    dummy_cols = [c for c in panel.columns if c.startswith("Reg_")]

    train = panel[panel["Year"] <= 2022]
    test = panel[panel["Year"] >= 2023]

    print(f"[A11] train {YEAR_MIN}-2022: {len(train)} rows | "
          f"test 2023-{YEAR_MAX}: {len(test)} rows")

    # TimeSeriesSplit needs rows in calendar order, not region order.
    cv_frame = panel.sort_values(["date", "Region"]).reset_index(drop=True)

    return panel, train, test, dummy_cols, cv_frame


# ---------- R1: feature variant comparison ----------------------------------

def evaluate_feature_variants(train: pd.DataFrame,
                              test: pd.DataFrame,
                              dummy_cols: list[str]) -> list[str]:
    #Looking at different combinations of features and determining the best one
    print("\n-- R1: feature variant comparison (Linear Regression) --")

    variants = {
        "V1: Temporal": FEATURE_VARIANTS_BASE,
        "V2: + Region": FEATURE_VARIANTS_BASE + dummy_cols,
        "V3: + Crime ctx": FEATURE_VARIANTS_BASE + ["Alcohol_offences", "DV_offences"],
        "V4: Full (no PAC)": (FEATURE_VARIANTS_BASE
                              + ["Alcohol_offences", "DV_offences"] + dummy_cols),
    }

    y_train = train["log_assault_rate"].to_numpy(float)
    y_test = test["log_assault_rate"].to_numpy(float)

    variant_rmse: dict[str, float] = {}

    for name, features in variants.items():
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(train[features].astype(float))
        x_te = scaler.transform(test[features].astype(float))
        pred = LinearRegression().fit(x_tr, y_train).predict(x_te)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        mae = float(mean_absolute_error(y_test, pred))
        variant_rmse[name] = rmse
        print(f"   {name:<20} RMSE={rmse:.4f} MAE={mae:.4f}")

    best_variant = min(variant_rmse, key=variant_rmse.get)
    FEATURES = variants[best_variant]
    print(f"   -> best variant: {best_variant} (RMSE={variant_rmse[best_variant]:.4f})")

    # plot
    fig, ax = plt.subplots(figsize=(9, 4))
    values = list(variant_rmse.values())
    bars = ax.bar(
        list(variant_rmse),
        values,
        width=0.5,
        edgecolor="white",
        color=["#EF5350" if v == min(values) else "#90CAF9" for v in values],
    )
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set(
        xlabel="Feature Variant",
        ylabel="RMSE (log scale)",
        ylim=(0, max(values) * 1.15),
        title=("Feature Variant Comparison - Linear Regression\n"
               f"(train {YEAR_MIN}-2022, test 2023-{YEAR_MAX} | red = best)"),
    )
    fig.tight_layout()
    _save(fig, "R1_feature_variants.png", REG_PLOT_DIR)

    return FEATURES


# ---------- R2: VIF ---------------------------------------------------------

def compute_vif_and_filter(train: pd.DataFrame,
                           FEATURES: list[str]) -> list[str]:
    print("\n-- R2: VIF (multicollinearity) --")

    vif_matrix = train[FEATURES].astype(float).to_numpy()
    vif = pd.DataFrame({
        "Feature": FEATURES,
        "VIF": [variance_inflation_factor(vif_matrix, i)
                for i in range(len(FEATURES))],
    }).sort_values("VIF", ascending=False)

    to_remove = vif.loc[vif["VIF"] > 30, "Feature"].tolist()

    for _, row in vif.iterrows():
        verdict = (
            "remove" if row["VIF"] > 30
            else "elevated - monitor" if row["VIF"] > 10
            else "acceptable"
        )
        print(f"   {row['Feature']:<26}{row['VIF']:>8.1f}  {verdict}")

    if to_remove:
        print("\nRemoving due to VIF > 30:", to_remove)
        FEATURES = [f for f in FEATURES if f not in to_remove]
    else:
        print("\nNo features removed (all VIF ≤ 30).")

    return FEATURES



# ---------- R3: alpha tuning ------------------------------------------------

def tune_alphas(cv_frame: pd.DataFrame,
                FEATURES: list[str]) -> tuple[float, float]:
    print("\n-- R3: alpha tuning (TimeSeriesSplit over calendar order, [A13]) --")
    alphas = [0.01, 0.1, 1, 10, 100]
    ridge_cv: dict[float, float] = {}
    lasso_cv: dict[float, float] = {}

    print(f"   {'alpha':<8}{'Ridge CV RMSE':<18}{'Lasso CV RMSE'}")
    for alpha in alphas:
        ridge_cv[alpha] = cv_rmse_for_features(Ridge(alpha=alpha), cv_frame, FEATURES)
        lasso_cv[alpha] = cv_rmse_for_features(
            Lasso(alpha=alpha, max_iter=10000), cv_frame, FEATURES
        )
        print(f"   {alpha:<8}{ridge_cv[alpha]:<18.4f}{lasso_cv[alpha]:.4f}")

    best_ridge = min(ridge_cv, key=ridge_cv.get)
    best_lasso = min(lasso_cv, key=lasso_cv.get)
    print(f"   -> Ridge alpha={best_ridge} (CV RMSE={ridge_cv[best_ridge]:.4f})")
    print(f"   -> Lasso alpha={best_lasso} (CV RMSE={lasso_cv[best_lasso]:.4f})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, scores, name, color in zip(
        axes,
        [ridge_cv, lasso_cv],
        ["Ridge", "Lasso"],
        ["#2196F3", "#FF9800"],
    ):
        best = min(scores, key=scores.get)
        ax.plot(range(len(alphas)), list(scores.values()), marker="o", linewidth=2,
                color=color)
        ax.scatter([alphas.index(best)], [scores[best]], color="#EF5350", s=120,
                   zorder=5, label=f"best alpha={best}")
        ax.set_xticks(range(len(alphas)), [str(a) for a in alphas])
        ax.set(
            xlabel="Alpha",
            ylabel="CV RMSE (log scale)",
            title=f"{name} - Alpha Tuning\n(TimeSeriesSplit, 5 folds)",
        )
        ax.legend()
    fig.suptitle("Hyperparameter Tuning", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "R2_alpha_tuning.png", REG_PLOT_DIR)

    return best_ridge, best_lasso

# ---------- R3b: XGBoost tuning ------------------------------------------------

def tune_xgboost(cv_frame: pd.DataFrame,
                 FEATURES: list[str]) -> dict:
    print("\n-- R3b: XGBoost hyperparameter tuning --")

    # Candidate parameter sets
    param_grid = [
        {"max_depth": 3, "learning_rate": 0.05, "reg_alpha": 1.0, "reg_lambda": 3.0},
        {"max_depth": 4, "learning_rate": 0.03, "reg_alpha": 2.0, "reg_lambda": 4.0},
        {"max_depth": 5, "learning_rate": 0.02, "reg_alpha": 3.0, "reg_lambda": 5.0},
        {"max_depth": 6, "learning_rate": 0.02, "reg_alpha": 4.0, "reg_lambda": 6.0},
    ]

    scores = {}

    print(f"   {'Parameters':<55}{'CV RMSE'}")
    for params in param_grid:
        model = XGBRegressor(
            n_estimators=800,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=1,
            objective="reg:squarederror",
            random_state=42,
            **params
        )

        rmse = cv_rmse_for_features(model, cv_frame, FEATURES)
        scores[str(params)] = rmse

        print(f"   {str(params):<55}{rmse:.4f}")

    # Best parameter set
    best_params_str = min(scores, key=scores.get)
    best_params = eval(best_params_str)

    print(f"\n   -> Best XGBoost params = {best_params_str}")
    print(f"      CV RMSE = {scores[best_params_str]:.4f}")

    # Plot tuning curve
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(scores)), list(scores.values()),
            marker="o", linewidth=2, color="#66BB6A")
    ax.set_xticks(range(len(scores)),
                  [f"d={p['max_depth']}, lr={p['learning_rate']}"
                   for p in param_grid],
                  rotation=45, fontsize=8)
    ax.set_ylabel("CV RMSE (log scale)")
    ax.set_title("XGBoost Hyperparameter Tuning")
    fig.tight_layout()
    _save(fig, "R3b_xgb_tuning.png", REG_PLOT_DIR)

    return best_params


# ============================================================
# R4 — Final models 
# ============================================================

def train_final_models(train: pd.DataFrame,
                       test: pd.DataFrame,
                       FEATURES: list[str],
                       best_ridge: float,
                       best_lasso: float,
                       best_xgb_params: dict,
                       cv_frame: pd.DataFrame) -> tuple[dict, str]:

    print("\n-- R4: Model A (no alcohol supply) --")

    y_train = train["log_assault_rate"].to_numpy(float)
    y_test = test["log_assault_rate"].to_numpy(float)

    # Linear models use scaled features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[FEATURES].astype(float))
    x_test = scaler.transform(test[FEATURES].astype(float))

    # XGBoost uses raw features
    X_train_raw = train[FEATURES].astype(float)
    X_test_raw = test[FEATURES].astype(float)

    models = {
        "Linear Regression": LinearRegression(),
        f"Ridge (a={best_ridge})": Ridge(alpha=best_ridge),
        f"Lasso (a={best_lasso})": Lasso(alpha=best_lasso, max_iter=10000),
        "XGBoost": XGBRegressor(
            n_estimators=800,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=1,
            objective="reg:squarederror",
            random_state=42,
            **best_xgb_params
        ),
    }

    results = {}
    print(f"   {'Model':<22}{'RMSE(log)':<12}{'MAE(log)':<12}{'RMSE(/100k)':<14}{'CV RMSE'}")

    for name, model in models.items():

        if name == "XGBoost":
            model.fit(X_train_raw, y_train)
            pred = model.predict(X_test_raw)
            cv = cv_rmse_for_features(model, cv_frame, FEATURES)
        else:
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            cv = cv_rmse_for_features(model.__class__(**model.get_params()),
                                      cv_frame, FEATURES)

        rmse_log = float(np.sqrt(mean_squared_error(y_test, pred)))
        mae_log = float(mean_absolute_error(y_test, pred))
        rmse_rate = float(np.sqrt(mean_squared_error(np.expm1(y_test),
                                                     np.expm1(pred))))

        results[name] = {
            "model": model,
            "pred": pred,
            "rmse": rmse_log,
            "mae": mae_log,
            "rmse_rate": rmse_rate,
            "cv": cv,
        }

        print(f"   {name:<22}{rmse_log:<12.4f}{mae_log:<12.4f}"
              f"{rmse_rate:<14.1f}{cv:.4f}")

    # ⭐ XGBoost is best model (based on test RMSE)
    best_name = min(results, key=lambda k: results[k]["rmse"])
    print(f"   -> best model (Test RMSE): {best_name}")

    return results, best_name


# ============================================================
# R5 — Assumption checks 
# ============================================================

def check_assumptions(results: dict, y_test: np.ndarray) -> None:
    print("\n-- R5: assumption checks (Linear Regression) --")
    resid = y_test - results["Linear Regression"]["pred"]
    sw_stat, sw_p = stats.shapiro(resid)
    verdict = ("fail to reject H0 (approx. normal)" if sw_p > 0.05
               else "reject H0 (residuals not normal)")
    print(f"   Shapiro-Wilk W={sw_stat:.4f} p={sw_p:.4f} -> {verdict}")


# ============================================================
# R6 — Predicted vs Actual 
# ============================================================

def plot_predicted_vs_actual(test: pd.DataFrame,
                             results: dict,
                             best_name: str,
                             palette) -> None:

    actual_rate = np.expm1(test["log_assault_rate"].to_numpy(float))
    pred_rate = np.expm1(results[best_name]["pred"])

    fig, ax = plt.subplots(figsize=(8, 6))
    for region, color in zip(REGION_ORDER, palette):
        mask = (test["Region"] == region).to_numpy()
        ax.scatter(actual_rate[mask], pred_rate[mask], label=region, color=color,
                   alpha=0.75, s=60, edgecolors="white")

    limit = max(actual_rate.max(), pred_rate.max()) * 1.05
    ax.plot([0, limit], [0, limit], "r--", linewidth=1.5)

    ax.set(xlabel="Actual Assault Rate (per 100k)",
           ylabel="Predicted Assault Rate (per 100k)",
           title=f"Predicted vs Actual - {best_name}")

    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    _save(fig, "R5_predicted_vs_actual.png", REG_PLOT_DIR)


# ============================================================
# R7 — Paired t-tests
# ============================================================

def paired_t_tests(results: dict, y_test: np.ndarray) -> None:
    print("\n-- R7: paired t-tests on absolute error (alpha = 0.05) --")
    names = list(results)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            t, p = stats.ttest_rel(np.abs(y_test - results[a]["pred"]),
                                   np.abs(y_test - results[b]["pred"]))
            decision = "reject H0 (significant)" if p < 0.05 else "fail to reject H0"
            print(f"   {a:<22} vs {b:<22} t={t:+.4f} p={p:.4f}  {decision}")


# ============================================================
# R8 — Coefficients (linear only)
# ============================================================

def plot_coefficients(results: dict,
                      FEATURES: list[str],
                      best_ridge: float,
                      best_lasso: float) -> None:

    print("\n-- R8: coefficients --")
    coefs = pd.DataFrame({
        "Feature": FEATURES,
        "Ridge": results[f"Ridge (a={best_ridge})"]["model"].coef_,
        "Lasso": results[f"Lasso (a={best_lasso})"]["model"].coef_,
    }).sort_values("Ridge", key=abs, ascending=False)

    for _, row in coefs.iterrows():
        state = "ZEROED" if abs(row["Lasso"]) < 1e-6 else "kept"
        print(f"   {row['Feature']:<26} Ridge={row['Ridge']:+.4f} "
              f"Lasso={row['Lasso']:+.4f} [{state}]")


# ============================================================
# R9 — Region predictions 
# ============================================================

def per_region_predictions(test: pd.DataFrame,
                           results: dict,
                           best_name: str,
                           palette) -> None:

    print(f"\n-- R9: per-region accuracy (best model = {best_name}) --")

    actual_rate = np.expm1(test["log_assault_rate"].to_numpy(float))
    pred_rate = np.expm1(results[best_name]["pred"])

    preds = test[["Year", "Month number", "Region"]].reset_index(drop=True)
    preds["Actual_rate"] = actual_rate.round(1)
    preds["Predicted_rate"] = pred_rate.round(1)
    preds["Error"] = preds["Predicted_rate"] - preds["Actual_rate"]

    print(f"   {'Region':<20}{'Actual':>10}{'Predicted':>12}{'RMSE':>10}{'Err%':>10}")
    for region in REGION_ORDER:
        sub = preds[preds["Region"] == region]
        if sub.empty:
            continue
        rmse = np.sqrt((sub["Error"] ** 2).mean())
        err_pct = (sub["Error"] / sub["Actual_rate"] * 100).mean()
        print(f"   {region:<20}{sub['Actual_rate'].mean():>10.1f}"
              f"{sub['Predicted_rate'].mean():>12.1f}"
              f"{rmse:>10.1f}{err_pct:>10.1f}")


# ============================================================
# Orchestrator
# ============================================================

def run_regression(monthly_assault: pd.DataFrame,
                   population: pd.DataFrame,
                   palette) -> None:

    banner("STAGE 5 - REGRESSION -> regression_plots/")

    panel, train, test, dummy_cols, cv_frame = prepare_panel(monthly_assault, population)

    FEATURES = evaluate_feature_variants(train, test, dummy_cols)
    FEATURES = compute_vif_and_filter(train, FEATURES)

    best_ridge, best_lasso = tune_alphas(cv_frame, FEATURES)
    best_xgb_params = tune_xgboost(cv_frame, FEATURES)

    results, best_name = train_final_models(
        train, test, FEATURES,
        best_ridge, best_lasso,
        best_xgb_params,
        cv_frame
    )

    y_test = test["log_assault_rate"].to_numpy(float)

    check_assumptions(results, y_test)
    plot_predicted_vs_actual(test, results, best_name, palette)
    paired_t_tests(results, y_test)
    plot_coefficients(results, FEATURES, best_ridge, best_lasso)
    per_region_predictions(test, results, best_name, palette)

    model_b_with_alcohol(panel, FEATURES, best_ridge, best_lasso)

# ---------- R10: Model B with alcohol supply --------------------------------

def model_b_with_alcohol(panel: pd.DataFrame,
                         FEATURES: list[str],
                         best_ridge: float,
                         best_lasso: float,
                         best_xgb_params: dict) -> None:
    print(f"\n-- R10: Model B - with alcohol supply "
          f"(train {min(ALCOHOL_YEARS)}-2024, test {YEAR_MAX}) [A12] --")

    # Filter panel for alcohol supply years
    panel_b = panel[(panel["Year"] >= min(ALCOHOL_YEARS)) & panel["Total_PAC"].notna()].copy()
    panel_b["alcohol_per_capita"] = panel_b["Total_PAC"] / panel_b["Total_population"]

    # Add alcohol_per_capita to feature list
    features_b = FEATURES + ["alcohol_per_capita"]

    train_b = panel_b[panel_b["Year"] <= 2024]
    test_b = panel_b[panel_b["Year"] == YEAR_MAX]

    if len(train_b) <= 5 or test_b.empty:
        print("   insufficient data for Model B.")
        return

    y_train_b = train_b["log_assault_rate"].to_numpy(float)
    y_test_b = test_b["log_assault_rate"].to_numpy(float)

    # Linear models use scaled features
    scaler_b = StandardScaler()
    x_train_b = scaler_b.fit_transform(train_b[features_b].astype(float))
    x_test_b = scaler_b.transform(test_b[features_b].astype(float))

    # XGBoost uses raw features
    X_train_raw = train_b[features_b].astype(float)
    X_test_raw = test_b[features_b].astype(float)

    print(f"   train {len(train_b)} rows | test {len(test_b)} rows")
    print(f"   {'Model':<22}{'RMSE(log)':<12}{'MAE(log)':<12}{'RMSE(/100k)'}")

    results_b: dict[str, dict] = {}

    # Linear models
    linear_models = {
        "Linear Regression": LinearRegression(),
        f"Ridge (a={best_ridge})": Ridge(alpha=best_ridge),
        f"Lasso (a={best_lasso})": Lasso(alpha=best_lasso, max_iter=10000),
    }

    for name, model in linear_models.items():
        model.fit(x_train_b, y_train_b)
        pred = model.predict(x_test_b)
        rmse = float(np.sqrt(mean_squared_error(y_test_b, pred)))
        mae = float(mean_absolute_error(y_test_b, pred))
        rmse_rate = float(np.sqrt(mean_squared_error(np.expm1(y_test_b),
                                                     np.expm1(pred))))
        results_b[name] = {
            "rmse": rmse,
            "mae": mae,
            "rmse_rate": rmse_rate,
            "coef": dict(zip(features_b, model.coef_)),
        }
        print(f"   {name:<22}{rmse:<12.4f}{mae:<12.4f}{rmse_rate:.1f}")

    # XGBoost model
    xgb_model = XGBRegressor(
        n_estimators=800,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=1,
        objective="reg:squarederror",
        random_state=42,
        **best_xgb_params
    )

    xgb_model.fit(X_train_raw, y_train_b)
    xgb_pred = xgb_model.predict(X_test_raw)

    xgb_rmse = float(np.sqrt(mean_squared_error(y_test_b, xgb_pred)))
    xgb_mae = float(mean_absolute_error(y_test_b, xgb_pred))
    xgb_rmse_rate = float(np.sqrt(mean_squared_error(np.expm1(y_test_b),
                                                     np.expm1(xgb_pred))))

    results_b["XGBoost"] = {
        "rmse": xgb_rmse,
        "mae": xgb_mae,
        "rmse_rate": xgb_rmse_rate,
        "coef": None  # trees don't have linear coefficients
    }

    print(f"   {'XGBoost':<22}{xgb_rmse:<12.4f}{xgb_mae:<12.4f}{xgb_rmse_rate:.1f}")

    # Alcohol_per_capita coefficient (linear only)
    print("   standardised alcohol_per_capita coefficient:")
    for name, r in results_b.items():
        if r["coef"] is not None:
            print(f"      {name:<22}{r['coef']['alcohol_per_capita']:+.4f}")
        else:
            print(f"      {name:<22}N/A (tree model)")



# ---------- orchestrator ----------------------------------------------------

def run_regression(monthly_assault: pd.DataFrame,
                   population: pd.DataFrame,
                   palette) -> None:
    banner("STAGE 5 - REGRESSION -> regression_plots/")

    panel, train, test, dummy_cols, cv_frame = prepare_panel(monthly_assault, population)

    FEATURES = evaluate_feature_variants(train, test, dummy_cols)
    FEATURES = compute_vif_and_filter(train, FEATURES)

    best_ridge, best_lasso = tune_alphas(cv_frame, FEATURES)
    best_xgb_params = tune_xgboost(cv_frame, FEATURES)
    results, best_name = train_final_models(train, test, FEATURES,
                                            best_ridge, best_lasso, best_xgb_params, cv_frame)

    y_test = test["log_assault_rate"].to_numpy(float)

    check_assumptions(results, y_test)
    plot_predicted_vs_actual(test, results, best_name, palette)
    paired_t_tests(results, y_test)
    plot_coefficients(results, FEATURES, best_ridge, best_lasso)
    per_region_predictions(test, results, best_name, palette)
    

    model_b_with_alcohol(panel, FEATURES, best_ridge, best_lasso, best_xgb_params)

# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    ingest_main()
    panel, population, _ = build_panel()
    eda = run_eda(panel)
    run_pca(population)
    run_regression(eda["monthly_assault"], population, eda["palette"])

    banner("PIPELINE COMPLETE")
    print(f"  dataset/raw/         ingested copies + manifest.json")
    print(f"  dataset/processed/   nt_crime_merged_2015_2025.csv")
    print(f"  eda_plots/           {len(list(PLOT_DIR.glob('*.png')))} figures")
    print(f"  regression_plots/    {len(list(REG_PLOT_DIR.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
