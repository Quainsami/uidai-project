import json
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd

BASE = Path(__file__).resolve().parents[1]
DATA_ENROL = BASE / "api_data_aadhar_enrolment"
DATA_DEMO = BASE / "api_data_aadhar_demographic"
DATA_BIO = BASE / "api_data_aadhar_biometric"
EXTRA = BASE / "uidai_data"
OUT = BASE / "analysis" / "outputs"
FIG = BASE / "analysis" / "figures"

OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 220,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "font.size": 10,
})

STATE_CANONICAL = {
    "andaman nicobar": "Andaman and Nicobar Islands",
    "andaman and nicobar islands": "Andaman and Nicobar Islands",
    "dadra nagar haveli": "Dadra and Nagar Haveli and Daman and Diu",
    "dadra and nagar haveli": "Dadra and Nagar Haveli and Daman and Diu",
    "daman diu": "Dadra and Nagar Haveli and Daman and Diu",
    "daman and diu": "Dadra and Nagar Haveli and Daman and Diu",
    "dadra and nagar haveli and daman and diu": "Dadra and Nagar Haveli and Daman and Diu",
    "nct of delhi": "Delhi",
    "jammu kashmir": "Jammu and Kashmir",
    "jammu and kashmir": "Jammu and Kashmir",
    "jammu kashmir union territory": "Jammu and Kashmir",
    "orissa": "Odisha",
    "odisha": "Odisha",
    "lakshdweep": "Lakshadweep",
    "chhatisgarh": "Chhattisgarh",
    "chhattisgarh": "Chhattisgarh",
    "telengana": "Telangana",
    "uttaranchal": "Uttarakhand",
    "pondicherry": "Puducherry",
}

PINCODE_STATE_MAP = None
PINCODE_DISTRICT_MAP = None
VALID_STATES = None


def _normalize_state_key(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.lower()
        .str.replace("&", "and")
        .str.replace(r"[^a-z0-9 ]+", "", regex=True)
        .str.replace(r"\\s+", " ", regex=True)
        .str.strip()
    )


def _load_pincode_lookup() -> None:
    global PINCODE_STATE_MAP, PINCODE_DISTRICT_MAP, VALID_STATES
    if PINCODE_STATE_MAP is not None:
        return
    pin_path = EXTRA / "pincodes_full.csv"
    if not pin_path.exists():
        return
    df = pd.read_csv(pin_path)
    df["Pincode"] = pd.to_numeric(df["Pincode"], errors="coerce").astype("Int64")
    df["state_key"] = _normalize_state_key(df["State"])
    df["state_canon"] = df["state_key"].map(STATE_CANONICAL).fillna(df["State"].astype(str).str.strip())
    # most common mapping per pincode
    PINCODE_STATE_MAP = df.groupby("Pincode")["state_canon"].agg(lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0])
    PINCODE_DISTRICT_MAP = df.groupby("Pincode")["DistrictsName"].agg(lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0])
    VALID_STATES = set(df["state_canon"].dropna().unique().tolist())


def _read_concat(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob("*.csv"))
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    df["state"] = df["state"].astype(str).str.strip()
    df["district"] = df["district"].astype(str).str.strip()
    df["pincode"] = pd.to_numeric(df["pincode"], errors="coerce").astype("Int64")
    _load_pincode_lookup()
    # normalize state labels to reduce duplicates
    state_key = _normalize_state_key(df["state"])
    df["state"] = state_key.map(STATE_CANONICAL).fillna(df["state"])

    # fill invalid state/district via pincode lookup
    if VALID_STATES:
        state_invalid = ~df["state"].isin(VALID_STATES) | df["state"].str.fullmatch(r"\\d+", na=False)
        df.loc[state_invalid, "state"] = df.loc[state_invalid, "pincode"].map(PINCODE_STATE_MAP)
    district_invalid = df["district"].str.fullmatch(r"\\d+", na=False)
    df.loc[district_invalid, "district"] = df.loc[district_invalid, "pincode"].map(PINCODE_DISTRICT_MAP)
    df = df[df["state"].notna()]
    return df


def _normalize_name(s: pd.Series) -> pd.Series:
    s = s.str.lower().str.strip()
    s = s.str.replace(r"[^a-z0-9 ]+", "", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True)
    # common renames
    s = s.str.replace("bengaluru", "bangalore")
    s = s.str.replace("mysuru", "mysore")
    s = s.str.replace("belagavi", "belgaum")
    s = s.str.replace("kalaburagi", "gulbarga")
    s = s.str.replace("vijayapura", "bijapur")
    s = s.str.replace("ysr", "kadapa")
    s = s.str.replace("chengalpattu", "kanchipuram")
    s = s.str.replace("palakkad", "palghat")
    s = s.str.replace("tiruvannamalai", "thiruvannamalai")
    s = s.str.replace("thiruvananthapuram", "trivandrum")
    return s


def _save_parquet(df: pd.DataFrame, name: str) -> None:
    df.to_parquet(OUT / f"{name}.parquet", index=False)


print("Loading datasets...")
enrol = _standardize(_read_concat(DATA_ENROL))
demo = _standardize(_read_concat(DATA_DEMO))
bio = _standardize(_read_concat(DATA_BIO))

# metrics
enrol["enrol_total"] = enrol[["age_0_5", "age_5_17", "age_18_greater"]].sum(axis=1)
demo["demo_total"] = demo[["demo_age_5_17", "demo_age_17_"]].sum(axis=1)
bio["bio_total"] = bio[["bio_age_5_17", "bio_age_17_"]].sum(axis=1)

print("Aggregating state and district daily tables...")

# state daily
enrol_state_daily = enrol.groupby(["date", "state"], as_index=False).agg({
    "age_0_5": "sum",
    "age_5_17": "sum",
    "age_18_greater": "sum",
    "enrol_total": "sum",
})
demo_state_daily = demo.groupby(["date", "state"], as_index=False).agg({
    "demo_age_5_17": "sum",
    "demo_age_17_": "sum",
    "demo_total": "sum",
})
bio_state_daily = bio.groupby(["date", "state"], as_index=False).agg({
    "bio_age_5_17": "sum",
    "bio_age_17_": "sum",
    "bio_total": "sum",
})

state_daily = enrol_state_daily.merge(demo_state_daily, on=["date", "state"], how="outer")\
    .merge(bio_state_daily, on=["date", "state"], how="outer")
state_daily = state_daily.fillna(0)
state_daily["update_total"] = state_daily["demo_total"] + state_daily["bio_total"]

# district daily
enrol_district_daily = enrol.groupby(["date", "state", "district"], as_index=False).agg({
    "age_0_5": "sum",
    "age_5_17": "sum",
    "age_18_greater": "sum",
    "enrol_total": "sum",
})
demo_district_daily = demo.groupby(["date", "state", "district"], as_index=False).agg({
    "demo_age_5_17": "sum",
    "demo_age_17_": "sum",
    "demo_total": "sum",
})
bio_district_daily = bio.groupby(["date", "state", "district"], as_index=False).agg({
    "bio_age_5_17": "sum",
    "bio_age_17_": "sum",
    "bio_total": "sum",
})


district_daily = enrol_district_daily.merge(demo_district_daily, on=["date", "state", "district"], how="outer")\
    .merge(bio_district_daily, on=["date", "state", "district"], how="outer")

for col in ["age_0_5", "age_5_17", "age_18_greater", "enrol_total",
            "demo_age_5_17", "demo_age_17_", "demo_total",
            "bio_age_5_17", "bio_age_17_", "bio_total"]:
    if col in district_daily.columns:
        district_daily[col] = district_daily[col].fillna(0)

district_daily["update_total"] = district_daily["demo_total"] + district_daily["bio_total"]

# pincode daily (for anomalies and city maps)
print("Aggregating pincode daily tables...")
enrol_pincode_daily = enrol.groupby(["date", "state", "district", "pincode"], as_index=False).agg({
    "enrol_total": "sum",
})
demo_pincode_daily = demo.groupby(["date", "state", "district", "pincode"], as_index=False).agg({
    "demo_total": "sum",
})
bio_pincode_daily = bio.groupby(["date", "state", "district", "pincode"], as_index=False).agg({
    "bio_total": "sum",
})

pincode_daily = enrol_pincode_daily.merge(demo_pincode_daily, on=["date", "state", "district", "pincode"], how="outer")\
    .merge(bio_pincode_daily, on=["date", "state", "district", "pincode"], how="outer")

pincode_daily[["enrol_total", "demo_total", "bio_total"]] = pincode_daily[["enrol_total", "demo_total", "bio_total"]].fillna(0)
pincode_daily["update_total"] = pincode_daily["demo_total"] + pincode_daily["bio_total"]
pincode_daily["update_to_enrol_ratio"] = np.where(
    pincode_daily["enrol_total"] > 0,
    pincode_daily["update_total"] / pincode_daily["enrol_total"],
    np.nan,
)

print("Saving outputs...")
_save_parquet(state_daily, "state_daily")
_save_parquet(district_daily, "district_daily")
_save_parquet(pincode_daily, "pincode_daily")

# State summary
state_summary = state_daily.groupby("state", as_index=False).agg({
    "age_0_5": "sum",
    "age_5_17": "sum",
    "age_18_greater": "sum",
    "enrol_total": "sum",
    "demo_total": "sum",
    "bio_total": "sum",
    "update_total": "sum",
})
state_summary["demo_share"] = np.where(state_summary["update_total"] > 0,
                                        state_summary["demo_total"] / state_summary["update_total"], 0)
state_summary["bio_share"] = np.where(state_summary["update_total"] > 0,
                                       state_summary["bio_total"] / state_summary["update_total"], 0)
state_summary["update_enrol_ratio"] = state_summary["update_total"] / state_summary["enrol_total"].replace(0, np.nan)

# District summary with per-capita
print("Loading census data...")
census = pd.read_csv(EXTRA / "india_districts_census_2011.csv")

census["state_norm"] = _normalize_name(census["State name"].astype(str))
census["district_norm"] = _normalize_name(census["District name"].astype(str))

# State-level census aggregates
census_state = census.groupby("state_norm", as_index=False).agg({
    "Population": "sum",
    "Households": "sum",
    "Housholds_with_Electric_Lighting": "sum",
    "Households_with_Internet": "sum",
})
state_summary["state_norm"] = _normalize_name(state_summary["state"].astype(str))
state_summary = state_summary.merge(census_state, on="state_norm", how="left")
state_summary["population"] = state_summary["Population"]
state_summary["updates_per_1k"] = np.where(state_summary["population"] > 0,
                                           state_summary["update_total"] / state_summary["population"] * 1000,
                                           np.nan)
state_summary["enrol_per_1k"] = np.where(state_summary["population"] > 0,
                                         state_summary["enrol_total"] / state_summary["population"] * 1000,
                                         np.nan)
state_summary["electricity_rate"] = np.where(state_summary["Households"] > 0,
                                             state_summary["Housholds_with_Electric_Lighting"] / state_summary["Households"],
                                             np.nan)
state_summary["internet_rate"] = np.where(state_summary["Households"] > 0,
                                          state_summary["Households_with_Internet"] / state_summary["Households"],
                                          np.nan)

# normalize district names in uidai data
uidai_district_year = district_daily.groupby(["state", "district"], as_index=False).agg({
    "enrol_total": "sum",
    "demo_total": "sum",
    "bio_total": "sum",
    "update_total": "sum",
})
uidai_district_year["state_norm"] = _normalize_name(uidai_district_year["state"].astype(str))
uidai_district_year["district_norm"] = _normalize_name(uidai_district_year["district"].astype(str))

uidai_district_year = uidai_district_year.merge(
    census,
    left_on=["state_norm", "district_norm"],
    right_on=["state_norm", "district_norm"],
    how="left",
    suffixes=("", "_census"),
)
uidai_district_year["population"] = uidai_district_year["Population"]
uidai_district_year["updates_per_1k"] = np.where(uidai_district_year["population"] > 0,
                                                 uidai_district_year["update_total"] / uidai_district_year["population"] * 1000,
                                                 np.nan)
uidai_district_year["enrol_per_1k"] = np.where(uidai_district_year["population"] > 0,
                                               uidai_district_year["enrol_total"] / uidai_district_year["population"] * 1000,
                                               np.nan)
uidai_district_year["update_enrol_ratio"] = uidai_district_year["update_total"] / uidai_district_year["enrol_total"].replace(0, np.nan)

_save_parquet(state_summary, "state_summary")
_save_parquet(uidai_district_year, "district_summary")

# Day-of-week patterns
state_daily["dow"] = state_daily["date"].dt.day_name()
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
# Identify months with dense daily coverage (>=10 non-zero update days)
state_daily["month"] = state_daily["date"].dt.to_period("M").astype(str)
daily_updates = state_daily.groupby("date", as_index=False)["update_total"].sum()
daily_updates["month"] = daily_updates["date"].dt.to_period("M").astype(str)
nonzero_by_month = daily_updates[daily_updates["update_total"] > 0].groupby("month").size()
daily_months = nonzero_by_month[nonzero_by_month >= 10].index.tolist()

daily_focus = state_daily[state_daily["month"].isin(daily_months)].copy()
dow_summary = daily_focus.groupby("dow", as_index=False).agg({
    "enrol_total": "mean",
    "demo_total": "mean",
    "bio_total": "mean",
    "update_total": "mean",
})
dow_summary["dow"] = pd.Categorical(dow_summary["dow"], categories=dow_order, ordered=True)
dow_summary = dow_summary.sort_values("dow")
_save_parquet(dow_summary, "dow_summary")

# Monthly trends
monthly = state_daily.groupby("month", as_index=False).agg({
    "enrol_total": "sum",
    "demo_total": "sum",
    "bio_total": "sum",
    "update_total": "sum",
})
_save_parquet(monthly, "monthly_summary")

# State-month matrix for heatmaps
state_monthly = state_daily.groupby(["month", "state"], as_index=False).agg({
    "enrol_total": "sum",
    "demo_total": "sum",
    "bio_total": "sum",
    "update_total": "sum",
})
_save_parquet(state_monthly, "state_monthly")

# Save daily focus window (for daily patterns)
_save_parquet(daily_focus, "daily_focus")

# District quadrant classification
district_focus = uidai_district_year.dropna(subset=["enrol_per_1k", "updates_per_1k"]).copy()
median_enrol = district_focus["enrol_per_1k"].median()
median_updates = district_focus["updates_per_1k"].median()
district_focus["quadrant"] = np.select(
    [
        (district_focus["enrol_per_1k"] >= median_enrol) & (district_focus["updates_per_1k"] >= median_updates),
        (district_focus["enrol_per_1k"] >= median_enrol) & (district_focus["updates_per_1k"] < median_updates),
        (district_focus["enrol_per_1k"] < median_enrol) & (district_focus["updates_per_1k"] >= median_updates),
    ],
    ["High Enrol / High Update", "High Enrol / Low Update", "Low Enrol / High Update"],
    default="Low Enrol / Low Update",
)
_save_parquet(district_focus, "district_focus")

# Anomaly detection for pincodes
ratio = pincode_daily.dropna(subset=["update_to_enrol_ratio"]).copy()
ratio["ratio_mean"] = ratio.groupby("pincode")["update_to_enrol_ratio"].transform("mean")
ratio["ratio_std"] = ratio.groupby("pincode")["update_to_enrol_ratio"].transform("std")
ratio["ratio_z"] = (ratio["update_to_enrol_ratio"] - ratio["ratio_mean"]) / ratio["ratio_std"]
ratio["ratio_z"] = ratio["ratio_z"].replace([np.inf, -np.inf], np.nan)

anomaly = ratio.sort_values("ratio_z", ascending=False).head(25)
_save_parquet(anomaly, "pincode_anomalies")

# City pincode totals for available geojsons
city_geojsons = {
    "bangalore": "bangalore_pincode_boundary.geojson",
    "mumbai": "mumbai_pincode_boundary.geojson",
    "chennai": "chennai_pincode_boundary.geojson",
    "hyderabad": "hyderabad_pincode_boundary.geojson",
    "kolkata": "kolkata_pincode_boundary.geojson",
    "ahmedabad": "ahmedabad_pincode_boundary.geojson",
}

pincode_year = pincode_daily.groupby(["pincode"], as_index=False).agg({
    "enrol_total": "sum",
    "update_total": "sum",
})

for city, filename in city_geojsons.items():
    geo_path = EXTRA / filename
    if not geo_path.exists():
        continue
    gdf = gpd.read_file(geo_path)
    pin_col = None
    for col in gdf.columns:
        if col.lower() in {"pin_code", "pincode", "pin"}:
            pin_col = col
            break
    if pin_col is None:
        continue
    gdf["pincode"] = pd.to_numeric(gdf[pin_col], errors="coerce").astype("Int64")
    gdf = gdf.merge(pincode_year, on="pincode", how="left")
    gdf[["enrol_total", "update_total"]] = gdf[["enrol_total", "update_total"]].fillna(0)
    gdf.to_file(OUT / f"{city}_pincode.geojson", driver="GeoJSON")

# State-level geojson for India map
state_geo_source = EXTRA / "india_states.geojson"
if state_geo_source.exists():
    india_geo = gpd.read_file(state_geo_source)
    india_geo["state_norm"] = _normalize_name(india_geo["st_nm"].astype(str))
    state_geo = india_geo.dissolve(by="state_norm", as_index=False)
    state_geo = state_geo.merge(
        state_summary[[
            "state_norm", "state", "enrol_total", "update_total", "update_enrol_ratio",
            "updates_per_1k", "bio_share"
        ]],
        on="state_norm",
        how="left",
    )
    state_geo["state"] = state_geo["state"].fillna(state_geo["state_norm"].str.title())
    state_geo.to_file(OUT / "india_states.geojson", driver="GeoJSON")

# --------- Visualizations ---------
print("Generating figures...")

# 1. Monthly national trends
monthly = monthly.sort_values("month")
plt.figure(figsize=(10, 5))
x = np.arange(len(monthly))
plt.bar(x, monthly["demo_total"], label="Demographic Updates", color="#ff7f0e")
plt.bar(x, monthly["bio_total"], bottom=monthly["demo_total"], label="Biometric Updates", color="#2ca02c")
plt.plot(x, monthly["enrol_total"], label="Enrolment", color="#1f77b4", marker="o")
plt.xticks(x, monthly["month"], rotation=45, ha="right")
plt.title("Monthly Aadhaar Activity (2025)")
plt.ylabel("Total Transactions")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "fig_01_monthly_trends.png")
plt.close()

# 2. Daily trends in dense months (7-day rolling)
daily_totals = daily_focus.groupby("date", as_index=False).agg({
    "enrol_total": "sum",
    "demo_total": "sum",
    "bio_total": "sum",
    "update_total": "sum",
}).sort_values("date")
rolling = daily_totals.set_index("date").rolling(7).mean().reset_index()

plt.figure(figsize=(10, 5))
plt.plot(rolling["date"], rolling["enrol_total"], label="Enrolment", color="#1f77b4")
plt.plot(rolling["date"], rolling["demo_total"], label="Demographic Updates", color="#ff7f0e")
plt.plot(rolling["date"], rolling["bio_total"], label="Biometric Updates", color="#2ca02c")
plt.title("Daily Aadhaar Activity (7-day rolling, dense months)")
plt.xlabel("Date")
plt.ylabel("Transactions")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "fig_02_daily_trends.png")
plt.close()

# 3. Day of week pattern (dense months only)
plt.figure(figsize=(9, 4.5))
width = 0.25
x = np.arange(len(dow_summary["dow"]))
plt.bar(x - width, dow_summary["enrol_total"], width=width, label="Enrolment", color="#1f77b4")
plt.bar(x, dow_summary["demo_total"], width=width, label="Demographic", color="#ff7f0e")
plt.bar(x + width, dow_summary["bio_total"], width=width, label="Biometric", color="#2ca02c")
plt.xticks(x, dow_summary["dow"], rotation=25)
plt.title("Average Daily Volume by Day of Week (dense months)")
plt.ylabel("Average Transactions")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "fig_03_dow_pattern.png")
plt.close()

# 4. Top states age mix
state_summary = state_summary.sort_values("enrol_total", ascending=False).head(12)
state_summary["age_0_5_share"] = state_summary["age_0_5"] / state_summary["enrol_total"]
state_summary["age_5_17_share"] = state_summary["age_5_17"] / state_summary["enrol_total"]
state_summary["age_18_share"] = state_summary["age_18_greater"] / state_summary["enrol_total"]

plt.figure(figsize=(10, 5))
bottom = np.zeros(len(state_summary))
plt.bar(state_summary["state"], state_summary["age_0_5_share"], label="0-5", color="#8dd3c7")
bottom += state_summary["age_0_5_share"].values
plt.bar(state_summary["state"], state_summary["age_5_17_share"], bottom=bottom, label="5-17", color="#ffffb3")
bottom += state_summary["age_5_17_share"].values
plt.bar(state_summary["state"], state_summary["age_18_share"], bottom=bottom, label="18+", color="#bebada")
plt.title("Age Mix of Enrolments in Top 12 States")
plt.ylabel("Share of Enrolments")
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "fig_04_state_age_mix.png")
plt.close()

# 5. Update mix by state
state_mix = state_summary.copy()
state_mix["demo_to_bio_ratio"] = np.where(state_mix["bio_total"] > 0, state_mix["demo_total"] / state_mix["bio_total"], np.nan)
plt.figure(figsize=(10, 5))
plt.scatter(state_mix["demo_to_bio_ratio"], state_mix["update_total"], s=60, color="#9467bd", alpha=0.7)
for _, row in state_mix.iterrows():
    plt.text(row["demo_to_bio_ratio"], row["update_total"], row["state"], fontsize=8, alpha=0.8)
plt.title("Update Mix by State (Demographic vs Biometric)")
plt.xlabel("Demographic/Biometric Ratio")
plt.ylabel("Total Updates")
plt.tight_layout()
plt.savefig(FIG / "fig_05_state_update_mix.png")
plt.close()

# 6. District intensity scatter
plt.figure(figsize=(8, 6))
subset = uidai_district_year.dropna(subset=["enrol_per_1k", "updates_per_1k"])
plt.scatter(subset["enrol_per_1k"], subset["updates_per_1k"], alpha=0.6, color="#1f77b4")
max_val = np.nanmax([subset["enrol_per_1k"].max(), subset["updates_per_1k"].max()])
plt.plot([0, max_val], [0, max_val], linestyle="--", color="#555", alpha=0.6)
plt.title("District Enrolment vs Update Intensity (per 1k population)")
plt.xlabel("Enrolments per 1k")
plt.ylabel("Updates per 1k")
plt.tight_layout()
plt.savefig(FIG / "fig_06_district_intensity.png")
plt.close()

# 7. Update-to-enrol ratio vs electricity access (district)
if "Housholds_with_Electric_Lighting" in uidai_district_year.columns:
    df_corr = uidai_district_year.dropna(subset=["update_total", "enrol_total", "Population", "Housholds_with_Electric_Lighting", "Households"]).copy()
    df_corr["update_enrol_ratio"] = df_corr["update_total"] / df_corr["enrol_total"].replace(0, np.nan)
    df_corr["electricity_rate"] = df_corr["Housholds_with_Electric_Lighting"] / df_corr["Households"]
    df_corr = df_corr.replace([np.inf, -np.inf], np.nan).dropna(subset=["update_enrol_ratio", "electricity_rate"])
    plt.figure(figsize=(8, 6))
    sns.regplot(data=df_corr, x="electricity_rate", y="update_enrol_ratio",
                scatter_kws={"alpha": 0.5}, line_kws={"color": "#d62728"})
    plt.title("Infrastructure Access vs Update-Heavy Districts")
    plt.xlabel("Households with Electricity (share)")
    plt.ylabel("Updates-to-Enrolment Ratio")
    plt.tight_layout()
    plt.savefig(FIG / "fig_07_infra_update_ratio.png")
    plt.close()

# 8. Pincode anomalies
anomaly_plot = anomaly.dropna(subset=["ratio_z"]).head(20).copy()
plt.figure(figsize=(9, 6))
plt.hlines(y=anomaly_plot["pincode"].astype(str), xmin=0, xmax=anomaly_plot["ratio_z"], color="#ff7f0e")
plt.scatter(anomaly_plot["ratio_z"], anomaly_plot["pincode"].astype(str), color="#ff7f0e")
plt.title("Top Pincode Anomalies (Update-to-Enrolment Ratio z-score)")
plt.xlabel("z-score")
plt.ylabel("Pincode")
plt.tight_layout()
plt.savefig(FIG / "fig_08_pincode_anomalies.png")
plt.close()

# 9-10. City maps for Bangalore and Mumbai
for city in ["bangalore", "mumbai"]:
    geo_path = OUT / f"{city}_pincode.geojson"
    if not geo_path.exists():
        continue
    gdf = gpd.read_file(geo_path)
    plt.figure(figsize=(7, 7))
    gdf.plot(column="update_total", cmap="viridis", legend=True, linewidth=0.2, edgecolor="#333")
    plt.title(f"{city.title()} Pincode Update Intensity (2025)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIG / f"fig_09_{city}_map.png")
    plt.close()

# 11. State-month heatmap (top states by updates)
top_states = state_summary.sort_values("update_total", ascending=False)["state"].head(12).tolist()
heat = state_monthly[state_monthly["state"].isin(top_states)].copy()
heat_pivot = heat.pivot_table(index="state", columns="month", values="update_total", aggfunc="sum").fillna(0)
heat_pivot = heat_pivot.loc[top_states]
plt.figure(figsize=(10, 5))
sns.heatmap(heat_pivot, cmap="YlGnBu", linewidths=0.5)
plt.title("Monthly Update Load by State (Top 12)")
plt.xlabel("Month")
plt.ylabel("State")
plt.tight_layout()
plt.savefig(FIG / "fig_10_state_month_heatmap.png")
plt.close()

# 12. District quadrant scatter
plt.figure(figsize=(8, 6))
palette = {
    "High Enrol / High Update": "#2ca02c",
    "High Enrol / Low Update": "#ff7f0e",
    "Low Enrol / High Update": "#d62728",
    "Low Enrol / Low Update": "#1f77b4",
}
for quad, group in district_focus.groupby("quadrant"):
    plt.scatter(group["enrol_per_1k"], group["updates_per_1k"], alpha=0.6, s=25, label=quad, color=palette.get(quad))
plt.axvline(median_enrol, color="#555", linestyle="--", linewidth=1)
plt.axhline(median_updates, color="#555", linestyle="--", linewidth=1)
plt.title("District Quadrants: Enrolment vs Update Intensity")
plt.xlabel("Enrolments per 1k")
plt.ylabel("Updates per 1k")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "fig_11_district_quadrants.png")
plt.close()

# 13. Update-to-enrol ratio distribution
ratio_vals = uidai_district_year["update_enrol_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
ratio_clip = ratio_vals[ratio_vals < ratio_vals.quantile(0.98)]
plt.figure(figsize=(8, 5))
plt.hist(ratio_clip, bins=40, color="#9467bd", alpha=0.8)
plt.title("Distribution of Update-to-Enrolment Ratio (Districts)")
plt.xlabel("Update / Enrolment Ratio")
plt.ylabel("Number of Districts")
plt.tight_layout()
plt.savefig(FIG / "fig_12_update_ratio_dist.png")
plt.close()

# 14-15. India state maps
state_geo_path = OUT / "india_states.geojson"
if state_geo_path.exists():
    state_geo = gpd.read_file(state_geo_path)
    if "updates_per_1k" in state_geo.columns:
        plt.figure(figsize=(7.5, 8))
        state_geo.plot(column="updates_per_1k", cmap="YlOrRd", legend=True, linewidth=0.4, edgecolor="#333")
        plt.title("India State Update Intensity (per 1k population)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(FIG / "fig_13_state_updates_per_1k.png")
        plt.close()
    if "update_enrol_ratio" in state_geo.columns:
        plt.figure(figsize=(7.5, 8))
        state_geo.plot(column="update_enrol_ratio", cmap="PuRd", legend=True, linewidth=0.4, edgecolor="#333")
        plt.title("India State Update-to-Enrolment Ratio")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(FIG / "fig_14_state_update_ratio.png")
        plt.close()

# Summary metrics for narrative
overall_totals = state_daily.groupby("date", as_index=False)[
    ["enrol_total", "demo_total", "bio_total", "update_total"]
].sum()
overall_update_ratio = (overall_totals["demo_total"].sum() + overall_totals["bio_total"].sum()) / overall_totals["enrol_total"].sum()

district_rank = uidai_district_year.dropna(subset=["update_enrol_ratio"]).copy()
district_rank = district_rank[district_rank["enrol_total"] >= 1000]
top_update_ratio = district_rank.sort_values("update_enrol_ratio", ascending=False)[
    ["state", "district", "update_enrol_ratio"]
].head(5)
top_updates_per_1k = uidai_district_year.dropna(subset=["updates_per_1k"]).sort_values("updates_per_1k", ascending=False)[
    ["state", "district", "updates_per_1k"]
].head(5)

summary = {
    "date_min": str(state_daily["date"].min().date()),
    "date_max": str(state_daily["date"].max().date()),
    "total_enrolments": int(overall_totals["enrol_total"].sum()),
    "total_demographic_updates": int(overall_totals["demo_total"].sum()),
    "total_biometric_updates": int(overall_totals["bio_total"].sum()),
    "overall_update_ratio": float(overall_update_ratio),
    "top_states_enrolment": state_summary.sort_values("enrol_total", ascending=False)["state"].head(5).tolist(),
    "top_states_updates": state_summary.sort_values("update_total", ascending=False)["state"].head(5).tolist(),
    "daily_focus_months": daily_months,
    "top_district_update_ratio": top_update_ratio.to_dict(orient="records"),
    "top_district_updates_per_1k": top_updates_per_1k.to_dict(orient="records"),
}

with open(OUT / "summary_metrics.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Done.")
