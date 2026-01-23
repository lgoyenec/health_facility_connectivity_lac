"""
Health Facility Connectivity Dashboard (LAC) — Thin Streamlit App
-----------------------------------------------------------------

Run:
  streamlit run app.py

Prerequisite (build prepared files once):
  python preprocess.py
"""

import json
import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import plotly.express as px

from shapely.ops import unary_union
from pyproj import CRS

# -----------------------------------------------------------------------------
# Streamlit setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Health Facility Connectivity — IDB (Internal)", layout="wide")

st.markdown(
    """
<style>
.sidebar-note {
    font-size: 0.85rem;
    color: #6B7280;
    font-style: italic;
    margin-top: -8px;
    margin-bottom: 8px;
}
div[data-testid="stMetricLabel"] > div {
    font-size: 0.9rem;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Health Facility Connectivity in Latin America and the Caribbean")

SPEED_COLS = ["fix_dl_mbps", "fix_ul_mbps", "mob_dl_mbps", "mob_ul_mbps"]

# -----------------------------------------------------------------------------
# IDB-inspired colors (not claiming official palette)
# -----------------------------------------------------------------------------
IDB_COLORS = {
    "navy":   "#003A70",
    "teal":   "#00A3A6",
    "orange": "#F28C28",
    "gray":   "#6B7280",
    "red":    "#D62728",
    "green":  "#2CA02C",
}
PLOTLY_DISCRETE = [
    IDB_COLORS["navy"],
    IDB_COLORS["teal"],
    IDB_COLORS["orange"],
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
]

# =============================================================================
# TRUE POP-UP disclaimer (requires st.dialog)
# =============================================================================
def show_modal_disclaimer_once():
    if "disclaimer_dismissed" not in st.session_state:
        st.session_state["disclaimer_dismissed"] = False

    if st.session_state["disclaimer_dismissed"]:
        return

    if not hasattr(st, "dialog"):
        st.error(
            "This app shows a true pop-up disclaimer using `st.dialog()`, "
            "but your Streamlit version does not support it.\n\n"
            "Upgrade Streamlit:\n  pip install -U streamlit\n\n"
            "Then restart the app."
        )
        st.stop()

    @st.dialog("Internal-use disclaimer", width="large")
    def _dlg():
        st.markdown(
            """
### Internal dashboard (not public)

This dashboard was developed for **internal reference** to help understand health facility connectivity patterns across **Latin America and the Caribbean**.

**Important:**
- **Do not share externally.**
- This is **not** an officially approved public dashboard.
- Use is restricted to internal specialist workflows within the **Inter-American Development Bank (IDB)**.
            """
        )
        if st.button("I understand — continue", type="primary", use_container_width=True):
            st.session_state["disclaimer_dismissed"] = True
            st.rerun()

    _dlg()


show_modal_disclaimer_once()

# =============================================================================
# Data loading (cached) — IMPORTANT: lazy load population to avoid OOM
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_facilities_countries_metadata() -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, Dict]:
    try:
        fac = gpd.read_parquet("./facilities_prepared.geoparquet")
        countries = gpd.read_parquet("./countries.geoparquet")
        with open("./metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        st.error(
            "Failed to load prepared files at startup.\n\n"
            "Make sure these files exist in the repository root:\n"
            "- facilities_prepared.geoparquet\n"
            "- countries.geoparquet\n"
            "- metadata.json\n\n"
            f"Error: {repr(e)}"
        )
        st.stop()

    for c in SPEED_COLS:
        if c in fac.columns:
            fac[c] = pd.to_numeric(fac[c], errors="coerce")
    for c in ["lat", "lon"]:
        if c in fac.columns:
            fac[c] = pd.to_numeric(fac[c], errors="coerce")

    return fac, countries, meta


@st.cache_resource(show_spinner=False)
def load_population_points() -> gpd.GeoDataFrame:
    try:
        pop = gpd.read_parquet("./population_points.geoparquet")
    except Exception as e:
        st.error(
            "Failed to load population_points.geoparquet.\n\n"
            "If this file is tracked via Git LFS, confirm Streamlit Cloud pulled the real file "
            "(not an LFS pointer).\n\n"
            f"Error: {repr(e)}"
        )
        st.stop()
    return pop


fac0, countries_gdf, meta = load_facilities_countries_metadata()
LAC_CRS = meta.get("crs", str(getattr(fac0, "crs", "")))

# =============================================================================
# Dashboard usage instructions
# =============================================================================
with st.expander("How to use this dashboard", expanded=False):
    st.markdown(
        """
### What controls do what?

**A) Selected speed (used everywhere)**
- **Direction**: Download vs Upload  
- **Network mode**:
  - Fixed = fixed broadband
  - Mobile = mobile
  - Best available = max(fixed, mobile)
  - Both (min) = min(fixed, mobile) (conservative)

**B) High-speed (used everywhere)**
- **Quantile** = computed **within each country × facility type**
- **Manual** = you set a cutoff per facility type

**C) Country focus**
- To keep performance stable, the country focus is **single-country only** (or **All LAC**).
- It affects **only**:
  - *Facilities Map*
  - *Population Coverage*
- Other tabs remain **LAC-wide** (because they already summarize across all countries).
        """
    )

# =============================================================================
# Helpers
# =============================================================================
def _stable_hash(payload: Dict) -> str:
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _is_hospital_type(tp: str) -> bool:
    return "hosp" in str(tp).lower()


@st.cache_data(show_spinner=False)
def add_selected_speed(_df: pd.DataFrame, direction: str, network_mode: str) -> pd.DataFrame:
    df = _df.copy()

    if direction == "Download":
        fix = df["fix_dl_mbps"]
        mob = df["mob_dl_mbps"]
    else:
        fix = df["fix_ul_mbps"]
        mob = df["mob_ul_mbps"]

    if network_mode == "Fixed":
        sel = fix
    elif network_mode == "Mobile":
        sel = mob
    elif network_mode == "Best available":
        sel = pd.concat([fix, mob], axis=1).max(axis=1, skipna=True)
    elif network_mode == "Both (min)":
        sel = pd.concat([fix, mob], axis=1).min(axis=1, skipna=True)
    else:
        sel = pd.concat([fix, mob], axis=1).max(axis=1, skipna=True)

    df["selected_speed_mbps"] = sel
    return df


@st.cache_data(show_spinner=False)
def filter_facilities_types_speed(_df: pd.DataFrame, types: List[str], require_speed: bool) -> pd.DataFrame:
    df = _df.copy()
    if types:
        df = df[df["tp_stbl"].isin(types)]
    if require_speed:
        df = df[df["has_speed"]]
    return df


@st.cache_data(show_spinner=False)
def compute_high_speed_flag_country_type(
    _df: pd.DataFrame,
    mode: str,
    quantile_spec: str,
    cutoff_map: Dict[str, float],
) -> Tuple[pd.DataFrame, Dict]:
    df = _df.copy()
    df["high_speed"] = False
    df.loc[df["selected_speed_mbps"].isna(), "high_speed"] = False

    meta_out: Dict = {}

    if mode == "Quantile":
        q = 0.75 if quantile_spec == "Quartile (top 25%)" else (2 / 3)
        for (ctry, t), sub in df.groupby(["country", "tp_stbl"], dropna=False):
            vals = sub["selected_speed_mbps"].dropna()
            thr = vals.quantile(q) if len(vals) else np.nan
            if pd.notna(thr):
                df.loc[sub.index, "high_speed"] = sub["selected_speed_mbps"] >= thr
        meta_out = {"mode": "Quantile", "quantile": q, "grouping": "country × tp_stbl"}
    else:
        for t, sub in df.groupby("tp_stbl", dropna=False):
            thr = cutoff_map.get(str(t), np.nan)
            if pd.notna(thr):
                df.loc[sub.index, "high_speed"] = df.loc[sub.index, "selected_speed_mbps"] >= thr
        meta_out = {"mode": "Manual", "grouping": "tp_stbl"}

    df.loc[df["selected_speed_mbps"].isna(), "high_speed"] = False
    return df, meta_out


def compute_kpis_map(df: pd.DataFrame) -> Dict[str, float]:
    total = len(df)
    if total == 0:
        return dict(total=0, pct_with_speed=0.0, pct_high_speed=0.0, pct_phc=0.0, pct_hosp=0.0)

    pct_with = float(df["has_speed"].mean() * 100) if "has_speed" in df.columns else 0.0
    pct_high = float(df["high_speed"].mean() * 100) if "high_speed" in df.columns else 0.0

    hosp_mask = df["tp_stbl"].apply(_is_hospital_type)
    pct_hosp = float(hosp_mask.mean() * 100)
    pct_phc = 100.0 - pct_hosp

    return dict(
        total=int(total),
        pct_with_speed=pct_with,
        pct_high_speed=pct_high,
        pct_phc=pct_phc,
        pct_hosp=pct_hosp,
    )


@st.cache_data(show_spinner=False)
def country_stats_table(_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    d = _df.dropna(subset=["selected_speed_mbps"]).copy()
    if d.empty:
        return pd.DataFrame(columns=[
            "Country", "Minimum speed (Mbps)", "P25 speed (Mbps)", "Median speed (Mbps)",
            "P75 speed (Mbps)", "N facilities", "% below threshold"
        ])

    g = d.groupby("country")["selected_speed_mbps"]
    out = pd.DataFrame({
        "Country": g.apply(lambda x: x.name).values,
        "Minimum speed (Mbps)": g.min().values,
        "P25 speed (Mbps)": g.quantile(0.25).values,
        "Median speed (Mbps)": g.median().values,
        "P75 speed (Mbps)": g.quantile(0.75).values,
        "N facilities": g.size().values,
    })

    below = (d["selected_speed_mbps"] < threshold).groupby(d["country"]).mean() * 100.0
    out["% below threshold"] = out["Country"].map(below.to_dict()).fillna(0.0)
    return out.sort_values("Median speed (Mbps)", ascending=True)


@st.cache_data(show_spinner=False)
def sample_for_distribution(_df: pd.DataFrame, per_country_cap: int = 1200, overall_cap: int = 45000) -> pd.DataFrame:
    d = _df.dropna(subset=["selected_speed_mbps"]).copy()
    if d.empty:
        return d
    parts = []
    for ctry, sub in d.groupby("country"):
        parts.append(sub.sample(n=min(per_country_cap, len(sub)), random_state=42))
    out = pd.concat(parts, ignore_index=True)
    if len(out) > overall_cap:
        out = out.sample(n=overall_cap, random_state=42)
    return out


@st.cache_data(show_spinner=False)
def regional_gaps(_df: pd.DataFrame, level: str, ref_stat: str, group_stat: str, country_filter: Optional[List[str]] = None) -> Tuple[pd.DataFrame, float]:
    d_all = _df["selected_speed_mbps"].dropna()
    lac_ref = float(d_all.quantile(0.25)) if ref_stat == "P25" else float(d_all.median())

    if level == "Country":
        keys = ["country"]
    elif level == "ADM1":
        keys = ["country", "ADM1_EN", "ADM1_PCODE"]
    else:
        keys = ["country", "ADM2_EN", "ADM2_PCODE"]

    d = _df.dropna(subset=["selected_speed_mbps"]).copy()
    if country_filter and level in ["ADM1", "ADM2"]:
        d = d[d["country"].isin(country_filter)].copy()

    if d.empty:
        return pd.DataFrame(columns=keys + ["Group value (Mbps)", "LAC reference (Mbps)", "Gap (Mbps)", "Gap direction"]), lac_ref

    if group_stat == "Median":
        agg = d.groupby(keys, as_index=False)["selected_speed_mbps"].median().rename(columns={"selected_speed_mbps": "Group value (Mbps)"})
    else:
        agg = d.groupby(keys, as_index=False)["selected_speed_mbps"].quantile(0.25).rename(columns={"selected_speed_mbps": "Group value (Mbps)"})

    agg["LAC reference (Mbps)"] = lac_ref
    agg["Gap (Mbps)"] = agg["Group value (Mbps)"] - lac_ref
    agg["Gap direction"] = np.where(agg["Gap (Mbps)"] < 0, "Below reference", "Above reference")
    return agg.sort_values("Gap (Mbps)"), lac_ref


# =============================================================================
# Population coverage — avoid huge serialization + always sample by default
# =============================================================================
@st.cache_data(show_spinner=True)
def population_coverage_cached(
    filter_hash: str,
    lac_crs_str: str,
    radius_km: float,
    pop_weight_col: Optional[str],
    _fac_gdf: gpd.GeoDataFrame,
    _pop_gdf: gpd.GeoDataFrame,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Cached by filter_hash (not by GeoDataFrames), while GeoDataFrames are passed
    as underscore args (unhashed) to avoid Streamlit hashing errors and huge payloads.
    """
    # Buffer in metric CRS
    c = CRS.from_user_input(lac_crs_str)
    metric_crs = c if (c and not c.is_geographic) else CRS.from_epsg(3857)

    fac_m = _fac_gdf.to_crs(metric_crs)
    pop_m = _pop_gdf.to_crs(metric_crs)

    radius_m = float(radius_km) * 1000.0
    union_poly = unary_union(list(fac_m.geometry.buffer(radius_m).values))
    union_gdf = gpd.GeoDataFrame({"_id": [1]}, geometry=[union_poly], crs=metric_crs)

    covered = gpd.sjoin(pop_m, union_gdf, how="inner", predicate="within").drop(columns=["index_right"], errors="ignore")

    if pop_weight_col and pop_weight_col in pop_m.columns:
        cov_sum = float(pd.to_numeric(covered[pop_weight_col], errors="coerce").fillna(0).sum())
        tot_sum = float(pd.to_numeric(pop_m[pop_weight_col], errors="coerce").fillna(0).sum())
        pct = 100.0 * cov_sum / tot_sum if tot_sum > 0 else 0.0
        overall = {"covered": cov_sum, "total": tot_sum, "pct_covered": pct, "metric": f"sum({pop_weight_col})"}
    else:
        cov_sum = int(len(covered))
        tot_sum = int(len(pop_m))
        pct = 100.0 * cov_sum / tot_sum if tot_sum > 0 else 0.0
        overall = {"covered": cov_sum, "total": tot_sum, "pct_covered": pct, "metric": "# population points"}

    if "country" not in pop_m.columns:
        return overall, pd.DataFrame()

    if pop_weight_col and pop_weight_col in pop_m.columns:
        tot = pop_m.groupby("country", as_index=False)[pop_weight_col].sum().rename(columns={pop_weight_col: "Total"})
        cov = covered.groupby("country", as_index=False)[pop_weight_col].sum().rename(columns={pop_weight_col: "Covered"})
    else:
        tot = pop_m.groupby("country", as_index=False).size().rename(columns={"size": "Total"})
        cov = covered.groupby("country", as_index=False).size().rename(columns={"size": "Covered"})

    tbl = tot.merge(cov, on="country", how="left").fillna({"Covered": 0})
    tbl["% covered"] = np.where(tbl["Total"] > 0, 100.0 * tbl["Covered"] / tbl["Total"], 0.0)
    tbl = tbl.sort_values("% covered", ascending=True)
    tbl = tbl.rename(columns={"country": "Country"})
    return overall, tbl


# =============================================================================
# Sidebar controls
# =============================================================================
with st.sidebar:
    st.header("Global controls")
    st.markdown('<div class="sidebar-note">Applies to all tabs.</div>', unsafe_allow_html=True)

    direction = st.radio("Direction", ["Download", "Upload"], horizontal=True)
    network_mode = st.selectbox("Network mode", ["Best available", "Fixed", "Mobile", "Both (min)"], index=0)

    all_types = sorted(fac0["tp_stbl"].dropna().unique().tolist())
    sel_types = st.multiselect("Facility types", all_types, default=all_types)

    require_speed = st.checkbox("Only facilities with any speed", value=False)

    st.divider()
    st.subheader("High-speed definition")
    hs_mode = st.radio("Mode", ["Quantile", "Manual by type"], index=0)
    if hs_mode == "Quantile":
        quantile_spec = st.selectbox("Quantile (within country × type)", ["Quartile (top 25%)", "Tercile (top 33%)"], index=0)
        cutoff_map = {}
    else:
        quantile_spec = "Quartile (top 25%)"
        cutoff_map = {str(t): st.number_input(f"Cutoff for {t} (Mbps)", min_value=0.0, value=10.0, step=1.0) for t in all_types}

    st.divider()
    country_threshold = st.number_input("Below-threshold if selected speed < (Mbps)", min_value=0.0, value=5.0, step=1.0)

    st.divider()
    st.header("Country focus")
    st.markdown('<div class="sidebar-note">Single-country only (or All LAC). Applies to <i>Facilities Map</i> and <i>Population Coverage</i>.</div>', unsafe_allow_html=True)

    all_countries = sorted(fac0["country"].dropna().unique().tolist())
    focus_option = ["All LAC"] + all_countries
    focus_country = st.selectbox("Focus country", options=focus_option, index=0)

# =============================================================================
# Compute global derived data
# =============================================================================
fac1 = add_selected_speed(fac0, direction, network_mode)
fac_global = filter_facilities_types_speed(fac1, sel_types, require_speed)
fac_global_hs, _ = compute_high_speed_flag_country_type(fac_global, hs_mode, quantile_spec, cutoff_map)

# Focus subset for Map + Population Coverage only
fac_focus = fac_global_hs.copy()
if focus_country != "All LAC":
    fac_focus = fac_focus[fac_focus["country"] == focus_country].copy()

tabs = st.tabs([
    "Facilities Map",
    "Country Distributions",
    "Regional Gaps",
    "Urban vs Rural",
    "Population Coverage"
])

# =============================================================================
# TAB 1: Facilities Map
# =============================================================================
with tabs[0]:
    st.subheader("Facilities map (focus country)")
    st.caption("Country focus affects only this tab and Population Coverage.")

    if focus_country == "All LAC":
        st.warning("Focus country = All LAC. The map can be heavy; the app will downsample automatically if needed.")
    else:
        st.info(f"Showing facilities for: **{focus_country}**")

    if fac_focus.empty:
        st.warning("No facilities match your current filters for the selected focus country.")
    else:
        k = compute_kpis_map(fac_focus)
        a, b, c, d, e = st.columns(5)
        a.metric("Total facilities", f"{k['total']:,}")
        b.metric("% with any speed", f"{k['pct_with_speed']:.1f}%")
        c.metric("% high-speed", f"{k['pct_high_speed']:.1f}%")
        d.metric("% Primary care", f"{k['pct_phc']:.1f}%")
        e.metric("% Hospitals", f"{k['pct_hosp']:.1f}%")

        plot_df = fac_focus.dropna(subset=["lat", "lon"]).copy()

        if len(plot_df) > 30000:
            st.warning(f"Map has {len(plot_df):,} points. Showing 30,000 for performance.")
            if not st.checkbox("Show all points (may be slow)", value=False):
                plot_df = plot_df.sample(n=30000, random_state=42)

        types_present = sorted(plot_df["tp_stbl"].dropna().unique().tolist())
        hosp_types = [t for t in types_present if _is_hospital_type(t)]
        non_hosp_types = [t for t in types_present if t not in hosp_types]

        color_map = {}
        for t in hosp_types:
            color_map[str(t)] = IDB_COLORS["orange"]
        for t in non_hosp_types:
            color_map[str(t)] = IDB_COLORS["navy"]

        j = 0
        for t in types_present:
            if str(t) not in color_map:
                color_map[str(t)] = PLOTLY_DISCRETE[j % len(PLOTLY_DISCRETE)]
                j += 1

        # Plot primary care first, then hospitals on top (better visibility)
        df_primary = plot_df[plot_df["tp_stbl"].isin(non_hosp_types)].copy() if non_hosp_types else plot_df.iloc[0:0].copy()
        df_hosp = plot_df[plot_df["tp_stbl"].isin(hosp_types)].copy() if hosp_types else plot_df.iloc[0:0].copy()

        base_df = df_primary if len(df_primary) else plot_df

        fig = px.scatter_mapbox(
            base_df,
            lat="lat",
            lon="lon",
            color="tp_stbl",
            color_discrete_map=color_map,
            hover_name="name",
            hover_data=[
                "country", "tp_stbl",
                "fix_dl_mbps", "fix_ul_mbps", "mob_dl_mbps", "mob_ul_mbps",
                "selected_speed_mbps", "high_speed", "speed_source"
            ],
            zoom=3.4 if focus_country != "All LAC" else 2.3,
            height=650,
            opacity=0.90,
        )
        fig.update_traces(marker={"size": 4})

        if len(df_hosp):
            fig2 = px.scatter_mapbox(
                df_hosp,
                lat="lat",
                lon="lon",
                color="tp_stbl",
                color_discrete_map=color_map,
                hover_name="name",
                hover_data=[
                    "country", "tp_stbl",
                    "fix_dl_mbps", "fix_ul_mbps", "mob_dl_mbps", "mob_ul_mbps",
                    "selected_speed_mbps", "high_speed", "speed_source"
                ],
                opacity=0.95,
            )
            fig2.update_traces(marker={"size": 5})
            for tr in fig2.data:
                fig.add_trace(tr)

        fig.update_layout(
            mapbox_style="open-street-map",
            margin=dict(l=0, r=0, t=0, b=0),
            legend_title_text="Facility type"
        )
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 2: Country Distributions
# =============================================================================
with tabs[1]:
    st.subheader("Country distributions (LAC-wide)")
    st.caption("This tab is LAC-wide (not affected by Country focus). Distributions are sampled for performance.")

    df_speed = fac_global_hs.dropna(subset=["selected_speed_mbps"]).copy()
    if df_speed.empty:
        st.warning("No facilities with selected speed under current global filters.")
    else:
        plot_kind = st.radio("Distribution plot", ["Box (sampled)", "Violin (sampled)"], horizontal=True, index=1)
        sampled = sample_for_distribution(fac_global_hs)

        if plot_kind.startswith("Box"):
            fig = px.box(sampled, x="country", y="selected_speed_mbps", points=False, height=520)
        else:
            fig = px.violin(sampled, x="country", y="selected_speed_mbps", box=True, points=False, height=520)

        fig.update_layout(xaxis_title="", yaxis_title="Selected speed (Mbps)", margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

        tbl = country_stats_table(fac_global_hs, threshold=float(country_threshold))
        st.markdown("**Country summary (selected speed)**")
        st.dataframe(tbl, use_container_width=True)

# =============================================================================
# TAB 3: Regional Gaps
# =============================================================================
with tabs[2]:
    st.subheader("Regional gaps (LAC-wide)")
    st.caption("Gap = group statistic − LAC reference. Negative = below reference; Positive = above reference.")

    level = st.selectbox("Group level", ["Country", "ADM1", "ADM2"], index=0)
    ref_stat = st.selectbox("LAC reference", ["Median", "P25"], index=0)
    group_stat = st.selectbox("Group statistic", ["Median", "P25"], index=0)

    country_filter = None
    if level in ["ADM1", "ADM2"]:
        st.markdown("**Optional filter (recommended for ADM1/ADM2):**")
        country_filter = st.multiselect("Filter to countries", options=sorted(fac_global_hs["country"].dropna().unique().tolist()), default=[])

    gap_tbl, lac_ref = regional_gaps(fac_global_hs, level, ref_stat, group_stat, country_filter=country_filter)
    st.caption(f"LAC reference ({ref_stat}) = {lac_ref:.2f} Mbps")

    show_only = st.radio("Display", ["All groups", "Only below reference", "Only above reference"], horizontal=True, index=0)
    n_groups = st.slider("Number of groups to display", 10, 400, 80, step=10)

    plot_tbl = gap_tbl.copy()
    if show_only == "Only below reference":
        plot_tbl = plot_tbl[plot_tbl["Gap (Mbps)"] < 0].copy()
    elif show_only == "Only above reference":
        plot_tbl = plot_tbl[plot_tbl["Gap (Mbps)"] > 0].copy()

    plot_tbl = plot_tbl.sort_values("Gap (Mbps)").head(n_groups).copy()

    if plot_tbl.empty:
        st.warning("No groups to plot under current selection.")
    else:
        if level == "Country":
            plot_tbl["Group"] = plot_tbl["country"]
        elif level == "ADM1":
            plot_tbl["Group"] = plot_tbl["ADM1_EN"].fillna("Unknown ADM1") + " — " + plot_tbl["country"]
        else:
            plot_tbl["Group"] = plot_tbl["ADM2_EN"].fillna("Unknown ADM2") + " — " + plot_tbl["country"]

        fig = px.bar(
            plot_tbl,
            x="Gap (Mbps)",
            y="Group",
            orientation="h",
            color="Gap direction",
            color_discrete_map={
                "Below reference": IDB_COLORS["orange"],
                "Above reference": IDB_COLORS["teal"],
            },
            height=760,
        )
        fig.update_layout(xaxis_title="Gap (Mbps)", yaxis_title="", margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Regional gaps table**")
    st.dataframe(gap_tbl, use_container_width=True)

# =============================================================================
# TAB 4: Urban vs Rural
# =============================================================================
with tabs[3]:
    st.subheader("Urban vs rural (LAC-wide)")
    st.caption("Urban = facility within urban polygons. Rural = outside. Not affected by Country focus.")

    if "is_urban" not in fac_global_hs.columns:
        st.error("Missing `is_urban` in prepared facilities. Re-run preprocess.py.")
    else:
        d = fac_global_hs.dropna(subset=["selected_speed_mbps"]).copy()
        if d.empty:
            st.warning("No facilities with speed under current global filters.")
        else:
            d["Area"] = np.where(d["is_urban"], "Urban", "Rural")
            d["Below threshold"] = d["selected_speed_mbps"] < float(country_threshold)

            overall = (
                d.groupby("Area")
                 .agg(
                     **{
                         "N facilities": ("selected_speed_mbps", "size"),
                         "Median speed (Mbps)": ("selected_speed_mbps", "median"),
                         "P25 speed (Mbps)": ("selected_speed_mbps", lambda x: x.quantile(0.25)),
                         "P75 speed (Mbps)": ("selected_speed_mbps", lambda x: x.quantile(0.75)),
                         "% below threshold": ("Below threshold", lambda x: 100 * x.mean()),
                         "% high-speed": ("high_speed", lambda x: 100 * x.mean()),
                     }
                 )
                 .reset_index()
            )

            st.markdown("### Overall: Urban vs Rural")
            st.dataframe(overall, use_container_width=True)

            fig = px.box(d, x="Area", y="selected_speed_mbps", points=False, height=420)
            fig.update_layout(xaxis_title="", yaxis_title="Selected speed (Mbps)", margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### By country: % below threshold (Urban vs Rural)")
            by_country = (
                d.groupby(["country", "Area"])
                 .agg(
                     **{
                         "N facilities": ("selected_speed_mbps", "size"),
                         "Median speed (Mbps)": ("selected_speed_mbps", "median"),
                         "% below threshold": ("Below threshold", lambda x: 100 * x.mean()),
                     }
                 )
                 .reset_index()
                 .rename(columns={"country": "Country"})
            )

            st.dataframe(by_country.sort_values(["% below threshold"], ascending=False), use_container_width=True)

            fig2 = px.bar(by_country, x="Country", y="% below threshold", color="Area", barmode="group", height=480)
            fig2.update_layout(xaxis_title="", yaxis_title=f"% below {country_threshold} Mbps", margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig2, use_container_width=True)

# =============================================================================
# TAB 5: Population Coverage  (ALWAYS SAMPLE SMALL BY DEFAULT)
# =============================================================================
with tabs[4]:
    st.subheader("Population coverage (focus country)")
    st.markdown(
        """
**Objective**: estimate how much population falls within a catchment radius of the selected facilities.
- We buffer facilities by a radius (km), union the buffers, then count/sum population points inside.
- If a population weight column exists, we sum it. Otherwise we count points (proxy).
        """
    )

    if focus_country == "All LAC":
        st.warning("For performance, Population Coverage requires selecting a single focus country.")
        st.stop()

    pop_gdf = load_population_points()

    pop_weight_col = meta.get("population_weight_col", None)
    if "country" not in pop_gdf.columns:
        st.error("population_points.geoparquet is missing `country`. Re-run preprocess.py.")
        st.stop()

    radius_km = st.slider("Catchment radius (km)", min_value=1, max_value=100, value=10, step=1)
    cov_fac_mode = st.radio("Facilities used for buffers", ["All filtered facilities", "High-speed only"], horizontal=True)
    high_speed_only = (cov_fac_mode == "High-speed only")

    fac_for_cov = fac_focus.copy()
    if high_speed_only:
        fac_for_cov = fac_for_cov[fac_for_cov["high_speed"]].copy()

    pop_for_cov_full = pop_gdf[pop_gdf["country"] == focus_country].copy()
    if pop_for_cov_full.empty:
        st.warning("No population points found for the selected country.")
        st.stop()

    # Always sample for stability (default small)
    st.markdown("### Performance settings (recommended to keep defaults)")
    ALWAYS_SAMPLE = st.checkbox("Use sampling (recommended)", value=True)
    DEFAULT_SAMPLE = 250_000
    HARD_CAP = 600_000  # keep it safe for Streamlit Cloud

    if ALWAYS_SAMPLE:
        sample_n = st.slider("Population sample size", 50_000, HARD_CAP, min(DEFAULT_SAMPLE, len(pop_for_cov_full)), step=50_000)
        pop_for_cov = pop_for_cov_full.sample(n=min(sample_n, len(pop_for_cov_full)), random_state=42).copy()
        st.caption(f"Using a random sample of {len(pop_for_cov):,} population points (out of {len(pop_for_cov_full):,}).")
    else:
        # still enforce hard cap
        if len(pop_for_cov_full) > HARD_CAP:
            st.warning(f"Too many population points ({len(pop_for_cov_full):,}). Sampling is required on Streamlit Cloud.")
            pop_for_cov = pop_for_cov_full.sample(n=HARD_CAP, random_state=42).copy()
        else:
            pop_for_cov = pop_for_cov_full

    payload = {
        "focus_country": focus_country,
        "types": sorted(sel_types),
        "require_speed": require_speed,
        "high_speed_only": high_speed_only,
        "direction": direction,
        "network_mode": network_mode,
        "radius_km": float(radius_km),
        "pop_points_used": int(len(pop_for_cov)),
        "hs_mode": hs_mode,
        "quantile_spec": quantile_spec,
        "threshold": float(country_threshold),
    }
    filter_hash = _stable_hash(payload)

    with st.spinner("Computing coverage..."):
        overall, cov_tbl = population_coverage_cached(
            filter_hash=filter_hash,
            lac_crs_str=LAC_CRS,
            radius_km=float(radius_km),
            pop_weight_col=pop_weight_col,
            _fac_gdf=fac_for_cov[["geometry"]].copy(),
            _pop_gdf=pop_for_cov[["geometry", "country"] + ([pop_weight_col] if pop_weight_col and pop_weight_col in pop_for_cov.columns else [])].copy(),
        )

    a, b, c = st.columns(3)
    if str(overall.get("metric", "")).startswith("sum"):
        a.metric("Covered population (sample)", f"{overall.get('covered', 0):,.0f}")
        b.metric("Total population (sample)", f"{overall.get('total', 0):,.0f}")
    else:
        a.metric("Covered population points (sample)", f"{overall.get('covered', 0):,}")
        b.metric("Total population points (sample)", f"{overall.get('total', 0):,}")
    c.metric("% covered (sample)", f"{overall.get('pct_covered', 0.0):.1f}%")

    covered = float(overall.get("covered", 0))
    total = float(overall.get("total", 0))
    not_cov = max(total - covered, 0)

    pie_df = pd.DataFrame({"Status": ["Covered", "Not covered"], "Value": [covered, not_cov]})
    pie_fig = px.pie(
        pie_df,
        names="Status",
        values="Value",
        title="Coverage in sample: covered vs not covered",
        color_discrete_sequence=[IDB_COLORS["teal"], IDB_COLORS["gray"]],
    )
    st.plotly_chart(pie_fig, use_container_width=True)

    st.markdown("### Note on interpretation")
    st.caption(
        "Coverage is computed using a sampled set of population points to keep the app stable. "
        "Use it as an indicative signal for relative comparisons (e.g., different radii or facility subsets)."
    )

st.caption("")
