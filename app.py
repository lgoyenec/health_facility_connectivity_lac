# app.py
# Run:
#   streamlit run app.py

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(
    page_title="Health facility connectivity (LAC)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Constants / style
# ---------------------------
IDB_COLORS = {
    "navy": "#002D72",     # IDB-ish dark blue
    "orange": "#FF6A13",   # contrast for hospitals
    "teal": "#00A3AD",
    "gray": "#6B7280",
    "light": "#F3F4F6",
    "red": "#C81E1E",
    "green": "#0E9F6E",
}

MAP_DEFAULT_CAP = 5_000
MAP_HARD_CAP = 15_000  # never send more than this to browser

POP_DEFAULT_CAP = 150_000  # population points sampled for the map/estimates
FAC_BUFFER_CAP = 2_000     # facilities used for buffers to prevent OOM

# ---------------------------
# Helpers
# ---------------------------
def _safe_pct(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return 100.0 * num / den


def _is_hospital_type(tp: str) -> bool:
    if tp is None:
        return False
    s = str(tp).lower()
    return ("hosp" in s) or ("hospital" in s)


def _hash_filters(items) -> str:
    s = json.dumps(items, sort_keys=True, default=str)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


@st.cache_resource
def load_metadata():
    p = Path("./metadata.json")
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_resource
def load_facilities():
    return gpd.read_parquet("./facilities_prepared.geoparquet")


@st.cache_resource
def load_countries():
    return gpd.read_parquet("./countries.geoparquet")


@st.cache_resource
def load_population_points():
    return gpd.read_parquet("./population_points.geoparquet")


def add_selected_speed(df: pd.DataFrame, direction: str, network_mode: str) -> pd.DataFrame:
    """
    Adds:
      - selected_speed_mbps
      - has_fix, has_mob, has_speed
      - speed_source
    Uses ONLY columns that preprocessing prepared.
    """
    out = df.copy()

    fix = "fix_dl_mbps" if direction == "Download" else "fix_ul_mbps"
    mob = "mob_dl_mbps" if direction == "Download" else "mob_ul_mbps"

    # ensure numeric (robust)
    for c in [fix, mob, "fix_dl_mbps", "fix_ul_mbps", "mob_dl_mbps", "mob_ul_mbps"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out["has_fix"] = out[fix].notna()
    out["has_mob"] = out[mob].notna()
    out["has_speed"] = out["has_fix"] | out["has_mob"]

    # Selected speed logic
    if network_mode == "Fixed":
        out["selected_speed_mbps"] = out[fix]
    elif network_mode == "Mobile":
        out["selected_speed_mbps"] = out[mob]
    elif network_mode == "Best available (max)":
        out["selected_speed_mbps"] = np.nanmax(np.vstack([out[fix].to_numpy(), out[mob].to_numpy()]), axis=0)
        out.loc[~out["has_speed"], "selected_speed_mbps"] = np.nan
    elif network_mode == "Both (min, conservative)":
        out["selected_speed_mbps"] = np.nanmin(np.vstack([out[fix].to_numpy(), out[mob].to_numpy()]), axis=0)
        out.loc[~out["has_speed"], "selected_speed_mbps"] = np.nan
    else:
        out["selected_speed_mbps"] = np.nan

    # Speed source category
    def _src(row):
        if row["has_fix"] and row["has_mob"]:
            return "Fixed+Mobile"
        if row["has_fix"]:
            return "Fixed only"
        if row["has_mob"]:
            return "Mobile only"
        return "No data"

    out["speed_source"] = out.apply(_src, axis=1)
    return out


def add_high_speed_flag(
    df: pd.DataFrame,
    mode: str,
    quantile_kind: str,
    type_scope: str,
    cutoff_by_type: dict,
) -> pd.DataFrame:
    """
    Adds boolean high_speed following your dashboard rules.

    IMPORTANT: Quantiles are computed WITHIN EACH COUNTRY and facility type.
    """
    out = df.copy()
    out["high_speed"] = False

    s = out["selected_speed_mbps"]

    # no speed -> never high speed
    out.loc[~out["has_speed"], "high_speed"] = False

    if mode == "Quantiles":
        q = 0.75 if quantile_kind == "Quartile (top 25%)" else (2.0 / 3.0)

        # Always compute within each country and tp_stbl (as requested)
        group_cols = ["country", "tp_stbl"] if type_scope == "Within facility type" else ["country"]
        # Compute threshold per group
        th = out.groupby(group_cols, dropna=False)["selected_speed_mbps"].quantile(q)
        th = th.reset_index().rename(columns={"selected_speed_mbps": "q_threshold"})
        out = out.merge(th, on=group_cols, how="left")
        out["high_speed"] = out["has_speed"] & (out["selected_speed_mbps"] >= out["q_threshold"])
        out.drop(columns=["q_threshold"], inplace=True, errors="ignore")

    else:
        # Manual cutoffs per tp_stbl
        out["high_speed"] = False
        for tp, cut in cutoff_by_type.items():
            if cut is None:
                continue
            m = (out["tp_stbl"] == tp) & out["has_speed"]
            out.loc[m, "high_speed"] = out.loc[m, "selected_speed_mbps"] >= float(cut)

    return out


def compute_map_kpis(df: pd.DataFrame) -> dict:
    total = len(df)
    pct_with = _safe_pct(df["has_speed"].sum(), total)
    pct_high = _safe_pct(df["high_speed"].sum(), total)

    # mix
    is_h = df["tp_stbl"].apply(_is_hospital_type)
    pct_hosp = _safe_pct(is_h.sum(), total)
    pct_phc = 100.0 - pct_hosp if total > 0 else 0.0

    return dict(
        total=total,
        pct_with_speed=pct_with,
        pct_high_speed=pct_high,
        pct_phc=pct_phc,
        pct_hosp=pct_hosp,
    )


def clean_latlon(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out = out.dropna(subset=["lat", "lon"])
    # LAC-ish bounds
    out = out[out["lat"].between(-60, 35) & out["lon"].between(-120, -20)]
    return out


def pretty_country_table(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    g = df[df["has_speed"]].copy()
    if g.empty:
        return pd.DataFrame(columns=[
            "Country", "Facilities (with speed)", "Minimum speed (Mbps)", "P25 (Mbps)",
            "Median (Mbps)", "P75 (Mbps)", f"% below {threshold:.1f} Mbps"
        ])

    stats = g.groupby("country")["selected_speed_mbps"].agg(
        count="count",
        min="min",
        median="median",
    )
    p25 = g.groupby("country")["selected_speed_mbps"].quantile(0.25).rename("p25")
    p75 = g.groupby("country")["selected_speed_mbps"].quantile(0.75).rename("p75")
    below = g.assign(below=g["selected_speed_mbps"] < threshold).groupby("country")["below"].mean().rename("pct_below")

    out = stats.join([p25, p75, below]).reset_index()
    out["pct_below"] = out["pct_below"] * 100.0

    out = out.rename(columns={
        "country": "Country",
        "count": "Facilities (with speed)",
        "min": "Minimum speed (Mbps)",
        "p25": "P25 (Mbps)",
        "median": "Median (Mbps)",
        "p75": "P75 (Mbps)",
        "pct_below": f"% below {threshold:.1f} Mbps",
    })

    for c in ["Minimum speed (Mbps)", "P25 (Mbps)", "Median (Mbps)", "P75 (Mbps)"]:
        out[c] = out[c].round(2)
    out[f"% below {threshold:.1f} Mbps"] = out[f"% below {threshold:.1f} Mbps"].round(1)

    out = out.sort_values("Median (Mbps)", ascending=False)
    return out


def chunked_union(polys, chunk_size=250):
    # Avoid huge unary_union memory spikes by unioning in chunks
    from shapely.ops import unary_union
    polys = [p for p in polys if p is not None and not p.is_empty]
    if not polys:
        return None
    chunks = []
    for i in range(0, len(polys), chunk_size):
        chunks.append(unary_union(polys[i:i + chunk_size]))
    return unary_union(chunks)


# ---------------------------
# Disclaimer "popup" gate
# ---------------------------
if "ack_disclaimer" not in st.session_state:
    st.session_state["ack_disclaimer"] = False

if not st.session_state["ack_disclaimer"]:
    st.title("Internal reference dashboard (IDB)")
    st.warning(
        "This dashboard is **for internal use within IDB** only and is **not publicly approved** for external sharing.\n\n"
        "It is intended to support operational design and internal analysis."
    )
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("I understand — continue"):
            st.session_state["ack_disclaimer"] = True
            st.rerun()
    with c2:
        st.caption("If you are not an IDB staff/consultant working on internal operations, please close this page.")
    st.stop()

# ---------------------------
# Load data
# ---------------------------
meta = load_metadata()
fac = load_facilities()
countries_gdf = load_countries()
pop_gdf = load_population_points()

# Basic validation / robustness
for col in ["country", "tp_stbl", "lat", "lon"]:
    if col not in fac.columns:
        st.error(f"Missing required column in facilities_prepared: {col}")
        st.stop()

# ---------------------------
# Sidebar controls
# ---------------------------
st.title("Health facility connectivity in Latin America and the Caribbean")

with st.sidebar:
    st.header("Global controls")
    st.caption("These controls apply to all tabs (speed definition and thresholds).")

    direction = st.radio("Direction", ["Download", "Upload"], index=0)
    network_mode = st.selectbox(
        "Network mode",
        ["Fixed", "Mobile", "Best available (max)", "Both (min, conservative)"],
        index=2,
    )

    st.divider()
    st.subheader("High-speed definition")
    hs_mode = st.radio("Mode", ["Quantiles", "Manual cutoffs"], index=0)

    # facility types
    facility_types = sorted([x for x in fac["tp_stbl"].dropna().unique().tolist()])[:20]

    if hs_mode == "Quantiles":
        quantile_kind = st.selectbox("Quantile", ["Tercile (top 33%)", "Quartile (top 25%)"], index=0)
        type_scope = st.selectbox("Compute quantiles", ["Within facility type", "Overall (ignore type)"], index=0)
        manual_cutoffs = {}
    else:
        quantile_kind = "Tercile (top 33%)"
        type_scope = "Within facility type"
        st.caption("Set a Mbps cutoff per facility type.")
        manual_cutoffs = {}
        # show at most 2 most common types (your earlier constraint)
        top_types = fac["tp_stbl"].value_counts().head(2).index.tolist()
        for tp in top_types:
            manual_cutoffs[tp] = st.number_input(f"Cutoff for {tp} (Mbps)", min_value=0.0, value=10.0, step=1.0)

    st.divider()
    st.header("Country focus")
    st.caption("This filter applies only to: *Facilities Map* and *Population Coverage*.")

    all_countries = sorted([c for c in fac["country"].dropna().unique().tolist()])
    focus_country = st.selectbox("Focus country", ["All LAC"] + all_countries, index=0)

    # Map-specific filters
    st.divider()
    st.subheader("Map filters")
    st.caption("Applied only in Facilities Map tab.")

    type_filter = st.multiselect("Facility type", options=sorted(fac["tp_stbl"].dropna().unique().tolist()),
                                default=sorted(fac["tp_stbl"].dropna().unique().tolist()))
    only_high_speed = st.checkbox("Show only high-speed facilities", value=False)

    if st.button("Clear filters"):
        # keep disclaimer acknowledged
        keep_ack = st.session_state.get("ack_disclaimer", True)
        st.session_state.clear()
        st.session_state["ack_disclaimer"] = keep_ack
        st.rerun()

# ---------------------------
# Derived data (fast, vectorized)
# ---------------------------
fac0 = add_selected_speed(fac, direction, network_mode)
fac1 = add_high_speed_flag(
    fac0,
    mode=hs_mode,
    quantile_kind=("Quartile (top 25%)" if quantile_kind.startswith("Quartile") else "Tercile (top 33%)"),
    type_scope=type_scope,
    cutoff_by_type=manual_cutoffs,
)

# Focus-country dataframe (ONLY used for tab 1 and tab 5)
if focus_country == "All LAC":
    fac_focus = fac1.copy()
else:
    fac_focus = fac1[fac1["country"] == focus_country].copy()

# Map filters (type + high-speed)
if type_filter:
    fac_focus = fac_focus[fac_focus["tp_stbl"].isin(type_filter)].copy()
if only_high_speed:
    fac_focus = fac_focus[fac_focus["high_speed"]].copy()

# ---------------------------
# Tabs
# ---------------------------
tabs = st.tabs([
    "Facilities Map",
    "Country Distributions",
    "Regional Gaps",
    "Urban vs Rural",
    "Population Coverage",
])

# =============================================================================
# TAB 1: Facilities Map
# =============================================================================
with tabs[0]:
    st.markdown(
        """
**How to use this tab**
- Choose **Direction** and **Network mode** in the sidebar (download/upload; fixed/mobile/best/min).
- Define **High-speed** using **Quantiles** (within country & type) or **Manual cutoffs**.
- Use **Focus country** to view one country or all LAC.
- Use **Map filters** to restrict facility types or show only high-speed.
        """
    )

    # KPIs must reflect current filtered fac_focus
    k = compute_map_kpis(fac_focus)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total facilities", f"{k['total']:,}")
    c2.metric("% with any speed", f"{k['pct_with_speed']:.1f}%")
    c3.metric("% high-speed", f"{k['pct_high_speed']:.1f}%")
    c4.metric("% Primary care", f"{k['pct_phc']:.1f}%")
    c5.metric("% Hospitals", f"{k['pct_hosp']:.1f}%")

    df_map = clean_latlon(fac_focus)
    if df_map.empty:
        st.warning("No facilities with valid lat/lon for the current selection.")
        st.stop()

    # Keep hover light to avoid OOM/crashes
    extended_hover = st.checkbox("Show extended hover details (slower)", value=False)
    hover_basic = {
        "country": True,
        "tp_stbl": True,
        "selected_speed_mbps": ":.2f",
        "high_speed": True,
    }
    hover_extended = {
        "country": True,
        "tp_stbl": True,
        "fix_dl_mbps": ":.2f",
        "fix_ul_mbps": ":.2f",
        "mob_dl_mbps": ":.2f",
        "mob_ul_mbps": ":.2f",
        "selected_speed_mbps": ":.2f",
        "high_speed": True,
        "speed_source": True,
    }
    hover_data = hover_extended if extended_hover else hover_basic

    # Downsample aggressively
    n = len(df_map)
    st.caption(f"Facilities in selection: {n:,} (map will sample for stability if needed).")

    # Reset show_more when focus_country changes
    if "prev_focus_country" not in st.session_state:
        st.session_state["prev_focus_country"] = focus_country
    if st.session_state["prev_focus_country"] != focus_country:
        st.session_state["show_more_points"] = False
        st.session_state["prev_focus_country"] = focus_country

    if n > MAP_DEFAULT_CAP:
        show_more = st.checkbox(
            f"Show up to {MAP_HARD_CAP:,} points (may be slow)",
            value=st.session_state.get("show_more_points", False),
            key="show_more_points",
        )
        if show_more:
            if n > MAP_HARD_CAP:
                df_map = df_map.sample(n=MAP_HARD_CAP, random_state=42)
        else:
            df_map = df_map.sample(n=MAP_DEFAULT_CAP, random_state=42)

    # Two-layer plot: primary care below, hospitals on top
    df_map["is_hosp"] = df_map["tp_stbl"].apply(_is_hospital_type)
    df_phc = df_map[~df_map["is_hosp"]].copy()
    df_hosp = df_map[df_map["is_hosp"]].copy()

    center_lat = float(df_map["lat"].mean())
    center_lon = float(df_map["lon"].mean())
    zoom = 4.0 if focus_country != "All LAC" else 2.3

    # Use plotly scatter_map (MapLibre) instead of deprecated scatter_mapbox
    # This reduces some mapbox-related baggage and is future-proof.
    if len(df_phc) == 0:
        base_df = df_map.copy()
    else:
        base_df = df_phc.copy()

    fig = px.scatter_map(
        base_df,
        lat="lat",
        lon="lon",
        color_discrete_sequence=[IDB_COLORS["navy"]],
        hover_name="name",
        hover_data=hover_data,
        opacity=0.85,
        zoom=zoom,
        center={"lat": center_lat, "lon": center_lon},
        height=650,
    )
    fig.update_traces(marker={"size": 4})

    if len(df_hosp) > 0:
        fig_h = px.scatter_map(
            df_hosp,
            lat="lat",
            lon="lon",
            color_discrete_sequence=[IDB_COLORS["orange"]],
            hover_name="name",
            hover_data=hover_data,
            opacity=0.95,
        )
        fig_h.update_traces(marker={"size": 5})
        for tr in fig_h.data:
            fig.add_trace(tr)

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h"),
        showlegend=False,
    )

    st.plotly_chart(fig, width="stretch")

# =============================================================================
# TAB 2: Country Distributions (NO CHOROPLETH to avoid MessageSizeError)
# =============================================================================
with tabs[1]:
    st.markdown(
        """
**What this tab shows**
- Country-by-country distribution of the **selected speed** (only facilities with speed).
- Summary table: minimum, P25, median, P75, and % below a user-defined threshold.

**Note**: The optional choropleth was removed because it repeatedly exceeded Streamlit’s message size limit.
        """
    )

    threshold = st.number_input("Low-connectivity threshold (Mbps)", min_value=0.0, value=5.0, step=1.0)

    df = fac1[fac1["has_speed"]].copy()
    if df.empty:
        st.warning("No facilities with speed available.")
        st.stop()

    # Lighter plot: sample per country
    max_per_country = st.slider("Max facilities per country in plot (performance)", 200, 5000, 800, 100)
    df_plot = df.groupby("country", group_keys=False).apply(
        lambda g: g.sample(n=min(len(g), max_per_country), random_state=42)
    )

    chart_type = st.selectbox("Chart type", ["Violin (recommended)", "Box"], index=0)

    if chart_type.startswith("Violin"):
        fig = px.violin(
            df_plot,
            x="country",
            y="selected_speed_mbps",
            color_discrete_sequence=[IDB_COLORS["teal"]],
            box=True,
            points=False,
        )
    else:
        fig = px.box(
            df_plot,
            x="country",
            y="selected_speed_mbps",
            points=False,
        )

    fig.update_layout(
        xaxis_title="Country",
        yaxis_title="Selected speed (Mbps)",
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Country summary table")
    tbl = pretty_country_table(df, threshold)
    st.dataframe(tbl, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 3: Regional Gaps
# =============================================================================
with tabs[2]:
    st.markdown(
        """
**What this tab shows**
Connectivity gaps relative to a LAC reference statistic:
- Choose the group level: **Country / ADM1 / ADM2**
- Choose the reference: **LAC median** or **LAC P25**
- Gap = (group statistic) − (LAC reference)
Negative gaps are **below** the reference; positive gaps are **above**.
        """
    )

    group_level = st.selectbox("Group level", ["Country", "ADM1", "ADM2"], index=0)  # default Country
    ref_kind = st.selectbox("LAC reference", ["Median", "P25"], index=0)
    top_n = st.slider("Show top N (by absolute gap)", 5, 50, 20, 5)

    df = fac1[fac1["has_speed"]].copy()
    if df.empty:
        st.warning("No facilities with speed available.")
        st.stop()

    ref_val = df["selected_speed_mbps"].median() if ref_kind == "Median" else df["selected_speed_mbps"].quantile(0.25)

    if group_level == "Country":
        key_cols = ["country"]
        label_col = "country"
    elif group_level == "ADM1":
        key_cols = ["country", "ADM1_EN"]
        label_col = "ADM1_EN"
    else:
        key_cols = ["country", "ADM2_EN"]
        label_col = "ADM2_EN"

    agg = df.groupby(key_cols)["selected_speed_mbps"].median().reset_index().rename(columns={"selected_speed_mbps": "Group median (Mbps)"})
    agg["LAC reference (Mbps)"] = ref_val
    agg["Gap (Mbps)"] = agg["Group median (Mbps)"] - ref_val

    # pick top by absolute gap
    agg["abs_gap"] = agg["Gap (Mbps)"].abs()
    view = agg.sort_values("abs_gap", ascending=False).head(top_n).copy()

    # build label
    if group_level == "Country":
        view["Group"] = view["country"]
    else:
        view["Group"] = view["country"].astype(str) + " — " + view[label_col].astype(str)

    # plot with two colors
    view["Direction"] = np.where(view["Gap (Mbps)"] >= 0, "Above reference", "Below reference")

    fig = px.bar(
        view.sort_values("Gap (Mbps)"),
        x="Gap (Mbps)",
        y="Group",
        color="Direction",
        color_discrete_map={
            "Above reference": IDB_COLORS["green"],
            "Below reference": IDB_COLORS["red"],
        },
        orientation="h",
        height=520,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")

    st.dataframe(
        view.drop(columns=["abs_gap"]),
        use_container_width=True,
        hide_index=True,
    )

# =============================================================================
# TAB 4: Urban vs Rural (include country stats)
# =============================================================================
with tabs[3]:
    st.markdown(
        """
**What this tab shows**
Connectivity comparison for **urban vs rural facilities** (based on whether the facility point falls within the urban polygons).
We show:
- Overall urban vs rural summary
- Country-level urban vs rural medians and % below threshold
        """
    )

    threshold_ur = st.number_input("Below-threshold cutoff (Mbps) for this tab", min_value=0.0, value=5.0, step=1.0)

    df = fac1[fac1["has_speed"]].copy()
    if df.empty:
        st.warning("No facilities with speed available.")
        st.stop()

    if "is_urban" not in df.columns:
        st.error("facilities_prepared.geoparquet is missing `is_urban` (should be created in preprocess.py).")
        st.stop()

    df["Area"] = np.where(df["is_urban"], "Urban", "Rural")
    df["below"] = df["selected_speed_mbps"] < threshold_ur

    # Overall KPIs
    overall = df.groupby("Area")["selected_speed_mbps"].agg(
        Facilities="count",
        Median="median",
    ).reset_index()
    overall["% below threshold"] = df.groupby("Area")["below"].mean().values * 100.0
    overall["Median"] = overall["Median"].round(2)
    overall["% below threshold"] = overall["% below threshold"].round(1)

    st.subheader("Overall urban vs rural")
    st.dataframe(overall, use_container_width=True, hide_index=True)

    fig = px.box(
        df.sample(n=min(len(df), 20_000), random_state=42),
        x="Area",
        y="selected_speed_mbps",
        points=False,
        color="Area",
        color_discrete_map={"Urban": IDB_COLORS["teal"], "Rural": IDB_COLORS["gray"]},
        height=420,
    )
    fig.update_layout(xaxis_title="", yaxis_title="Selected speed (Mbps)")
    st.plotly_chart(fig, width="stretch")

    # Country stats
    st.subheader("Country-level urban vs rural summary")
    ctab = (
        df.groupby(["country", "Area"])
        .agg(
            Facilities=("selected_speed_mbps", "count"),
            Median=("selected_speed_mbps", "median"),
            P25=("selected_speed_mbps", lambda x: x.quantile(0.25)),
            P75=("selected_speed_mbps", lambda x: x.quantile(0.75)),
            Pct_below=("below", "mean"),
        )
        .reset_index()
    )
    ctab["Median"] = ctab["Median"].round(2)
    ctab["P25"] = ctab["P25"].round(2)
    ctab["P75"] = ctab["P75"].round(2)
    ctab["Pct_below"] = (ctab["Pct_below"] * 100.0).round(1)

    ctab = ctab.rename(columns={
        "country": "Country",
        "Area": "Area",
        "Facilities": "Facilities (with speed)",
        "Median": "Median (Mbps)",
        "P25": "P25 (Mbps)",
        "P75": "P75 (Mbps)",
        "Pct_below": f"% below {threshold_ur:.1f} Mbps",
    })

    st.dataframe(ctab, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 5: Population Coverage (focus country only, robust + sampled)
# =============================================================================
with tabs[4]:
    st.subheader("Population coverage")
    st.markdown(
        """
**Objective**: estimate how much population falls within a catchment radius of facilities.
- We buffer facilities by a radius (km), then count/sum population points inside.
- This is an **approximation** based on point population data.

**Important**: This tab uses the **Focus country** selector in the sidebar.
        """
    )

    if focus_country == "All LAC":
        st.warning("For performance, please select a single Focus country to compute population coverage.")
        st.stop()

    # Filter facilities + population to focus country
    fac_cov = fac1[fac1["country"] == focus_country].copy()
    fac_cov = fac_cov[fac_cov["has_speed"]].copy()

    if fac_cov.empty:
        st.warning("No facilities with speed for the selected country. Coverage cannot be computed.")
        st.stop()

    if "country" not in pop_gdf.columns:
        st.error("population_points.geoparquet must include `country` (precomputed or joined in preprocessing).")
        st.stop()

    pop_cov = pop_gdf[pop_gdf["country"] == focus_country].copy()

    if pop_cov.empty:
        st.warning("No population points found for the selected country.")
        st.stop()

    # detect weight col from metadata
    pop_weight_col = meta.get("population_weight_col", None)
    has_weight = pop_weight_col is not None and pop_weight_col in pop_cov.columns

    radius_km = st.slider("Catchment radius (km)", 1, 50, 10, 1)

    # sample pop points (hard) to prevent crashes
    if len(pop_cov) > POP_DEFAULT_CAP:
        st.info(f"Population points: {len(pop_cov):,}. Using a sample of {POP_DEFAULT_CAP:,} for stability.")
        pop_cov = pop_cov.sample(n=POP_DEFAULT_CAP, random_state=42)

    # sample facilities used for buffers
    if len(fac_cov) > FAC_BUFFER_CAP:
        st.info(f"Facilities: {len(fac_cov):,}. Using a sample of {FAC_BUFFER_CAP:,} for buffering to keep the app stable.")
        fac_cov = fac_cov.sample(n=FAC_BUFFER_CAP, random_state=42)

    # caching key
    cov_key = _hash_filters({
        "country": focus_country,
        "radius_km": radius_km,
        "direction": direction,
        "mode": network_mode,
        "hs_mode": hs_mode,
        "type_filter": type_filter,
        "only_high_speed": only_high_speed,
    })

    @st.cache_data(show_spinner=False)
    def compute_coverage_cached(_key: str) -> dict:
        # Rebuild small GDFs inside cache to avoid hashing GeoDataFrames
        f = fac1[fac1["country"] == focus_country].copy()
        f = f[f["has_speed"]].copy()
        f = f[["geometry", "country"]].dropna(subset=["geometry"]).copy()

        # same sampling as above, deterministic
        if len(f) > FAC_BUFFER_CAP:
            f = f.sample(n=FAC_BUFFER_CAP, random_state=42)

        p = pop_gdf[pop_gdf["country"] == focus_country].copy()
        p = p[["geometry", "country"] + ([pop_weight_col] if (pop_weight_col and pop_weight_col in pop_gdf.columns) else [])]
        p = p.dropna(subset=["geometry"]).copy()
        if len(p) > POP_DEFAULT_CAP:
            p = p.sample(n=POP_DEFAULT_CAP, random_state=42)

        # Use EPSG:3857 for buffering (meters)
        f_m = f.to_crs(3857)
        buffers = f_m.geometry.buffer(radius_km * 1000.0)

        # Chunked union (prevents memory spike)
        union_poly = chunked_union(list(buffers), chunk_size=250)
        if union_poly is None or union_poly.is_empty:
            return {"covered_points": 0, "total_points": len(p), "covered_pop": None, "total_pop": None}

        # Convert union back to pop CRS for containment test
        union_gdf = gpd.GeoDataFrame(geometry=[union_poly], crs=3857).to_crs(p.crs)

        # Bounding-box prefilter (fast)
        minx, miny, maxx, maxy = union_gdf.total_bounds
        p_bb = p.cx[minx:maxx, miny:maxy].copy()
        if p_bb.empty:
            return {"covered_points": 0, "total_points": len(p), "covered_pop": None, "total_pop": None}

        # Exact within
        covered_mask = p_bb.within(union_gdf.geometry.iloc[0])
        covered_points = int(covered_mask.sum())
        total_points = int(len(p))

        if pop_weight_col and pop_weight_col in p.columns:
            total_pop = float(pd.to_numeric(p[pop_weight_col], errors="coerce").fillna(0).sum())
            covered_pop = float(pd.to_numeric(p_bb.loc[covered_mask, pop_weight_col], errors="coerce").fillna(0).sum())
            return {
                "covered_points": covered_points,
                "total_points": total_points,
                "covered_pop": covered_pop,
                "total_pop": total_pop,
            }

        return {
            "covered_points": covered_points,
            "total_points": total_points,
            "covered_pop": None,
            "total_pop": None,
        }

    res = compute_coverage_cached(cov_key)

    # Output KPIs + pie
    if has_weight and res["total_pop"] and res["total_pop"] > 0:
        covered = res["covered_pop"]
        total = res["total_pop"]
        pct = 100.0 * covered / total if total > 0 else 0.0

        a, b, c = st.columns(3)
        a.metric("Estimated covered population", f"{covered:,.0f}")
        b.metric("Estimated total population (sample-based)", f"{total:,.0f}")
        c.metric("% covered (approx.)", f"{pct:.1f}%")

        fig = go.Figure(
            data=[go.Pie(
                labels=["Covered", "Not covered"],
                values=[covered, max(total - covered, 0)],
                hole=0.45,
            )]
        )
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

        st.caption("Interpretation: approximation based on population point weights and sampled points for performance.")
    else:
        covered = res["covered_points"]
        total = res["total_points"]
        pct = 100.0 * covered / total if total > 0 else 0.0

        a, b, c = st.columns(3)
        a.metric("Covered population points", f"{covered:,}")
        b.metric("Total population points (sample)", f"{total:,}")
        c.metric("% points covered (proxy)", f"{pct:.1f}%")

        fig = go.Figure(
            data=[go.Pie(
                labels=["Covered points", "Not covered points"],
                values=[covered, max(total - covered, 0)],
                hole=0.45,
            )]
        )
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

        st.warning(
            "No population weight column was detected. Results show **# population points covered** (proxy), not people."
        )
