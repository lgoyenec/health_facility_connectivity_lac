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
# Page setup (call once)
# ---------------------------
st.set_page_config(
    page_title="Health facility connectivity in Latin America and Caribbean",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Constants / style
# ---------------------------
IDB_COLORS = {
    "navy": "#002D72",     # PHC
    "orange": "#FF6A13",   # Hospitals
    "teal": "#00A3AD",
    "gray": "#6B7280",
    "red": "#C81E1E",
    "green": "#0E9F6E",
}

# Map caps (stability)
MAP_DEFAULT_CAP = 5_000
MAP_HARD_CAP = 15_000  # never send more than this to the browser

DISCLAIMER_TEXT = (
    "This dashboard is **for internal use within IDB** only and is **not publicly approved** for external sharing.\n\n"
    "It is intended to support operational design and internal analysis."
)

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


def _compute_center_zoom_from_points(df: pd.DataFrame):
    """
    Best-effort center/zoom from lat/lon extent (no external mapbox token).
    """
    if df.empty or ("lat" not in df.columns) or ("lon" not in df.columns):
        return dict(lat=-15, lon=-60), 3

    d = df.dropna(subset=["lat", "lon"])
    if d.empty:
        return dict(lat=-15, lon=-60), 3

    lat_min, lat_max = float(d["lat"].min()), float(d["lat"].max())
    lon_min, lon_max = float(d["lon"].min()), float(d["lon"].max())
    center = dict(lat=(lat_min + lat_max) / 2.0, lon=(lon_min + lon_max) / 2.0)

    # heuristic zoom based on bounding box width in degrees
    lat_rng = max(lat_max - lat_min, 1e-6)
    lon_rng = max(lon_max - lon_min, 1e-6)
    extent = max(lat_rng, lon_rng)

    if extent > 80:
        zoom = 2
    elif extent > 40:
        zoom = 3
    elif extent > 20:
        zoom = 4
    elif extent > 10:
        zoom = 5
    elif extent > 5:
        zoom = 6
    elif extent > 2.5:
        zoom = 7
    else:
        zoom = 8

    return center, zoom


@st.cache_resource(show_spinner=False)
def load_metadata():
    p = Path("./metadata.json")
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_facilities():
    return gpd.read_parquet("./facilities_prepared.geoparquet")


@st.cache_resource(show_spinner=False)
def load_population_coverage():
    # Precomputed in preprocess.py to avoid heavy spatial ops in Streamlit
    return pd.read_parquet("./population_coverage_by_country.parquet")


def add_selected_speed(df: pd.DataFrame, direction: str, network_mode: str) -> pd.DataFrame:
    """
    Adds:
      - selected_speed_mbps
      - has_fix, has_mob, has_speed
      - speed_source
    """
    out = df.copy()

    fix = "fix_dl_mbps" if direction == "Download" else "fix_ul_mbps"
    mob = "mob_dl_mbps" if direction == "Download" else "mob_ul_mbps"

    # Ensure numeric robustly
    for c in [fix, mob, "fix_dl_mbps", "fix_ul_mbps", "mob_dl_mbps", "mob_ul_mbps"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out["has_fix"] = out[fix].notna()
    out["has_mob"] = out[mob].notna()
    out["has_speed"] = out["has_fix"] | out["has_mob"]

    if network_mode == "Fixed":
        out["selected_speed_mbps"] = out[fix]
    elif network_mode == "Mobile":
        out["selected_speed_mbps"] = out[mob]
    elif network_mode == "Best available (max)":
        out["selected_speed_mbps"] = np.nanmax(
            np.vstack([out[fix].to_numpy(), out[mob].to_numpy()]),
            axis=0,
        )
        out.loc[~out["has_speed"], "selected_speed_mbps"] = np.nan
    elif network_mode == "Both (min, conservative)":
        out["selected_speed_mbps"] = np.nanmin(
            np.vstack([out[fix].to_numpy(), out[mob].to_numpy()]),
            axis=0,
        )
        out.loc[~out["has_speed"], "selected_speed_mbps"] = np.nan
    else:
        out["selected_speed_mbps"] = np.nan

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
    cutoff_type_a: float,
    cutoff_type_b: float,
) -> pd.DataFrame:
    out = df.copy()
    out["high_speed"] = False

    # Identify two unique facility types (as provided in prep)
    types = [str(x) for x in out["tp_stbl"].dropna().unique().tolist()]
    types_sorted = sorted(types)
    t_a = types_sorted[0] if len(types_sorted) > 0 else None
    t_b = types_sorted[1] if len(types_sorted) > 1 else None

    if mode == "Quantile":
        q = 0.75 if quantile_kind == "Quartile (Q4 = high-speed)" else (2 / 3)
        if type_scope == "Within each facility type":
            for t in types_sorted:
                m = out["tp_stbl"] == t
                vals = out.loc[m, "selected_speed_mbps"].dropna()
                if len(vals) > 0:
                    thr = float(vals.quantile(q))
                    out.loc[m, "high_speed"] = out.loc[m, "selected_speed_mbps"] >= thr
        else:
            vals = out["selected_speed_mbps"].dropna()
            if len(vals) > 0:
                thr = float(vals.quantile(q))
                out["high_speed"] = out["selected_speed_mbps"] >= thr

    else:  # Manual cutoffs
        if t_a is not None:
            m = out["tp_stbl"] == t_a
            out.loc[m, "high_speed"] = out.loc[m, "selected_speed_mbps"] >= float(cutoff_type_a)
        if t_b is not None:
            m = out["tp_stbl"] == t_b
            out.loc[m, "high_speed"] = out.loc[m, "selected_speed_mbps"] >= float(cutoff_type_b)

    out["high_speed"] = out["high_speed"] & out["has_speed"]
    return out


def _make_facility_map(df: pd.DataFrame, show_all: bool, center: dict, zoom: int):
    """
    Scatter map with stability caps. Uses Plotly scattermapbox.
    Legend is placed inside the map for readability.
    """
    d = df.copy()
    if not show_all and len(d) > MAP_DEFAULT_CAP:
        d = d.sample(n=MAP_DEFAULT_CAP, random_state=42)

    if len(d) > MAP_HARD_CAP:
        d = d.sample(n=MAP_HARD_CAP, random_state=42)

    # Two traces by facility type for clarity
    phc = d[~d["tp_stbl"].apply(_is_hospital_type)].copy()
    hosp = d[d["tp_stbl"].apply(_is_hospital_type)].copy()

    fig = go.Figure()

    if not phc.empty:
        fig.add_trace(
            go.Scattermapbox(
                lat=phc["lat"],
                lon=phc["lon"],
                mode="markers",
                name="Primary health care centers",
                marker=dict(size=6, color=IDB_COLORS["navy"], opacity=0.75),
                customdata=np.stack(
                    [
                        phc.get("name", pd.Series([""] * len(phc))),
                        phc.get("country", pd.Series([""] * len(phc))),
                        phc.get("tp_stbl", pd.Series([""] * len(phc))),
                        phc.get("fix_dl_mbps", pd.Series([np.nan] * len(phc))),
                        phc.get("fix_ul_mbps", pd.Series([np.nan] * len(phc))),
                        phc.get("mob_dl_mbps", pd.Series([np.nan] * len(phc))),
                        phc.get("mob_ul_mbps", pd.Series([np.nan] * len(phc))),
                        phc.get("mob_ul_mbps", pd.Series([np.nan] * len(phc))),
                        phc.get("selected_speed_mbps", pd.Series([np.nan] * len(phc))),
                        phc.get("high_speed", pd.Series([False] * len(phc))),
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Country: %{customdata[1]}<br>"
                    "Type: %{customdata[2]}<br>"
                    "Fixed DL: %{customdata[3]:.2f} Mbps<br>"
                    "Fixed UL: %{customdata[4]:.2f} Mbps<br>"
                    "Mobile DL: %{customdata[5]:.2f} Mbps<br>"
                    "Mobile UL: %{customdata[6]:.2f} Mbps<br>"
                    "<b>Selected speed</b>: %{customdata[8]:.2f} Mbps<br>"
                    "High-speed: %{customdata[9]}<extra></extra>"
                ),
            )
        )

    if not hosp.empty:
        fig.add_trace(
            go.Scattermapbox(
                lat=hosp["lat"],
                lon=hosp["lon"],
                mode="markers",
                name="Hospitals",
                marker=dict(size=6, color=IDB_COLORS["orange"], opacity=0.80),
                customdata=np.stack(
                    [
                        hosp.get("name", pd.Series([""] * len(hosp))),
                        hosp.get("country", pd.Series([""] * len(hosp))),
                        hosp.get("tp_stbl", pd.Series([""] * len(hosp))),
                        hosp.get("fix_dl_mbps", pd.Series([np.nan] * len(hosp))),
                        hosp.get("fix_ul_mbps", pd.Series([np.nan] * len(hosp))),
                        hosp.get("mob_dl_mbps", pd.Series([np.nan] * len(hosp))),
                        hosp.get("mob_ul_mbps", pd.Series([np.nan] * len(hosp))),
                        hosp.get("selected_speed_mbps", pd.Series([np.nan] * len(hosp))),
                        hosp.get("high_speed", pd.Series([False] * len(hosp))),
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Country: %{customdata[1]}<br>"
                    "Type: %{customdata[2]}<br>"
                    "Fixed DL: %{customdata[3]:.2f} Mbps<br>"
                    "Fixed UL: %{customdata[4]:.2f} Mbps<br>"
                    "Mobile DL: %{customdata[5]:.2f} Mbps<br>"
                    "Mobile UL: %{customdata[6]:.2f} Mbps<br>"
                    "<b>Selected speed</b>: %{customdata[7]:.2f} Mbps<br>"
                    "High-speed: %{customdata[8]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(zoom=int(zoom), center=center),
        margin=dict(l=0, r=0, t=0, b=0),
        height=650,
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=0.02,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1,
            font=dict(size=12,color="#111827"),
        ),
    )
    return fig


# =============================================================================
# Disclaimer (non-blocking: fall back when st.dialog is unavailable)
# =============================================================================
if "disclaimer_ack" not in st.session_state:
    st.session_state["disclaimer_ack"] = False


def _render_disclaimer():
    st.markdown("### Internal use notice")
    st.markdown(DISCLAIMER_TEXT)
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("I understand"):
            st.session_state["disclaimer_ack"] = True
            st.rerun()


# Try true pop-up if supported
try:
    if not st.session_state["disclaimer_ack"]:
        dlg = getattr(st, "dialog", None)
        if dlg is not None:

            @st.dialog("Internal use notice")
            def _dlg():
                _render_disclaimer()

            _dlg()
            st.stop()
        else:
            st.warning("Internal use notice")
            _render_disclaimer()
            st.stop()
except Exception:
    if not st.session_state["disclaimer_ack"]:
        st.warning("Internal use notice")
        _render_disclaimer()
        st.stop()

# =============================================================================
# Load prepared data (fast)
# =============================================================================
meta = load_metadata()
fac0 = load_facilities()

# =============================================================================
# Sidebar controls
# =============================================================================
st.sidebar.markdown("## Global controls")
st.sidebar.caption("Apply to all tabs")

direction = st.sidebar.radio("Selected speed direction", ["Download", "Upload"], index=0)
network_mode = st.sidebar.radio(
    "Network mode",
    ["Fixed", "Mobile", "Best available (max)", "Both (min, conservative)"],
    index=2,
)

st.sidebar.markdown("---")
st.sidebar.markdown("## Country-level analysis")
st.sidebar.caption("*This filter applies only to* **Facilities Map** *and* **Population coverage**.")

countries_list = sorted([str(x) for x in fac0["country"].dropna().unique().tolist()])
focus_country = st.sidebar.selectbox("Focus country", ["All LAC"] + countries_list, index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("## Facility filters")
types_list = sorted([str(x) for x in fac0["tp_stbl"].dropna().unique().tolist()])
type_filter = st.sidebar.multiselect("Facility type", options=types_list, default=types_list)

only_high_speed = st.sidebar.checkbox("Show only high-speed facilities (map only)", value=False)

# High-speed definition
st.sidebar.markdown("---")
st.sidebar.markdown("## High-speed definition")

hs_mode = st.sidebar.radio("Mode", ["Quantile", "Manual cutoffs"], index=0)

if hs_mode == "Quantile":
    quantile_kind = st.sidebar.radio(
        "Quantile",
        ["Tercile (top tercile = high-speed)", "Quartile (Q4 = high-speed)"],
        index=0,
    )
    type_scope = st.sidebar.radio("Compute quantiles", ["Within each facility type", "Overall"], index=0)
    cutoff_a = 0.0
    cutoff_b = 0.0
else:
    quantile_kind = "Tercile (top tercile = high-speed)"
    type_scope = "Within each facility type"
    t_sorted = sorted(types_list)
    t_a = t_sorted[0] if len(t_sorted) > 0 else "Type A"
    t_b = t_sorted[1] if len(t_sorted) > 1 else "Type B"
    cutoff_a = st.sidebar.number_input(f"Cutoff for {t_a} (Mbps)", min_value=0.0, value=10.0, step=1.0)
    cutoff_b = st.sidebar.number_input(f"Cutoff for {t_b} (Mbps)", min_value=0.0, value=10.0, step=1.0)

st.sidebar.markdown("---")
if st.sidebar.button("Clear filters"):
    st.session_state.clear()
    st.rerun()

# =============================================================================
# Main title + how to use (collapsible)
# =============================================================================
st.title("Health facility connectivity in Latin America and Caribbean")

if "help_first_load" not in st.session_state:
    st.session_state["help_first_load"] = True

help_expanded = bool(st.session_state["help_first_load"])
with st.expander("How to use this dashboard", expanded=help_expanded):
    st.markdown(
        """
- Use **Selected speed direction** and **Network mode** to define the speed metric displayed and analyzed.
- Use **High-speed definition** to classify facilities as high-speed (quantiles or manual thresholds).
- Use **Facility filters** to include/exclude facility types across the dashboard.
- Use **Focus country** to filter and zoom in the **Facilities Map** and **Population coverage** tabs.
        """
    )

# After the first render, collapse it on subsequent reruns (e.g., when switching tabs)
if st.session_state["help_first_load"]:
    st.session_state["help_first_load"] = False

# =============================================================================
# Derived datasets (lightweight)
# =============================================================================
fac1 = add_selected_speed(fac0, direction, network_mode)
fac1 = fac1[fac1["tp_stbl"].isin(type_filter)].copy()
fac1 = add_high_speed_flag(
    fac1,
    mode=hs_mode,
    quantile_kind=quantile_kind,
    type_scope=type_scope,
    cutoff_type_a=cutoff_a,
    cutoff_type_b=cutoff_b,
)

# =============================================================================
# Tabs
# =============================================================================
tabs = st.tabs(
    [
        "Facilities Map",
        "Country Distributions",
        "Regional Gaps",
        "Urban vs Rural",
        "Population coverage",
    ]
)

# =============================================================================
# TAB 1: Facilities Map
# =============================================================================
with tabs[0]:
    st.subheader("Facilities map")
    st.markdown(
        """
**What this tab shows**
- A map of facilities (primary health care centers vs hospitals).
- Hover to see fixed/mobile download & upload speeds, your selected speed, and the high-speed flag.
- Use **Focus country** (sidebar) to zoom and filter this map.
        """
    )

    # Filter for map only
    fac_map = fac1.copy()
    if focus_country != "All LAC":
        fac_map = fac_map[fac_map["country"] == focus_country].copy()

    # Map only high speed toggle
    if only_high_speed:
        fac_map = fac_map[fac_map["high_speed"]].copy()

    # KPIs (react to current map filters)
    total_fac = len(fac_map)
    with_speed = int(fac_map["has_speed"].sum()) if "has_speed" in fac_map.columns else 0
    high_speed = int(fac_map["high_speed"].sum()) if "high_speed" in fac_map.columns else 0

    phc_n = int((~fac_map["tp_stbl"].apply(_is_hospital_type)).sum()) if "tp_stbl" in fac_map.columns else 0
    hosp_n = int((fac_map["tp_stbl"].apply(_is_hospital_type)).sum()) if "tp_stbl" in fac_map.columns else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total facilities", f"{total_fac:,}")
    c2.metric("% with any speed", f"{_safe_pct(with_speed, total_fac):.1f}%")
    c3.metric("% high-speed", f"{_safe_pct(high_speed, total_fac):.1f}%")
    c4.metric("% primary health centers", f"{_safe_pct(phc_n, total_fac):.1f}%")
    c5.metric("% hospitals", f"{_safe_pct(hosp_n, total_fac):.1f}%")

    show_all = st.checkbox(
        "Show all points (may be slow)",
        value=False,
        help=f"Default cap: {MAP_DEFAULT_CAP:,}. Hard cap: {MAP_HARD_CAP:,}.",
    )

    if total_fac == 0:
        st.warning("No facilities match the current filters.")
        st.stop()

    fac_plot = fac_map.dropna(subset=["lat", "lon"])
    center, zoom = _compute_center_zoom_from_points(fac_plot)

    fig = _make_facility_map(fac_plot, show_all=show_all, center=center, zoom=zoom)
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 2: Country Distributions
# =============================================================================
with tabs[1]:
    st.subheader("Country distributions")
    st.markdown(
        """
**What this tab shows**
- Country-by-country distributions of the selected speed metric.
- A summary table with min/median/p25/p75 and % below a user-defined threshold.
        """
    )

    threshold = st.slider("Threshold (Mbps) to flag below-threshold facilities", 0.0, 50.0, 5.0, 0.5)

    tmp = fac1.copy()
    tmp = tmp[tmp["has_speed"]].copy()

    if tmp.empty:
        st.warning("No facilities with speed under the current filters.")
        st.stop()

    def _q(x, q):
        x = x.dropna()
        return float(x.quantile(q)) if len(x) else np.nan

    country_stats = (
        tmp.groupby("country", as_index=False)
        .agg(
            min_speed=("selected_speed_mbps", "min"),
            median_speed=("selected_speed_mbps", "median"),
            p25=("selected_speed_mbps", lambda x: _q(x, 0.25)),
            p75=("selected_speed_mbps", lambda x: _q(x, 0.75)),
            n=("selected_speed_mbps", "count"),
        )
    )
    below = tmp.assign(below=lambda d: d["selected_speed_mbps"] < threshold).groupby("country")["below"].mean().reset_index()
    below["pct_below_threshold"] = 100.0 * below["below"]
    country_stats = country_stats.merge(below[["country", "pct_below_threshold"]], on="country", how="left")

    plot_kind = st.radio("Distribution plot", ["Boxplot", "Violin"], index=0, horizontal=True)

    if plot_kind == "Boxplot":
        fig = px.box(
            tmp,
            x="country",
            y="selected_speed_mbps",
            points=False,
            title="Selected speed distribution by country",
        )
    else:
        fig = px.violin(
            tmp,
            x="country",
            y="selected_speed_mbps",
            box=True,
            points=False,
            title="Selected speed distribution by country",
        )

    fig.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    show = country_stats.sort_values("median_speed", ascending=False).copy()
    show = show.rename(
        columns={
            "country": "Country",
            "min_speed": "Minimum speed (Mbps)",
            "median_speed": "Median speed (Mbps)",
            "p25": "P25 (Mbps)",
            "p75": "P75 (Mbps)",
            "pct_below_threshold": "% below threshold",
            "n": "# facilities with speed",
        }
    )
    st.dataframe(show, use_container_width=True)

# =============================================================================
# TAB 3: Regional Gaps
# =============================================================================
with tabs[2]:
    st.subheader("Regional gaps")
    st.markdown(
        """
**What this tab shows**
- Connectivity gaps relative to a LAC reference (median or p25).
- Negative gaps mean below the LAC reference; positive gaps mean above the reference.
- For ADM1/ADM2, select a **country** to keep the plot readable.
        """
    )

    group_level = st.selectbox("Group level", ["Country", "ADM1", "ADM2"], index=0)
    ref_kind = st.radio("LAC reference", ["Median", "P25"], index=0, horizontal=True)

    ref_series = fac1["selected_speed_mbps"].dropna()
    if ref_series.empty:
        st.warning("No speed values available under current filters.")
        st.stop()

    ref_val = float(ref_series.median()) if ref_kind == "Median" else float(ref_series.quantile(0.25))

    # Extra filter when ADM1/ADM2
    chosen_country = None
    if group_level in ("ADM1", "ADM2"):
        default_country = focus_country if focus_country != "All LAC" else (countries_list[0] if countries_list else None)
        chosen_country = st.selectbox("Country (required for ADM1/ADM2)", options=countries_list, index=(countries_list.index(default_country) if default_country in countries_list else 0))
        fac_gap = fac1[fac1["country"] == chosen_country].copy()
    else:
        fac_gap = fac1.copy()

    if group_level == "Country":
        key_cols = ["country"]
        label_col = "country"
    elif group_level == "ADM1":
        key_cols = ["country", "ADM1_EN"]
        label_col = "ADM1_EN"
    else:
        key_cols = ["country", "ADM2_EN"]
        label_col = "ADM2_EN"

    g = (
        fac_gap[fac_gap["has_speed"]]
        .dropna(subset=key_cols)
        .groupby(key_cols, as_index=False)["selected_speed_mbps"]
        .median()
    )
    if g.empty:
        st.warning("No grouped results available for the current selection.")
        st.stop()

    g["gap_vs_lac"] = g["selected_speed_mbps"] - ref_val
    g["gap_direction"] = np.where(g["gap_vs_lac"] >= 0, "Above reference", "Below reference")

    # show both good and bad; sort by gap
    g = g.sort_values("gap_vs_lac")

    show_n = st.slider("Number of groups to display (sorted by gap)", 10, 100, 30, 1)

    # Choose slice centered around extremes: show most negative and most positive if possible
    if len(g) <= show_n:
        g_show = g.copy()
    else:
        half = show_n // 2
        g_show = pd.concat([g.head(half), g.tail(show_n - half)], ignore_index=True)
        g_show = g_show.sort_values("gap_vs_lac")

    fig = px.bar(
        g_show,
        x="gap_vs_lac",
        y=label_col,
        orientation="h",
        color="gap_direction",
        color_discrete_map={
            "Below reference": IDB_COLORS["red"],
            "Above reference": IDB_COLORS["green"],
        },
        title=f"Connectivity gap vs LAC reference ({ref_kind})",
    )
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        g.rename(
            columns={
                "selected_speed_mbps": "Group median speed (Mbps)",
                "gap_vs_lac": "Gap vs LAC reference (Mbps)",
                label_col: "Group",
                "country": "Country",
                "gap_direction": "Direction",
            }
        ),
        use_container_width=True,
    )

# =============================================================================
# TAB 4: Urban vs Rural
# =============================================================================
with tabs[3]:
    st.subheader("Urban vs rural")
    st.markdown(
        """
**What this tab shows**
- Facilities are classified as **urban** if they fall inside the precomputed urban polygons; otherwise **rural**.
- Compares connectivity statistics and distributions between urban and rural facilities.
        """
    )

    tmp = fac1.copy()
    tmp = tmp[tmp["has_speed"]].copy()

    if tmp.empty:
        st.warning("No facilities with speed under the current filters.")
        st.stop()

    tmp["Urban/Rural"] = np.where(tmp["is_urban"], "Urban", "Rural")

    stats_ur = tmp.groupby("Urban/Rural", as_index=False).agg(
        n=("selected_speed_mbps", "count"),
        median_speed=("selected_speed_mbps", "median"),
        p25=("selected_speed_mbps", lambda x: float(x.quantile(0.25)) if len(x.dropna()) else np.nan),
        p75=("selected_speed_mbps", lambda x: float(x.quantile(0.75)) if len(x.dropna()) else np.nan),
        pct_high_speed=("high_speed", lambda x: 100.0 * float(np.mean(x)) if len(x) else 0.0),
    )

    st.dataframe(
        stats_ur.rename(
            columns={
                "n": "# facilities with speed",
                "median_speed": "Median speed (Mbps)",
                "p25": "P25 (Mbps)",
                "p75": "P75 (Mbps)",
                "pct_high_speed": "% high-speed",
            }
        ),
        use_container_width=True,
    )

    # Use palette for urban/rural
    fig = px.box(
        tmp,
        x="Urban/Rural",
        y="selected_speed_mbps",
        points=False,
        title="Selected speed (Mbps) by urban/rural",
        color="Urban/Rural",
        color_discrete_map={"Urban": IDB_COLORS["teal"], "Rural": IDB_COLORS["gray"]},
    )
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Country-level urban vs rural summary**")
    by_country = tmp.groupby(["country", "Urban/Rural"], as_index=False)["selected_speed_mbps"].median()
    fig2 = px.bar(
        by_country,
        x="country",
        y="selected_speed_mbps",
        color="Urban/Rural",
        barmode="group",
        color_discrete_map={"Urban": IDB_COLORS["teal"], "Rural": IDB_COLORS["gray"]},
        title="Median selected speed by country and urban/rural",
    )
    fig2.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
    st.plotly_chart(fig2, use_container_width=True)

# =============================================================================
# TAB 5: Population Coverage (precomputed)
# =============================================================================
with tabs[4]:
    st.subheader("Population coverage")
    st.markdown(
        """
**What this tab shows**
- A *best-effort* estimate of population “covered” within a catchment radius around facilities.
- Results are **precomputed** in preprocessing to keep the app fast and stable.
- Use **Focus country** (sidebar) to select one country at a time for this tab.
        """
    )

    if focus_country == "All LAC":
        st.warning("For performance, please select a single Focus country to view population coverage.")
        st.stop()

    cov_df = load_population_coverage()

    if "country" not in cov_df.columns or "radius_km" not in cov_df.columns:
        st.error("population_coverage_by_country.parquet is missing required columns (`country`, `radius_km`). Please rerun preprocess.py.")
        st.stop()

    cov_country = cov_df[cov_df["country"] == focus_country].copy()
    if cov_country.empty:
        st.warning("No precomputed population coverage found for the selected country. Please rerun preprocess.py.")
        st.stop()

    radii = sorted([int(x) for x in cov_country["radius_km"].dropna().unique().tolist()])
    default_radius = 10 if 10 in radii else (radii[0] if radii else 10)

    radius_km = st.selectbox(
        "Catchment radius (km)",
        options=radii,
        index=radii.index(default_radius) if default_radius in radii else 0
    )

    row = cov_country[cov_country["radius_km"] == int(radius_km)].head(1)
    if row.empty:
        st.warning("No results for that radius. Please select another radius.")
        st.stop()

    r = row.iloc[0]
    has_weight = bool(r.get("has_weight", False)) and pd.notna(r.get("total_pop", None))

    if has_weight and float(r.get("total_pop", 0) or 0) > 0:
        covered = float(r["covered_pop"] or 0)
        total = float(r["total_pop"] or 0)
        pct = 100.0 * covered / total if total > 0 else 0.0

        a, b, c = st.columns(3)
        a.metric("Estimated covered population", f"{covered:,.0f}")
        b.metric("Estimated total population", f"{total:,.0f}")
        c.metric("% covered (approx.)", f"{pct:.1f}%")

        fig = go.Figure(
            data=[go.Pie(
                labels=["Covered", "Not covered"],
                values=[covered, max(total - covered, 0)],
                hole=0.45,
                marker=dict(colors=[IDB_COLORS["green"], IDB_COLORS["gray"]]),
            )]
        )
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Interpretation: approximation based on population point weights. Values are precomputed for performance.")
    else:
        covered = int(r.get("covered_points", 0) or 0)
        total = int(r.get("total_points", 0) or 0)
        pct = 100.0 * covered / total if total > 0 else 0.0

        a, b, c = st.columns(3)
        a.metric("Covered population points", f"{covered:,}")
        b.metric("Total population points", f"{total:,}")
        c.metric("% points covered (proxy)", f"{pct:.1f}%")

        fig = go.Figure(
            data=[go.Pie(
                labels=["Covered points", "Not covered points"],
                values=[covered, max(total - covered, 0)],
                hole=0.45,
                marker=dict(colors=[IDB_COLORS["green"], IDB_COLORS["gray"]]),
            )]
        )
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.warning("No population weight column was detected. Results show **# population points covered** (proxy), not people.")

    with st.expander("Show results by radius for this country", expanded=False):
        tmp = cov_country.copy()
        show_cols = ["radius_km", "covered_pop", "total_pop", "covered_points", "total_points", "has_weight"]
        show_cols = [c for c in show_cols if c in tmp.columns]
        tmp = tmp[show_cols].sort_values("radius_km")
        tmp = tmp.rename(
            columns={
                "radius_km": "Catchment radius (km)",
                "covered_pop": "Covered population",
                "total_pop": "Total population",
                "covered_points": "Covered population points",
                "total_points": "Total population points",
                "has_weight": "Has population weights",
            }
        )
        st.dataframe(tmp, use_container_width=True)
