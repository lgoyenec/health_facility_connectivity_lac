"""
Preprocess geospatial inputs for the Streamlit dashboard
-------------------------------------------------------

Run:
  python preprocess.py

Outputs (in same folder):
  ./facilities_prepared.geoparquet
  ./countries.geoparquet
  ./population_points.geoparquet
  ./metadata.json
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

warnings.filterwarnings("ignore", category=UserWarning)

OUT_FAC = Path("./facilities_prepared.geoparquet")
OUT_COU = Path("./countries.geoparquet")
OUT_POP = Path("./population_points.geoparquet")
OUT_META = Path("./metadata.json")

# ---------------------------------------------------------------------------
# BASE CODE (MUST REMAIN VERBATIM; DO NOT EDIT)
# ---------------------------------------------------------------------------
# -- Libraries ----------------------------------------------------------------
#------------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sns.set_style("darkgrid")

# -- Import datasets ----------------------------------------------------------
#------------------------------------------------------------------------------
lac     = gpd.read_file('./ocha-shp/_output/lac.shp')
shp0    = gpd.read_file('./ocha-shp/_output/lac-level-0.shp')
shp1    = gpd.read_file('./ocha-shp/_output/lac-level-1.shp')
shp2    = gpd.read_file('./ocha-shp/_output/lac-level-2.shp')
phc     = gpd.read_file('../../44-Health accessibility/Accessibility Index/data/raw/MasterList_PrimaryCare.shp')
hosp    = gpd.read_file('../../44-Health accessibility/Accessibility Index/data/raw/MasterList_Hospitals.shp')
hcenter = gpd.read_file('../../44-Health accessibility/Accessibility Index/data/raw/MasterList_AllLAC.shp')
fixed   = gpd.read_file('./ookla-connectivity/fixed_connectivity_h3.geojson') 
mobile  = gpd.read_file('./ookla-connectivity/mobile_connectivity_h3.geojson')
urban   = gpd.read_file('./shapefiles/urban_areas_lac.shp')
poplac  = pd.read_parquet('./meta-pop/total_population/LAC_total_population.parquet')

# -- Select IADB countries ----------------------------------------------------
#------------------------------------------------------------------------------
# Create dissolve LAC shapefile
if "lac" not in globals():    
    shp0_ = shp0.copy()
    shp0_['geometry'] = shp0_['geometry'].buffer(0.00001)
    shp0_ = shp0_.dissolve()
    shp0_.to_file('./ocha-shp/_output/lac.shp')

# Create LAC health centers
if "hcenter" not in globals():
    # Make sure CRS matches
    phc  = phc .to_crs(lac.crs)
    hosp = hosp.to_crs(lac.crs)
    
    # Keep only facilities within LAC
    phc_lac  = gpd.sjoin(phc,  lac, how="inner", predicate="within")
    hosp_lac = gpd.sjoin(hosp, lac, how="inner", predicate="within")
    
    # Drop join index column(s)
    phc_lac  = phc_lac .drop(columns=["index_right"], errors="ignore")
    hosp_lac = hosp_lac.drop(columns=["index_right"], errors="ignore")
    
    del phc, hosp
    
    # Master health centers 
    hcenter = gpd.GeoDataFrame(pd.concat([phc_lac, hosp_lac], ignore_index=True), crs=lac.crs)
    hcenter = hcenter[['ADM0_EN','ADM0_PCODE','country','lat','lon','tp_stbl','name','address','source','geometry']]
    hcenter.to_file('../../44-Health accessibility/Accessibility Index/data/raw/MasterList_AllLAC.shp')

# Rename down/upload internet variables 
fixed .rename(columns = {'dl_mbps':'fix_dl_mbps','ul_mbps':'fix_ul_mbps'}, inplace = True)
mobile.rename(columns = {'dl_mbps':'mob_dl_mbps','ul_mbps':'mob_ul_mbps'}, inplace = True)

# Make sure CRS matches
fixed  = fixed .to_crs(lac.crs)
mobile = mobile.to_crs(lac.crs)

# Assing internet to facilities
hc_speed = gpd.sjoin(
    hcenter,
    fixed.drop(columns = ['h3','sum_tests']),
    how="left",
    predicate="within"
).drop(columns=["index_right"], errors="ignore")

hc_speed = gpd.sjoin(
    hc_speed,
    mobile.drop(columns = ['h3','sum_tests']),
    how="left",
    predicate="within"
).drop(columns=["index_right"], errors="ignore")
# ---------------------------------------------------------------------------
# END BASE CODE
# ---------------------------------------------------------------------------


def detect_population_weight_column(df: pd.DataFrame) -> str | None:
    """Detect likely population weight column name (case-insensitive)."""
    candidates = ["population", "pop", "total_pop", "tot_pop", "pop_total", "poblacion", "pob"]
    cols_l = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols_l:
            return cols_l[cand]
    # fallback: any numeric column that looks like pop
    for c in df.columns:
        cl = c.lower()
        if "pop" in cl or "pob" in cl:
            return c
    return None


def main():
    # -----------------------
    # 1) Facilities prepared
    # -----------------------
    fac = hc_speed.copy()

    # Ensure expected columns exist
    for c in ["fix_dl_mbps", "fix_ul_mbps", "mob_dl_mbps", "mob_ul_mbps", "lat", "lon"]:
        if c in fac.columns:
            fac[c] = pd.to_numeric(fac[c], errors="coerce")

    fac["has_fix"] = fac[["fix_dl_mbps", "fix_ul_mbps"]].notna().any(axis=1)
    fac["has_mob"] = fac[["mob_dl_mbps", "mob_ul_mbps"]].notna().any(axis=1)
    fac["has_speed"] = fac["has_fix"] | fac["has_mob"]

    def _speed_source(row):
        if row["has_fix"] and row["has_mob"]:
            return "fixed+mobile"
        if row["has_fix"]:
            return "fixed"
        if row["has_mob"]:
            return "mobile"
        return "none"

    fac["speed_source"] = fac.apply(_speed_source, axis=1)

    # Use facility 'country' as primary key (per your request)
    if "country" not in fac.columns:
        # STRICTLY NECESSARY EDIT: ensure country exists if missing
        fac["country"] = fac["ADM0_EN"]  # STRICTLY NECESSARY EDIT

    # Admin joins (precompute; app must NOT do spatial joins)
    shp1_ = shp1[["ADM1_EN", "ADM1_PCODE", "geometry"]].to_crs(fac.crs)
    shp2_ = shp2[["ADM2_EN", "ADM2_PCODE", "geometry"]].to_crs(fac.crs)
    urban_ = urban[["geometry"]].to_crs(fac.crs)

    fac = gpd.sjoin(fac, shp1_, how="left", predicate="within").drop(columns=["index_right"], errors="ignore")
    fac = gpd.sjoin(fac, shp2_, how="left", predicate="within").drop(columns=["index_right"], errors="ignore")

    # Urban flag
    fac_u = gpd.sjoin(fac[["geometry"]], urban_, how="left", predicate="within")
    fac["is_urban"] = fac_u["index_right"].notna().values

    # Keep only what the app needs
    keep_cols = [
        "ADM0_EN", "ADM0_PCODE", "country",
        "ADM1_EN", "ADM1_PCODE",
        "ADM2_EN", "ADM2_PCODE",
        "lat", "lon", "tp_stbl", "name", "address", "source",
        "geometry",
        "fix_dl_mbps", "fix_ul_mbps", "mob_dl_mbps", "mob_ul_mbps",
        "has_fix", "has_mob", "has_speed", "speed_source",
        "is_urban",
    ]
    keep_cols = [c for c in keep_cols if c in fac.columns]
    fac_out = fac[keep_cols].copy()

    # Write facilities
    fac_out.to_parquet(OUT_FAC, index=False)

    # -----------------------
    # 2) Countries polygons
    # -----------------------
    # Create a 'country' column aligned with facilities.country
    # Here we use shp0 ADM0_EN as country label (most stable)
    shp0_ = shp0[["ADM0_EN", "ADM0_PCODE", "geometry"]].copy().to_crs(lac.crs)
    shp0_["country"] = shp0_["ADM0_EN"]

    countries = shp0_.dissolve(by="country", as_index=False)[["country", "geometry"]]
    # keep ADM0_EN as display label too
    countries["ADM0_EN"] = countries["country"]

    countries.to_parquet(OUT_COU, index=False)

    # -----------------------
    # 3) Population points
    # -----------------------
    pop_df = poplac.copy()
    if not {"latitude", "longitude"}.issubset(pop_df.columns):
        raise ValueError("poplac must have lat/lon columns per requirements.")

    pop_weight_col = detect_population_weight_column(pop_df)

    pop_gdf = gpd.GeoDataFrame(
        pop_df,
        geometry=gpd.points_from_xy(pop_df["longitude"], pop_df["latitude"]),
        crs="EPSG:4326",
    )

    # Transform to lac CRS for consistent spatial operations
    pop_gdf = pop_gdf.to_crs(lac.crs)

    # Keep minimal columns
    pop_keep = ["geometry"]
    if pop_weight_col and pop_weight_col in pop_gdf.columns:
        pop_keep.append(pop_weight_col)

    pop_gdf = pop_gdf[pop_keep].copy()

    # IMPORTANT FIX: assign population points to country ONCE (so app can filter before heavy ops)
    pop_with_country = gpd.sjoin(
        pop_gdf,
        countries[["country", "geometry"]].to_crs(pop_gdf.crs),
        how="left",
        predicate="within",
    ).drop(columns=["index_right"], errors="ignore")

    pop_with_country.to_parquet(OUT_POP, index=False)

    # -----------------------
    # 4) Metadata
    # -----------------------
    meta = {
        "crs": str(lac.crs),
        "facilities_count": int(len(fac_out)),
        "countries_count": int(len(countries)),
        "population_points_count": int(len(pop_with_country)),
        "facility_types": sorted([str(x) for x in fac_out["tp_stbl"].dropna().unique().tolist()]) if "tp_stbl" in fac_out.columns else [],
        "countries_list": sorted([str(x) for x in fac_out["country"].dropna().unique().tolist()]) if "country" in fac_out.columns else [],
        "population_weight_col": pop_weight_col,
    }

    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("✅ Wrote:")
    print(" -", OUT_FAC)
    print(" -", OUT_COU)
    print(" -", OUT_POP)
    print(" -", OUT_META)


if __name__ == "__main__":
    main()
