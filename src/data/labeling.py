"""
Turns Alaska Fire Service fire-perimeter history into HIGH / MODERATE / NO_RISK
labels for a set of (location, date) sample points — the ground-truth labels the
CNN-LSTM is trained to predict.

Labeling rule (this is the actual research decision the project needs):
  HIGH     = the point actually burned (fell inside a fire perimeter) at some point
             between the sample date and (sample date + horizon).
  MODERATE = a fire occurred in that same window within `buffer_km` of the point,
             but the point itself wasn't inside the burned perimeter.
  NO_RISK  = neither of the above.

This is a retrospective/historical labeling approach: for training, we look forward
from each sample date to see what actually happened, so the model learns to predict
that outcome from features only available *at* the sample date (imagery + weather).
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point


def prepare_fire_dates(fires_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Parse PERIMETERDATE; fall back to July 1 of FIREYEAR when missing (peak fire season)."""
    fires = fires_gdf.copy()
    col = fires["PERIMETERDATE"]

    if pd.api.types.is_numeric_dtype(col):
        # raw values are Unix epoch milliseconds, not plain date strings/ns
        fires["PERIMETERDATE"] = pd.to_datetime(col, unit="ms", errors="coerce")
    else:
        fires["PERIMETERDATE"] = pd.to_datetime(col, errors="coerce")

    missing = fires["PERIMETERDATE"].isna()
    if missing.any() and "FIREYEAR" in fires.columns:
        fallback_year = fires.loc[missing, "FIREYEAR"].astype(int).astype(str)
        fires.loc[missing, "PERIMETERDATE"] = pd.to_datetime(fallback_year + "-07-01")
    return fires.dropna(subset=["PERIMETERDATE"])


def generate_sample_grid(
    bounds: tuple[float, float, float, float],  # (west, south, east, north)
    sample_dates: list[str],
    spacing_deg: float = 0.1,
) -> gpd.GeoDataFrame:
    """Generate a regular grid of (lat, lon) points across the AOI, repeated for each
    sample date. spacing_deg ~0.1 degrees is roughly 8-11 km at Alaska's latitude —
    tune it based on how many samples you want / can afford to label & pull imagery for."""
    west, south, east, north = bounds
    lons = np.arange(west, east, spacing_deg)
    lats = np.arange(south, north, spacing_deg)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    rows = []
    for date_str in sample_dates:
        for lon, lat in zip(lon_grid.ravel(), lat_grid.ravel()):
            rows.append({"sample_date": pd.Timestamp(date_str), "lon": lon, "lat": lat})

    df = pd.DataFrame(rows)
    geometry = [Point(xy) for xy in zip(df["lon"], df["lat"])]
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")


def assign_risk_labels(
    fires_gdf: gpd.GeoDataFrame,
    points_gdf: gpd.GeoDataFrame,
    horizon_months: int = 1,
    buffer_km: float = 30.0,
    high_buffer_km: float = 1.0,
) -> gpd.GeoDataFrame:
    """
    high_buffer_km: radius (in km) around the point that counts as "this area
    burned" for a HIGH label — should roughly match your patch footprint
    (e.g. a 32px patch at 10m/px covers ~0.32km across, so ~0.5-1km is
    reasonable), rather than requiring exact point-in-polygon containment,
    which is unrealistically strict for a sparse sample grid.
    buffer_km: wider radius for MODERATE — nearby but not directly burned.
    """
    fires = prepare_fire_dates(fires_gdf)

    fires_m = fires.to_crs(epsg=3338)
    points_m = points_gdf.to_crs(epsg=3338)

    high_buffer_m = high_buffer_km * 1000
    buffer_m = buffer_km * 1000
    fires_m["high_buffer_geom"] = fires_m.geometry.buffer(high_buffer_m)
    fires_m["buffer_geom"] = fires_m.geometry.buffer(buffer_m)

    labels = np.full(len(points_m), "NO_RISK", dtype=object)

    for i, (_, pt) in enumerate(points_m.iterrows()):
        horizon_end = pt["sample_date"] + pd.DateOffset(months=horizon_months)
        window = fires_m[
            (fires_m["PERIMETERDATE"] >= pt["sample_date"]) & (fires_m["PERIMETERDATE"] <= horizon_end)
        ]
        if window.empty:
            continue

        point_geom = pt.geometry
        if window["high_buffer_geom"].contains(point_geom).any():
            labels[i] = "HIGH"
        elif window["buffer_geom"].contains(point_geom).any():
            labels[i] = "MODERATE"

    out = points_gdf.copy()
    out["risk_label"] = labels
    return out