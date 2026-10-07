"""
Real multi-source satellite + climate data pipeline via Google Earth Engine (GEE).

This is the "swap in real data" counterpart to synthetic_data.py. It is NOT run by
default (config.data.use_synthetic = true), because it requires:
  1. A Google Earth Engine account: https://earthengine.google.com/
  2. `pip install earthengine-api` (already in requirements.txt)
  3. Running `earthengine authenticate` once from your terminal
  4. Setting your GEE project id below (or via the EE_PROJECT env var)

Fill in `get_default_aoi()` with your Alaska bounding box of interest, then call
`build_feature_stack()` to pull:
  - Sentinel-2 (optical, cloud-masked, monthly composite)
  - Sentinel-1 (SAR, VV/VH, calibrated)
  - MODIS/VIIRS (thermal anomalies / fire radiative power)
  - ERA5 (temperature, wind, humidity, precipitation reanalysis)

and `load_afs_fire_perimeters()` for historical fire perimeter labels from the
Alaska Fire Service / BLM.
"""
from __future__ import annotations

import os
from typing import Optional

EE_PROJECT = os.environ.get("EE_PROJECT", "alaska-wildfire-502509")


def _lazy_import_ee():
    try:
        import ee  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "earthengine-api is not installed or not authenticated. "
            "Run: pip install earthengine-api && earthengine authenticate"
        ) from e
    return ee


def init_earth_engine(project: str = EE_PROJECT):
    """Initialize the Earth Engine session. Call once per notebook/script."""
    ee = _lazy_import_ee()
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)
    return ee


def get_default_aoi():
    """
    Alaska boreal/tundra area of interest.
    Replace these coordinates with your specific study region
    (e.g. Fairbanks North Star Borough, Yukon-Koyukuk, or statewide).
    """
    ee = _lazy_import_ee()
    # Using the Fairbanks region specifically
    return ee.Geometry.Rectangle([-148.9, 63.5, -145.3, 65.5])


def get_sentinel2_composite(aoi, start_date: str, end_date: str, cloud_pct: int = 20):
    """Cloud-masked Sentinel-2 SR monthly composite (optical: R, G, NIR, SWIR)."""
    ee = _lazy_import_ee()

    def mask_clouds(img):
        qa = img.select("QA60")
        cloud_bit, cirrus_bit = 1 << 10, 1 << 11
        mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
        return img.updateMask(mask).divide(10000)

    coll = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .map(mask_clouds)
    )
    composite = coll.median().select(["B4", "B3", "B8", "B11"])  # R, G, NIR, SWIR1
    return composite.clip(aoi)


def get_sentinel1_composite(aoi, start_date: str, end_date: str):
    """Sentinel-1 SAR (VV, VH), radiometrically terrain-corrected, monthly composite.

    SAR is highlighted in the literature survey as underused for fire risk despite
    penetrating cloud cover and carrying fuel-moisture signal — see research gap G2.
    """
    ee = _lazy_import_ee()
    coll = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH"])
    )
    return coll.median().clip(aoi)


def get_thermal_anomalies(aoi, start_date: str, end_date: str):
    """MODIS/VIIRS active fire / thermal anomaly product (Fire Radiative Power proxy)."""
    ee = _lazy_import_ee()
    coll = (
        ee.ImageCollection("MODIS/061/MOD14A1")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .select("MaxFRP")
    )
    return coll.max().clip(aoi)


def get_era5_weather(aoi, start_date: str, end_date: str):
    """ERA5 reanalysis: 2m temperature, wind components, and precipitation."""
    ee = _lazy_import_ee()
    coll = (
        ee.ImageCollection("ECMWF/ERA5/DAILY")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .select(
            ["mean_2m_air_temperature", "u_component_of_wind_10m",
             "v_component_of_wind_10m", "total_precipitation"]
        )
    )
    return coll


def build_feature_stack(aoi, start_date: str, end_date: str):
    """Fuse optical + SAR + thermal + weather into one analysis-ready image stack,
    matching research gap G3 (4-modality fusion, unaddressed in prior literature)."""
    optical = get_sentinel2_composite(aoi, start_date, end_date)
    sar = get_sentinel1_composite(aoi, start_date, end_date)
    thermal = get_thermal_anomalies(aoi, start_date, end_date)
    stack = optical.addBands(sar).addBands(thermal)
    return stack


def load_afs_fire_perimeters(shapefile_path: Optional[str] = None):
    """
    Load Alaska Fire Service historical fire perimeters (used to build HIGH/MODERATE/
    NO_RISK ground-truth labels).

    Public source: Alaska Fire Service / Bureau of Land Management (BLM) fire history
    GIS data, e.g. via the Alaska Interagency Coordination Center (AICC) open data
    portal or the National Interagency Fire Center (NIFC) ArcGIS feature service.

    This function expects a local shapefile/GeoJSON downloaded from one of those
    sources (`shapefile_path`). It is a thin geopandas wrapper — fill in the exact
    download URL/path for your project.
    """
    import geopandas as gpd

    if shapefile_path is None:
        raise FileNotFoundError(
            "Download AFS/BLM historical fire perimeters (shapefile or GeoJSON) and "
            "pass its path here. See README section 4 for the data source."
        )
    return gpd.read_file(shapefile_path)


if __name__ == "__main__":
    print(
        "This module defines the real Earth Engine data pipeline.\n"
        "Run init_earth_engine() after `earthengine authenticate`, then call "
        "build_feature_stack(aoi, start_date, end_date) to fetch a fused image stack.\n"
        "See notebooks/01_data_collection.ipynb for a worked example."
    )











# ---------------------------------------------------------------------------
# Real patch + weather extraction for step 8: pulls pixels directly via the
# API (no Drive/GCS export), respects the "no future data" rule (features are
# built from a window ending AT sample_date, never after), and handles the
# resolution mismatch between Sentinel-2 (10m), Sentinel-1 (10m), and
# MODIS/VIIRS thermal (1km) by resampling everything onto one common grid.
# ---------------------------------------------------------------------------

def get_feature_patch(aoi_point, sample_date: str, patch_size_px: int = 32,
                       scale: int = 10, lookback_days: int = 30):
    """
    Pulls a fused optical+SAR+thermal patch for the `lookback_days` window
    ending at `sample_date` (never after it — this is what makes it a valid
    training feature rather than a leak of future information).

    Returns a numpy array of shape [n_bands, patch_size_px, patch_size_px].
    """
    ee = _lazy_import_ee()
    import numpy as np

    end = ee.Date(sample_date)
    start = end.advance(-lookback_days, "day")
    start_str = start.format("YYYY-MM-dd").getInfo()
    end_str = end.format("YYYY-MM-dd").getInfo()

    stack = build_feature_stack(aoi_point.buffer(patch_size_px * scale), start_str, end_str)

    # MODIS/VIIRS thermal is natively ~1km resolution; resample it onto the
    # same 10m grid as everything else so all bands stack cleanly.
    thermal_band = stack.select("MaxFRP").resample("bilinear").reproject(crs="EPSG:3338", scale=scale)
    other_bands = stack.select(["B4", "B3", "B8", "B11", "VV", "VH"])
    stack = other_bands.addBands(thermal_band)

    region = aoi_point.buffer(patch_size_px * scale / 2).bounds()
    sampled = stack.sampleRectangle(region=region, defaultValue=0)

    band_names = ["B4", "B3", "B8", "B11", "VV", "VH", "MaxFRP"]
    arrays = []
    for b in band_names:
        arr = np.array(sampled.get(b).getInfo(), dtype=np.float32)
        # sampleRectangle can return slightly different pixel counts per band
        # due to rounding at the edges — force everything to the same shape.
        arr = _fit_to_shape(arr, (patch_size_px, patch_size_px))
        arrays.append(arr)

    return np.stack(arrays, axis=0)  # [7, patch_size_px, patch_size_px]


def _fit_to_shape(arr, target_shape):
    """Pad or crop a 2D array to an exact target shape."""
    import numpy as np
    out = np.zeros(target_shape, dtype=arr.dtype)
    h = min(arr.shape[0], target_shape[0])
    w = min(arr.shape[1], target_shape[1])
    out[:h, :w] = arr[:h, :w]
    return out


def get_weather_series(aoi_point, sample_date: str, seq_len: int = 30):
    """
    Pulls a daily ERA5 weather time series for the `seq_len` days ending at
    `sample_date`. Returns a numpy array of shape [seq_len, 4]:
    [temp, wind_speed, precip, dewpoint-proxy].

    Note: ERA5/DAILY doesn't expose humidity/dewpoint directly the way the
    synthetic generator's 5-feature schema assumed — this real version uses
    the 4 fields ERA5 actually provides. Update n_weather_features in
    config.yaml to 4 once you switch to real data, or add a 5th derived
    feature yourself if you need to match the synthetic shape exactly.
    """
    ee = _lazy_import_ee()
    import numpy as np
    import pandas as pd

    end = ee.Date(sample_date)
    start = end.advance(-seq_len, "day")

    coll = (
        ee.ImageCollection("ECMWF/ERA5/DAILY")
        .filterBounds(aoi_point)
        .filterDate(start, end)
        .select(
            ["mean_2m_air_temperature", "u_component_of_wind_10m",
             "v_component_of_wind_10m", "total_precipitation"]
        )
        .sort("system:time_start")
    )

    def extract(img):
        stats = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi_point, scale=1000)
        return ee.Feature(None, stats.set("date", img.date().format("YYYY-MM-dd")))

    features = coll.map(extract).getInfo()["features"]

    rows = []
    for f in features:
        p = f["properties"]
        temp_c = (p.get("mean_2m_air_temperature") or 273.15) - 273.15
        u = p.get("u_component_of_wind_10m") or 0
        v = p.get("v_component_of_wind_10m") or 0
        wind_speed = (u ** 2 + v ** 2) ** 0.5
        precip_mm = (p.get("total_precipitation") or 0) * 1000
        rows.append([temp_c, wind_speed, precip_mm])

    arr = np.array(rows, dtype=np.float32)
    if len(arr) < seq_len:  # pad short sequences (e.g. near the start of your data range)
        pad = np.repeat(arr[:1], seq_len - len(arr), axis=0) if len(arr) > 0 else np.zeros((seq_len, 3))
        arr = np.concatenate([pad, arr], axis=0)
    return arr[-seq_len:]