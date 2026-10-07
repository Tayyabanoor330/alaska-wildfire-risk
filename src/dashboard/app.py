"""
Streamlit GIS dashboard — Alaska Wildfire Risk.

Two modes:
  1. Demo grid  — a synthetic, illustrative risk surface across the AOI (fast, always works)
  2. Real model — actual CNN-LSTM predictions on your real extracted Fairbanks patches
                  (data/processed/patch_*.npz), compared against their true historical labels

Run with:
    streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import os
import sys

import folium
import numpy as np
import pandas as pd
import streamlit as st
import torch
import yaml
from folium.plugins import HeatMap
from streamlit_folium import st_folium

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

RISK_COLORS = {"NO_RISK": "#2ecc71", "MODERATE": "#f1c40f", "HIGH": "#e74c3c"}
RISK_LABELS = ["NO_RISK", "MODERATE", "HIGH"]
ALASKA_BOUNDS = {"lat_min": 54.0, "lat_max": 71.5, "lon_min": -168.0, "lon_max": -130.0}




BASE_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
BASE_TILE_ATTRIBUTION = "&copy; OpenStreetMap contributors"


@st.cache_data
def generate_synthetic_risk_grid(horizon: int, n_points: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed + horizon)
    lats = rng.uniform(ALASKA_BOUNDS["lat_min"], ALASKA_BOUNDS["lat_max"], n_points)
    lons = rng.uniform(ALASKA_BOUNDS["lon_min"], ALASKA_BOUNDS["lon_max"], n_points)

    latent = (71.5 - lats) / (71.5 - 54.0)
    horizon_bias = {1: 0.0, 3: 0.05, 6: 0.10}[horizon]
    risk_score = np.clip(latent + horizon_bias + rng.normal(0, 0.15, n_points), 0, 1)
    risk_class = pd.cut(risk_score, bins=[-0.01, 0.4, 0.7, 1.01], labels=RISK_LABELS).astype(str)

    return pd.DataFrame({"lat": lats, "lon": lons, "risk_score": risk_score, "risk_class": risk_class,
                          "true_label": "N/A (synthetic demo)"})


@st.cache_data
@st.cache_data
def load_real_predictions(checkpoint_path: str, labeled_points_path: str, processed_dir: str) -> pd.DataFrame | None:
    """Runs your actual trained model on your real extracted patches and returns
    a DataFrame with real coordinates, the model's prediction, and the true
    historical label for comparison."""
    from src.data.real_data import AlaskaWildfireRealDataset
    from src.models.cnn_lstm import build_model
    from src.training.train import load_config

    if not os.path.exists(checkpoint_path) or not os.path.exists(labeled_points_path):
        return None

    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "..", "config", "config.yaml"))
    dataset = AlaskaWildfireRealDataset(processed_dir=processed_dir, labeled_points_path=labeled_points_path)
    cfg["data"]["n_weather_features"] = dataset.n_weather_features

    model = build_model(cfg, dataset.n_spatial_channels, dataset.n_weather_features)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    rows = []
    with torch.no_grad():
        for i, idx in enumerate(dataset.indices):
            sample = dataset[i]
            logits = model(sample["spatial"].unsqueeze(0), sample["weather"].unsqueeze(0))
            probs = torch.softmax(logits, dim=1).numpy()[0]
            pred_class = RISK_LABELS[probs.argmax()]
            true_class = RISK_LABELS[sample["label"].item()]

            point = dataset.gdf.loc[idx].geometry
            rows.append({
                "lat": point.y, "lon": point.x,
                "risk_score": float(probs.max()),
                "risk_class": pred_class,
                "true_label": true_class,
            })

    return pd.DataFrame(rows)


def build_map(df: pd.DataFrame, horizon_label: str, show_heatmap: bool = True) -> folium.Map:
    center_lat = df["lat"].mean() if len(df) else 64.2008
    center_lon = df["lon"].mean() if len(df) else -149.4937
    zoom = 7 if len(df) and df["lat"].std() < 2 else 4

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles=None) 
    folium.TileLayer(tiles=BASE_TILE_URL, attr=BASE_TILE_ATTRIBUTION, name="Map").add_to(m)

    marker_group = folium.FeatureGroup(name="Risk points").add_to(m)
    for _, row in df.iterrows():
        match_note = ""
        if row["true_label"] != "N/A (synthetic demo)":
            match_note = " ✓ matches actual history" if row["risk_class"] == row["true_label"] else \
                         f" (actual: {row['true_label']})"
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5,
            color=RISK_COLORS[row["risk_class"]],
            fill=True,
            fill_color=RISK_COLORS[row["risk_class"]],
            fill_opacity=0.75,
            weight=1,
            popup=folium.Popup(
                f"<b>{row['risk_class']}</b> (confidence {row['risk_score']:.0%})<br>"
                f"{horizon_label}{match_note}",
                max_width=220,
            ),
        ).add_to(marker_group)

    if show_heatmap and len(df):
        heat_group = folium.FeatureGroup(name="Risk heatmap", show=False).add_to(m)
        HeatMap(df[["lat", "lon", "risk_score"]].values.tolist(), radius=18, blur=22, max_zoom=8).add_to(heat_group)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def render_legend():
    st.markdown("**Risk levels**")
    for name, color in RISK_COLORS.items():
        st.markdown(
            f'<span style="display:inline-block;width:12px;height:12px;border-radius:50%;'
            f'background:{color};margin-right:6px;"></span>{name.replace("_", " ").title()}',
            unsafe_allow_html=True,
        )


def main():
    st.set_page_config(page_title="Alaska Wildfire Risk Dashboard", layout="wide", page_icon="🔥")

    st.title("Alaska Wildfire Risk — Interactive GIS Dashboard")
    st.caption("Hybrid CNN-LSTM wildfire risk classifier for the Fairbanks North Star Borough, Alaska.")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    processed_dir = os.path.join(project_root, "data", "processed")
    synthetic_checkpoint = os.path.join(project_root, "outputs", "checkpoints", "cnn_lstm_synthetic_demo.pt")

    REAL_HORIZON_FILES = {
        1: ("cnn_lstm_real_fairbanks_h1.pt", "labeled_points.geojson"),
        3: ("cnn_lstm_real_fairbanks_h3.pt", "labeled_points_h3.geojson"),
        6: ("cnn_lstm_real_fairbanks_h6.pt", "labeled_points_h6.geojson"),
    }

    has_real_model = any(
        os.path.exists(os.path.join(project_root, "outputs", "checkpoints", ckpt))
        for ckpt, _ in REAL_HORIZON_FILES.values()
    )
    has_synthetic_model = os.path.exists(synthetic_checkpoint)

    with st.sidebar:
        st.header("Controls")

        mode_options = ["Demo grid (illustrative)"]
        if has_real_model:
            mode_options.insert(0, "My trained model (real Fairbanks data)")
        mode = st.radio("Data source", mode_options)

        if mode == "My trained model (real Fairbanks data)":
            horizon = st.selectbox("Prediction horizon", [1, 3, 6], format_func=lambda h: f"{h}-month")
            st.caption("Shows your CNN-LSTM's actual predictions on real satellite patches, "
                       "compared against real historical fire outcomes.")
        else:
            horizon = st.selectbox("Prediction horizon", [1, 3, 6], format_func=lambda h: f"{h}-month")

        show_heatmap = st.checkbox("Show heatmap overlay", value=False)

        st.divider()
        render_legend()

        st.divider()
        st.caption("Project: Hybrid CNN-LSTM Wildfire Risk Prediction\nStudy area: Fairbanks North Star Borough, AK")

    if mode == "Demo grid (illustrative)":
        df = generate_synthetic_risk_grid(horizon)
        horizon_label = f"{horizon}-month horizon (illustrative demo)"
    else:
        ckpt_name, labels_name = REAL_HORIZON_FILES[horizon]
        checkpoint_path = os.path.join(project_root, "outputs", "checkpoints", ckpt_name)
        labeled_points_path = os.path.join(processed_dir, labels_name)

        df = load_real_predictions(checkpoint_path, labeled_points_path, processed_dir)
        horizon_label = f"{horizon}-month horizon (real model output)"
        if df is None or len(df) == 0:
            st.warning(f"No real model available for the {horizon}-month horizon yet — falling back to the demo grid.")
            df = generate_synthetic_risk_grid(horizon)
            horizon_label = f"{horizon}-month horizon (illustrative demo)"

    col_map, col_stats = st.columns([3, 1])

    with col_stats:
        st.metric("Points shown", len(df))
        st.metric("HIGH risk", int((df["risk_class"] == "HIGH").sum()))
        st.metric("MODERATE risk", int((df["risk_class"] == "MODERATE").sum()))
        st.bar_chart(df["risk_class"].value_counts())

        if mode != "Demo grid (illustrative)":
            accuracy = (df["risk_class"] == df["true_label"]).mean()
            st.metric("Match rate vs. real history", f"{accuracy:.0%}")

    with col_map:
        m = build_map(df, horizon_label, show_heatmap=show_heatmap)
        st_folium(m, width=None, height=600, use_container_width=True)

    with st.expander("About this dashboard"):
        st.markdown(
            "This prototype visualizes wildfire risk classification for the Fairbanks North Star "
            "Borough, Alaska, produced by a hybrid CNN-LSTM model trained on Sentinel-1/2, "
            "MODIS/VIIRS, and ERA5 data fused with historical Alaska Fire Service fire perimeters. "
            "**Demo grid** mode shows an illustrative synthetic surface for exploring the interface. "
            "**My trained model** mode (when available) shows the model's actual predictions on real "
            "extracted satellite patches, with each point's predicted class compared against what "
            "actually happened historically at that location and date."
        )


if __name__ == "__main__":
    main()