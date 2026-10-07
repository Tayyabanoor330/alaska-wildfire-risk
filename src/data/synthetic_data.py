"""
Synthetic multi-modal dataset generator.

Mimics the shape and rough statistics of the real fused dataset described in the
research plan:
  - Optical patch  (Sentinel-2 subset: R, G, NIR, SWIR)      -> [C_opt, H, W]
  - SAR patch      (Sentinel-1: VV, VH)                       -> [C_sar, H, W]
  - Thermal patch  (MODIS/VIIRS FRP / brightness temp)         -> [C_thermal, H, W]
  - Weather series (ERA5/NOAA: temp, wind, humidity, precip,
                     dewpoint), daily, for `weather_seq_len` days -> [T, F]
  - Label: one of NO_RISK / MODERATE / HIGH, per horizon (1/3/6 month)

This lets every notebook / script in the repo run immediately without any
satellite API credentials. Swap in `gee_pipeline.py` for real data once you have
Earth Engine access.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class AlaskaWildfireSyntheticDataset(Dataset):
    """Synthetic stand-in for the fused multi-modal Alaska wildfire dataset."""

    def __init__(self, cfg: dict, horizon: int = 1, split: str = "train", seed: int = 42):
        assert horizon in cfg["data"]["horizons"], f"horizon must be one of {cfg['data']['horizons']}"
        self.cfg = cfg
        self.horizon = horizon
        self.split = split

        d = cfg["data"]
        self.n_samples = d["n_samples"]
        self.patch_size = d["patch_size"]
        self.n_optical = d["n_optical_bands"]
        self.n_sar = d["n_sar_bands"]
        self.n_thermal = d["n_thermal_bands"]
        self.seq_len = d["weather_seq_len"]
        self.n_weather = d["n_weather_features"]
        self.classes = d["classes"]

        rng = np.random.default_rng(seed + horizon)  # different-but-reproducible per horizon
        self._generate(rng)

    # ------------------------------------------------------------------
    def _generate(self, rng: np.random.Generator):
        n, p = self.n_samples, self.patch_size

        # --- risk labels: skewed toward NO_RISK / MODERATE, like real wildfire data ---
        # Longer horizons -> slightly higher uncertainty/risk, mimicking compounding drought signal
        base_probs = np.array([0.55, 0.32, 0.13])
        horizon_shift = {1: 0.0, 3: 0.03, 6: 0.06}[self.horizon]
        probs = base_probs + np.array([-horizon_shift, 0.0, horizon_shift])
        probs = np.clip(probs, 0.01, None)
        probs = probs / probs.sum()
        labels = rng.choice(3, size=n, p=probs)  # 0=NO_RISK, 1=MODERATE, 2=HIGH

        # --- underlying "dryness/fuel" latent factor drives every modality ---
        # HIGH risk -> lower NDVI/moisture, higher thermal anomaly, higher SAR backscatter dryness signal
        latent = labels.astype(np.float32) / 2.0 + rng.normal(0, 0.1, size=n).astype(np.float32)
        latent = np.clip(latent, 0, 1)

        # Optical patches: NDVI-like vegetation vigor drops with rising latent risk
        optical = rng.normal(loc=0.5 - 0.3 * latent[:, None, None, None],
                              scale=0.08,
                              size=(n, self.n_optical, p, p)).astype(np.float32)
        optical = np.clip(optical, 0, 1)

        # SAR patches: backscatter increases (drier canopy/soil) with latent risk
        sar = rng.normal(loc=-12 + 6 * latent[:, None, None, None],
                          scale=1.5,
                          size=(n, self.n_sar, p, p)).astype(np.float32)

        # Thermal patches: hotspot intensity increases with latent risk
        thermal = rng.gamma(shape=2.0 + 3.0 * latent[:, None, None, None],
                             scale=1.0,
                             size=(n, self.n_thermal, p, p)).astype(np.float32)

        # Weather time series: warmer/drier/windier trending into higher risk
        t = np.linspace(0, 1, self.seq_len)[None, :, None]
        trend = latent[:, None, None]
        noise = rng.normal(0, 1, size=(n, self.seq_len, self.n_weather)).astype(np.float32)
        temp = 10 + 15 * trend + 5 * np.sin(2 * np.pi * t) + noise[..., 0:1]
        wind = 3 + 8 * trend + noise[..., 1:2]
        humidity = np.clip(70 - 40 * trend + noise[..., 2:3], 5, 100)
        precip = np.clip(5 - 4.5 * trend + 0.5 * noise[..., 3:4], 0, None)
        dewpoint = temp - (100 - humidity) / 5 + 0.3 * noise[..., 4:5]
        weather = np.concatenate([temp, wind, humidity, precip, dewpoint], axis=-1).astype(np.float32)

        # spatial tile id, used for spatial-blocked splitting (prevents leakage)
        n_tiles = max(8, n // 40)
        tile_ids = rng.integers(0, n_tiles, size=n)

        self.optical = optical
        self.sar = sar
        self.thermal = thermal
        self.weather = weather
        self.labels = labels.astype(np.int64)
        self.tile_ids = tile_ids

    # ------------------------------------------------------------------
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        spatial = np.concatenate(
            [self.optical[idx], self.sar[idx], self.thermal[idx]], axis=0
        )  # [C_total, H, W]
        sample = {
            "spatial": torch.from_numpy(spatial),
            "weather": torch.from_numpy(self.weather[idx]),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "tile_id": int(self.tile_ids[idx]),
        }
        return sample

    @property
    def n_spatial_channels(self) -> int:
        return self.n_optical + self.n_sar + self.n_thermal


def spatial_block_split(dataset: AlaskaWildfireSyntheticDataset, cfg: dict, seed: int = 42):
    """Split by spatial tile_id (not by individual sample) to avoid data leakage,
    matching the '70/15/15 split with spatial blocking' called for in the methodology."""
    rng = np.random.default_rng(seed)
    unique_tiles = np.unique(dataset.tile_ids)
    rng.shuffle(unique_tiles)

    n_tiles = len(unique_tiles)
    n_train = int(cfg["split"]["train"] * n_tiles)
    n_val = int(cfg["split"]["val"] * n_tiles)

    train_tiles = set(unique_tiles[:n_train])
    val_tiles = set(unique_tiles[n_train:n_train + n_val])
    test_tiles = set(unique_tiles[n_train + n_val:])

    train_idx = [i for i, t in enumerate(dataset.tile_ids) if t in train_tiles]
    val_idx = [i for i, t in enumerate(dataset.tile_ids) if t in val_tiles]
    test_idx = [i for i, t in enumerate(dataset.tile_ids) if t in test_tiles]

    return train_idx, val_idx, test_idx
