"""
Real dataset loader: reads the patches exported in step 8 (data/processed/patch_{i}.npz)
and matches each one back to its (lon, lat) AND its label for a given horizon, since a
single set of extracted patches can serve multiple prediction horizons — the horizon only
changes the label (whether a fire happened within N months), not the underlying satellite
imagery or weather data.
"""
import glob
import os
import re

import geopandas as gpd
import numpy as np
import torch
from torch.utils.data import Dataset

LABEL_MAP = {"NO_RISK": 0, "MODERATE": 1, "HIGH": 2}


class AlaskaWildfireRealDataset(Dataset):
    def __init__(self, processed_dir: str, labeled_points_path: str, tile_size_deg: float = 0.5):
        self.processed_dir = processed_dir
        self.gdf = gpd.read_file(labeled_points_path)

        pattern = re.compile(r"patch_(\d+)\.npz$")
        files = glob.glob(os.path.join(processed_dir, "patch_*.npz"))

        self.indices = []
        for f in files:
            m = pattern.search(os.path.basename(f))
            if m and int(m.group(1)) in self.gdf.index:
                self.indices.append(int(m.group(1)))
        self.indices.sort()

        if len(self.indices) == 0:
            raise RuntimeError(
                f"No matching patches found in {processed_dir}. "
                "Did step 8's extraction loop finish and save any .npz files?"
            )

        lons = self.gdf.loc[self.indices, "geometry"].x.values
        lats = self.gdf.loc[self.indices, "geometry"].y.values
        tile_x = (lons // tile_size_deg).astype(int)
        tile_y = (lats // tile_size_deg).astype(int)
        self.tile_ids = np.array([f"{x}_{y}" for x, y in zip(tile_x, tile_y)])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        data = np.load(os.path.join(self.processed_dir, f"patch_{idx}.npz"))

        # Label always comes from the geojson (this horizon's labels), NOT from
        # whatever was baked into the .npz file when it was originally extracted.
        label_str = self.gdf.loc[idx, "risk_label"]
        label = LABEL_MAP[label_str]

        return {
            "spatial": torch.from_numpy(data["spatial"].astype(np.float32)),
            "weather": torch.from_numpy(data["weather"].astype(np.float32)),
            "label": torch.tensor(label, dtype=torch.long),
        }

    @property
    def n_spatial_channels(self) -> int:
        return self[0]["spatial"].shape[0]

    @property
    def n_weather_features(self) -> int:
        return self[0]["weather"].shape[1]


def spatial_block_split_real(dataset, train_frac=0.7, val_frac=0.15, seed=42):
    rng = np.random.default_rng(seed)
    unique_tiles = np.unique(dataset.tile_ids)
    rng.shuffle(unique_tiles)

    n = len(unique_tiles)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    train_tiles = set(unique_tiles[:n_train])
    val_tiles = set(unique_tiles[n_train:n_train + n_val])
    test_tiles = set(unique_tiles[n_train + n_val:])

    train_idx = [i for i, t in enumerate(dataset.tile_ids) if t in train_tiles]
    val_idx = [i for i, t in enumerate(dataset.tile_ids) if t in val_tiles]
    test_idx = [i for i, t in enumerate(dataset.tile_ids) if t in test_tiles]
    return train_idx, val_idx, test_idx