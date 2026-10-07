"""Sanity-check unit tests. Run with: pytest tests/"""
import os
import sys

import torch
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.synthetic_data import AlaskaWildfireSyntheticDataset, spatial_block_split
from src.models.cnn_lstm import build_model, FocalLoss


def load_test_config():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["n_samples"] = 64  # keep tests fast
    return cfg


def test_synthetic_dataset_shapes():
    cfg = load_test_config()
    ds = AlaskaWildfireSyntheticDataset(cfg, horizon=1)
    sample = ds[0]
    p = cfg["data"]["patch_size"]
    expected_channels = cfg["data"]["n_optical_bands"] + cfg["data"]["n_sar_bands"] + cfg["data"]["n_thermal_bands"]
    assert sample["spatial"].shape == (expected_channels, p, p)
    assert sample["weather"].shape == (cfg["data"]["weather_seq_len"], cfg["data"]["n_weather_features"])
    assert sample["label"].item() in (0, 1, 2)


def test_spatial_block_split_no_leakage():
    cfg = load_test_config()
    ds = AlaskaWildfireSyntheticDataset(cfg, horizon=1)
    train_idx, val_idx, test_idx = spatial_block_split(ds, cfg)
    train_tiles = {ds.tile_ids[i] for i in train_idx}
    val_tiles = {ds.tile_ids[i] for i in val_idx}
    test_tiles = {ds.tile_ids[i] for i in test_idx}
    assert train_tiles.isdisjoint(val_tiles)
    assert train_tiles.isdisjoint(test_tiles)
    assert val_tiles.isdisjoint(test_tiles)


def test_model_forward_pass():
    cfg = load_test_config()
    ds = AlaskaWildfireSyntheticDataset(cfg, horizon=1)
    model = build_model(cfg, ds.n_spatial_channels, cfg["data"]["n_weather_features"])

    batch = [ds[i] for i in range(4)]
    spatial = torch.stack([b["spatial"] for b in batch])
    weather = torch.stack([b["weather"] for b in batch])

    logits = model(spatial, weather)
    assert logits.shape == (4, cfg["model"]["num_classes"])


def test_focal_loss_runs():
    logits = torch.randn(8, 3)
    targets = torch.randint(0, 3, (8,))
    loss_fn = FocalLoss(gamma=2.0)
    loss = loss_fn(logits, targets)
    assert loss.item() >= 0
