"""
Hybrid CNN-LSTM architecture for Alaska wildfire risk classification.

Matches the "Model Architecture" slide:
  Monthly satellite patches -> CNN encoder -> spatial feature maps  --\
                                                                        >-- Fusion -> FC -> HIGH/MOD/NO RISK
  ERA5+NOAA weather time-series -> LSTM sequence encoder             --/

Trained separately per prediction horizon (1 / 3 / 6 month), or with the horizon
passed in as a conditioning feature — see `HORIZON_CONDITIONING` below.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """Spatial branch: encodes a fused optical+SAR+thermal patch."""

    def __init__(self, in_channels: int, out_channels_list=(16, 32, 64), dropout: float = 0.3):
        super().__init__()
        layers = []
        c_in = in_channels
        for c_out in out_channels_list:
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_channels_list[-1]

    def forward(self, x):
        x = self.conv(x)              # [B, C, H', W']
        x = self.pool(x).flatten(1)   # [B, C]
        return self.dropout(x)


class LSTMEncoder(nn.Module):
    """Temporal branch: encodes the daily weather (ERA5/NOAA) time series."""

    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dim = hidden_size

    def forward(self, x):
        # x: [B, T, F]
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]  # [B, hidden_size] — final layer's last hidden state


class HybridCNNLSTM(nn.Module):
    """Fuses spatial (CNN) + temporal (LSTM) branches and classifies fire risk."""

    def __init__(
        self,
        n_spatial_channels: int,
        n_weather_features: int,
        cnn_out_channels=(16, 32, 64),
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 1,
        fusion_hidden_size: int = 128,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.cnn = CNNEncoder(n_spatial_channels, cnn_out_channels, dropout)
        self.lstm = LSTMEncoder(n_weather_features, lstm_hidden_size, lstm_num_layers, dropout)

        fusion_in = self.cnn.out_dim + self.lstm.out_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_size, num_classes),
        )

    def forward(self, spatial: torch.Tensor, weather: torch.Tensor) -> torch.Tensor:
        """
        spatial: [B, C, H, W]  fused optical+SAR+thermal patch
        weather: [B, T, F]     daily ERA5/NOAA time series
        returns: [B, num_classes] raw logits (HIGH / MODERATE / NO RISK)
        """
        spatial_feat = self.cnn(spatial)
        temporal_feat = self.lstm(weather)
        fused = torch.cat([spatial_feat, temporal_feat], dim=1)
        return self.fusion(fused)


def build_model(cfg: dict, n_spatial_channels: int, n_weather_features: int) -> HybridCNNLSTM:
    m = cfg["model"]
    return HybridCNNLSTM(
        n_spatial_channels=n_spatial_channels,
        n_weather_features=n_weather_features,
        cnn_out_channels=tuple(m["cnn_out_channels"]),
        lstm_hidden_size=m["lstm_hidden_size"],
        lstm_num_layers=m["lstm_num_layers"],
        fusion_hidden_size=m["fusion_hidden_size"],
        num_classes=m["num_classes"],
        dropout=m["dropout"],
    )


class FocalLoss(nn.Module):
    """Focal loss for class imbalance (HIGH risk is rare) — per the training strategy slide."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()
