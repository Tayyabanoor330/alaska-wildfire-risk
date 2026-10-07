# AI-Powered Wildfire Risk Prediction for Alaska's Boreal Forests

A hybrid **CNN-LSTM** deep learning system for classifying wildfire risk (NO_RISK / MODERATE / HIGH)
at 1-, 3-, and 6-month horizons, trained on **real** multi-modal satellite and climate data
(Sentinel-1 SAR, Sentinel-2 optical, MODIS thermal, ERA5 weather) fused with real historical
Alaska Fire Service fire perimeter records, for the Fairbanks North Star Borough, Alaska.

This repo contains a complete, working research pipeline — real data extraction via Google
Earth Engine, a trained and validated model at all three prediction horizons, a classical
ML baseline comparison, and an interactive GIS dashboard — not a prototype or demo.

## Key results

| Horizon | CNN-LSTM Accuracy (mean of 5 runs) | CNN-LSTM Macro F1 | Best Baseline Accuracy |
|---|---|---|---|
| 1-month | 81.8% ± 4.7% | 0.446 ± 0.063 | 93.9% (Logistic Regression / XGBoost) |
| 3-month | 74.6% ± 1.6% | 0.557 ± 0.020 | — |
| 6-month | 74.3% ± 1.5% | 0.568 ± 0.031 | — |

Classical baseline models (Logistic Regression, Random Forest, XGBoost) outperformed the
CNN-LSTM on raw accuracy at every horizon — an expected finding given the real dataset size
(1,586 samples), since deep architectures generally need more data than classical methods to
realize their advantage. Longer horizons produced meaningfully more balanced (higher macro F1)
and more stable (lower variance) results despite lower raw accuracy. Full discussion in
[`docs/research_summary.md`](docs/research_summary.md) and the full report.

## What this repo includes

- A **real Google Earth Engine data pipeline** (`src/data/gee_pipeline.py`) that pulls actual
  Sentinel-1 SAR, Sentinel-2 optical, and MODIS thermal imagery, fused with ERA5 weather data.
- **Real fire-risk labeling logic** (`src/data/labeling.py`) that derives HIGH/MODERATE/NO_RISK
  labels from real Alaska Fire Service historical fire perimeter records.
- The **Hybrid CNN-LSTM model** (spatial CNN branch + temporal LSTM branch + fusion head).
- A training pipeline with **70/15/15 spatially-blocked splits, focal loss, SMOTE oversampling**,
  and baseline comparisons (Logistic Regression, Random Forest, XGBoost).
- Evaluation code producing **accuracy, macro-F1, confusion matrices**, averaged across
  multiple training runs for robustness.
- A **Folium/Streamlit GIS dashboard** showing real trained-model predictions on real satellite
  patches, compared against real historical fire outcomes, switchable across all three horizons.
- A **synthetic data generator** (`src/data/synthetic_data.py`), retained for quickly testing
  the pipeline's code paths without needing satellite data access.

> **Note on data:** due to file size, the extracted satellite patches, trained model
> checkpoints, and raw fire perimeter shapefile are not included in this repo. See
> `notebooks/01_data_collection.ipynb` and `notebooks/02_real_data_training.ipynb` to
> reproduce them (requires a free Google Earth Engine account).

---

## What is not included in this repository

Due to file size, the extracted satellite patches, trained model
checkpoints, and raw fire perimeter data are not included in this repo.
See `notebooks/01_data_collection.ipynb` and
`notebooks/02_real_data_training.ipynb` to reproduce them.


## 1. Project layout
```
alaska-wildfire-prediction/
├── README.md
├── requirements.txt
├── config/
│ └── config.yaml # all hyperparameters & paths in one place
├── docs/
│ └── research_summary.md # literature review, methodology, full results
├── notebooks/
│ ├── 01_data_collection.ipynb # real GEE extraction, fire-perimeter labeling
│ └── 02_real_data_training.ipynb # real training, SMOTE, evaluation, all 3 horizons
├── src/
│ ├── data/
│ │ ├── gee_pipeline.py # real Sentinel-1/2 + MODIS + ERA5 pipeline (GEE)
│ │ ├── labeling.py # fire-perimeter-based risk labeling logic
│ │ ├── real_data.py # PyTorch Dataset for real extracted patches
│ │ └── synthetic_data.py # synthetic data generator (for quick pipeline testing)
│ ├── models/
│ │ └── cnn_lstm.py # Hybrid CNN-LSTM architecture (PyTorch)
│ ├── training/
│ │ └── train.py # training loop, focal loss, splits, baselines
│ ├── evaluation/
│ │ └── metrics.py # accuracy / F1 / confusion matrix helpers
│ └── dashboard/
│ └── app.py # Streamlit app: real + synthetic risk dashboard
├── data/
│ ├── raw/ # (gitignored) downloaded fire perimeter data
│ └── processed/ # (gitignored) extracted patches + label files
├── outputs/
│ ├── checkpoints/ # (gitignored) trained model weights per horizon
│ └── figures/ # confusion matrices and training curves
└── tests/
└── test_model.py # sanity-check unit tests for model & data shapes
```


## 2. Setup

### Option A — plain venv / pip (works in VS Code or terminal)

```bash
cd alaska-wildfire-prediction
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Option B — Jupyter

```bash
pip install -r requirements.txt
jupyter lab
```

### Option C — VS Code

Open the folder in VS Code, select the `.venv` interpreter, install the "Python" and
"Jupyter" extensions if prompted, then open any `.ipynb` file and run cells directly.

## 3. Reproducing the real results

This requires a free [Google Earth Engine](https://earthengine.google.com/) account.

1. `pip install earthengine-api` (already in `requirements.txt`), then run `earthengine authenticate`.
2. Set your GEE project ID in `src/data/gee_pipeline.py` (or via the `EE_PROJECT` environment variable).
3. Download the Alaska Fire Service historical fire perimeter dataset (GeoJSON) from the
   [Alaska Fire Service predictive services page](https://fire.ak.blm.gov/predsvcs/maps.php)
   and place it under `data/raw/`.
4. Run `notebooks/01_data_collection.ipynb` top to bottom — this builds the sample grid,
   labels it against real fire history, and extracts real satellite + weather patches via
   the Earth Engine API (this step can take a long time depending on sample count, since
   each patch is a live API call).
5. Run `notebooks/02_real_data_training.ipynb` top to bottom — this relabels the extracted
   patches for all three horizons, trains the CNN-LSTM with SMOTE across 5 random seeds per
   horizon, evaluates against classical baselines, and saves checkpoints for the dashboard.

## 4. Quickstart without real data (pipeline test only)

```bash
pip install -r requirements.txt
python -m src.training.train --epochs 5 --synthetic
```

This trains on synthetic, randomly generated data purely to confirm the code pipeline runs
end to end. **Results from this mode are not meaningful and are never reported as project
findings** — they exist only to validate the architecture and training loop.

## 5. Running the dashboard

```bash
streamlit run src/dashboard/app.py
```

Supports three modes: real trained-model predictions (switchable across 1/3/6-month
horizons), a clearly-labeled synthetic-model demo, and an illustrative synthetic risk grid.

## 6. What's implemented

| Component | Status |
|---|---|
| Hybrid CNN-LSTM architecture | ✅ fully implemented and trained on real data |
| Real Sentinel-1/2 + MODIS + ERA5 extraction via GEE | ✅ fully implemented |
| Real fire-perimeter-based labeling (1/3/6-month horizons) | ✅ fully implemented |
| Training loop (focal loss, spatial blocking, SMOTE) | ✅ fully implemented |
| Baselines (Logistic Regression, Random Forest, XGBoost) | ✅ fully implemented |
| Evaluation (accuracy, macro-F1, confusion matrix, multi-seed averaging) | ✅ fully implemented |
| Folium/Streamlit GIS dashboard (real + synthetic modes) | ✅ fully implemented |
| Synthetic data generator | ✅ retained for quick pipeline testing |

## 7. Background and full results

See [`docs/research_summary.md`](docs/research_summary.md) for the full literature review,
research gap analysis, methodology, complete results across all three horizons, and
disclosed limitations.