# Research Summary — AI-Powered Wildfire Risk Prediction for Alaska


## Why this matters
- Alaska's burned area is projected to increase up to **169%** by 2050.
- A recent fire season burned **1.68M acres** — roughly **2×** the 10-year state average.
- **Zero** published CNN-LSTM models exist specifically for Alaska's boreal/tundra ecosystem.

## Problem statement
No large-scale AI fire-risk prediction system exists for Alaska's unique combination of
boreal forest, tundra, and permafrost. Existing models are trained on California,
Mediterranean, or Australian data and don't capture Alaska's distinct climate, vegetation,
and ignition patterns.

## What this project actually built
1. **Real multi-source data pipeline** — fused Sentinel-2 (optical), Sentinel-1 (SAR), and
   MODIS (thermal) satellite imagery with ERA5 weather data, pulled directly via the Google
   Earth Engine API, for the Fairbanks North Star Borough, Alaska.
2. **Hybrid CNN-LSTM model** — a 3-class wildfire risk classifier (NO_RISK / MODERATE /
   HIGH), trained and validated at **all three proposed prediction horizons**: 1, 3, and
   6 months.
3. **Real evaluation against classical baselines** — Logistic Regression, Random Forest,
   and XGBoost, trained and tested on identical real data at each horizon.
4. **Interactive GIS dashboard** — a Streamlit/Folium application supporting all three
   horizons, visualizing real model predictions against real historical fire outcomes.

## Study area and data

**Study area:** Fairbanks North Star Borough, Interior Alaska (bounding box:
148.9°W–145.3°W, 63.5°N–65.5°N).

**1,586 real samples** were extracted (same satellite imagery and weather data reused
across all three horizons, since only the label — whether a fire occurred within the
window — depends on the horizon). Class distribution by horizon:

| Horizon | NO_RISK | MODERATE | HIGH |
|---|---|---|---|
| 1-month | 1,198 | 334 | 54 |
| 3-month | 938 | 514 | 134 |
| 6-month | 937 | 515 | 134 |

**Labeling rule:** a point was labeled HIGH if a real fire's recorded date fell within
the horizon window and within 10 km of the point; MODERATE if within 30 km; otherwise
NO_RISK.

## Key Results — all three horizons

| Horizon | CNN-LSTM Accuracy | CNN-LSTM Macro F1 | Best Baseline Accuracy |
|---|---|---|---|
| 1-month | 81.8% ± 4.7% | 0.446 ± 0.063 | 93.9% (LogReg / XGBoost) |
| 3-month | 74.6% ± 1.6% | 0.557 ± 0.020 | — |
| 6-month | 74.3% ± 1.5% | 0.568 ± 0.031 | — |

**Key finding:** a clear accuracy-versus-balance tradeoff emerged across horizons. The
1-month model has the highest raw accuracy but the lowest macro F1 and widest run-to-run
variance, reflecting its more imbalanced class distribution. The 3- and 6-month models
have lower raw accuracy but meaningfully higher macro F1 and roughly 3x tighter variance,
indicating more genuine, balanced, and reliable classification across all three risk
classes. Classical baselines outperformed the CNN-LSTM on raw accuracy at every horizon,
an expected result given deep architectures generally need more data than classical
methods to realize their advantage.

## Key Methodological Decisions
- **Study area:** Fairbanks North Star Borough — chosen for real, documented fire
  activity and to keep data extraction tractable.
- **Reused patches across horizons:** satellite/weather data was extracted once and
  relabeled for each horizon, since the underlying data doesn't depend on the horizon —
  only whether a fire later occurred does.
- **Class imbalance handling:** SMOTE oversampling applied to training data only (never
  validation/test), at each horizon independently.
- **Data leakage prevention:** spatial blocking (grouping by geographic tile before
  splitting) used for all train/val/test splits.
- **Missing SAR data handling:** Sentinel-1 coverage gaps were filled with a placeholder
  value rather than dropping affected samples, to avoid sampling bias.
- **Result reporting:** all results averaged across 5 independent training runs (random
  seeds) rather than a single run, to account for variability at this data scale.

## Limitations (Disclosed)
- Scoped to one borough, not statewide Alaska.
- Some samples lack real SAR coverage (filled with a placeholder value).
- HIGH-risk threshold (10 km) is an empirical modeling choice, not physically derived.
- ERA5 weather data provided 3 usable features, not the 5 originally planned.
- SMOTE was applied to flattened pixel vectors, not true image-domain augmentation.
- The accuracy-versus-balance tradeoff across horizons has only been observed in this
  single study area and dataset; generalization to other regions is untested.

## Interactive dashboard
A Streamlit/Folium dashboard visualizes the trained model's real predictions on real
extracted satellite patches, each compared against its true historical fire outcome,
with a toggleable heatmap overlay and summary statistics. A separate, clearly labeled
synthetic-data demo mode illustrates the pipeline concept without risk of confusing
fake and real results.

## Key references
1. Sun C. (2022). CNNs for South American wildfire imagery. *arXiv:2201.09671*.
2. Ghali & Akhloufi (2023). DL approaches for wildland fires. *Fire, 6(5), 192*.
3. Marjani et al. (2024). CNN-BiLSTM for near-real-time wildfire spread. *Remote Sens., 16(8)*.
4. Jain et al. (2020). ML applications in wildfire science. *Environmental Reviews, 28(4)*.
5. Gupta & Kim (2024). Geospatial wildfire risk modeling. *IEEE JSTARS, 17*.
6. Dong et al. (2024). DL framework for Fire Radiative Power prediction. *IEEE JSTARS, 17*.
7. Phillips et al. (2022). Escalating carbon emissions from boreal wildfires. *Science Advances, 8(17)*.
8. Kasischke & Turetsky (2006). Changes in North American boreal fire regime. *Can. J. For. Res., 34*.
9. Alaska Fire Service (AFS). Historical fire perimeters database. Bureau of Land Management.