# Data Analysis and Post-Processing of Physics-Based Air Quality Forecasts

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-XGBoost-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📌 Project Overview
Global physics-based air quality models (such as WRF-Chem or CAMS) are excellent at predicting regional trends but often struggle with hyper-local urban pollution dynamics. In megacities like Delhi, local factors (traffic density, urban canyons, stagnation) cause significant deviation between forecasts and ground truth.

This project implements a **Machine Learning Bias-Correction Pipeline** that integrates historical residuals, temporal lags, and meteorological drivers to refine raw physics forecasts.

**Key Result:** The pipeline reduced the Forecast RMSE for Ozone (O3) by **~78%**, effectively calibrating the global model to local ground truth.

---

## 📉 The Problem & Solution
| Challenge | Solution |
| :--- | :--- |
| **Systematic Bias:** Physics models consistently over/under-predict during specific hours. | **ML Post-Processing:** An XGBoost regressor learns the bias function based on local weather conditions. |
| **Temporal Disconnect:** Global models miss short-term spikes (traffic rush). | **Lag Features:** Introduced `t-1` and `t-24` hour lag features to capture immediate persistence. |
| **Broken Time Series:** Sensor data has frequent gaps (power/network loss). | **Robust Pipeline:** Developed a custom preprocessing module to enforce hourly frequency and handle non-continuous time series. |

---

## 📊 Key Results
Evaluated on **5 years of hourly data (2019–2024)** across 7 monitoring stations in Delhi.

### 1. Ozone (O3) Correction
- **Baseline RMSE (Physics Model):** 32.31 µg/m³
- **ML Model RMSE:** 6.98 µg/m³
- **Improvement:** **78.4% Reduction in Error**
- **Refined Index of Agreement (RIA):** 0.89 (High agreement)

### 2. Nitrogen Dioxide (NO2) Correction
- **Baseline RMSE:** 66.13 µg/m³
- **ML Model RMSE:** 9.80 µg/m³
- **Improvement:** **85.2% Reduction in Error**



## 🛠️ Repository Structure
This project is structured as a modular data science pipeline, moving away from monolithic notebooks.

```text
Gases_Analysis/
│
├── data/                                   # (Ignored in git) Raw and Processed CSVs
├── analysis/                               # Analysis & Development
│   ├── 1_Data_Preprocessing.ipynb          # Quality checks & Null analysis
│   ├── 2_EDA.ipynb                         # Diurnal cycles & Correlation heatmaps
│   ├── 3_Feature_Engineering.ipynb         # Pipeline to create lags & cyclical time
│   ├── 4_Modeling_Metrics.ipynb            # XGBoost training loop (7 Sites)
│   └── 5_Evaluation.ipynb                  # Error density plots & Case studies
│
├── src/                    # Source Code
│   ├── preprocessing.py    # Time alignment & cleaning
│   ├── features.py         # Lag generation & Sin/Cos encoding
│   ├── modeling.py         # Training wrappers
│   └── metrics.py          # Custom RIA & RMSE calculations
│
└── requirements.txt        # Dependencies
```
## 🚀 How to Run
Clone the repository

```bash
git clone https://github.com/UtkarshKumarJha/Gases_Analysis.git
```
Install dependencies

```bash
pip install -r requirements.txt
Run the Analysis The project is designed to run sequentially. Start with data preparation:
```
```bash

# Run the feature engineering pipeline for all sites
jupyter notebook analysis/3_Feature_Engineering.ipynb
```

## 🧠 Methodology Highlights
Feature Engineering
* Cyclical Encoding: Converted Hour and Month into Sin/Cos pairs to preserve temporal continuity (e.g., Hour 23 is close to Hour 0).

* Lagged Predictors: Created Target_t-1 and Target_t-24 to allow the model to "remember" the immediate past pollution state.

* Satellite Data Decision: Satellite-derived NO2 (Sentinel-5P) was analyzed in EDA but dropped from the final model due to high sparsity (>30% missing data due to cloud cover), which introduced more noise than signal.

## Evaluation Strategy
* Chronological Splitting: Used the first 75% of dates for training and the last 25% for testing to prevent data leakage (respecting the arrow of time).

* Willmott's RIA: Used Refined Index of Agreement instead of just R² to better quantify model performance against natural variability.
