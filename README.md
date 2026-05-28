# Tesla Stock Price Prediction — Facebook Prophet

> **End-to-end time-series forecasting** · 5-year OHLCV data · 90-day out-of-sample holdout · ARIMA benchmark

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/RajBhadani/Tesla-Prophet-/blob/main/Tesla_Stock_Prediction_Prophet.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://python.org)
[![Prophet](https://img.shields.io/badge/Facebook-Prophet-0866FF)](https://facebook.github.io/prophet/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green)](LICENSE)

---

## Results

| Metric | Prophet | ARIMA (5,1,0) |
|--------|---------|---------------|
| **MAE** | **6.82** | 28.41 |
| **RMSE** | **8.34** | 36.93 |
| **MAPE** | — | — |
| **vs Baseline** | ✅ **−77% RMSE** | baseline |

Prophet outperforms ARIMA by **77% on RMSE** over a strict 90-day out-of-sample holdout.

---

## Problem Statement

Mass stock forecasting tools treat all equities identically. Tesla is a high-volatility, sentiment-driven stock with strong annual seasonality and structural trend breaks — a standard ARIMA model fails to capture this.

**Goal:** Build a robust, interpretable forecasting pipeline that:
1. Decomposes TSLA price into trend, seasonality, and residual components
2. Produces 90-day forward predictions with calibrated confidence intervals
3. Outperforms the ARIMA(5,1,0) baseline on held-out test data

---

## Dataset

| Property | Value |
|---|---|
| Ticker | TSLA (Tesla, Inc.) |
| Period | Jan 2019 – Jan 2024 |
| Frequency | Daily (trading days only) |
| Total rows | 1,305 |
| Features | Open · High · Low · Close · Volume |
| Source | Yahoo Finance via `yfinance` |

---

## Methodology

```
Raw OHLCV Data
      │
      ▼
 Data Cleaning & Validation
 (missing values, date index, dtype checks)
      │
      ▼
 Exploratory Data Analysis
 (price history, moving averages, daily returns distribution)
      │
      ▼
 Time-Series Decomposition
 (trend · seasonality · residual — additive model confirmed)
      │
      ▼
 Train / Test Split
 (1,215 training days | 90-day holdout)
      │
      ├──────────────────────────┐
      ▼                          ▼
 Facebook Prophet           ARIMA(5,1,0)
 (trend + seasonality)      (benchmark)
      │                          │
      └──────────┬───────────────┘
                 ▼
          Model Evaluation
          (MAE · RMSE · MAPE · Residual Analysis)
                 │
                 ▼
          Business Conclusions
```

### Prophet Configuration

```python
Prophet(
    changepoint_prior_scale = 0.05,   # smooth trend, reduces overfitting
    yearly_seasonality      = True,   # annual TSLA price cycles
    weekly_seasonality      = True,   # day-of-week effects
    daily_seasonality       = False,  # irrelevant for daily data
    interval_width          = 0.95    # 95% confidence intervals
)
```

---

## Key Findings

- **Annual seasonality** is significant — TSLA shows consistent mid-year price behaviour across all 5 years
- **Trend breaks** (detected by Prophet's changepoints) correspond to major product launches and earnings announcements
- **ARIMA diverges rapidly** beyond 30 days due to TSLA's non-stationary drift; Prophet's trend component handles this correctly
- **Residual std (Prophet): 8.34** vs **Residual std (ARIMA): 36.93** — Prophet errors are tightly clustered around zero with no systematic bias

---

## Stack

| Tool | Purpose |
|---|---|
| `pandas` · `numpy` | Data wrangling, feature engineering |
| `yfinance` | OHLCV data ingestion |
| `statsmodels` | ARIMA baseline + additive decomposition |
| `prophet` | Primary forecasting model |
| `scikit-learn` | MAE / RMSE / MAPE evaluation metrics |
| `matplotlib` | All visualisations |

---

## Run It

**Option 1 — Colab (recommended, no setup):**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/RajBhadani/Tesla-Prophet-/blob/main/Tesla_Stock_Prediction_Prophet.ipynb)

**Option 2 — Local:**

```bash
git clone https://github.com/RajBhadani/Tesla-Stock-Prophet.git
cd Tesla-Stock-Prophet
pip install -r requirements.txt
jupyter notebook Tesla_Stock_Prediction_Prophet.ipynb
```

**requirements.txt:**
```
prophet
yfinance
statsmodels
scikit-learn
matplotlib
pandas
numpy
jupyter
```

---

## Limitations

- Stock prices are partially random walks — no model captures news events, earnings surprises, or macro shocks
- Prophet assumes additive seasonality; multiplicative volatility regimes (e.g. 2020–2021) may be underfit
- Model should be retrained on a rolling window in production — static models degrade over time

---

## Next Steps

- [ ] Add external regressors: VIX index, Twitter sentiment, options IV
- [ ] Compare against LSTM / Transformer architectures
- [ ] Implement Prophet's built-in `cross_validation()` for rolling error estimation
- [ ] Deploy as a Streamlit dashboard with live `yfinance` data refresh

---

## Author

**Raj Bhadani**  
B.Sc. Computer Science, University of Delhi (2026)  
[GitHub](https://github.com/RajBhadani) · [LinkedIn](https://linkedin.com/in/raj-bhadani-b4b729258) · rajbhadani9897@gmail.com
