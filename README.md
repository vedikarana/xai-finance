# 📈 XAI Finance — Explainable AI for Stock Market Prediction

**Seminar Project | T.Y.B.Tech CSE (AIDS) | MIT-WPU | Semester VI (2025-26)**  
**Student:** Vedika Rana | **PRN:** 1032233559 | **Roll No.:** 49 | **Panel B**

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Streamlit app
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
xai_finance/
│
├── app.py                        ← Main Streamlit dashboard
│
├── utils/
│   └── data_pipeline.py          ← Stock data fetching & feature engineering
│
├── models/
│   └── ml_models.py              ← Random Forest, XGBoost, LSTM training & evaluation
│
├── explainers/
│   └── xai_explainers.py         ← SHAP, LIME, Feature Importance
│
└── requirements.txt
```

---

## 🧠 Project Overview

This project implements a complete **end-to-end Explainable AI pipeline** for stock market
direction prediction (UP / DOWN next day).

### Data Pipeline
- Fetches real-time historical data via **yfinance**
- Engineers **25+ technical indicators** as ML features:
  - Momentum: RSI, Stochastic Oscillator
  - Trend: MACD, EMA (12/26), SMA (20/50)
  - Volatility: Bollinger Bands, ATR
  - Volume: On-Balance Volume (OBV)
  - Price: 1-day, 5-day, 10-day returns, log returns, lag features

### Machine Learning Models
| Model | Type | Strength |
|-------|------|----------|
| **Random Forest** | Ensemble (Bagging) | Robust, interpretable via feature importance |
| **XGBoost** | Ensemble (Boosting) | High accuracy, built-in importance |
| **LSTM** | Deep Learning (RNN) | Captures temporal patterns in sequences |

### XAI Techniques
| Technique | Scope | Description |
|-----------|-------|-------------|
| **SHAP** | Global + Local | Game-theory based feature attribution |
| **LIME** | Local | Local linear approximation for single predictions |
| **Feature Importance** | Global | Built-in impurity/gain-based importance |

---

## 📊 Dashboard Tabs

1. **📊 Stock Data** — Candlestick chart, correlation heatmap, feature data
2. **🤖 Model Performance** — Accuracy, AUC-ROC, confusion matrices, classification reports
3. **🔍 SHAP Explanations** — Beeswarm, bar chart, waterfall plot
4. **🧩 LIME Explanations** — Per-sample local explanation with feature weights
5. **📌 Feature Importance** — RF & XGBoost comparison + radar chart

---

## 📚 References

1. S. S. A. Shahid et al., "Interpretability in Financial Forecasting: XAI in Pakistan Stock Exchange," *IEEE Access*, 2024.
2. M. A. Khan et al., "Explainable AI-Supported Financial Forecasting for EV Charging Infrastructure Investments," *Proc. IEEE Int. Conf.*, 2024.
3. M. A. Khan, A. A. Khan, and S. S. Naqvi, "An explainable deep learning approach for stock market trend prediction," *PLoS ONE*, vol. 19, no. 11, 2024.
4. "Explainable-AI Powered stock price prediction using time series transformers," *arXiv:2506.06345*, Jun. 2025.
5. "Explainable AI for Stock Price Prediction," *Proc. ACM Int. Conf.* DOI: 10.1145/3784833.3784870, 2026.

---

## ⚠️ Disclaimer

This project is for educational/research purposes only. Predictions should not be used
for actual investment decisions.
