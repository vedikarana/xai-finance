"""
╔══════════════════════════════════════════════════════════════════════════╗
║   StockSense AI — Smart Stock Analysis with Clear Reasoning              ║
║   Built by Vedika Rana — CFA L1 Candidate | SEBI Verified Advisor        ║
║                                                                           ║
╚══════════════════════════════════════════════════════════════════════════╝

Run:   streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─── Project Modules ──────────────────────────────────────────────────────────
from utils.data_pipeline import (
    fetch_stock_data, engineer_features, fetch_news_sentiment,
    get_feature_columns, prepare_train_test,
)
from models.ml_models import (
    train_random_forest, train_xgboost, train_lstm,
    evaluate_model, prepare_lstm_data,
)
from utils.robo_advisor import (
    RISK_PROFILES, STOCK_UNIVERSE, parse_portfolio_csv,
    analyse_uploaded_portfolio, design_portfolio,
    plot_sector_pie, plot_allocation_bar, plot_risk_radar,
    generate_portfolio_csv,
)
from utils.portfolio_xai import (
    build_portfolio, aggregate_portfolio_shap, portfolio_signal_table,
    plot_portfolio_shap_bar, plot_portfolio_heatmap,
    portfolio_diversification_score, plot_return_correlation,
)
from explainers.xai_explainers import (
    compute_shap_values, plot_shap_summary, plot_shap_bar,
    plot_shap_waterfall, explain_with_lime,
    plot_feature_importance, get_top_features_df,
    compute_counterfactuals, plot_counterfactual,
    compute_lstm_attention, plot_lstm_attention,
    run_backtest, plot_equity_curve,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page Config & Global Styling
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="StockSense AI — Smart Stock Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    /* ── Base — Groww bright white ── */
    html, body, .stApp {
        background: #f7f8fa !important;
        color: #1a1a2e;
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e8eaf0;
    }
    [data-testid="stSidebar"] label { color: #4b5563 !important; font-size: 0.82rem; font-weight: 600; }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown div { color: #4b5563 !important; font-size: 0.82rem; }
    [data-testid="stSidebar"] .stSelectbox > div,
    [data-testid="stSidebar"] .stTextInput > div > div {
        background: #f7f8fa !important;
        border: 1.5px solid #e0e3eb !important;
        border-radius: 10px !important;
        color: #1a1a2e !important;
    }

    /* ── Main area ── */
    .block-container { padding: 1.5rem 2rem 3rem 2rem !important; max-width: 1400px; }

    /* ── Tabs — Groww green pill ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #ffffff;
        border-radius: 14px;
        padding: 5px 6px;
        gap: 4px;
        border: 1.5px solid #e8eaf0;
        box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        padding: 10px 22px;
        color: #6b7280;
        font-weight: 600;
        font-size: 0.88rem;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background: #00b386 !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 2px 10px rgba(0,179,134,0.35);
    }

    /* ── Buttons — Groww green ── */
    .stButton > button {
        background: linear-gradient(135deg, #00b386, #00916e);
        color: white !important;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 13px 24px;
        letter-spacing: 0.3px;
        transition: all 0.2s ease;
        width: 100%;
        box-shadow: 0 4px 14px rgba(0,179,134,0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,179,134,0.45);
    }

    /* ── Cards ── */
    .g-card {
        background: #ffffff;
        border: 1.5px solid #e8eaf0;
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.05);
    }

    /* ── Insight boxes ── */
    .insight-box {
        background: #f0fdf9;
        border: 1.5px solid #a7f3d0;
        border-radius: 12px;
        padding: 14px 18px;
        margin: 12px 0;
        font-size: 0.9rem;
        color: #065f46;
        line-height: 1.75;
    }
    .insight-box b { color: #047857; }
    .insight-box .title {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #059669;
        margin-bottom: 6px;
    }
    .warn-box {
        background: #fffbeb;
        border: 1.5px solid #fde68a;
        border-radius: 12px;
        padding: 14px 18px;
        margin: 12px 0;
        color: #92400e;
        font-size: 0.9rem;
    }
    .info-box {
        background: #eff6ff;
        border: 1.5px solid #bfdbfe;
        border-radius: 12px;
        padding: 14px 18px;
        margin: 12px 0;
        font-size: 0.9rem;
        color: #1e40af;
        line-height: 1.75;
    }

    /* ── Section headers ── */
    .groww-header {
        font-size: 1.25rem;
        font-weight: 800;
        color: #111827;
        margin: 0 0 4px 0;
        letter-spacing: -0.3px;
    }
    .groww-subtext {
        font-size: 0.88rem;
        color: #6b7280;
        margin-bottom: 18px;
        line-height: 1.6;
    }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1.5px solid #e8eaf0;
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricValue"] { color: #111827 !important; font-size: 1.5rem !important; font-weight: 800 !important; }
    [data-testid="stMetricLabel"] { color: #6b7280 !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 1px; }

    /* ── Progress ── */
    .stProgress > div > div { background: linear-gradient(90deg, #00b386, #00d4ff) !important; border-radius: 6px; }

    /* ── DataFrame ── */
    [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1.5px solid #e8eaf0; box-shadow: 0 1px 4px rgba(0,0,0,0.05); }

    /* ── Divider ── */
    hr { border-color: #e8eaf0 !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: #f1f5f9; }
    ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }

    /* ── Radio ── */
    .stRadio label {
        background: #ffffff !important;
        border: 1.5px solid #e0e3eb !important;
        border-radius: 10px !important;
        padding: 8px 16px !important;
        color: #374151 !important;
        font-size: 0.88rem;
        font-weight: 600;
    }

    /* ── Hero ── */
    .groww-hero {
        background: linear-gradient(135deg, #ffffff 0%, #f0fdf9 100%);
        border: 1.5px solid #d1fae5;
        border-radius: 20px;
        padding: 28px 36px;
        margin-bottom: 20px;
        box-shadow: 0 2px 12px rgba(0,179,134,0.1);
    }
    .groww-hero h1 { font-size: 1.9rem; font-weight: 900; color: #111827; margin: 0 0 6px; letter-spacing: -0.5px; }
    .groww-hero span.accent { color: #00b386; }
    .groww-hero p { color: #6b7280; margin: 0; font-size: 0.92rem; }

    /* ── Chart wrapper ── */
    .chart-wrap {
        background: #ffffff;
        border: 1.5px solid #e8eaf0;
        border-radius: 16px;
        padding: 4px 4px 0 4px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.05);
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Exchange & Currency Config
# ─────────────────────────────────────────────────────────────────────────────

# ── Global ticker search: any valid yfinance ticker works ──────────────────
EXCHANGES = {
    "🇮🇳  India (NSE)": {
        "currency": "₹", "currency_name": "INR", "flag": "🇮🇳",
        "stocks": {
            "Reliance Industries": "RELIANCE.NS", "TCS": "TCS.NS",
            "Infosys": "INFY.NS", "HDFC Bank": "HDFCBANK.NS",
            "Wipro": "WIPRO.NS", "ICICI Bank": "ICICIBANK.NS",
            "Bajaj Finance": "BAJFINANCE.NS", "HCL Tech": "HCLTECH.NS",
            "Adani Ports": "ADANIPORTS.NS", "SBI": "SBIN.NS",
            "Maruti Suzuki": "MARUTI.NS", "Sun Pharma": "SUNPHARMA.NS",
            "Titan": "TITAN.NS", "Asian Paints": "ASIANPAINT.NS",
            "Hindustan Unilever": "HINDUNILVR.NS", "Kotak Bank": "KOTAKBANK.NS",
            "L&T": "LT.NS", "Axis Bank": "AXISBANK.NS",
            "Dr Reddy": "DRREDDY.NS", "Nestle India": "NESTLEIND.NS",
            "ITC": "ITC.NS", "ONGC": "ONGC.NS", "NTPC": "NTPC.NS",
            "Power Grid": "POWERGRID.NS", "Tech Mahindra": "TECHM.NS",
            "M&M": "M&M.NS", "JSW Steel": "JSWSTEEL.NS",
            "Tata Motors": "TATAMOTORS.NS", "Tata Steel": "TATASTEEL.NS",
            "UltraTech Cement": "ULTRACEMCO.NS",
        },
    },
    "🇮🇳  India (BSE)": {
        "currency": "₹", "currency_name": "INR", "flag": "🇮🇳",
        "stocks": {
            "Reliance (BSE)": "500325.BO", "TCS (BSE)": "532540.BO",
            "Infosys (BSE)": "500209.BO", "HDFC Bank (BSE)": "500180.BO",
            "Wipro (BSE)": "507685.BO", "ICICI Bank (BSE)": "532174.BO",
            "SBI (BSE)": "500112.BO", "ITC (BSE)": "500875.BO",
            "Tata Motors (BSE)": "500570.BO", "Bajaj Auto (BSE)": "532977.BO",
        },
    },
    "🇺🇸  USA (NYSE/NASDAQ)": {
        "currency": "$", "currency_name": "USD", "flag": "🇺🇸",
        "stocks": {
            "Apple": "AAPL", "Microsoft": "MSFT", "Google": "GOOGL",
            "Amazon": "AMZN", "NVIDIA": "NVDA", "Tesla": "TSLA",
            "Meta": "META", "Netflix": "NFLX", "JPMorgan": "JPM",
            "Coca-Cola": "KO", "Walt Disney": "DIS", "Visa": "V",
            "Mastercard": "MA", "Salesforce": "CRM", "Adobe": "ADBE",
            "Intel": "INTC", "AMD": "AMD", "Qualcomm": "QCOM",
            "Broadcom": "AVGO", "Oracle": "ORCL", "Cisco": "CSCO",
            "IBM": "IBM", "Goldman Sachs": "GS", "Morgan Stanley": "MS",
            "Bank of America": "BAC", "Wells Fargo": "WFC",
            "Johnson & Johnson": "JNJ", "Pfizer": "PFE", "UnitedHealth": "UNH",
            "Exxon Mobil": "XOM", "Chevron": "CVX", "Boeing": "BA",
            "Caterpillar": "CAT", "3M": "MMM", "Nike": "NKE",
            "Starbucks": "SBUX", "McDonald's": "MCD", "Walmart": "WMT",
            "Target": "TGT", "Home Depot": "HD", "Costco": "COST",
            "Berkshire Hathaway": "BRK-B", "Palantir": "PLTR",
            "Snowflake": "SNOW", "Datadog": "DDOG", "CrowdStrike": "CRWD",
            "Uber": "UBER", "Airbnb": "ABNB", "Shopify": "SHOP",
            "Spotify": "SPOT", "Twitter/X": "X", "Rivian": "RIVN",
        },
    },
    "🇯🇵  Japan (TSE)": {
        "currency": "¥", "currency_name": "JPY", "flag": "🇯🇵",
        "stocks": {
            "Toyota": "7203.T", "Sony": "6758.T", "SoftBank": "9984.T",
            "Honda": "7267.T", "Keyence": "6861.T", "Mitsubishi": "8058.T",
            "Nintendo": "7974.T", "Fast Retailing": "9983.T",
            "Panasonic": "6752.T", "Canon": "7751.T", "Hitachi": "6501.T",
            "Fujitsu": "6702.T", "NTT": "9432.T", "Sharp": "6753.T",
            "Bridgestone": "5108.T", "Olympus": "7733.T",
        },
    },
    "🇸🇬  Singapore (SGX)": {
        "currency": "S$", "currency_name": "SGD", "flag": "🇸🇬",
        "stocks": {
            "DBS Group": "D05.SI", "OCBC Bank": "O39.SI", "UOB": "U11.SI",
            "Singapore Airlines": "C6L.SI", "CapitaLand": "9CI.SI",
            "Keppel Corp": "BN4.SI", "Wilmar Intl": "F34.SI",
            "Genting Singapore": "G13.SI", "Jardine Matheson": "J36.SI",
            "ST Engineering": "S63.SI", "ComfortDelGro": "C52.SI",
        },
    },
    "🇨🇳  China (HK-listed)": {
        "currency": "HK$", "currency_name": "HKD", "flag": "🇨🇳",
        "stocks": {
            "Tencent": "0700.HK", "Alibaba HK": "9988.HK",
            "Meituan": "3690.HK", "JD.com HK": "9618.HK",
            "NetEase HK": "9999.HK", "Xiaomi": "1810.HK",
            "BYD HK": "1211.HK", "CNOOC": "0883.HK",
            "China Mobile": "0941.HK", "ICBC": "1398.HK",
            "Ping An": "2318.HK", "Baidu HK": "9888.HK",
        },
    },
    "🇨🇳  China (US-listed ADR)": {
        "currency": "$", "currency_name": "USD", "flag": "🇨🇳",
        "stocks": {
            "Alibaba ADR": "BABA", "JD.com ADR": "JD",
            "Baidu ADR": "BIDU", "NIO ADR": "NIO",
            "PDD Holdings": "PDD", "Li Auto": "LI",
            "Xpeng": "XPEV", "NetEase ADR": "NTES",
            "Trip.com": "TCOM", "iQIYI": "IQ",
        },
    },
    "🇬🇧  UK (LSE)": {
        "currency": "£", "currency_name": "GBP", "flag": "🇬🇧",
        "stocks": {
            "HSBC": "HSBA.L", "BP": "BP.L", "Unilever": "ULVR.L",
            "AstraZeneca": "AZN.L", "Shell": "SHEL.L", "Barclays": "BARC.L",
            "GSK": "GSK.L", "Vodafone": "VOD.L", "Rolls-Royce": "RR.L",
            "Diageo": "DGE.L", "Glencore": "GLEN.L", "Rio Tinto": "RIO.L",
            "BT Group": "BT-A.L", "Lloyds Bank": "LLOY.L",
        },
    },
    "🇩🇪  Germany (XETRA)": {
        "currency": "€", "currency_name": "EUR", "flag": "🇩🇪",
        "stocks": {
            "SAP": "SAP.DE", "Siemens": "SIE.DE", "Volkswagen": "VOW3.DE",
            "BMW": "BMW.DE", "Deutsche Bank": "DBK.DE", "Bayer": "BAYN.DE",
            "Adidas": "ADS.DE", "BASF": "BAS.DE", "Allianz": "ALV.DE",
            "Deutsche Telekom": "DTE.DE", "Mercedes-Benz": "MBG.DE",
            "Infineon": "IFX.DE", "Deutsche Post": "DHL.DE",
        },
    },
    "🇭🇰  Hong Kong (HKEX)": {
        "currency": "HK$", "currency_name": "HKD", "flag": "🇭🇰",
        "stocks": {
            "HSBC HK": "0005.HK", "AIA Group": "1299.HK",
            "Alibaba HK": "9988.HK", "Tencent": "0700.HK",
            "Meituan": "3690.HK", "CK Hutchison": "0001.HK",
            "Sun Hung Kai": "0016.HK", "MTR Corp": "0066.HK",
            "Hang Seng Bank": "0011.HK", "CLP Holdings": "0002.HK",
        },
    },
    "🇰🇷  South Korea (KRX)": {
        "currency": "₩", "currency_name": "KRW", "flag": "🇰🇷",
        "stocks": {
            "Samsung Electronics": "005930.KS", "SK Hynix": "000660.KS",
            "LG Electronics": "066570.KS", "Hyundai Motor": "005380.KS",
            "POSCO": "005490.KS", "Kakao": "035720.KS",
            "Naver": "035420.KS", "LG Chem": "051910.KS",
            "Kia Motors": "000270.KS", "SK Telecom": "017670.KS",
        },
    },
    "🇦🇺  Australia (ASX)": {
        "currency": "A$", "currency_name": "AUD", "flag": "🇦🇺",
        "stocks": {
            "BHP Group": "BHP.AX", "Commonwealth Bank": "CBA.AX",
            "CSL": "CSL.AX", "ANZ Bank": "ANZ.AX",
            "Westpac": "WBC.AX", "NAB": "NAB.AX",
            "Macquarie": "MQG.AX", "Wesfarmers": "WES.AX",
            "Woolworths": "WOW.AX", "Rio Tinto ASX": "RIO.AX",
            "Fortescue": "FMG.AX", "Woodside": "WDS.AX",
        },
    },
    "🇧🇷  Brazil (B3)": {
        "currency": "R$", "currency_name": "BRL", "flag": "🇧🇷",
        "stocks": {
            "Petrobras": "PETR4.SA", "Vale": "VALE3.SA",
            "Itaú Unibanco": "ITUB4.SA", "Bradesco": "BBDC4.SA",
            "Ambev": "ABEV3.SA", "Embraer": "EMBR3.SA",
            "Magazine Luiza": "MGLU3.SA", "WEG": "WEGE3.SA",
        },
    },
    "🌐  Custom Ticker": {
        "currency": "$", "currency_name": "USD", "flag": "🌐",
        "stocks": {"Enter ticker manually below": "CUSTOM"},
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📈 XAI Finance")
    st.markdown("**Explainable AI for Stock Market**")
    st.divider()

    st.markdown("#### 🌍 Step 1 — Choose Exchange")
    exchange_choice = st.selectbox(
        "Stock Exchange", list(EXCHANGES.keys()), index=0, label_visibility="collapsed"
    )
    exc = EXCHANGES[exchange_choice]
    CURRENCY   = exc["currency"]
    CURR_NAME  = exc["currency_name"]

    st.markdown("#### 📌 Step 2 — Choose Stock")
    stock_names = list(exc["stocks"].keys())
    stock_choice = st.selectbox("Stock", stock_names, index=0, label_visibility="collapsed")
    TICKER = exc["stocks"][stock_choice]

    custom = st.text_input("✏️ Or type custom ticker:", placeholder="e.g. RELIANCE.NS, TSLA")
    if custom.strip():
        TICKER = custom.strip().upper()

    st.divider()
    st.markdown("#### ⚙️ Step 3 — Settings")
    PERIOD      = st.selectbox("📅 Period", ["1y", "2y", "3y", "5y"], index=1)
    RF_TREES    = st.slider("🌲 Accuracy (Forest)",        50, 500, 200, 50)
    XGB_TREES   = st.slider("⚡ Accuracy (Trend)",  50, 500, 200, 50)
    LSTM_EPOCHS = st.slider("🧠 Memory Depth (AI)",     10, 60,  30,  5)
    SAMPLE_IDX  = st.slider("🔍 XAI Sample #",    0,  49,  0)

    st.divider()
    run_btn = st.button("🚀  Train & Explain", use_container_width=True)

    st.divider()
    st.markdown("""
    <div style='font-size:0.78rem; color:#6b7280; line-height:1.8; padding:8px 0'>
    <b style='color:#111827'>Vedika Rana</b><br>
    <span style='color:#00b386; font-size:0.74rem'>CFA Level I Candidate &nbsp;|&nbsp; SEBI Verified Advisor</span>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class='groww-hero'>
  <h1>📈 <span class='accent'>StockSense</span> AI</h1>
  <p>Smart stock predictions with clear reasoning — know <b>exactly why</b> the AI thinks a stock will go UP or DOWN</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────────────────────

def _fresh_state():
    return dict(
        trained=False, df=None, df_feat=None,
        X_train=None, X_test=None, y_train=None, y_test=None,
        scaler=None, features=None,
        rf=None, xgb_model=None, lstm_model=None,
        lstm_X_test=None, lstm_y_test=None,
        rf_metrics=None, xgb_metrics=None, lstm_metrics=None,
        shap_vals_rf=None, shap_vals_xgb=None, shap_X_rf=None, shap_X_xgb=None,
        sentiment_series=None, sentiment_available=False,
        backtest_rf=None, backtest_xgb=None,
        currency="$", curr_name="USD", ticker="", stock_name="",
    )

if "state" not in st.session_state:
    st.session_state.state = _fresh_state()
S = st.session_state.state


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Big Price Card (green/red)
# ─────────────────────────────────────────────────────────────────────────────

def _stat_card(label, value, color):
    return (
        f'<div style="background:#f7f8fa; border:1.5px solid #e8eaf0; border-radius:12px;'
        f'padding:12px 18px; text-align:center; min-width:90px">'
        f'<div style="font-size:0.68rem; color:#9ca3af; text-transform:uppercase;'
        f'letter-spacing:1px; margin-bottom:5px; font-weight:600">{label}</div>'
        f'<div style="font-size:1rem; font-weight:800; color:{color}">{value}</div>'
        f'</div>'
    )


def render_price_ticker_bar(df, currency, curr_name, stock_name, ticker):
    latest   = df["Close"].iloc[-1]
    prev     = df["Close"].iloc[-2]
    chg_abs  = latest - prev
    chg_pct  = (chg_abs / prev) * 100
    is_up    = chg_abs >= 0

    arrow    = "▲" if is_up else "▼"
    color    = "#00916e" if is_up else "#dc2626"
    bg       = "#f0fdf9"  if is_up else "#fef2f2"
    border   = "#a7f3d0"  if is_up else "#fecaca"
    badge_bg = "#dcfce7"  if is_up else "#fee2e2"
    badge_c  = "#15803d"  if is_up else "#b91c1c"
    signal   = "BULLISH"  if is_up else "BEARISH"
    icon     = "📈"       if is_up else "📉"

    high_52 = df["High"].max()
    low_52  = df["Low"].min()
    avg_vol = df["Volume"].mean()

    stat_cards = (
        _stat_card("52W High",   f"{currency}{high_52:,.2f}", "#15803d") +
        _stat_card("52W Low",    f"{currency}{low_52:,.2f}",  "#b91c1c") +
        _stat_card("Prev Close", f"{currency}{prev:,.2f}",    "#374151")
    )

    price_str  = f"{currency}{latest:,.2f}"
    change_str = f"{arrow} {currency}{abs(chg_abs):,.2f}"
    pct_str    = f"{arrow} {abs(chg_pct):.2f}%"
    vol_str    = f"Avg Vol: {avg_vol/1e6:.2f}M"

    html = (
        f'<div style="background:{bg}; border:1.5px solid {border}; border-radius:18px;'
        f'padding:22px 28px; margin-bottom:18px; display:flex; flex-wrap:wrap;'
        f'align-items:center; gap:28px; box-shadow:0 2px 12px rgba(0,0,0,0.05);">'

        f'<div style="min-width:170px">'
        f'<div style="display:flex; align-items:center; gap:8px; margin-bottom:6px">'
        f'<span style="background:#e8eaf0; border-radius:7px; padding:3px 10px;'
        f'font-size:0.72rem; font-weight:700; color:#4b5563; letter-spacing:1.5px">{ticker}</span>'
        f'<span style="background:{badge_bg}; border-radius:7px; padding:3px 10px;'
        f'font-size:0.72rem; font-weight:800; color:{badge_c}; letter-spacing:1px">{icon} {signal}</span>'
        f'</div>'
        f'<div style="font-size:1.2rem; font-weight:800; color:#111827">{stock_name}</div>'
        f'<div style="font-size:0.78rem; color:#9ca3af; margin-top:2px">{curr_name} · {vol_str}</div>'
        f'</div>'

        f'<div style="min-width:160px">'
        f'<div style="font-size:0.68rem; color:#9ca3af; text-transform:uppercase;'
        f'letter-spacing:1.5px; margin-bottom:4px; font-weight:600">Current Price (LTP)</div>'
        f'<div style="font-size:2.8rem; font-weight:900; color:#111827;'
        f'letter-spacing:-1px; line-height:1">{price_str}</div>'
        f'</div>'

        f'<div style="min-width:150px">'
        f'<div style="font-size:0.68rem; color:#9ca3af; text-transform:uppercase;'
        f'letter-spacing:1.5px; margin-bottom:4px; font-weight:600">Today\'s Change</div>'
        f'<div style="font-size:1.6rem; font-weight:800; color:{color}; line-height:1.2">{change_str}</div>'
        f'<div style="font-size:1.05rem; font-weight:700; color:{color}">{pct_str}</div>'
        f'</div>'

        f'<div style="display:flex; gap:12px; flex-wrap:wrap; margin-left:auto">'
        f'{stat_cards}'
        f'</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_mini_stats(df, currency):
    data = []
    for d in [1, 2, 3, 5, 10]:
        if len(df) > d:
            r = ((df["Close"].iloc[-1] - df["Close"].iloc[-d-1]) / df["Close"].iloc[-d-1]) * 100
            data.append((f"{d}D", r))

    cols = st.columns(len(data))
    for col, (lbl, val) in zip(cols, data):
        color  = "#00916e" if val >= 0 else "#dc2626"
        arrow  = "▲" if val >= 0 else "▼"
        bg     = "#f0fdf9" if val >= 0 else "#fef2f2"
        border = "#a7f3d0" if val >= 0 else "#fecaca"
        col.markdown(
            f'<div style="background:{bg}; border:1.5px solid {border}; border-radius:12px;'
            f'padding:14px 8px; text-align:center; box-shadow:0 1px 4px rgba(0,0,0,0.04)">'
            f'<div style="font-size:0.68rem; color:#9ca3af; text-transform:uppercase;'
            f'letter-spacing:1.5px; margin-bottom:6px; font-weight:600">{lbl} Return</div>'
            f'<div style="font-size:1.3rem; font-weight:800; color:{color}">{arrow} {abs(val):.2f}%</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Plotly candlestick
# ─────────────────────────────────────────────────────────────────────────────

def candlestick_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.04, row_heights=[0.75, 0.25],
    )
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color="#00916e", increasing_fillcolor="#00916e",
        decreasing_line_color="#dc2626", decreasing_fillcolor="#dc2626",
        name="Price",
    ), row=1, col=1)
    vol_colors = np.where(
        df["Close"] >= df["Open"],
        "rgba(0,145,110,0.3)",
        "rgba(220,38,38,0.3)",
    )
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=list(vol_colors),
        name="Volume",
    ), row=2, col=1)
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        title=dict(text=f"<b>{ticker}</b> — Price Chart + Volume",
                   font=dict(size=16, color="#111827")),
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(t=60, b=30, l=40, r=20),
        legend=dict(bgcolor="#f7f8fa", bordercolor="#e8eaf0"),
        font=dict(color="#374151"),
    )
    fig.update_xaxes(gridcolor="#f1f5f9", linecolor="#e8eaf0")
    fig.update_yaxes(gridcolor="#f1f5f9", linecolor="#e8eaf0")
    return fig


def confusion_heatmap(cm, title):
    fig = px.imshow(
        cm, text_auto=True,
        color_continuous_scale=[[0,"#f0fdf9"],[0.5,"#6ee7b7"],[1,"#00916e"]],
        labels=dict(x="What AI Predicted", y="What Actually Happened"),
        x=["📉 DOWN", "📈 UP"], y=["📉 DOWN", "📈 UP"],
        title=title,
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        height=300,
        margin=dict(t=50, b=30),
        font=dict(color="#374151", size=13),
        coloraxis_showscale=False,
    )
    fig.update_traces(textfont=dict(size=16, color="#111827"))
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Training Pipeline (runs when button is clicked)
# ─────────────────────────────────────────────────────────────────────────────

if run_btn:
    S.update(_fresh_state())   # reset
    S["currency"]   = CURRENCY
    S["curr_name"]  = CURR_NAME
    S["ticker"]     = TICKER
    S["stock_name"] = stock_choice if not custom.strip() else TICKER

    progress = st.progress(0, text="⏳ Fetching stock data…")

    try:
        # 1 — Data
        df = fetch_stock_data(TICKER, PERIOD)
        if df.empty or len(df) < 100:
            st.error(f"❌ Couldn't fetch enough data for **{TICKER}**. Check the ticker symbol.")
            st.stop()
        S["df"] = df
        progress.progress(10, "📐 Engineering features…")

        # 2a — News Sentiment (FinBERT feature 26)
        progress.progress(12, "📰 Fetching news sentiment (FinBERT)…")
        sentiment_series = fetch_news_sentiment(TICKER, df.index)
        S["sentiment_series"] = sentiment_series
        has_sentiment = sentiment_series.abs().sum() > 0
        S["sentiment_available"] = has_sentiment

        # 2b — Features
        df_feat = engineer_features(df, sentiment_series)
        S["df_feat"] = df_feat
        progress.progress(20, "✂️ Preparing train/test split…")

        # 3 — Split
        X_train, X_test, y_train, y_test, scaler, features = prepare_train_test(df_feat)
        S.update(dict(X_train=X_train, X_test=X_test,
                      y_train=y_train, y_test=y_test,
                      scaler=scaler, features=features))
        progress.progress(30, "🌲 Training Forest Predictor…")

        # 4 — Random Forest
        rf = train_random_forest(X_train, y_train, n_estimators=RF_TREES)
        S["rf"] = rf
        S["rf_metrics"] = evaluate_model(rf, X_test, y_test, "sklearn")
        progress.progress(45, "⚡ Training Trend Predictor…")

        # 5 — XGBoost
        xgb_model = train_xgboost(X_train, y_train, n_estimators=XGB_TREES)
        S["xgb_model"] = xgb_model
        S["xgb_metrics"] = evaluate_model(xgb_model, X_test, y_test, "xgb")
        progress.progress(60, "🧠 Training Memory Predictor (takes ~30s)…")

        # 6 — LSTM
        lstm_model, lstm_history, lstm_X_test, lstm_y_test = train_lstm(
            X_train, y_train, X_test, y_test, epochs=LSTM_EPOCHS
        )
        S["lstm_model"]  = lstm_model
        S["lstm_X_test"] = lstm_X_test
        S["lstm_y_test"] = lstm_y_test
        if lstm_model is not None:
            S["lstm_metrics"] = evaluate_model(lstm_model, lstm_X_test, lstm_y_test, "lstm")
        else:
            S["lstm_metrics"] = {
                "accuracy": 0, "auc": 0,
                "report": {}, "confusion": [[0,0],[0,0]],
                "y_pred": [], "y_prob": [],
                "note": "LSTM disabled — tensorflow not installed on this server",
            }
        progress.progress(75, "🔍 Computing SHAP values (RF)…")

        # 7 — SHAP RF
        _, shap_vals_rf, shap_X_rf = compute_shap_values(rf, X_train, X_test, "tree")
        S["shap_vals_rf"] = shap_vals_rf
        S["shap_X_rf"]    = shap_X_rf
        progress.progress(88, "🔍 Computing SHAP values (XGB)…")

        # 8 — SHAP XGB
        _, shap_vals_xgb, shap_X_xgb = compute_shap_values(xgb_model, X_train, X_test, "tree")
        S["shap_vals_xgb"] = shap_vals_xgb
        S["shap_X_xgb"]    = shap_X_xgb

        progress.progress(90, "📊 Running backtests…")
        # 9 — Backtest RF and XGB
        df_test_slice = df_feat.iloc[-len(X_test):]
        S["backtest_rf"]  = run_backtest(rf,        X_test, df_test_slice)
        S["backtest_xgb"] = run_backtest(xgb_model, X_test, df_test_slice)

        progress.progress(100, "✅ All done!")
        S["trained"] = True
        st.success(f"🎉 Pipeline complete for **{TICKER}**! Explore the tabs below.")

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.exception(e)
        st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Dashboard Tabs
# ─────────────────────────────────────────────────────────────────────────────

if not S["trained"]:
    st.markdown("""
    <div class='insight-box'>
    <div class='title'>👋 Welcome! Here's how to get started</div>
    <b>Step 1 —</b> Choose your country's stock exchange from the left sidebar (India, USA, Japan etc.)<br>
    <b>Step 2 —</b> Pick a company you're interested in (e.g. Reliance, Apple, Toyota)<br>
    <b>Step 3 —</b> Click the green <b>Train & Explain</b> button<br><br>
    Our AI will then study 2 years of real stock data, train 3 different prediction models, and explain
    <em>exactly why</em> it thinks the stock will go <b>UP 📈</b> or <b>DOWN 📉</b> tomorrow.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    for col, icon, title, desc in [
        (col1, "🤖", "3 Smart AI Predictors", "Forest, Trend & Memory predictors analyse price patterns and news together"),
        (col2, "🔍", "4 XAI Methods", "SHAP, LIME, Counterfactuals & LSTM Attention explain every decision"),
        (col3, "🌍", "7 Global Markets", "India, USA, Japan, UK, Singapore, Germany & Hong Kong"),
    ]:
        col.markdown(
            f'<div class="g-card" style="text-align:center; padding:24px 16px">'
            f'<div style="font-size:2rem">{icon}</div>'
            f'<div style="font-weight:800; color:#111827; margin:8px 0 6px; font-size:1rem">{title}</div>'
            f'<div style="font-size:0.85rem; color:#6b7280; line-height:1.6">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.stop()

# ── Tabs ──
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "📊 Stock Data",
    "📊 Prediction Results",
    "🔍 Why This Prediction?",
    "🧩 Today's Reasoning",
    "📌 Key Signals",
    "📰 News Sentiment",
    "🔮 What Would Change It?",
    "📈 Strategy Performance",
    "💼 My Portfolio",
    "✨ Build My Portfolio",
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — Stock Data
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    df      = S["df"]
    df_feat = S["df_feat"]
    cur     = S["currency"]
    cur_nm  = S["curr_name"]
    sname   = S["stock_name"]
    tick    = S["ticker"]

    st.markdown("<div class='groww-header'>📊 Stock Overview</div>", unsafe_allow_html=True)
    st.markdown("<div class='groww-subtext'>Here's a snapshot of the stock you selected — price, recent performance, and historical trends.</div>", unsafe_allow_html=True)

    render_price_ticker_bar(df, cur, cur_nm, sname, tick)
    render_mini_stats(df, cur)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Insight before chart ──────────────────────────────────────────
    latest = df["Close"].iloc[-1]
    prev   = df["Close"].iloc[-2]
    chg_pct = ((latest - prev) / prev) * 100
    trend_5d = ((df["Close"].iloc[-1] - df["Close"].iloc[-6]) / df["Close"].iloc[-6]) * 100
    trend_word = "risen" if trend_5d > 0 else "fallen"
    today_word = "gained" if chg_pct > 0 else "lost"

    st.markdown(f"""
    <div class='insight-box'>
    <div class='title'>📖 What does this chart tell you?</div>
    <b>{sname}</b> has {today_word} <b>{abs(chg_pct):.2f}%</b> today and has {trend_word} <b>{abs(trend_5d):.2f}%</b> over the last 5 days.
    The <b>green candles</b> below show days the stock went UP, and <b>red candles</b> show days it went DOWN.
    The <b>bar chart at the bottom</b> shows trading volume — higher volume means more people were buying/selling that day,
    which often signals a stronger move. Use this chart to spot trends before making investment decisions.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(candlestick_chart(df, tick), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Correlation heatmap ───────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='groww-header' style='font-size:1.1rem'>🔥 How Are Technical Signals Related?</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='insight-box'>
    <div class='title'>📖 What does this heatmap tell you?</div>
    This shows how different technical indicators (like RSI, MACD, Moving Averages) move together.
    <b>Dark green = strong positive relationship</b> (both go up together), <b>light = weak or no relationship</b>,
    <b>red = opposite movement</b> (when one goes up, the other goes down).
    <br>When indicators are <em>not</em> highly correlated with each other, they give the AI more unique information
    — making predictions more reliable. This helps us pick the best features to feed into our models.
    </div>
    """, unsafe_allow_html=True)

    sample_feats = ["RSI","MACD","ATR","BB_Width","Return_1d","Return_5d","EMA_12","EMA_26","SMA_20","OBV"]
    corr = df_feat[sample_feats].corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale=[[0,"#dc2626"],[0.5,"#f9fafb"],[1,"#00916e"]],
        title="Technical Indicator Correlation — Closer to 1.0 = Move Together",
        height=460,
    )
    fig_corr.update_layout(
        template="plotly_white", paper_bgcolor="#ffffff",
        font=dict(color="#374151"),
        margin=dict(t=50, b=20),
    )
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='groww-header' style='font-size:1.1rem'>📋 Recent Data Used by the AI</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='insight-box'>
    <div class='title'>📖 What is this table?</div>
    These are the exact numbers fed into our AI models. Each row is one trading day.
    <b>RSI</b> (0-100) measures if the stock is overbought (>70) or oversold (<30).
    <b>MACD</b> shows momentum direction. <b>ATR</b> measures how volatile the stock is.
    <b>Target = 1</b> means the stock went UP the next day, <b>Target = 0</b> means it went DOWN.
    This is what the AI learns from!
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(
        df_feat[["Close","RSI","MACD","BB_Width","ATR","OBV","Target"]].tail(30)
        .style.background_gradient(cmap="RdYlGn", subset=["RSI"])
        .format({"Close": f"{cur}{{:,.2f}}", "RSI": "{:.1f}", "MACD": "{:.3f}",
                 "BB_Width": "{:.3f}", "ATR": "{:.2f}"}),
        use_container_width=True, height=320,
    )

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — Model Performance
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='groww-header'>🤖 How Good Are Our AI Models?</div>", unsafe_allow_html=True)
    st.markdown("<div class='groww-subtext'>We trained 3 different AI models on this stock's data. Here's how well each one can predict whether the price will go UP or DOWN tomorrow.</div>", unsafe_allow_html=True)

    rf_m   = S["rf_metrics"]
    xgb_m  = S["xgb_metrics"]
    lstm_m = S["lstm_metrics"]

    # ── Insight box ───────────────────────────────────────────────────
    best_model = max([("Forest Predictor 🌲", rf_m.get("accuracy", 0)),
                      ("Trend Predictor ⚡", xgb_m.get("accuracy", 0)),
                      ("Memory Predictor 🧠", lstm_m.get("accuracy", 0))], key=lambda x: x[1])
    st.markdown(f"""
    <div class='insight-box'>
    <div class='title'>🏆 Quick Summary</div>
    Your best performing model is <b>{best_model[0]}</b> with <b>{best_model[1]}% accuracy</b> on test data.
    Think of accuracy like this — if the AI analysed 100 trading days, it correctly predicted UP or DOWN on {best_model[1]:.0f} of them.
    <b>AUC-ROC</b> above 60% means the model is better than random guessing. Above 70% is considered good for stock prediction.
    </div>
    """, unsafe_allow_html=True)

    # ── Score cards ───────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    for col, m, name, icon, desc in [
        (col1, rf_m,   "Forest Predictor", "🌲", "Analyses 200 different viewpoints and takes a majority vote"),
        (col2, xgb_m,  "Trend Predictor", "⚡", "Learns from past mistakes to get smarter with each step"),
        (col3, lstm_m, "Memory Predictor", "🧠", "Reads the last 10 days like a chart — remembers patterns over time"),
    ]:
        acc_color = "#00916e" if m.get("accuracy", 0) >= 55 else "#dc2626"
        auc_color = "#00916e" if m.get("auc", 0) >= 60 else "#dc2626"
        col.markdown(
            f'<div class="g-card">'
            f'<div style="font-size:1.6rem">{icon}</div>'
            f'<div style="font-weight:800; color:#111827; font-size:1rem; margin:6px 0 2px">{name}</div>'
            f'<div style="font-size:0.78rem; color:#9ca3af; margin-bottom:14px">{desc}</div>'
            f'<div style="display:flex; gap:12px">'
            f'<div style="flex:1; background:#f7f8fa; border-radius:10px; padding:10px; text-align:center">'
            f'<div style="font-size:0.7rem; color:#9ca3af; font-weight:600; text-transform:uppercase; letter-spacing:1px">Accuracy</div>'
            f'<div style="font-size:1.6rem; font-weight:900; color:{acc_color}">{m.get("accuracy", 0)}%</div>'
            f'</div>'
            f'<div style="flex:1; background:#f7f8fa; border-radius:10px; padding:10px; text-align:center">'
            f'<div style="font-size:0.7rem; color:#9ca3af; font-weight:600; text-transform:uppercase; letter-spacing:1px">AUC-ROC</div>'
            f'<div style="font-size:1.6rem; font-weight:900; color:{auc_color}">{m.get("auc", 0)}%</div>'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Bar comparison ────────────────────────────────────────────────
    st.markdown("<div class='groww-header' style='font-size:1.1rem'>📊 Side-by-Side Model Comparison</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='insight-box'>
    <div class='title'>📖 How to read this chart</div>
    The taller the bar, the better the model performed.
    <b>Accuracy</b> = % of correct UP/DOWN predictions out of all predictions made.
    <b>AUC-ROC</b> = how well the model separates "UP days" from "DOWN days" — 50% = coin flip, 100% = perfect.
    A higher AUC-ROC means the model is more confident and reliable, not just lucky.
    </div>
    """, unsafe_allow_html=True)

    comp_df = pd.DataFrame({
        "Model":    ["Forest Predictor 🌲", "Trend Predictor ⚡", "Memory Predictor 🧠"],
        "Accuracy": [rf_m.get("accuracy", 0), xgb_m.get("accuracy", 0), lstm_m.get("accuracy", 0)],
        "AUC-ROC":  [rf_m.get("auc", 0),      xgb_m.get("auc", 0),      lstm_m.get("auc", 0)],
    })
    fig_bar = px.bar(
        comp_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
        x="Model", y="Score", color="Metric", barmode="group",
        color_discrete_sequence=["#00916e", "#3b82f6"],
        title="Model Accuracy vs AUC-ROC",
        height=380, text_auto=".1f",
    )
    fig_bar.update_layout(
        template="plotly_white", paper_bgcolor="#ffffff",
        yaxis=dict(range=[40, 100], title="Score (%)"),
        font=dict(color="#374151"),
        legend=dict(bgcolor="#f7f8fa"),
    )
    fig_bar.update_traces(textposition="outside", textfont_size=13)
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Confusion matrices ────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='groww-header' style='font-size:1.1rem'>🎯 Did the AI Predict Correctly?</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='insight-box'>
    <div class='title'>📖 How to read the Confusion Matrix</div>
    This grid shows where the AI got it right and where it made mistakes.
    <b>Top-left</b> = correctly predicted DOWN days (good!).
    <b>Bottom-right</b> = correctly predicted UP days (good!).
    <b>Top-right</b> = predicted UP but stock went DOWN (false alarm).
    <b>Bottom-left</b> = predicted DOWN but stock went UP (missed opportunity).
    You want the diagonal numbers (top-left and bottom-right) to be as large as possible!
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🌲 Forest Predictor**")
        st.plotly_chart(confusion_heatmap(rf_m.get("confusion", [[0,0],[0,0]]), "Forest Predictor"), use_container_width=True)
    with c2:
        st.markdown("**⚡ Trend Predictor**")
        st.plotly_chart(confusion_heatmap(xgb_m.get("confusion", [[0,0],[0,0]]), "Trend Predictor"), use_container_width=True)
    with c3:
        st.markdown("**🧠 Memory Predictor**")
        st.plotly_chart(confusion_heatmap(lstm_m.get("confusion", [[0,0],[0,0]]), "Memory Predictor"), use_container_width=True)

    # ── Classification reports ────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='groww-header' style='font-size:1.1rem'>📄 Detailed Report Card</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='insight-box'>
    <div class='title'>📖 What do Precision, Recall, F1 mean?</div>
    <b>Precision</b> — When the AI said "stock will go UP", how often was it actually right? High precision = fewer false Buy signals.<br>
    <b>Recall</b> — Out of all actual UP days, how many did the AI catch? High recall = fewer missed buying opportunities.<br>
    <b>F1 Score</b> — A balanced mix of both. Closer to 1.0 = better overall performance.
    </div>
    """, unsafe_allow_html=True)

    cr1, cr2, cr3 = st.columns(3)
    for col, m, label in [(cr1, rf_m, "🌲 Forest Predictor"), (cr2, xgb_m, "⚡ Trend Predictor"), (cr3, lstm_m, "🧠 Memory Predictor")]:
        with col:
            st.markdown(f"**{label}**")
            rpt = m.get("report", {})
            rpt_df = pd.DataFrame({
                "Prediction":  ["📉 DOWN", "📈 UP", "Overall"],
                # sklearn version-safe: try string keys then integer keys
                "Precision":   [round((rpt.get("0") or rpt.get(0) or {}).get("precision",0),3), round((rpt.get("1") or rpt.get(1) or {}).get("precision",0),3), "—"],
                "Recall":      [round((rpt.get("0") or rpt.get(0) or {}).get("recall",0),3),    round((rpt.get("1") or rpt.get(1) or {}).get("recall",0),3),    "—"],
                "F1 Score":    [round((rpt.get("0") or rpt.get(0) or {}).get("f1-score",0),3),  round((rpt.get("1") or rpt.get(1) or {}).get("f1-score",0),3),  round(rpt.get("accuracy",0),3)],
            })
            st.dataframe(rpt_df, hide_index=True, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — SHAP Explanations
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("<div class='groww-header'>🔍 Why Did the AI Make This Prediction?</div>", unsafe_allow_html=True)
    st.markdown("<div class='groww-subtext'>SHAP shows you the exact reason behind every AI decision — like asking the AI to 'show its work'.</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='insight-box'>
    <div class='title'>💡 What is SHAP? (In plain English)</div>
    Imagine you're predicting if it will rain tomorrow. You might look at clouds, humidity, and wind.
    SHAP does the same for stocks — it tells you <b>which signals (like RSI, MACD, volume) pushed the AI towards saying BUY or SELL</b>,
    and by how much. <b>Green bars = pushed towards BUY</b>. <b>Red bars = pushed towards SELL</b>.
    This helps you trust the AI's decision because you can see exactly what it was "thinking."
    </div>
    """, unsafe_allow_html=True)

    model_choice = st.radio("Choose AI model to explain:", ["🌲 Forest Predictor", "⚡ Trend Predictor"], horizontal=True)
    is_rf     = "Forest" in model_choice  # works for both old and new name
    shap_vals = S["shap_vals_rf"]  if is_rf else S["shap_vals_xgb"]
    X_shap    = S["shap_X_rf"]     if is_rf else S["shap_X_xgb"]
    X_test    = S["X_test"]
    features  = S["features"]
    # hard-sync: rows AND cols must match before any plot
    _nr = min(len(shap_vals), len(X_shap))
    _nc = min(shap_vals.shape[1], X_shap.shape[1], len(features))
    shap_vals = shap_vals[:_nr, :_nc]
    X_shap    = X_shap[:_nr, :_nc]
    feat_shap = list(features[:_nc])

    sub1, sub2, sub3 = st.tabs(["🐝 Overall Impact", "📊 Top Signals Ranked", "💧 One Specific Day"])

    with sub1:
        st.markdown("**What factors matter most across all predictions?**")
        st.markdown("""
        <div class='info-box'>
        Each dot = one trading day from our test data. Dots to the <b>right (positive)</b> = that signal pushed the AI to say BUY.
        Dots to the <b>left (negative)</b> = pushed towards SELL. The <b>colour</b> shows whether that feature's value was high (red) or low (blue) on that day.
        Features at the <b>top</b> matter most to the AI's decision.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        fig = plot_shap_summary(shap_vals, X_shap, feat_shap)
        st.pyplot(fig, use_container_width=True)
        plt.close("all")
        st.markdown('</div>', unsafe_allow_html=True)

    with sub2:
        st.markdown("**Which signals does the AI rely on the most?**")
        st.markdown("""
        <div class='info-box'>
        This bar chart ranks every signal by how much it influenced the AI's decisions <b>on average</b>.
        The longer the bar, the more important that signal is. If <b>RSI</b> is at the top, it means the AI
        relies heavily on whether the stock is overbought/oversold to make its call.
        Use this to understand what drives this particular stock's movements.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        fig = plot_shap_bar(shap_vals, feat_shap)
        st.pyplot(fig, use_container_width=True)
        plt.close("all")
        st.markdown('</div>', unsafe_allow_html=True)

    with sub3:
        st.markdown(f"**Breaking down the AI's decision for Day #{SAMPLE_IDX} in our test data**")
        model = S["rf"] if "Forest" in model_choice else S["xgb_model"]  # Forest = RF
        prob  = model.predict_proba(X_test[[SAMPLE_IDX]])[0]
        pred  = "📈 BUY (UP)" if prob[1] > 0.5 else "📉 SELL (DOWN)"
        conf  = max(prob) * 100

        st.markdown(
            f'<div class="insight-box">'
            f'<div class="title">🤖 AI Decision for this day</div>'
            f'The AI predicted: <b style="font-size:1.1rem">{pred}</b>'
            f' with <b>{conf:.1f}% confidence</b>.<br>'
            f'BUY probability = {prob[1]*100:.1f}% &nbsp;|&nbsp; SELL probability = {prob[0]*100:.1f}%<br><br>'
            f'The chart below shows <b>exactly which signals</b> made the AI lean towards this decision.'
            f' Green bars = reasons to BUY, Red bars = reasons to SELL.'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        fig = plot_shap_waterfall(shap_vals, X_shap, feat_shap, idx=min(SAMPLE_IDX, len(shap_vals)-1))
        st.pyplot(fig, use_container_width=True)
        plt.close("all")
        st.markdown('</div>', unsafe_allow_html=True)


with tab4:
    st.markdown("<div class='groww-header'>🧩 Explain a Single Trading Day</div>", unsafe_allow_html=True)
    st.markdown("<div class='groww-subtext'>LIME zooms in on one specific prediction and explains it in simple terms — like asking the AI 'why did you say BUY today specifically?'</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='insight-box'>
    <div class='title'>💡 What is LIME? (In plain English)</div>
    SHAP looks at the big picture across all days. LIME focuses on <b>one specific day</b>.
    It asks: "What would need to change for the AI to predict the opposite?"
    For example, if LIME says <b>"RSI &gt; 65 → SELL"</b>, it means on this particular day, the stock being
    overbought (RSI high) was the main reason the AI said SELL.
    This is like getting a personalised explanation, not just an average one.
    </div>
    """, unsafe_allow_html=True)

    lime_model_choice = st.radio("Choose model:", ["🌲 Forest Predictor", "⚡ Trend Predictor"], horizontal=True, key="lime_radio")
    lime_model = S["rf"] if "Forest" in lime_model_choice else S["xgb_model"]

    prob_lime = lime_model.predict_proba(S["X_test"][[SAMPLE_IDX]])[0]
    pred_lime = "📈 BUY (UP)" if prob_lime[1] > 0.5 else "📉 SELL (DOWN)"

    with st.spinner("Calculating explanation for this day…"):
        lime_exp, lime_fig = explain_with_lime(
            lime_model, S["X_train"], S["X_test"], S["features"],
            idx=SAMPLE_IDX, num_features=12,
        )

    st.markdown(
        f'<div class="insight-box">'
        f'<div class="title">📅 Analysing Day #{SAMPLE_IDX} from test data</div>'
        f'AI prediction: <b>{pred_lime}</b> — Confidence: BUY {prob_lime[1]*100:.1f}% | SELL {prob_lime[0]*100:.1f}%<br>'
        f'The chart below shows the <b>top signals that drove this specific prediction</b>. '
        f'<b>Green (right side)</b> = pushed AI to say BUY. <b>Red (left side)</b> = pushed AI to say SELL.'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.pyplot(lime_fig, use_container_width=True)
    plt.close("all")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**📋 Signal-by-Signal Breakdown**")
    st.markdown("""
    <div class='info-box'>
    Each row below is one signal (like RSI or MACD) and how much it pushed the AI towards BUY or SELL on this day.
    The <b>Feature Rule</b> column shows the exact condition (e.g., "RSI > 65.2") and the <b>Weight</b> shows
    how strongly it pushed the prediction. Positive = towards BUY, Negative = towards SELL.
    </div>
    """, unsafe_allow_html=True)

    available_labels = list(lime_exp.local_exp.keys())
    use_label = 1 if 1 in available_labels else available_labels[0]
    exp_list  = lime_exp.as_list(label=use_label)
    lime_df   = pd.DataFrame(exp_list, columns=["Signal Condition", "Influence Score"])
    lime_df["Decision Push"] = lime_df["Influence Score"].apply(
        lambda x: "🟢 Towards BUY" if x > 0 else "🔴 Towards SELL"
    )
    lime_df["Influence Score"] = lime_df["Influence Score"].round(4)
    lime_df["Strength"] = lime_df["Influence Score"].abs().apply(
        lambda x: "🔥 Strong" if x > 0.05 else ("⚡ Medium" if x > 0.02 else "💧 Weak")
    )
    st.dataframe(lime_df, hide_index=True, use_container_width=True)

    st.markdown("""
    <div class='warn-box'>
    ⚠️ <b>Important:</b> This explanation is only for this one trading day (Day #{idx}).
    Different days may have different reasons. Use the SHAP tab for overall patterns across all days.
    </div>
    """.format(idx=SAMPLE_IDX), unsafe_allow_html=True)


with tab5:
    st.markdown("<div class='groww-header'>📌 What Does the AI Pay Attention To?</div>", unsafe_allow_html=True)
    st.markdown("<div class='groww-subtext'>Feature Importance shows which stock signals the AI considers most useful when deciding to predict UP or DOWN.</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='insight-box'>
    <div class='title'>💡 Why does Feature Importance matter for investors?</div>
    If the AI says <b>RSI is the most important signal</b> for this stock, it means this stock tends to
    reverse direction when it becomes overbought or oversold. That's a real, actionable insight!
    Knowing which signals matter helps you focus on what to watch when monitoring this stock manually.
    Different stocks have different "key signals" — what works for Reliance may not work for TCS.
    </div>
    """, unsafe_allow_html=True)

    fi1, fi2 = st.columns(2)
    with fi1:
        st.markdown("#### 🌲 Random Forest — Top Signals")
        st.markdown("""<div class='info-box' style='font-size:0.84rem'>
        Each bar = one signal. The longer the bar, the more the Random Forest
        relied on it to make predictions. This model uses <b>200 trees voting together</b>,
        so importance is averaged across all of them — making it very stable and reliable.
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        fig = plot_feature_importance(S["rf"], S["features"], model_name="Forest Predictor")
        st.pyplot(fig, use_container_width=True)
        plt.close("all")
        st.markdown('</div>', unsafe_allow_html=True)
        st.dataframe(get_top_features_df(S["rf"], S["features"]), hide_index=True, use_container_width=True)

    with fi2:
        st.markdown("#### ⚡ XGBoost — Top Signals")
        st.markdown("""<div class='info-box' style='font-size:0.84rem'>
        XGBoost importance is based on how many times a signal was used to make a split decision
        across all its trees — weighted by how much it improved the prediction each time.
        Compare this with Random Forest above to see if both models agree on what matters!
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        fig = plot_feature_importance(S["xgb_model"], S["features"], model_name="Trend Predictor")
        st.pyplot(fig, use_container_width=True)
        plt.close("all")
        st.markdown('</div>', unsafe_allow_html=True)
        st.dataframe(get_top_features_df(S["xgb_model"], S["features"]), hide_index=True, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='groww-header' style='font-size:1.1rem'>🕸️ Do Both Models Agree?</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='insight-box'>
    <div class='title'>📖 How to read the Radar Chart</div>
    This spider-web chart compares what Random Forest (green) and XGBoost (blue) consider important.
    <b>If both shapes overlap closely</b> — great! Both models agree on what drives this stock.
    <b>If they differ a lot</b> — the models use different strategies, which can actually make their
    combined predictions more robust (like getting a second opinion from a different doctor).
    Signals that <b>both models rate highly</b> are the most trustworthy indicators for this stock.
    </div>
    """, unsafe_allow_html=True)

    top_feats = get_top_features_df(S["rf"], S["features"], top_n=8)["Feature"].tolist()
    rf_imp    = [S["rf"].feature_importances_[S["features"].index(f)] for f in top_feats]
    xgb_imp   = [S["xgb_model"].feature_importances_[S["features"].index(f)] for f in top_feats]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=rf_imp + [rf_imp[0]], theta=top_feats + [top_feats[0]],
        fill="toself", name="Random Forest 🌲",
        line=dict(color="#00916e", width=2.5),
        fillcolor="rgba(0,145,110,0.12)",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=xgb_imp + [xgb_imp[0]], theta=top_feats + [top_feats[0]],
        fill="toself", name="XGBoost ⚡",
        line=dict(color="#3b82f6", width=2.5),
        fillcolor="rgba(59,130,246,0.12)",
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="#f7f8fa",
            radialaxis=dict(visible=True, color="#9ca3af", gridcolor="#e8eaf0"),
            angularaxis=dict(color="#374151"),
        ),
        template="plotly_white",
        paper_bgcolor="#ffffff",
        legend=dict(bgcolor="#f7f8fa", bordercolor="#e8eaf0", font=dict(color="#374151")),
        height=480,
        font=dict(color="#374151"),
        title=dict(text="Signal Importance — Random Forest vs XGBoost", font=dict(color="#111827", size=15)),
    )
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig_radar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='groww-header' style='font-size:1.1rem'>🎓 Which XAI Method Should You Use?</div>", unsafe_allow_html=True)
    summary = pd.DataFrame({
        "Method":          ["SHAP 🔍",                       "LIME 🧩",                            "Feature Importance 📌"],
        "Best For":        ["Understanding overall patterns", "Explaining one specific prediction",  "Quick overview of key signals"],
        "Answers":         ["Why does the AI usually predict UP/DOWN?", "Why did it say BUY/SELL today specifically?", "What does the AI pay most attention to?"],
        "Coverage":        ["All predictions (global)",      "One day at a time (local)",            "All predictions (global)"],
        "Reliability":     ["⭐⭐⭐⭐⭐ Very high",             "⭐⭐⭐⭐ High",                          "⭐⭐⭐ Good"],
        "Speed":           ["🐢 Slower",                      "⚡ Medium",                            "🚀 Fast"],
    })
    st.dataframe(summary, hide_index=True, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 6 — News Sentiment (FinBERT)
# ──────────────────────────────────────────────────────────────────────────────
with tab6:
    st.markdown("<div class='groww-header'>📰 News Sentiment Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='groww-subtext'>Real-time news scored by FinBERT — the 26th feature that captures external shocks missed by price data.</div>", unsafe_allow_html=True)

    sentiment_series = S.get("sentiment_series")
    has_sentiment    = S.get("sentiment_available", False)

    if not has_sentiment or sentiment_series is None:
        st.markdown("""
        <div class='info-box'>
        <div class='title'>ℹ️ Sentiment Running in Neutral Mode</div>
        FinBERT requires the <code>transformers</code> library and an internet connection to fetch headlines.
        The model trained using <b>Sentiment_Score = 0.0 (neutral)</b> as a placeholder.<br><br>
        To enable real sentiment: <code>pip install transformers torch</code> — then re-run training.
        The Sentiment_Score feature slot is already in the model — it will automatically activate.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='insight-box'>
        <div class='title'>📖 What is FinBERT?</div>
        FinBERT is Google's BERT language model fine-tuned on 10,000 financial news articles.
        It reads each news headline and outputs a score from <b>–1.0 (very negative)</b> to <b>+1.0 (very positive)</b>.
        This score is added as the <b>26th feature</b> alongside 25 technical indicators — giving the AI context
        about <em>why</em> the market might move beyond what price charts can show.
        </div>
        """, unsafe_allow_html=True)

        # Sentiment over time chart
        sent_df = sentiment_series.to_frame("Sentiment")
        sent_df = sent_df[sent_df["Sentiment"] != 0]

        if not sent_df.empty:
            fig_sent = go.Figure()
            pos = sent_df[sent_df["Sentiment"] >= 0]
            neg = sent_df[sent_df["Sentiment"] < 0]
            fig_sent.add_trace(go.Bar(x=pos.index, y=pos["Sentiment"], name="Positive 📈",
                                       marker_color="#00916e", opacity=0.8))
            fig_sent.add_trace(go.Bar(x=neg.index, y=neg["Sentiment"], name="Negative 📉",
                                       marker_color="#dc2626", opacity=0.8))
            fig_sent.add_hline(y=0, line_color="#9ca3af", line_width=1)
            fig_sent.update_layout(
                template="plotly_white", paper_bgcolor="#ffffff",
                title="FinBERT Sentiment Score Over Time",
                height=380, barmode="relative",
                font=dict(color="#374151"),
                legend=dict(bgcolor="#f7f8fa"),
                margin=dict(t=50, b=30),
            )
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            st.plotly_chart(fig_sent, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Stats
            c1, c2, c3, c4 = st.columns(4)
            avg_sent = sent_df["Sentiment"].mean()
            latest_s = sent_df["Sentiment"].iloc[-1]
            pos_days = (sent_df["Sentiment"] > 0).sum()
            neg_days = (sent_df["Sentiment"] < 0).sum()
            for col, lbl, val, color in [
                (c1, "Average Sentiment", f"{avg_sent:+.3f}", "#00916e" if avg_sent > 0 else "#dc2626"),
                (c2, "Latest Score",      f"{latest_s:+.3f}", "#00916e" if latest_s > 0 else "#dc2626"),
                (c3, "Positive Days",     str(pos_days), "#00916e"),
                (c4, "Negative Days",     str(neg_days), "#dc2626"),
            ]:
                col.markdown(
                    f'<div style="background:#ffffff;border:1.5px solid #e8eaf0;border-radius:12px;'
                    f'padding:16px;text-align:center">'
                    f'<div style="font-size:0.72rem;color:#9ca3af;text-transform:uppercase;letter-spacing:1px;'
                    f'font-weight:600;margin-bottom:6px">{lbl}</div>'
                    f'<div style="font-size:1.6rem;font-weight:800;color:{color}">{val}</div>'
                    f'</div>', unsafe_allow_html=True)

    # FinBERT explanation table — always show
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='groww-header' style='font-size:1.1rem'>🧠 How FinBERT Reads Headlines</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='insight-box'>
    <div class='title'>📖 Real Examples — Why Context Beats Price Data</div>
    Traditional models ONLY see numbers (price, volume). FinBERT reads the <b>meaning</b> of news, so it catches
    events <em>before</em> they fully show up in the price chart.
    </div>
    """, unsafe_allow_html=True)
    examples = pd.DataFrame({
        "Headline Example": [
            "Reliance Industries beats Q3 earnings by 18%",
            "CEO of Infosys resigns amid board dispute",
            "RBI keeps interest rates unchanged",
            "SEBI launches probe into insider trading",
            "TCS wins $1.2B deal with US bank",
        ],
        "FinBERT Score": ["+0.91", "–0.87", "+0.12", "–0.74", "+0.88"],
        "Sentiment": ["🟢 Very Positive", "🔴 Very Negative", "🟡 Slightly Positive",
                      "🔴 Negative", "🟢 Very Positive"],
        "Effect on Model": [
            "Pushes BUY signal strongly",
            "Pushes SELL signal strongly",
            "Minimal effect",
            "Pushes SELL signal",
            "Pushes BUY signal strongly",
        ],
    })
    st.dataframe(examples, hide_index=True, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 7 — Counterfactual Explanations
# ──────────────────────────────────────────────────────────────────────────────
with tab7:
    st.markdown("<div class='groww-header'>🔮 Counterfactual Explanations</div>", unsafe_allow_html=True)
    st.markdown("<div class='groww-subtext'>'What would need to change for the AI to decide differently?' — The most actionable XAI method for investors.</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='insight-box'>
    <div class='title'>📖 What is a Counterfactual Explanation?</div>
    Normal explanations say <em>'the AI predicted SELL because RSI is high'</em>.
    A counterfactual goes further: <em>'if RSI dropped below 58 AND volume increased by 20%, the prediction would flip to BUY'</em>.
    This tells you <b>exactly what to watch for</b> — specific thresholds, not just vague reasons.
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 2])
    with col_a:
        cf_model_choice = st.selectbox("Model to explain", ["Random Forest", "XGBoost"], key="cf_model")
        cf_day = st.slider("Trading Day to analyse", 0, len(S["X_test"]) - 1,
                           len(S["X_test"]) // 2, key="cf_day")
        run_cf = st.button("🔮 Generate Counterfactual", key="cf_btn",
                           type="primary", use_container_width=True)

    model_cf = S["rf"] if cf_model_choice == "Random Forest" else S["xgb_model"]
    X_test_cf = S["X_test"]
    features_cf = S["features"]

    # Show current prediction for selected day
    with col_b:
        prob = model_cf.predict_proba(X_test_cf[cf_day].reshape(1, -1))[0]
        pred_cls   = int(np.argmax(prob))
        confidence = prob[pred_cls] * 100
        pred_label = "📈 BUY (UP)" if pred_cls == 1 else "📉 SELL (DOWN)"
        flip_label = "📉 SELL (DOWN)" if pred_cls == 1 else "📈 BUY (UP)"
        color_p    = "#00916e" if pred_cls == 1 else "#dc2626"
        st.markdown(f"""
        <div style="background:#ffffff;border:1.5px solid #e8eaf0;border-radius:14px;padding:20px;margin-top:8px">
        <div style="font-size:0.78rem;color:#9ca3af;font-weight:600;text-transform:uppercase;letter-spacing:1px">
            Day #{cf_day} Prediction — {cf_model_choice}</div>
        <div style="font-size:2rem;font-weight:900;color:{color_p};margin:8px 0">{pred_label}</div>
        <div style="font-size:1rem;color:#374151">Confidence: <b>{confidence:.1f}%</b></div>
        <div style="font-size:0.88rem;color:#9ca3af;margin-top:6px">
            Question: <em>"What would need to change to flip this to <b>{flip_label}</b>?"</em></div>
        </div>
        """, unsafe_allow_html=True)

    if run_cf:
        with st.spinner("🔮 Searching for minimum changes to flip prediction…"):
            cf_df = compute_counterfactuals(model_cf, X_test_cf, features_cf, idx=cf_day)

        if cf_df.empty:
            st.markdown("<div class='warn-box'>No counterfactual found — prediction confidence is very low (near 50/50), so the AI is already uncertain.</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='insight-box'>
            <div class='title'>🔮 Counterfactual Found!</div>
            To flip from <b>{pred_label}</b> → <b>{flip_label}</b>, these signals would need to change:
            </div>
            """, unsafe_allow_html=True)

            st.dataframe(cf_df, hide_index=True, use_container_width=True)
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            st.pyplot(plot_counterfactual(cf_df, pred_label, flip_label), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # Always show LSTM attention if model exists
    if S.get("lstm_model") is not None and S.get("lstm_X_test") is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='groww-header' style='font-size:1.1rem'>👁️ LSTM Attention Map</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='insight-box'>
        <div class='title'>📖 What is LSTM Attention?</div>
        The LSTM reads the last 10 trading days as a sequence before predicting. The attention map shows
        <b>which of those 10 days had the most influence</b> on today's prediction.
        A brighter cell = that past day's pattern mattered more than others.
        </div>
        """, unsafe_allow_html=True)

        lstm_X_test = S["lstm_X_test"]
        attn_idx = st.slider("Trading Day for attention map", 0,
                             max(0, len(lstm_X_test) - 1),
                             len(lstm_X_test) // 2, key="attn_idx")
        attn_weights = compute_lstm_attention(S["lstm_model"], lstm_X_test, idx=attn_idx)
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.pyplot(plot_lstm_attention(attn_weights), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Explain the top attention day
        top_day = int(np.argmax(attn_weights))
        days_ago = len(attn_weights) - top_day
        st.markdown(f"""
        <div class='insight-box'>
        <div class='title'>🎯 Key Finding</div>
        The LSTM paid most attention to <b>Day –{days_ago}</b> (i.e. {days_ago} trading days ago) with
        <b>{attn_weights[top_day]:.1%}</b> of its total attention weight.
        This means the price/volume pattern from that specific day had the strongest influence
        on today's prediction.
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 8 — Backtesting Engine
# ──────────────────────────────────────────────────────────────────────────────
with tab8:
    st.markdown("<div class='groww-header'>📈 Backtesting Engine</div>", unsafe_allow_html=True)
    st.markdown("<div class='groww-subtext'>Test the AI strategy on 2 years of real historical data. See Sharpe ratio, max drawdown, win rate and total return.</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='insight-box'>
    <div class='title'>📖 What is Backtesting?</div>
    Backtesting replays the model's predictions on <em>past</em> data to see how a trading strategy
    would have performed. <b>Sharpe Ratio</b> measures risk-adjusted return (>1 = good, >2 = excellent).
    <b>Max Drawdown</b> is the worst peak-to-trough loss. <b>Win Rate</b> is % of correct BUY trades.
    </div>
    """, unsafe_allow_html=True)

    bt_col1, bt_col2 = st.columns(2)

    for col, model_name, bt_key in [
        (bt_col1, "🌲 Random Forest", "backtest_rf"),
        (bt_col2, "⚡ XGBoost",       "backtest_xgb"),
    ]:
        bt_data = S.get(bt_key)
        if bt_data is None:
            col.info("Train models first to see backtesting results.")
            continue

        metrics    = bt_data["metrics"]
        equity_df  = bt_data["equity_curve"]
        trades_df  = bt_data["trades"]
        ticker_bt  = S["ticker"]
        currency_bt = S["currency"]

        col.markdown(f"<div class='groww-header' style='font-size:1.05rem'>{model_name} Strategy</div>",
                     unsafe_allow_html=True)

        # Metric cards
        metric_cols = col.columns(3)
        metric_items = list(metrics.items())
        colors_m = {
            "Total Return": "#00916e" if "+" in metrics["Total Return"] else "#dc2626",
            "CAGR":         "#00916e" if "+" in metrics["CAGR"] else "#dc2626",
            "Sharpe Ratio": "#6366f1",
            "Max Drawdown": "#dc2626",
            "Win Rate":     "#00916e",
            "Total Trades": "#374151",
        }
        for i, (k, v) in enumerate(metric_items):
            mc = metric_cols[i % 3]
            mc.markdown(
                f'<div style="background:#ffffff;border:1.5px solid #e8eaf0;border-radius:10px;'
                f'padding:12px 8px;text-align:center;margin-bottom:8px">'
                f'<div style="font-size:0.65rem;color:#9ca3af;text-transform:uppercase;'
                f'letter-spacing:0.8px;font-weight:600;margin-bottom:4px">{k}</div>'
                f'<div style="font-size:1.2rem;font-weight:800;color:{colors_m.get(k,"#374151")}">{v}</div>'
                f'</div>', unsafe_allow_html=True)

        # Equity curve
        col.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        col.pyplot(plot_equity_curve(equity_df, ticker_bt, currency_bt),
                   use_container_width=True)
        col.markdown('</div>', unsafe_allow_html=True)

    # Trade Log
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='groww-header' style='font-size:1.1rem'>📋 Trade Log — Random Forest</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='insight-box'>
    <div class='title'>📖 What is this table?</div>
    Every row is one day the model said <b>BUY</b> with enough confidence (>55%).
    The <b>Return</b> column shows what actually happened that day — a green ✅ win or red ❌ loss.
    This gives you transparency into exactly when and why the strategy traded.
    </div>
    """, unsafe_allow_html=True)

    bt_rf = S.get("backtest_rf")
    if bt_rf and not bt_rf["trades"].empty:
        trades_show = bt_rf["trades"].copy()
        st.dataframe(trades_show, hide_index=True, use_container_width=True,
                     height=min(400, 35 * len(trades_show) + 40))
    else:
        st.info("No trades recorded — try lowering the confidence threshold.")

    # Confidence threshold re-run
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='groww-header' style='font-size:1.0rem'>⚙️ Adjust Confidence Threshold</div>",
                unsafe_allow_html=True)
    threshold = st.slider("Only trade when model confidence exceeds:", 0.50, 0.90, 0.55, 0.01,
                          key="bt_threshold")
    if st.button("🔄 Re-run Backtest with New Threshold", key="rerun_bt"):
        with st.spinner("Re-running backtests…"):
            df_feat_bt   = S["df_feat"]
            X_test_bt    = S["X_test"]
            df_test_bt   = df_feat_bt.iloc[-len(X_test_bt):]
            S["backtest_rf"]  = run_backtest(S["rf"],        X_test_bt, df_test_bt, threshold)
            S["backtest_xgb"] = run_backtest(S["xgb_model"], X_test_bt, df_test_bt, threshold)
        st.rerun()

    # Updated XAI comparison table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='groww-header' style='font-size:1.1rem'>🎓 Complete XAI Method Comparison</div>",
                unsafe_allow_html=True)
    full_summary = pd.DataFrame({
        "Method":      ["SHAP 🔍", "LIME 🧩", "Counterfactuals 🔮", "LSTM Attention 👁️", "Feature Importance 📌"],
        "Best For":    ["Overall signal importance", "One specific day's decision",
                        "Actionable trading thresholds", "Deep learning transparency",
                        "Quick model overview"],
        "Answers":     ["Why does AI usually predict UP?", "Why BUY/SELL today specifically?",
                        "What exact values would flip the call?", "Which past days drove today's prediction?",
                        "What does AI pay most attention to?"],
        "Scope":       ["Global", "Local", "Local (What-if)", "Local (Temporal)", "Global"],
        "Reliability": ["⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐", "⭐⭐⭐"],
        "Novel?":      ["Standard", "Standard", "✨ Research Grade", "✨ Research Grade", "Standard"],
    })
    st.dataframe(full_summary, hide_index=True, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 9 — Portfolio XAI
# ──────────────────────────────────────────────────────────────────────────────
with tab9:
    st.markdown("<div class='groww-header'>💼 Portfolio XAI</div>", unsafe_allow_html=True)
    st.markdown("<div class='groww-subtext'>Analyse a basket of stocks together — see which signals drive your entire portfolio, not just one stock.</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='insight-box'>
    <div class='title'>📖 What is Portfolio-level XAI?</div>
    Single-stock XAI tells you why <em>one</em> stock got a BUY signal. Portfolio XAI asks:
    <b>across all your stocks, which signals matter most?</b> Are your stocks truly diversified,
    or do they all depend on the same indicator (e.g. RSI)? If they do, one bad RSI reading
    could crash your entire portfolio at once.
    </div>
    """, unsafe_allow_html=True)

    # ── Stock picker ──────────────────────────────────────────────────────────
    st.markdown("#### Select your portfolio stocks")
    col_l, col_r = st.columns([3, 1])
    with col_l:
        portfolio_input = st.text_input(
            "Enter ticker symbols separated by commas",
            value="RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS",
            help="Use .NS for NSE, .BO for BSE, no suffix for US stocks",
            key="port_tickers",
        )
    with col_r:
        port_period = st.selectbox("Data period", ["6mo", "1y", "2y"], index=1, key="port_period")

    tickers_raw = [t.strip().upper() for t in portfolio_input.split(",") if t.strip()]

    if len(tickers_raw) < 2:
        st.warning("Please enter at least 2 ticker symbols.")
        st.stop()

    if len(tickers_raw) > 10:
        st.warning("Maximum 10 stocks for performance. Using first 10.")
        tickers_raw = tickers_raw[:10]

    run_portfolio = st.button("🚀 Analyse Portfolio", type="primary",
                              key="run_portfolio", use_container_width=False)

    if "portfolio_data" not in st.session_state:
        st.session_state.portfolio_data = None

    if run_portfolio:
        prog = st.progress(0, text="Starting portfolio analysis…")
        status_box = st.empty()

        def _progress(i, n, ticker):
            pct = int((i / n) * 90)
            prog.progress(pct, text=f"Training models for {ticker} ({i+1}/{n})…")
            status_box.markdown(f"<div class='info-box'>⏳ Processing <b>{ticker}</b>…</div>",
                                unsafe_allow_html=True)

        portfolio_data = build_portfolio(
            tickers_raw,
            period=port_period,
            sentiment_enabled=False,
            progress_cb=_progress,
        )
        prog.progress(100, text="✅ Done!")
        status_box.empty()
        st.session_state.portfolio_data = portfolio_data
        st.success(f"✅ Portfolio analysis complete for {len(tickers_raw)} stocks!")

    pdata = st.session_state.portfolio_data
    if pdata is None:
        st.info("👆 Enter your stocks and click **Analyse Portfolio** to begin.")
        st.stop()

    # ── Signal summary table ──────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='groww-header' style='font-size:1.1rem'>📋 Portfolio Signal Summary</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='insight-box'>
    <div class='title'>📖 What does this table show?</div>
    Each row is one stock in your portfolio. <b>Signal</b> = what the AI recommends today.
    <b>Confidence</b> = how sure the model is. Low confidence (&lt;60%) means the stock
    is near the decision boundary — treat those signals with caution.
    </div>
    """, unsafe_allow_html=True)

    sig_df = portfolio_signal_table(pdata, currency=S.get("currency", "₹"))
    st.dataframe(sig_df, hide_index=True, use_container_width=True)

    # ── Diversification score ─────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='groww-header' style='font-size:1.1rem'>🎯 XAI Diversification Score</div>", unsafe_allow_html=True)
    div_result = portfolio_diversification_score(pdata)

    score  = div_result["score"]
    color  = "#00916e" if score >= 70 else "#f59e0b" if score >= 40 else "#dc2626"
    badge  = "🟢 Well Diversified" if score >= 70 else "🟡 Moderate" if score >= 40 else "🔴 Concentrated Risk"

    c1, c2, c3 = st.columns(3)
    for col, lbl, val, clr in [
        (c1, "Diversification Score", f"{score}/100", color),
        (c2, "Dominant Signal",       div_result["dominant_feature"], "#6366f1"),
        (c3, "Stocks Sharing Top Signal", f"{div_result['agreement_pct']:.0f}%", color),
    ]:
        col.markdown(
            f'<div style="background:#ffffff;border:1.5px solid #e8eaf0;border-radius:12px;'
            f'padding:16px;text-align:center">'
            f'<div style="font-size:0.72rem;color:#9ca3af;text-transform:uppercase;'
            f'letter-spacing:1px;font-weight:600;margin-bottom:6px">{lbl}</div>'
            f'<div style="font-size:1.6rem;font-weight:800;color:{clr}">{val}</div>'
            f'</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class='insight-box' style='margin-top:12px'>
    <div class='title'>{badge}</div>
    {div_result["interpretation"]}
    </div>
    """, unsafe_allow_html=True)

    # ── Portfolio SHAP bar ────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='groww-header' style='font-size:1.1rem'>📊 Portfolio-wide Feature Importance (SHAP)</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='insight-box'>
    <div class='title'>📖 How to read this chart</div>
    Each bar = average importance of that signal <b>across all your stocks</b>.
    <b>Darker bars</b> = more stocks agree that feature matters.
    <b>Lighter bars</b> = only a few stocks are driven by that feature.
    </div>
    """, unsafe_allow_html=True)

    port_model_choice = st.radio("Model for SHAP", ["Random Forest", "XGBoost"],
                                  horizontal=True, key="port_model")
    shap_key = "shap_rf" if "Forest" in port_model_choice else "shap_xgb"
    agg_df   = aggregate_portfolio_shap(pdata, shap_key)

    if not agg_df.empty:
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.pyplot(plot_portfolio_shap_bar(agg_df, top_n=15), use_container_width=True)
        plt.close("all")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Full feature importance table**")
        st.dataframe(agg_df, hide_index=True, use_container_width=True)

    # ── Cross-stock heatmap ───────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='groww-header' style='font-size:1.1rem'>🗺️ Which Signal Drives Which Stock?</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='insight-box'>
    <div class='title'>📖 How to read this heatmap</div>
    Each row = one stock. Each column = a technical signal.
    <b>Darker green</b> = that signal strongly drives that stock's AI prediction.
    Use this to spot if two stocks in your portfolio are driven by identical signals —
    that means they'll likely move together (low diversification).
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.pyplot(plot_portfolio_heatmap(pdata, shap_key, top_features=10), use_container_width=True)
    plt.close("all")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Return correlation ────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='groww-header' style='font-size:1.1rem'>📉 Return Correlation</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='insight-box'>
    <div class='title'>📖 What does correlation mean here?</div>
    A value of <b>+1.0</b> means two stocks move together perfectly — not diversified.
    A value of <b>0.0</b> means they move independently — good diversification.
    A value of <b>−1.0</b> means they move opposite — perfect hedge.
    Ideally your portfolio should have low correlation between stocks.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.pyplot(plot_return_correlation(pdata), use_container_width=True)
    plt.close("all")
    st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 10 — Robo-Advisor & Portfolio Designer
# ──────────────────────────────────────────────────────────────────────────────
with tab10:
    st.markdown("<div class='groww-header'>🤖 Robo-Advisor & Portfolio Designer</div>", unsafe_allow_html=True)
    st.markdown("<div class='groww-subtext'>Upload your portfolio for AI analysis, or let the AI design a new portfolio tailored to your risk tolerance, capital and time horizon.</div>", unsafe_allow_html=True)

    robo_tab1, robo_tab2 = st.tabs(["📂 Analyse My Portfolio", "✨ Design New Portfolio"])

    # ══════════════════════════════════════════════════════════════════════════
    # SUB-TAB A — Analyse uploaded portfolio
    # ══════════════════════════════════════════════════════════════════════════
    with robo_tab1:
        st.markdown("""
        <div class='insight-box'>
        <div class='title'>📖 How this works</div>
        Upload a CSV of your holdings → AI trains on each stock → gives you
        <b>BUY MORE / HOLD / SELL</b> recommendation per stock based on both
        the AI signal AND your current profit/loss position.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**CSV format required:**")
        sample_csv = "Ticker,Shares,Avg_Buy_Price\nRELIANCE.NS,10,2500\nTCS.NS,5,3800\nAAPL,8,175\n"
        st.code(sample_csv, language="csv")
        st.download_button("⬇️ Download sample CSV", sample_csv,
                           file_name="sample_portfolio.csv", mime="text/csv")

        uploaded = st.file_uploader("Upload your portfolio CSV", type=["csv"], key="port_upload")

        if uploaded:
            try:
                port_df = parse_portfolio_csv(uploaded)
                st.markdown(f"✅ **{len(port_df)} stocks detected**")
                st.dataframe(port_df, hide_index=True, use_container_width=True)

                up_currency = st.selectbox("Your portfolio currency",
                                           ["₹ INR", "$ USD", "¥ JPY", "£ GBP", "€ EUR",
                                            "HK$ HKD", "S$ SGD", "A$ AUD"],
                                           key="up_currency")
                cur_sym = up_currency.split()[0]

                if st.button("🔍 Analyse My Portfolio", type="primary", key="run_analyse"):
                    prog2 = st.progress(0, "Starting analysis…")
                    status2 = st.empty()

                    def _p2(i, n, t):
                        prog2.progress(int(i/n*95), f"Analysing {t} ({i+1}/{n})…")
                        status2.markdown(f"<div class='info-box'>⏳ Processing <b>{t}</b>…</div>",
                                         unsafe_allow_html=True)

                    result_df = analyse_uploaded_portfolio(port_df, progress_cb=_p2)
                    prog2.progress(100, "✅ Done!")
                    status2.empty()

                    st.session_state["analysed_portfolio"] = result_df

            except Exception as e:
                st.error(f"❌ {e}")

        if "analysed_portfolio" in st.session_state and st.session_state["analysed_portfolio"] is not None:
            rdf = st.session_state["analysed_portfolio"]
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='groww-header' style='font-size:1.1rem'>📋 AI Recommendation per Stock</div>", unsafe_allow_html=True)

            # colour-code the recommendation column
            def _colour(val):
                if "SELL" in str(val) and "🔴" in str(val):
                    return "background-color:#fef2f2; color:#b91c1c"
                if "🟢" in str(val):
                    return "background-color:#f0fdf4; color:#15803d"
                if "🟡" in str(val):
                    return "background-color:#fffbeb; color:#92400e"
                return ""

            styled = rdf.style.applymap(_colour, subset=["Recommendation"])
            st.dataframe(styled, hide_index=True, use_container_width=True)

            # Summary counts
            sells  = rdf["Recommendation"].str.contains("🔴").sum()
            holds  = rdf["Recommendation"].str.contains("⚪|⏸️").sum()
            adds   = rdf["Recommendation"].str.contains("🟢").sum()
            yellows= rdf["Recommendation"].str.contains("🟡").sum()

            cs = st.columns(4)
            for col, lbl, val, clr in [
                (cs[0], "Strong SELL", sells,   "#dc2626"),
                (cs[1], "Consider",    yellows,  "#f59e0b"),
                (cs[2], "Hold / Add",  adds,     "#00916e"),
                (cs[3], "Hold (flat)", holds,    "#6366f1"),
            ]:
                col.markdown(
                    f'<div style="background:#ffffff;border:1.5px solid #e8eaf0;border-radius:12px;'
                    f'padding:14px;text-align:center">'
                    f'<div style="font-size:0.72rem;color:#9ca3af;text-transform:uppercase;'
                    f'letter-spacing:1px;font-weight:600;margin-bottom:6px">{lbl}</div>'
                    f'<div style="font-size:1.8rem;font-weight:800;color:{clr}">{val}</div>'
                    f'</div>', unsafe_allow_html=True)

            # Sector breakdown of uploaded portfolio
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='groww-header' style='font-size:1.1rem'>🥧 Sector Breakdown of Your Portfolio</div>", unsafe_allow_html=True)
            sec_counts = rdf.groupby("Sector").size().to_dict()
            col_p, col_t = st.columns([1, 1])
            with col_p:
                st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                st.pyplot(plot_sector_pie(sec_counts, "Your portfolio sectors"), use_container_width=True)
                plt.close("all")
                st.markdown('</div>', unsafe_allow_html=True)
            with col_t:
                st.markdown("""
                <div class='insight-box' style='margin-top:8px'>
                <div class='title'>📖 Is your portfolio diversified?</div>
                Ideally no single sector should exceed 35% of your portfolio.
                High concentration in one sector means one bad news event
                could hurt your entire portfolio at once.
                </div>
                """, unsafe_allow_html=True)
                sec_df = pd.DataFrame(list(sec_counts.items()),
                                      columns=["Sector", "Stocks"])
                sec_df["Share %"] = (sec_df["Stocks"] / sec_df["Stocks"].sum() * 100).round(1)
                st.dataframe(sec_df, hide_index=True, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SUB-TAB B — Design new portfolio
    # ══════════════════════════════════════════════════════════════════════════
    with robo_tab2:
        st.markdown("""
        <div class='insight-box'>
        <div class='title'>📖 How portfolio design works</div>
        Tell us your <b>capital, risk appetite and time horizon</b> →
        AI scores 15–20 stocks from your chosen market using RF + XGBoost signals,
        1-year momentum, volatility, and sector diversification rules →
        outputs an optimised allocation you can download and act on.
        </div>
        """, unsafe_allow_html=True)

        # ── Inputs ────────────────────────────────────────────────────────────
        col1, col2 = st.columns(2)
        with col1:
            invest_capital = st.number_input("💰 Total investment amount",
                                              min_value=1000, max_value=100_000_000,
                                              value=100_000, step=1000, key="invest_cap")
            invest_currency = st.selectbox("Currency",
                                            ["₹ INR","$ USD","£ GBP","€ EUR",
                                             "¥ JPY","HK$ HKD","S$ SGD","A$ AUD"],
                                            key="invest_cur")
            invest_market = st.selectbox("Market to invest in",
                                          ["India","USA","Global"], key="invest_mkt")

        with col2:
            risk_choice = st.selectbox("Risk tolerance", list(RISK_PROFILES.keys()),
                                        index=1, key="risk_choice")
            time_horizon = st.slider("Investment time horizon (years)",
                                      1, 20, 5, key="time_horizon")
            st.markdown(f"""
            <div style='background:#f7f8fa;border:1.5px solid #e8eaf0;
            border-radius:10px;padding:12px;margin-top:8px;font-size:0.85rem;color:#374151'>
            <b>{risk_choice.split()[0]}</b>: {RISK_PROFILES[risk_choice]['description']}<br>
            Target return: <b>{RISK_PROFILES[risk_choice]['target_return']}</b> ·
            Volatility: <b>{RISK_PROFILES[risk_choice]['volatility']}</b>
            </div>
            """, unsafe_allow_html=True)

        cur_sym2 = invest_currency.split()[0]

        # Risk radar
        st.markdown("<br>", unsafe_allow_html=True)
        rcol1, rcol2 = st.columns([1, 2])
        with rcol1:
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            st.pyplot(plot_risk_radar(risk_choice), use_container_width=True)
            plt.close("all")
            st.markdown('</div>', unsafe_allow_html=True)
        with rcol2:
            profile_data = RISK_PROFILES[risk_choice]
            st.markdown(f"""
            <div class='insight-box'>
            <div class='title'>📊 Your Profile Summary</div>
            <b>Equity allocation:</b> {profile_data['equity_pct']*100:.0f}%
            (stocks) + {profile_data['bond_pct']*100:.0f}% (bonds/cash)<br>
            <b>Max drawdown tolerance:</b> {profile_data['max_dd_tolerance']*100:.0f}%<br>
            <b>Preferred sectors:</b> {', '.join(profile_data['preferred_sectors'])}<br>
            <b>Min stocks in portfolio:</b> {profile_data['min_stocks']}<br>
            <b>Investment amount in equities:</b>
            {cur_sym2}{invest_capital * profile_data['equity_pct']:,.0f}
            </div>
            """, unsafe_allow_html=True)

        design_btn = st.button("✨ Design My Portfolio", type="primary",
                                key="design_btn", use_container_width=False)

        if "designed_portfolio" not in st.session_state:
            st.session_state["designed_portfolio"] = None

        if design_btn:
            prog3 = st.progress(0, "Scoring stocks…")
            status3 = st.empty()

            def _p3(i, n, t):
                prog3.progress(int(i/n*90), f"Evaluating {t} ({i+1}/{n})…")
                status3.markdown(f"<div class='info-box'>⏳ Scoring <b>{t}</b>…</div>",
                                  unsafe_allow_html=True)

            result = design_portfolio(
                capital=invest_capital,
                risk_profile=risk_choice,
                time_horizon_years=time_horizon,
                market=invest_market,
                currency=cur_sym2,
                progress_cb=_p3,
            )
            prog3.progress(100, "✅ Portfolio designed!")
            status3.empty()

            if "error" in result:
                st.error(f"❌ {result['error']}")
            else:
                st.session_state["designed_portfolio"] = result
                st.success("✅ Portfolio designed! Scroll down to see your allocation.")

        dp = st.session_state.get("designed_portfolio")
        if dp and "error" not in dp:
            alloc_df     = dp["allocation"]
            metrics      = dp["metrics"]
            sec_wts      = dp["sector_weights"]

            # ── Metrics row ────────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='groww-header' style='font-size:1.1rem'>📈 Portfolio Metrics</div>", unsafe_allow_html=True)

            metric_items = [(k, v) for k, v in metrics.items() if k != "Horizon Note"]
            mcols = st.columns(4)
            colors_m = {
                "Expected Return (1Y)": "#00916e",
                "Portfolio Volatility":  "#f59e0b",
                "Sharpe Score":          "#6366f1",
                "Equity Allocation":     "#00916e",
                "Bond/Cash Allocation":  "#374151",
                "Target Return (profile)":"#00916e",
                "Stocks Selected":        "#374151",
                "Sectors Covered":        "#6366f1",
            }
            for idx, (k, v) in enumerate(metric_items):
                mcol = mcols[idx % 4]
                clr  = colors_m.get(k, "#374151")
                mcol.markdown(
                    f'<div style="background:#ffffff;border:1.5px solid #e8eaf0;'
                    f'border-radius:10px;padding:12px 8px;text-align:center;margin-bottom:8px">'
                    f'<div style="font-size:0.65rem;color:#9ca3af;text-transform:uppercase;'
                    f'letter-spacing:0.8px;font-weight:600;margin-bottom:4px">{k}</div>'
                    f'<div style="font-size:1.2rem;font-weight:800;color:{clr}">{v}</div>'
                    f'</div>', unsafe_allow_html=True)

            # Horizon note
            st.markdown(f"""
            <div class='insight-box'>
            <div class='title'>⏳ Time Horizon Advice ({time_horizon} years)</div>
            {metrics['Horizon Note']}
            </div>
            """, unsafe_allow_html=True)

            # ── Allocation table ───────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='groww-header' style='font-size:1.1rem'>📋 Recommended Stock Allocation</div>", unsafe_allow_html=True)
            st.dataframe(alloc_df, hide_index=True, use_container_width=True)

            # ── Charts row ─────────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            ch1, ch2 = st.columns(2)
            with ch1:
                st.markdown("**Sector allocation**")
                st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                st.pyplot(plot_sector_pie(sec_wts), use_container_width=True)
                plt.close("all")
                st.markdown('</div>', unsafe_allow_html=True)
            with ch2:
                st.markdown("**Stock weight allocation**")
                st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                st.pyplot(plot_allocation_bar(alloc_df, cur_sym2), use_container_width=True)
                plt.close("all")
                st.markdown('</div>', unsafe_allow_html=True)

            # ── Download ───────────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            csv_bytes = generate_portfolio_csv(alloc_df)
            st.download_button(
                "⬇️ Download Portfolio as CSV",
                csv_bytes,
                file_name=f"xai_portfolio_{risk_choice.split()[0].lower()}_{invest_market}.csv",
                mime="text/csv",
                type="primary",
            )

            st.markdown("""
            <div class='warn-box'>
            ⚠️ <b>Disclaimer:</b> This is an AI-generated suggestion based on historical data and model predictions.
            It is <b>not financial advice</b>. Always consult a SEBI-registered investment advisor before investing.
            Past performance does not guarantee future returns.
            </div>
            """, unsafe_allow_html=True)
