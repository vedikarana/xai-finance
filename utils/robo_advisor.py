"""
Robo-Advisor & Portfolio Designer — XAI Finance v2
Features:
  - Upload existing portfolio (CSV) → BUY/SELL/HOLD recommendation per stock
  - Risk tolerance profiling → Conservative / Moderate / Aggressive
  - AI-designed portfolio based on risk + capital + time horizon
  - Sector diversification scoring
  - News sentiment overlay per stock
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Sector metadata
# ─────────────────────────────────────────────────────────────────────────────

SECTOR_MAP = {
    # India NSE
    "RELIANCE.NS": "Energy", "TCS.NS": "Technology", "INFY.NS": "Technology",
    "HDFCBANK.NS": "Finance", "WIPRO.NS": "Technology", "ICICIBANK.NS": "Finance",
    "BAJFINANCE.NS": "Finance", "HCLTECH.NS": "Technology", "SBIN.NS": "Finance",
    "SUNPHARMA.NS": "Healthcare", "MARUTI.NS": "Automotive", "TITAN.NS": "Consumer",
    "ASIANPAINT.NS": "Consumer", "HINDUNILVR.NS": "Consumer", "KOTAKBANK.NS": "Finance",
    "LT.NS": "Industrials", "AXISBANK.NS": "Finance", "DRREDDY.NS": "Healthcare",
    "ITC.NS": "Consumer", "ONGC.NS": "Energy", "NTPC.NS": "Utilities",
    "TATAMOTORS.NS": "Automotive", "TATASTEEL.NS": "Materials", "JSWSTEEL.NS": "Materials",
    # USA
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "AMZN": "Consumer", "NVDA": "Technology", "TSLA": "Automotive",
    "META": "Technology", "NFLX": "Entertainment", "JPM": "Finance",
    "KO": "Consumer", "DIS": "Entertainment", "V": "Finance",
    "MA": "Finance", "CRM": "Technology", "ADBE": "Technology",
    "INTC": "Technology", "AMD": "Technology", "QCOM": "Technology",
    "GS": "Finance", "MS": "Finance", "BAC": "Finance",
    "JNJ": "Healthcare", "PFE": "Healthcare", "UNH": "Healthcare",
    "XOM": "Energy", "CVX": "Energy", "BA": "Industrials",
    # China
    "BABA": "Technology", "JD": "Consumer", "BIDU": "Technology",
    "NIO": "Automotive", "PDD": "Consumer", "0700.HK": "Technology",
    "9988.HK": "Technology", "1211.HK": "Automotive",
    # Japan
    "7203.T": "Automotive", "6758.T": "Technology", "9984.T": "Technology",
    "7267.T": "Automotive", "7974.T": "Entertainment",
    # Default
    "DEFAULT": "Other",
}

def get_sector(ticker: str) -> str:
    return SECTOR_MAP.get(ticker.upper(), SECTOR_MAP.get(ticker, "Other"))


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Risk profile
# ─────────────────────────────────────────────────────────────────────────────

RISK_PROFILES = {
    "Conservative 🛡️": {
        "description": "Capital preservation first. Low volatility, stable dividends.",
        "max_single_stock": 0.08,
        "sector_caps": {"Technology": 0.20, "Finance": 0.25, "DEFAULT": 0.15},
        "preferred_sectors": ["Finance", "Healthcare", "Consumer", "Utilities", "Energy"],
        "equity_pct": 0.40,
        "bond_pct":   0.60,
        "min_stocks": 10,
        "max_dd_tolerance": 0.10,
        "target_return": "6–9% p.a.",
        "volatility": "Low",
    },
    "Moderate ⚖️": {
        "description": "Balanced growth and protection. Diversified across sectors.",
        "max_single_stock": 0.12,
        "sector_caps": {"Technology": 0.30, "Finance": 0.25, "DEFAULT": 0.20},
        "preferred_sectors": ["Technology", "Finance", "Healthcare", "Consumer", "Industrials"],
        "equity_pct": 0.70,
        "bond_pct":   0.30,
        "min_stocks": 8,
        "max_dd_tolerance": 0.20,
        "target_return": "10–14% p.a.",
        "volatility": "Medium",
    },
    "Aggressive 🚀": {
        "description": "Maximum growth potential. High volatility accepted.",
        "max_single_stock": 0.20,
        "sector_caps": {"Technology": 0.50, "DEFAULT": 0.30},
        "preferred_sectors": ["Technology", "Automotive", "Entertainment", "Healthcare"],
        "equity_pct": 0.95,
        "bond_pct":   0.05,
        "min_stocks": 5,
        "max_dd_tolerance": 0.35,
        "target_return": "15–25% p.a.",
        "volatility": "High",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Parse uploaded CSV portfolio
# ─────────────────────────────────────────────────────────────────────────────

def parse_portfolio_csv(uploaded_file) -> pd.DataFrame:
    """
    Accept CSV with columns: Ticker, Shares, Avg_Buy_Price
    Returns DataFrame with added columns: Sector, Investment
    """
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # flexible column name detection
        ticker_col = next((c for c in df.columns if "ticker" in c or "symbol" in c or "stock" in c), None)
        shares_col = next((c for c in df.columns if "share" in c or "qty" in c or "quantity" in c or "units" in c), None)
        price_col  = next((c for c in df.columns if "price" in c or "cost" in c or "buy" in c or "avg" in c), None)

        if not ticker_col:
            raise ValueError("No 'Ticker' column found. CSV needs: Ticker, Shares, Avg_Buy_Price")

        df = df.rename(columns={
            ticker_col: "Ticker",
            shares_col: "Shares"   if shares_col else None,
            price_col:  "Avg_Buy_Price" if price_col else None,
        }).dropna(subset=["Ticker"])

        df["Ticker"]        = df["Ticker"].str.strip().str.upper()
        df["Shares"]        = pd.to_numeric(df.get("Shares", 1),        errors="coerce").fillna(1)
        df["Avg_Buy_Price"] = pd.to_numeric(df.get("Avg_Buy_Price", 0), errors="coerce").fillna(0)
        df["Investment"]    = df["Shares"] * df["Avg_Buy_Price"]
        df["Sector"]        = df["Ticker"].apply(get_sector)

        return df[["Ticker", "Shares", "Avg_Buy_Price", "Investment", "Sector"]]
    except Exception as e:
        raise ValueError(f"Could not parse CSV: {e}\n\nExpected format: Ticker, Shares, Avg_Buy_Price")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Analyse uploaded portfolio → BUY/SELL/HOLD per stock
# ─────────────────────────────────────────────────────────────────────────────

def analyse_uploaded_portfolio(portfolio_df: pd.DataFrame,
                                progress_cb=None) -> pd.DataFrame:
    """
    For each ticker in the uploaded portfolio:
      - Fetch current price
      - Train RF + XGBoost
      - Get signal + confidence
      - Compute P&L from buy price
      - Recommend: SELL if SELL signal + loss, HOLD if uncertain, BUY MORE if strong BUY
    """
    from utils.data_pipeline import fetch_stock_data, engineer_features, prepare_train_test
    from models.ml_models import train_random_forest, train_xgboost

    results = []
    n = len(portfolio_df)

    for i, row in portfolio_df.iterrows():
        ticker   = row["Ticker"]
        shares   = row["Shares"]
        buy_px   = row["Avg_Buy_Price"]
        sector   = row["Sector"]

        if progress_cb:
            progress_cb(i, n, ticker)

        try:
            df      = fetch_stock_data(ticker, "1y")
            cur_px  = float(df["Close"].iloc[-1])
            pl_pct  = ((cur_px - buy_px) / buy_px * 100) if buy_px > 0 else 0

            df_feat = engineer_features(df)
            X_train, X_test, y_train, y_test, scaler, features = prepare_train_test(df_feat)
            rf  = train_random_forest(X_train, y_train)
            xgb = train_xgboost(X_train, y_train)

            prob_rf  = rf.predict_proba(X_test[-1].reshape(1, -1))[0]
            prob_xgb = xgb.predict_proba(X_test[-1].reshape(1, -1))[0]

            # ensemble vote (average probabilities)
            avg_prob_up   = (prob_rf[1] + prob_xgb[1]) / 2
            avg_prob_down = 1 - avg_prob_up
            confidence    = max(avg_prob_up, avg_prob_down) * 100

            # signal
            if avg_prob_up >= 0.65:
                signal = "📈 BUY MORE"
            elif avg_prob_down >= 0.65:
                signal = "📉 SELL"
            else:
                signal = "⏸️ HOLD"

            # final recommendation combining signal + P&L
            if signal == "📉 SELL" and pl_pct < -5:
                recommendation = "🔴 SELL — AI bearish + in loss"
            elif signal == "📉 SELL" and pl_pct > 15:
                recommendation = "🟡 CONSIDER SELLING — AI bearish but in profit"
            elif signal == "📈 BUY MORE" and pl_pct > 0:
                recommendation = "🟢 HOLD / ADD — AI bullish + in profit"
            elif signal == "📈 BUY MORE" and pl_pct < 0:
                recommendation = "🟡 HOLD — AI bullish but currently in loss"
            else:
                recommendation = "⚪ HOLD — Insufficient confidence to act"

            results.append({
                "Ticker":         ticker,
                "Sector":         sector,
                "Shares":         shares,
                "Buy Price":      f"{buy_px:.2f}" if buy_px > 0 else "—",
                "Current Price":  f"{cur_px:.2f}",
                "P&L %":          f"{pl_pct:+.1f}%",
                "AI Signal":      signal,
                "Confidence":     f"{confidence:.1f}%",
                "Recommendation": recommendation,
            })
        except Exception as e:
            results.append({
                "Ticker": ticker, "Sector": sector,
                "Shares": shares, "Buy Price": "—",
                "Current Price": "Error", "P&L %": "—",
                "AI Signal": "❌ Error", "Confidence": "—",
                "Recommendation": f"Could not fetch: {str(e)[:40]}",
            })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  AI Portfolio Designer
# ─────────────────────────────────────────────────────────────────────────────

# Curated universe by exchange + sector
STOCK_UNIVERSE = {
    "India": {
        "Technology":  ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
        "Finance":     ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"],
        "Energy":      ["RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS"],
        "Healthcare":  ["SUNPHARMA.NS", "DRREDDY.NS"],
        "Consumer":    ["HINDUNILVR.NS", "ITC.NS", "TITAN.NS", "ASIANPAINT.NS"],
        "Automotive":  ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS"],
        "Industrials": ["LT.NS"],
        "Materials":   ["TATASTEEL.NS", "JSWSTEEL.NS"],
    },
    "USA": {
        "Technology":    ["AAPL", "MSFT", "GOOGL", "NVDA", "AMD", "CRM", "ADBE"],
        "Finance":       ["JPM", "GS", "MS", "BAC", "V", "MA"],
        "Healthcare":    ["JNJ", "PFE", "UNH"],
        "Consumer":      ["AMZN", "KO", "WMT", "COST", "MCD"],
        "Energy":        ["XOM", "CVX"],
        "Automotive":    ["TSLA"],
        "Entertainment": ["NFLX", "DIS"],
        "Industrials":   ["BA", "CAT"],
    },
    "Global": {
        "Technology":  ["TSM", "ASML", "SAP", "7203.T", "005930.KS"],
        "Finance":     ["HSBA.L", "0005.HK", "D05.SI"],
        "Energy":      ["BP.L", "SHEL.L"],
        "Consumer":    ["ULVR.L", "DGE.L"],
        "Healthcare":  ["AZN.L", "NVO", "ROG.SW"],
        "Automotive":  ["7203.T", "7267.T", "MBG.DE", "VOW3.DE"],
    },
}


def design_portfolio(
    capital: float,
    risk_profile: str,
    time_horizon_years: int,
    market: str = "India",
    currency: str = "₹",
    progress_cb=None,
) -> dict:
    """
    Designs an AI-driven portfolio given:
      - capital (total investment amount)
      - risk_profile (Conservative / Moderate / Aggressive)
      - time_horizon_years
      - market (India / USA / Global)

    Returns:
      { "allocation": DataFrame, "metrics": dict, "sector_weights": dict }
    """
    from utils.data_pipeline import fetch_stock_data, engineer_features, prepare_train_test
    from models.ml_models import train_random_forest, train_xgboost

    profile  = RISK_PROFILES[risk_profile]
    universe = STOCK_UNIVERSE.get(market, STOCK_UNIVERSE["India"])
    sectors  = profile["preferred_sectors"]

    # Pick candidate stocks from preferred sectors
    candidates = []
    for sector in sectors:
        if sector in universe:
            candidates.extend(universe[sector])

    # Remove duplicates, cap at 20 to keep runtime reasonable
    candidates = list(dict.fromkeys(candidates))[:20]

    scored = []
    n = len(candidates)

    for i, ticker in enumerate(candidates):
        if progress_cb:
            progress_cb(i, n, ticker)
        try:
            df      = fetch_stock_data(ticker, "1y")
            df_feat = engineer_features(df)
            X_train, X_test, y_train, y_test, scaler, features = prepare_train_test(df_feat)

            rf  = train_random_forest(X_train, y_train)
            xgb = train_xgboost(X_train, y_train)

            prob_rf  = rf.predict_proba(X_test[-1].reshape(1, -1))[0][1]
            prob_xgb = xgb.predict_proba(X_test[-1].reshape(1, -1))[0][1]
            bull_prob = (prob_rf + prob_xgb) / 2

            # 1-year return for momentum score
            ret_1y   = float((df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0] * 100)
            # volatility
            daily_ret = df["Close"].pct_change().dropna()
            vol       = float(daily_ret.std() * np.sqrt(252) * 100)
            # Sharpe-like score: return / volatility, weighted by AI bull prob
            score = (bull_prob * 0.5) + (max(ret_1y, 0) / max(vol, 1) * 0.5)

            scored.append({
                "Ticker":    ticker,
                "Sector":    get_sector(ticker),
                "Bull Prob": round(bull_prob * 100, 1),
                "1Y Return": round(ret_1y, 1),
                "Volatility":round(vol, 1),
                "Score":     round(score, 4),
                "Price":     round(float(df["Close"].iloc[-1]), 2),
            })
        except Exception:
            continue

    if not scored:
        return {"error": "No stocks could be scored. Check internet connection."}

    scored_df = pd.DataFrame(scored).sort_values("Score", ascending=False)

    # Select top stocks respecting sector caps
    max_single = profile["max_single_stock"]
    sector_caps = profile["sector_caps"]
    selected   = []
    sector_wts = {}

    for _, row in scored_df.iterrows():
        sector   = row["Sector"]
        cap      = sector_caps.get(sector, sector_caps.get("DEFAULT", 0.20))
        cur_wt   = sector_wts.get(sector, 0)
        if cur_wt < cap and len(selected) < 15:
            selected.append(row)
            sector_wts[sector] = cur_wt + max_single

    if not selected:
        selected = scored_df.head(profile["min_stocks"]).to_dict("records")

    sel_df = pd.DataFrame(selected)

    # Assign weights: higher score → higher weight, capped at max_single_stock
    raw_wts = sel_df["Score"].values
    raw_wts = raw_wts / raw_wts.sum()
    raw_wts = np.clip(raw_wts, 0, max_single)
    raw_wts = raw_wts / raw_wts.sum()   # renormalise

    sel_df["Weight %"] = (raw_wts * 100).round(1)
    sel_df["Amount"]   = (raw_wts * capital * profile["equity_pct"]).round(2)
    sel_df["Shares (approx)"] = (sel_df["Amount"] / sel_df["Price"]).round(2)

    # Portfolio-level metrics
    port_return = float((sel_df["Weight %"] / 100 * sel_df["1Y Return"]).sum())
    port_vol    = float((sel_df["Weight %"] / 100 * sel_df["Volatility"]).sum())
    sharpe      = port_return / max(port_vol, 0.01)

    # Sector breakdown
    sec_wts = sel_df.groupby("Sector")["Weight %"].sum().to_dict()

    # Time-horizon adjustment note
    if time_horizon_years <= 1:
        horizon_note = "Short-term: prioritise liquidity. Consider reducing equity allocation by 10%."
    elif time_horizon_years <= 3:
        horizon_note = "Medium-term: current allocation is well-suited."
    else:
        horizon_note = "Long-term: consider increasing equity allocation by 5–10% for higher compounding."

    metrics = {
        "Expected Return (1Y)":   f"{port_return:+.1f}%",
        "Portfolio Volatility":    f"{port_vol:.1f}%",
        "Sharpe Score":            f"{sharpe:.2f}",
        "Equity Allocation":       f"{profile['equity_pct']*100:.0f}%",
        "Bond/Cash Allocation":    f"{profile['bond_pct']*100:.0f}%",
        "Target Return (profile)": profile["target_return"],
        "Stocks Selected":         str(len(sel_df)),
        "Sectors Covered":         str(len(sec_wts)),
        "Horizon Note":            horizon_note,
    }

    out_df = sel_df[[
        "Ticker", "Sector", "Bull Prob", "1Y Return",
        "Volatility", "Weight %", "Amount", "Shares (approx)", "Score"
    ]].reset_index(drop=True)
    out_df.columns = [
        "Ticker", "Sector", "AI Bull Prob %", "1Y Return %",
        "Volatility %", "Weight %", f"Amount ({currency})", "Shares (approx)", "AI Score"
    ]

    return {
        "allocation":    out_df,
        "metrics":       metrics,
        "sector_weights":sec_wts,
        "scored_all":    scored_df,
        "profile":       profile,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Charts
# ─────────────────────────────────────────────────────────────────────────────

def plot_sector_pie(sector_weights: dict, title: str = "Sector Allocation") -> plt.Figure:
    colors = ["#00b386","#6366f1","#f59e0b","#ec4899","#06b6d4",
              "#84cc16","#f97316","#8b5cf6","#14b8a6","#ef4444"]
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#ffffff")
    wedges, texts, autotexts = ax.pie(
        list(sector_weights.values()),
        labels=list(sector_weights.keys()),
        colors=colors[:len(sector_weights)],
        autopct="%1.1f%%", startangle=90,
        pctdistance=0.82, textprops={"fontsize": 10, "color": "#374151"},
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
    ax.set_title(title, color="#111827", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def plot_allocation_bar(alloc_df: pd.DataFrame, currency: str = "₹") -> plt.Figure:
    col = [c for c in alloc_df.columns if "Amount" in c][0]
    top = alloc_df.nlargest(15, "Weight %")
    fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.5)))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f7f8fa")
    bars = ax.barh(top["Ticker"][::-1], top["Weight %"][::-1],
                   color="#00b386", alpha=0.82)
    ax.set_xlabel("Portfolio Weight (%)", color="#374151", fontsize=11)
    ax.set_title("Stock allocation by weight", color="#111827",
                 fontsize=13, fontweight="bold")
    ax.tick_params(colors="#374151")
    for spine in ax.spines.values():
        spine.set_edgecolor("#e8eaf0")
    for bar, (_, row) in zip(bars[::-1], top[::-1].iterrows()):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f'{currency}{row[col]:,.0f}',
                va="center", fontsize=8.5, color="#374151")
    plt.tight_layout()
    return fig


def plot_risk_radar(profile_name: str) -> plt.Figure:
    """Radar chart showing risk profile dimensions."""
    labels   = ["Return", "Risk", "Diversification", "Liquidity", "Stability"]
    profiles = {
        "Conservative 🛡️": [4, 2, 9, 8, 9],
        "Moderate ⚖️":      [6, 5, 7, 7, 6],
        "Aggressive 🚀":    [9, 9, 4, 5, 3],
    }
    values = profiles.get(profile_name, profiles["Moderate ⚖️"])
    values += values[:1]

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 4), subplot_kw={"polar": True})
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f7f8fa")
    ax.plot(angles, values, color="#00b386", linewidth=2)
    ax.fill(angles, values, color="#00b386", alpha=0.18)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9, color="#374151")
    ax.set_yticks([2, 4, 6, 8])
    ax.set_yticklabels(["2","4","6","8"], fontsize=7, color="#9ca3af")
    ax.set_ylim(0, 10)
    ax.set_title(f"Risk Profile — {profile_name.split()[0]}",
                 color="#111827", fontsize=11, fontweight="bold", pad=12)
    ax.grid(color="#e8eaf0", linewidth=0.5)
    ax.spines["polar"].set_color("#e8eaf0")
    plt.tight_layout()
    return fig


def generate_portfolio_csv(alloc_df: pd.DataFrame) -> bytes:
    """Return portfolio as downloadable CSV bytes."""
    return alloc_df.to_csv(index=False).encode("utf-8")
