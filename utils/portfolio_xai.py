"""
Portfolio-Level XAI Module — XAI Finance v2
Explains a basket of stocks together using SHAP aggregation,
correlation analysis, and portfolio-level risk signals.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Train models for every ticker in the portfolio
# ─────────────────────────────────────────────────────────────────────────────

def build_portfolio(
    tickers: list,
    period: str = "1y",
    sentiment_enabled: bool = False,
    progress_cb=None,
) -> dict:
    """
    Train RF + XGBoost for each ticker and collect SHAP values.

    Returns a dict keyed by ticker:
        {
          ticker: {
            "df": raw DataFrame,
            "df_feat": featured DataFrame,
            "rf": model,
            "xgb": model,
            "shap_rf": np.array,
            "shap_xgb": np.array,
            "X_shap": np.array,
            "features": list[str],
            "rf_metrics": dict,
            "xgb_metrics": dict,
            "last_pred": {"signal", "confidence", "price"},
          }
        }
    """
    from utils.data_pipeline import (
        fetch_stock_data, engineer_features,
        fetch_news_sentiment, prepare_train_test,
    )
    from models.ml_models import train_random_forest, train_xgboost, evaluate_model
    from explainers.xai_explainers import compute_shap_values

    results = {}
    n = len(tickers)

    for i, ticker in enumerate(tickers):
        if progress_cb:
            progress_cb(i, n, ticker)
        try:
            df = fetch_stock_data(ticker, period)
            if df is None or len(df) < 80:
                continue

            sentiment = None
            if sentiment_enabled:
                sentiment = fetch_news_sentiment(ticker, df.index)

            df_feat = engineer_features(df, sentiment)
            X_train, X_test, y_train, y_test, scaler, features = prepare_train_test(df_feat)

            rf  = train_random_forest(X_train, y_train)
            xgb = train_xgboost(X_train, y_train)

            _, shap_rf,  X_shap_rf  = compute_shap_values(rf,  X_train, X_test, "tree")
            _, shap_xgb, X_shap_xgb = compute_shap_values(xgb, X_train, X_test, "tree")

            # align cols
            nc = min(shap_rf.shape[1], shap_xgb.shape[1], len(features))
            shap_rf  = shap_rf[:, :nc]
            shap_xgb = shap_xgb[:, :nc]
            feats    = features[:nc]

            # last-day prediction
            last_x  = X_test[-1].reshape(1, -1)
            prob_rf  = rf.predict_proba(last_x)[0]
            signal   = "BUY" if prob_rf[1] > 0.5 else "SELL"
            conf     = max(prob_rf) * 100
            price    = float(df["Close"].iloc[-1])

            results[ticker] = {
                "df":         df,
                "df_feat":    df_feat,
                "rf":         rf,
                "xgb":        xgb,
                "shap_rf":    shap_rf,
                "shap_xgb":   shap_xgb,
                "X_shap":     X_shap_rf,
                "features":   feats,
                "rf_metrics": evaluate_model(rf,  X_test, y_test, "sklearn"),
                "xgb_metrics":evaluate_model(xgb, X_test, y_test, "xgb"),
                "last_pred":  {"signal": signal, "confidence": conf, "price": price},
            }
        except Exception as e:
            results[ticker] = {"error": str(e)}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Aggregate SHAP across the portfolio
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_portfolio_shap(portfolio: dict, model_key: str = "shap_rf") -> pd.DataFrame:
    """
    Mean |SHAP value| per feature across all tickers.
    Returns a DataFrame: Feature | mean_importance | std | n_stocks
    """
    all_fi = {}
    for ticker, data in portfolio.items():
        if "error" in data:
            continue
        shap_vals = data.get(model_key)
        features  = data.get("features", [])
        if shap_vals is None:
            continue
        mean_abs = np.abs(shap_vals).mean(axis=0)
        nc = min(len(mean_abs), len(features))
        for i in range(nc):
            fname = features[i]
            all_fi.setdefault(fname, []).append(float(mean_abs[i]))

    rows = []
    for feat, vals in all_fi.items():
        rows.append({
            "Feature":          feat,
            "Mean Importance":  round(np.mean(vals), 5),
            "Std":              round(np.std(vals),  5),
            "Stocks Agreeing":  len(vals),
        })
    df = pd.DataFrame(rows).sort_values("Mean Importance", ascending=False).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Per-stock signal summary
# ─────────────────────────────────────────────────────────────────────────────

def portfolio_signal_table(portfolio: dict, currency: str = "₹") -> pd.DataFrame:
    """One-row-per-stock summary table."""
    rows = []
    for ticker, data in portfolio.items():
        if "error" in data:
            rows.append({
                "Ticker":     ticker,
                "Price":      "—",
                "Signal":     "ERROR",
                "Confidence": "—",
                "RF Accuracy":"—",
                "XGB Accuracy":"—",
                "Status":     f"❌ {data['error'][:40]}",
            })
            continue
        pred = data["last_pred"]
        rfm  = data["rf_metrics"]
        xgbm = data["xgb_metrics"]
        rows.append({
            "Ticker":      ticker,
            "Price":       f"{currency}{pred['price']:,.2f}",
            "Signal":      "📈 BUY" if pred["signal"] == "BUY" else "📉 SELL",
            "Confidence":  f"{pred['confidence']:.1f}%",
            "RF Accuracy": f"{rfm.get('accuracy', 0)*100:.1f}%",
            "XGB Accuracy":f"{xgbm.get('accuracy', 0)*100:.1f}%",
            "Status":      "✅ OK",
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Portfolio-level SHAP bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_portfolio_shap_bar(agg_df: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """
    Horizontal bar chart of mean |SHAP| across the portfolio.
    Colour intensity encodes how many stocks agree on that feature.
    """
    top = agg_df.head(top_n).copy()
    max_stocks = top["Stocks Agreeing"].max()
    alphas = (top["Stocks Agreeing"] / max_stocks).values

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.45)))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f7f8fa")

    colors = [f"rgba(0,{int(145*a)},{int(110*a)},1)" for a in alphas]
    bars = ax.barh(top["Feature"][::-1], top["Mean Importance"][::-1],
                   color="#00b386", alpha=0.75)

    # shade by agreement
    for bar, a in zip(bars, alphas[::-1]):
        bar.set_alpha(max(0.25, a))

    ax.set_xlabel("Mean |SHAP Value| across portfolio", color="#374151", fontsize=11)
    ax.set_title("Portfolio-level feature importance (SHAP)", color="#111827",
                 fontsize=14, fontweight="bold")
    ax.tick_params(colors="#374151")
    for spine in ax.spines.values():
        spine.set_edgecolor("#e8eaf0")

    # legend
    p1 = mpatches.Patch(color="#00b386", alpha=1.0,  label="All stocks agree")
    p2 = mpatches.Patch(color="#00b386", alpha=0.35, label="Few stocks agree")
    ax.legend(handles=[p1, p2], facecolor="#f7f8fa", labelcolor="#374151", fontsize=9)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Cross-stock SHAP heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_portfolio_heatmap(portfolio: dict,
                           model_key: str = "shap_rf",
                           top_features: int = 10) -> plt.Figure:
    """
    Heatmap: rows = tickers, cols = top features, values = mean |SHAP|.
    Lets you instantly see which stock is driven by which signal.
    """
    agg = aggregate_portfolio_shap(portfolio, model_key)
    top_feats = agg["Feature"].head(top_features).tolist()

    tickers, matrix = [], []
    for ticker, data in portfolio.items():
        if "error" in data:
            continue
        shap_vals = data.get(model_key)
        features  = data.get("features", [])
        if shap_vals is None:
            continue
        mean_abs = np.abs(shap_vals).mean(axis=0)
        nc = min(len(mean_abs), len(features))
        fi_map = {features[i]: float(mean_abs[i]) for i in range(nc)}
        row = [fi_map.get(f, 0.0) for f in top_feats]
        tickers.append(ticker)
        matrix.append(row)

    if not matrix:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No data to display", ha="center", va="center",
                transform=ax.transAxes, color="#374151")
        fig.patch.set_facecolor("#ffffff")
        return fig

    mat = np.array(matrix)
    fig, ax = plt.subplots(figsize=(max(8, top_features * 0.9),
                                    max(3, len(tickers) * 0.6)))
    fig.patch.set_facecolor("#ffffff")
    im = ax.imshow(mat, cmap="YlGn", aspect="auto")

    ax.set_xticks(range(len(top_feats)))
    ax.set_xticklabels(top_feats, rotation=35, ha="right",
                       fontsize=9, color="#374151")
    ax.set_yticks(range(len(tickers)))
    ax.set_yticklabels(tickers, fontsize=10, color="#111827")
    ax.set_title("Which signal drives which stock? (mean |SHAP|)",
                 color="#111827", fontsize=13, fontweight="bold")

    for i in range(len(tickers)):
        for j in range(len(top_feats)):
            ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                    fontsize=7.5,
                    color="white" if mat[i, j] > mat.max() * 0.65 else "#374151")

    plt.colorbar(im, ax=ax, label="Mean |SHAP|", fraction=0.02, pad=0.02)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Portfolio diversification score (XAI-based)
# ─────────────────────────────────────────────────────────────────────────────

def portfolio_diversification_score(portfolio: dict,
                                    model_key: str = "shap_rf") -> dict:
    """
    If all stocks rely on the SAME top feature → concentrated risk.
    Score 0–100: higher = more diversified explanation drivers.

    Returns:
        { "score": float, "dominant_feature": str,
          "agreement_pct": float, "interpretation": str }
    """
    top_features = []
    for ticker, data in portfolio.items():
        if "error" in data:
            continue
        shap_vals = data.get(model_key)
        features  = data.get("features", [])
        if shap_vals is None:
            continue
        mean_abs = np.abs(shap_vals).mean(axis=0)
        nc = min(len(mean_abs), len(features))
        if nc == 0:
            continue
        top_features.append(features[np.argmax(mean_abs[:nc])])

    if not top_features:
        return {"score": 0, "dominant_feature": "—",
                "agreement_pct": 0, "interpretation": "No data"}

    from collections import Counter
    counts = Counter(top_features)
    dominant, dom_count = counts.most_common(1)[0]
    agreement_pct = dom_count / len(top_features) * 100
    # score: 100 = all different, 0 = all same
    score = round(100 - agreement_pct, 1)

    if score >= 70:
        interp = "Well diversified — stocks are driven by different signals"
    elif score >= 40:
        interp = "Moderate concentration — some shared signal dependency"
    else:
        interp = f"Concentrated risk — most stocks driven by '{dominant}'"

    return {
        "score":            score,
        "dominant_feature": dominant,
        "agreement_pct":    round(agreement_pct, 1),
        "interpretation":   interp,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Price return correlation heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_return_correlation(portfolio: dict) -> plt.Figure:
    """Correlation of daily returns across all portfolio stocks."""
    ret_dict = {}
    for ticker, data in portfolio.items():
        if "error" in data:
            continue
        df = data.get("df")
        if df is None:
            continue
        ret_dict[ticker] = df["Close"].pct_change().dropna()

    if len(ret_dict) < 2:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "Need ≥ 2 stocks for correlation",
                ha="center", va="center", transform=ax.transAxes, color="#374151")
        fig.patch.set_facecolor("#ffffff")
        return fig

    ret_df = pd.DataFrame(ret_dict).dropna()
    corr   = ret_df.corr()

    fig, ax = plt.subplots(figsize=(max(5, len(ret_dict) * 0.9),
                                    max(4, len(ret_dict) * 0.8)))
    fig.patch.set_facecolor("#ffffff")
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")

    tickers = list(corr.columns)
    ax.set_xticks(range(len(tickers)))
    ax.set_yticks(range(len(tickers)))
    ax.set_xticklabels(tickers, rotation=35, ha="right",
                       fontsize=10, color="#374151")
    ax.set_yticklabels(tickers, fontsize=10, color="#374151")
    ax.set_title("Return correlation across portfolio",
                 color="#111827", fontsize=13, fontweight="bold")

    for i in range(len(tickers)):
        for j in range(len(tickers)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                    fontsize=9,
                    color="white" if abs(corr.values[i,j]) > 0.6 else "#374151")

    plt.colorbar(im, ax=ax, label="Pearson correlation",
                 fraction=0.025, pad=0.02)
    plt.tight_layout()
    return fig
