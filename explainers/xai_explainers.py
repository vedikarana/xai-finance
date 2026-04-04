"""
XAI Explainers for Finance Project
Implements: SHAP, LIME, Feature Importance
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# SHAP Explanations
# ─────────────────────────────────────────────────────────────────────────────

def _fix_xgb_base_score(model):
    """
    Fix for SHAP + XGBoost version mismatch:
    XGBoost >=2.0 stores base_score as a string like '[5.3932583E-1]'
    which older SHAP cannot convert to float.
    We patch it to 0.5 (neutral) before passing to TreeExplainer.
    """
    import xgboost as xgb
    try:
        config = model.get_booster().save_config()
        import json
        cfg = json.loads(config)
        bs  = cfg["learner"]["learner_model_param"]["base_score"]
        # strip brackets if present e.g. '[5.39E-1]' -> 0.5
        bs_clean = str(bs).strip("[]")
        float(bs_clean)   # test if it's already valid
    except Exception:
        # patch: set base_score to plain 0.5
        model.get_booster().set_param("base_score", 0.5)
    return model


def compute_shap_values(model, X_train, X_test, model_type="tree"):
    """
    Compute SHAP values for a trained model.

    Returns:
        (explainer, shap_values, X_explained)
        X_explained is the slice of X_test that shap_values correspond to.
        Always use X_explained (not raw X_test) when plotting.
    """
    if model_type == "tree":
        import xgboost as xgb
        if isinstance(model, xgb.XGBClassifier):
            model = _fix_xgb_base_score(model)

        try:
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            X_explained = X_test
        except Exception:
            background  = shap.sample(X_train, 50)
            explainer   = shap.KernelExplainer(model.predict_proba, background)
            X_explained = X_test[:100]
            shap_values = explainer.shap_values(X_explained)[1]
            # col safety
            n = min(shap_values.shape[1], X_explained.shape[1])
            return explainer, shap_values[:, :n], X_explained[:, :n]

        # For binary classification RF returns list; take class-1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        # New SHAP versions return 3-D array for classifiers
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
    else:
        background  = shap.sample(X_train, 100)
        explainer   = shap.KernelExplainer(model.predict_proba, background)
        X_explained = X_test[:50]
        shap_values = explainer.shap_values(X_explained)[1]
        n = min(shap_values.shape[1], X_explained.shape[1])
        return explainer, shap_values[:, :n], X_explained[:, :n]

    # col + row safety: trim to matching size
    n_rows = min(len(shap_values), len(X_explained))
    n_cols = min(shap_values.shape[1], X_explained.shape[1])
    shap_values = shap_values[:n_rows, :n_cols]
    X_explained = X_explained[:n_rows, :n_cols]

    return explainer, shap_values, X_explained


def plot_shap_summary(shap_values, X_test, feature_names, max_display=15):
    """Return a SHAP beeswarm summary plot as a matplotlib Figure."""
    # enforce rows AND cols match
    n_rows = min(len(shap_values), len(X_test))
    n_cols = min(shap_values.shape[1], X_test.shape[1], len(feature_names))
    sv = shap_values[:n_rows, :n_cols]
    Xt = X_test[:n_rows, :n_cols]
    fn = list(feature_names[:n_cols])
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(sv, Xt, feature_names=fn, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.patch.set_facecolor("#ffffff")
    plt.tight_layout()
    return fig


def plot_shap_bar(shap_values, feature_names, top_n=15):
    """Return a SHAP mean absolute bar chart as a matplotlib Figure."""
    n_cols   = min(shap_values.shape[1], len(feature_names))
    shap_values = shap_values[:, :n_cols]
    feature_names = feature_names[:n_cols]
    mean_abs = np.abs(shap_values).mean(axis=0)
    indices  = np.argsort(mean_abs)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f7f8fa")

    bars = ax.barh(
        [feature_names[i] for i in indices],
        mean_abs[indices],
        color="#00d4ff",
        alpha=0.85,
    )
    ax.set_xlabel("Mean |SHAP Value|", color="#111827", fontsize=12)
    ax.set_title("Feature Importance via SHAP", color="#111827", fontsize=14, fontweight="bold")
    ax.tick_params(colors="#374151")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    plt.tight_layout()
    return fig


def plot_shap_waterfall(shap_values, X_test, feature_names, idx=0):
    """Waterfall plot for a single prediction."""
    n_rows = min(len(shap_values), len(X_test))
    n_cols = min(shap_values.shape[1], X_test.shape[1], len(feature_names))
    shap_values   = shap_values[:n_rows, :n_cols]
    X_test        = X_test[:n_rows, :n_cols]
    feature_names = list(feature_names[:n_cols])
    idx  = min(idx, n_rows - 1)
    vals = shap_values[idx]
    sorted_idx = np.argsort(np.abs(vals))[-12:]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f7f8fa")

    colors = ["#ff4d4d" if v < 0 else "#00d4ff" for v in vals[sorted_idx]]
    ax.barh([feature_names[i] for i in sorted_idx], vals[sorted_idx], color=colors, alpha=0.85)
    ax.set_xlabel("SHAP Value (impact on prediction)", color="#111827", fontsize=11)
    ax.set_title(f"SHAP Waterfall — Sample #{idx}", color="#111827", fontsize=14, fontweight="bold")
    ax.tick_params(colors="#374151")
    ax.axvline(0, color="#111827", linewidth=0.8)

    red_patch  = mpatches.Patch(color="#ff4d4d", label="Pushes prediction DOWN (Sell)")
    blue_patch = mpatches.Patch(color="#00d4ff", label="Pushes prediction UP (Buy)")
    ax.legend(handles=[red_patch, blue_patch], facecolor="#f7f8fa", labelcolor="#111827")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# LIME Explanations
# ─────────────────────────────────────────────────────────────────────────────

def explain_with_lime(model, X_train, X_test, feature_names, idx=0, num_features=12):
    """
    Generate a LIME explanation for a single prediction.
    
    Returns:
        lime_exp object and a matplotlib Figure
    """
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=["DOWN (0)", "UP (1)"],
        mode="classification",
        discretize_continuous=True,
    )
    lime_exp = explainer.explain_instance(
        X_test[idx],
        model.predict_proba,
        num_features=num_features,
        top_labels=2,          # always request both labels so one is always available
    )

    # Use whichever label LIME actually returned (0 or 1)
    available_labels = list(lime_exp.local_exp.keys())
    use_label = 1 if 1 in available_labels else available_labels[0]

    # Build Figure manually (dark theme)
    exp_list = lime_exp.as_list(label=use_label)
    labels   = [e[0] for e in exp_list]
    values   = [e[1] for e in exp_list]
    colors   = ["#ff4d4d" if v < 0 else "#00d4ff" for v in values]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f7f8fa")
    ax.barh(labels[::-1], values[::-1], color=colors[::-1], alpha=0.85)
    ax.set_xlabel("LIME Weight", color="#111827", fontsize=11)
    ax.set_title(f"LIME Explanation — Sample #{idx}", color="#111827", fontsize=14, fontweight="bold")
    ax.tick_params(colors="#374151")
    ax.axvline(0, color="#111827", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    plt.tight_layout()
    return lime_exp, fig


# ─────────────────────────────────────────────────────────────────────────────
# Built-in Feature Importance
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names, top_n=15, model_name="Model"):
    """
    Plot built-in feature importances (RF / XGBoost).
    
    Returns a matplotlib Figure.
    """
    importances = model.feature_importances_
    indices     = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f7f8fa")

    gradient_colors = plt.cm.plasma(np.linspace(0.3, 0.9, top_n))
    ax.barh(
        [feature_names[i] for i in indices],
        importances[indices],
        color=gradient_colors,
        alpha=0.9,
    )
    ax.set_xlabel("Importance Score", color="#111827", fontsize=12)
    ax.set_title(f"{model_name} — Built-in Feature Importance",
                 color="#111827", fontsize=14, fontweight="bold")
    ax.tick_params(colors="#374151")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    plt.tight_layout()
    return fig


def get_top_features_df(model, feature_names, top_n=10):
    """Return a DataFrame of top features and their importance scores."""
    importances = model.feature_importances_
    df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).head(top_n).reset_index(drop=True)
    df["Importance"] = df["Importance"].round(4)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Counterfactual Explanations
# ─────────────────────────────────────────────────────────────────────────────

def compute_counterfactuals(model, X_test, feature_names, idx=0, n_steps=200):
    """
    Find the minimum feature changes needed to flip the model's prediction.
    Uses gradient-free perturbation search.

    Returns a DataFrame with columns:
        Feature | Current Value | Counterfactual Value | Delta | Direction
    """
    sample   = X_test[idx].copy()
    original = model.predict_proba(sample.reshape(1, -1))[0]
    orig_cls = int(np.argmax(original))
    target   = 1 - orig_cls          # flip class

    best_cf  = None
    best_dist = np.inf

    rng = np.random.default_rng(42)
    for _ in range(n_steps):
        perturbed = sample.copy()
        # perturb a random subset of features
        n_perturb = rng.integers(1, max(2, len(feature_names) // 3))
        feat_idx  = rng.choice(len(feature_names), size=n_perturb, replace=False)
        for fi in feat_idx:
            perturbed[fi] += rng.normal(0, 0.5)
        pred_cls = int(np.argmax(model.predict_proba(perturbed.reshape(1, -1))[0]))
        if pred_cls == target:
            dist = np.linalg.norm(perturbed - sample)
            if dist < best_dist:
                best_dist = dist
                best_cf   = perturbed.copy()

    if best_cf is None:
        # no flip found — return empty
        return pd.DataFrame(columns=["Feature","Current Value","Counterfactual Value","Change","Signal"])

    rows = []
    for i, fname in enumerate(feature_names):
        delta = best_cf[i] - sample[i]
        if abs(delta) > 0.01:           # only show meaningful changes
            rows.append({
                "Feature":               fname,
                "Current Value":         round(float(sample[i]), 4),
                "Counterfactual Value":  round(float(best_cf[i]), 4),
                "Change":                round(float(delta), 4),
                "Signal":                "🟢 Increase" if delta > 0 else "🔴 Decrease",
            })
    df_cf = pd.DataFrame(rows).sort_values("Change", key=abs, ascending=False).head(8)
    return df_cf


def plot_counterfactual(cf_df, orig_pred_label, cf_pred_label):
    """Bar chart of feature deltas needed to flip the prediction."""
    if cf_df.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No counterfactual found (prediction already uncertain)",
                ha="center", va="center", transform=ax.transAxes, color="#374151")
        fig.patch.set_facecolor("#ffffff")
        return fig

    colors = ["#00916e" if v > 0 else "#dc2626" for v in cf_df["Change"]]
    fig, ax = plt.subplots(figsize=(10, max(4, len(cf_df) * 0.55)))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f7f8fa")
    ax.barh(cf_df["Feature"], cf_df["Change"], color=colors, alpha=0.85)
    ax.axvline(0, color="#9ca3af", linewidth=1)
    ax.set_xlabel("Feature Change Needed", color="#374151", fontsize=11)
    ax.set_title(
        f"Counterfactual: To flip  {orig_pred_label} → {cf_pred_label}\n"
        f"these signals would need to change:",
        color="#111827", fontsize=13, fontweight="bold"
    )
    ax.tick_params(colors="#374151")
    for spine in ax.spines.values():
        spine.set_edgecolor("#e8eaf0")
    green_p = mpatches.Patch(color="#00916e", label="Needs to increase")
    red_p   = mpatches.Patch(color="#dc2626", label="Needs to decrease")
    ax.legend(handles=[green_p, red_p], facecolor="#f7f8fa", labelcolor="#374151")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# LSTM Attention Map
# ─────────────────────────────────────────────────────────────────────────────

def compute_lstm_attention(lstm_model, X_lstm, idx=0):
    """
    Approximate attention weights for the LSTM by measuring the gradient
    of the output w.r.t. each timestep's input (gradient * input saliency).
    Works even without a formal Attention layer.

    Returns:
        attention_weights: np.array of shape (timesteps,)
    """
    try:
        import tensorflow as tf
        sample = X_lstm[idx:idx+1]          # (1, timesteps, features)
        sample_tensor = tf.constant(sample, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(sample_tensor)
            pred = lstm_model(sample_tensor, training=False)
        grads = tape.gradient(pred, sample_tensor)   # (1, timesteps, features)
        if grads is None:
            return np.ones(sample.shape[1]) / sample.shape[1]
        # saliency = mean(|grad * input|) across feature dimension per timestep
        saliency = np.abs(grads.numpy()[0] * sample[0]).mean(axis=1)
        saliency = saliency / (saliency.sum() + 1e-9)
        return saliency
    except Exception:
        ts = X_lstm.shape[1] if X_lstm.ndim == 3 else 10
        return np.ones(ts) / ts


def plot_lstm_attention(attention_weights):
    """Heatmap of LSTM attention across the 10-day lookback window."""
    ts = len(attention_weights)
    labels = [f"Day –{ts - i}" for i in range(ts)]

    fig, ax = plt.subplots(figsize=(10, 2.8))
    fig.patch.set_facecolor("#ffffff")
    im = ax.imshow(
        attention_weights.reshape(1, -1),
        cmap="YlOrRd", aspect="auto", vmin=0, vmax=attention_weights.max()
    )
    ax.set_xticks(range(ts))
    ax.set_xticklabels(labels, fontsize=10, color="#374151")
    ax.set_yticks([])
    ax.set_title(
        "LSTM Attention Map — Which past trading days influenced today's prediction most?",
        color="#111827", fontsize=13, fontweight="bold"
    )

    # annotate each cell
    for i, w in enumerate(attention_weights):
        ax.text(i, 0, f"{w:.2%}", ha="center", va="center",
                color="white" if w > attention_weights.max() * 0.6 else "#374151",
                fontsize=9.5, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Attention Weight", fraction=0.02, pad=0.02)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Backtesting Engine
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(model, X_test, df_test: pd.DataFrame,
                 confidence_threshold: float = 0.55) -> dict:
    """
    Simple long-only backtest.
    BUY when model confidence > threshold, hold cash otherwise.

    Returns dict with:
        equity_curve, trade_log, metrics (Sharpe, MaxDrawdown, WinRate, CAGR)
    """
    probs      = model.predict_proba(X_test)[:, 1]   # P(UP)
    close      = df_test["Close"].values[-len(X_test):]
    dates      = df_test.index[-len(X_test):]

    capital    = 1.0
    equity     = [capital]
    trades     = []
    wins       = 0
    total      = 0

    for i in range(len(probs) - 1):
        ret_next = (close[i+1] - close[i]) / close[i]
        if probs[i] >= confidence_threshold:          # model says BUY
            capital *= (1 + ret_next)
            total   += 1
            if ret_next > 0:
                wins += 1
            trades.append({
                "Date":       dates[i].strftime("%Y-%m-%d"),
                "Action":     "BUY",
                "Confidence": f"{probs[i]*100:.1f}%",
                "Return":     f"{ret_next*100:+.2f}%",
                "Signal":     "🟢 Win" if ret_next > 0 else "🔴 Loss",
            })
        equity.append(capital)

    equity_arr = np.array(equity)
    daily_ret  = np.diff(equity_arr) / (equity_arr[:-1] + 1e-9)

    # Sharpe ratio (annualised, risk-free = 0)
    sharpe  = (daily_ret.mean() / (daily_ret.std() + 1e-9)) * np.sqrt(252)

    # Max drawdown
    peak    = np.maximum.accumulate(equity_arr)
    dd      = (equity_arr - peak) / (peak + 1e-9)
    max_dd  = dd.min() * 100

    # CAGR
    n_years = len(equity_arr) / 252
    cagr    = ((equity_arr[-1]) ** (1 / max(n_years, 0.01)) - 1) * 100

    win_rate = (wins / max(total, 1)) * 100
    total_return = (equity_arr[-1] - 1) * 100

    metrics = {
        "Total Return":    f"{total_return:+.2f}%",
        "CAGR":            f"{cagr:+.2f}%",
        "Sharpe Ratio":    f"{sharpe:.3f}",
        "Max Drawdown":    f"{max_dd:.2f}%",
        "Win Rate":        f"{win_rate:.1f}%",
        "Total Trades":    str(total),
    }

    equity_df = pd.DataFrame({
        "Date":     dates[:len(equity_arr)],
        "Equity":   equity_arr,
    }).set_index("Date")

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["Date","Action","Confidence","Return","Signal"])

    return {"equity_curve": equity_df, "trades": trades_df, "metrics": metrics}


def plot_equity_curve(equity_df: pd.DataFrame, ticker: str, currency: str) -> "plt.Figure":
    """Line chart of portfolio equity over time vs buy-and-hold."""
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")

    ax.plot(equity_df.index, equity_df["Equity"],
            color="#00b386", linewidth=2.2, label="AI Strategy")
    ax.axhline(1.0, color="#9ca3af", linewidth=1, linestyle="--", label="Starting Capital (1.0)")

    # shade profit/loss areas
    ax.fill_between(equity_df.index, 1.0, equity_df["Equity"],
                    where=equity_df["Equity"] >= 1.0,
                    alpha=0.12, color="#00b386")
    ax.fill_between(equity_df.index, 1.0, equity_df["Equity"],
                    where=equity_df["Equity"] < 1.0,
                    alpha=0.12, color="#ef4444")

    ax.set_title(f"📊 Backtest Equity Curve — {ticker}", fontsize=14,
                 fontweight="bold", color="#111827")
    ax.set_ylabel("Portfolio Value (started at 1.0)", color="#374151")
    ax.tick_params(colors="#374151")
    ax.legend(facecolor="#f7f8fa", labelcolor="#374151")
    for spine in ax.spines.values():
        spine.set_edgecolor("#e8eaf0")
    plt.tight_layout()
    return fig
