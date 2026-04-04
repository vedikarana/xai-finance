"""
Data Pipeline — XAI Finance v2
Fetches stock data + news sentiment (FinBERT) and engineers 26 features.
"""

import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend    import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume   import OnBalanceVolumeIndicator
warnings.filterwarnings("ignore")


def fetch_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    for attempt in range(3):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, auto_adjust=True)
            if df is not None and not df.empty and len(df) > 50:
                df.dropna(inplace=True)
                df.index = pd.to_datetime(df.index).tz_localize(None)
                return df
            df = yf.download(ticker, period=period, auto_adjust=True,
                             progress=False, show_errors=False)
            if df is not None and not df.empty and len(df) > 50:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.dropna(inplace=True)
                df.index = pd.to_datetime(df.index).tz_localize(None)
                return df
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
                continue
            raise ValueError(f"Could not fetch '{ticker}': {e}")
    raise ValueError(f"No data returned for '{ticker}'.")


def fetch_news_sentiment(ticker: str, df_index: pd.DatetimeIndex) -> pd.Series:
    """
    Score recent headlines with FinBERT. Returns daily sentiment [-1,+1].
    Falls back to 0.0 (neutral) if FinBERT / network unavailable.
    """
    sentiment_series = pd.Series(0.0, index=df_index, name="Sentiment_Score")
    try:
        from transformers import pipeline as hf_pipeline
        pipe = hf_pipeline("text-classification", model="ProsusAI/finbert",
                           truncation=True, max_length=512)
        stock = yf.Ticker(ticker)
        news  = getattr(stock, "news", []) or []
        if not news:
            return sentiment_series
        label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
        records = []
        for item in news[:50]:
            title = item.get("title", "")
            ts    = item.get("providerPublishTime", None)
            if not title or not ts:
                continue
            try:
                r = pipe(title)[0]
                score = label_map.get(r["label"].lower(), 0.0) * r["score"]
                date  = pd.to_datetime(ts, unit="s").normalize()
                records.append({"date": date, "score": score})
            except Exception:
                continue
        if records:
            news_df   = pd.DataFrame(records).groupby("date")["score"].mean()
            full_idx  = pd.date_range(df_index.min(), df_index.max(), freq="D")
            smoothed  = news_df.reindex(full_idx).ffill().fillna(0.0).rolling(7, min_periods=1).mean()
            for d in df_index:
                dn = pd.Timestamp(d).normalize()
                if dn in smoothed.index:
                    sentiment_series[d] = float(smoothed[dn])
    except Exception:
        pass
    return sentiment_series


def engineer_features(df: pd.DataFrame,
                      sentiment_series: pd.Series = None) -> pd.DataFrame:
    df = df.copy()
    df["Return_1d"]    = df["Close"].pct_change(1)
    df["Return_5d"]    = df["Close"].pct_change(5)
    df["Return_10d"]   = df["Close"].pct_change(10)
    df["Log_Return"]   = np.log(df["Close"] / df["Close"].shift(1))
    df["RSI"]          = RSIIndicator(close=df["Close"], window=14).rsi()
    stoch              = StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"], window=14)
    df["Stoch_K"]      = stoch.stoch()
    df["Stoch_D"]      = stoch.stoch_signal()
    macd               = MACD(close=df["Close"])
    df["MACD"]         = macd.macd()
    df["MACD_Signal"]  = macd.macd_signal()
    df["MACD_Hist"]    = macd.macd_diff()
    df["EMA_12"]       = EMAIndicator(close=df["Close"], window=12).ema_indicator()
    df["EMA_26"]       = EMAIndicator(close=df["Close"], window=26).ema_indicator()
    df["SMA_20"]       = SMAIndicator(close=df["Close"], window=20).sma_indicator()
    df["SMA_50"]       = SMAIndicator(close=df["Close"], window=50).sma_indicator()
    df["EMA_Cross"]    = df["EMA_12"] - df["EMA_26"]
    bb                 = BollingerBands(close=df["Close"], window=20)
    df["BB_High"]      = bb.bollinger_hband()
    df["BB_Low"]       = bb.bollinger_lband()
    df["BB_Width"]     = bb.bollinger_wband()
    df["ATR"]          = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14).average_true_range()
    df["OBV"]          = OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"]).on_balance_volume()
    df["Volume_MA"]    = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / (df["Volume_MA"] + 1e-9)
    for lag in [1, 2, 3, 5]:
        df[f"Close_Lag_{lag}"] = df["Close"].shift(lag)
    # Feature 26
    if sentiment_series is not None and len(sentiment_series) > 0:
        df["Sentiment_Score"] = sentiment_series.reindex(df.index).fillna(0.0)
    else:
        df["Sentiment_Score"] = 0.0
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)
    return df


def get_feature_columns() -> list:
    return [
        "Return_1d","Return_5d","Return_10d","Log_Return",
        "RSI","Stoch_K","Stoch_D",
        "MACD","MACD_Signal","MACD_Hist",
        "EMA_12","EMA_26","SMA_20","SMA_50","EMA_Cross",
        "BB_High","BB_Low","BB_Width",
        "ATR","OBV","Volume_MA","Volume_Ratio",
        "Close_Lag_1","Close_Lag_2","Close_Lag_3","Close_Lag_5",
        "Sentiment_Score",
    ]


def prepare_train_test(df: pd.DataFrame, test_size: float = 0.2) -> tuple:
    from sklearn.preprocessing import StandardScaler
    features = [f for f in get_feature_columns() if f in df.columns]
    X        = df[features].values
    y        = df["Target"].values
    split    = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    scaler   = StandardScaler()
    X_train  = scaler.fit_transform(X_train)
    X_test   = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler, features
