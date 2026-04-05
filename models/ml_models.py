"""
ML Models for XAI Finance Project
Includes: Random Forest, XGBoost, LSTM
Compatible with TensorFlow 2.x + Keras 2 and Keras 3
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score,
)
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Keras import helper — works with TF2+Keras2 and Keras3
# ─────────────────────────────────────────────────────────────────────────────

def _get_keras():
    """Returns keras — gracefully disabled if not installed."""
    try:
        import keras
        return keras
    except ImportError:
        pass
    try:
        import tensorflow as tf
        return tf.keras
    except ImportError:
        return None  # LSTM disabled — tensorflow not installed


# ─────────────────────────────────────────────────────────────────────────────
# Random Forest
# ─────────────────────────────────────────────────────────────────────────────

def train_random_forest(X_train, y_train, n_estimators=200, max_depth=10):
    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=42, n_jobs=-1, class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    return rf


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost
# ─────────────────────────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train, n_estimators=200, max_depth=6, lr=0.05):
    xgb_model = xgb.XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=lr, eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)
    return xgb_model


# ─────────────────────────────────────────────────────────────────────────────
# LSTM
# ─────────────────────────────────────────────────────────────────────────────

def prepare_lstm_data(X_train, X_test, y_train, y_test, timesteps=10):
    def make_seq(X, y, ts):
        Xs, ys = [], []
        for i in range(ts, len(X)):
            Xs.append(X[i - ts: i])
            ys.append(y[i])
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)
    Xtr, ytr = make_seq(X_train, y_train, timesteps)
    Xte, yte = make_seq(X_test,  y_test,  timesteps)
    return Xtr, Xte, ytr, yte


def build_lstm(input_shape):
    keras = _get_keras()
    if keras is None:
        return None
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(32, return_sequences=False),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1,  activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_lstm(X_train, y_train, X_test, y_test, timesteps=10, epochs=30):
    keras = _get_keras()
    if keras is None:
        # TensorFlow not installed — return dummy so app doesnt crash
        Xtr, Xte, ytr, yte = prepare_lstm_data(X_train, X_test, y_train, y_test, timesteps)
        return None, None, Xte, yte

    Xtr, Xte, ytr, yte = prepare_lstm_data(X_train, X_test, y_train, y_test, timesteps)
    model = build_lstm(input_shape=(timesteps, X_train.shape[1]))

    cb = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=0
    )
    try:
        model.fit(Xtr, ytr, validation_data=(Xte, yte),
                  epochs=epochs, batch_size=32, callbacks=[cb], verbose=0)
    except Exception as e:
        warnings.warn(f"LSTM training error ({e}), retrying simpler model...")
        model = keras.Sequential([
            keras.layers.Input(shape=(timesteps, X_train.shape[1])),
            keras.layers.LSTM(32),
            keras.layers.Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(Xtr, ytr, epochs=10, batch_size=32, verbose=0)

    return model, None, Xte, yte


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_type="sklearn"):
    if model is None:
        raise ValueError("Model is None — training failed.")

    y_test = np.array(y_test).astype(int)

    if model_type == "lstm":
        try:
            raw    = model.predict(X_test, verbose=0)
        except Exception:
            raw    = model(X_test, training=False)
        y_prob = np.array(raw).ravel().astype(float)
        y_pred = (y_prob > 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
        "auc":       round(roc_auc_score(y_test, y_prob) * 100, 2),
        "report":    classification_report(y_test, y_pred, output_dict=True),
        "confusion": confusion_matrix(y_test, y_pred).tolist(),
        "y_pred":    y_pred,
        "y_prob":    y_prob,
    }
