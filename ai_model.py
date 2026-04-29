# ai_model.py  — REPLACE ENTIRELY WITH THIS

import numpy as np
import joblib
import tensorflow as tf
import os

# ── Load models once at startup ──────────────────────────────────────
_BASE = os.path.dirname(__file__)

_cnn = tf.keras.models.load_model(os.path.join(_BASE, "best_model.h5"))
_xgb = joblib.load(os.path.join(_BASE, "model.pkl"))

_LABELS = ["normal", "moderate", "severe"]   # adjust if your labels differ

def _build_features(data: dict) -> np.ndarray:
    """
    Build the feature vector your models were trained on.
    Order MUST match your training CSV columns.
    """
    temp      = float(data.get("temp", 0))
    humidity  = float(data.get("humidity", 0))
    lux       = float(data.get("lux", 0))
    eye_temp  = float(data.get("eye_temp", 0))
    blink     = float(data.get("blink_rate", 0))
    temp_diff = temp - eye_temp
    blink_norm = blink / 60.0
    # Matches DED_Final_Sensor_Fusion.csv column order:
    return np.array([[temp, humidity, lux, eye_temp, blink, temp_diff, blink_norm]])


def predict(data: dict) -> dict:
    features = _build_features(data)

    # ── XGBoost prediction ──────────────────────────────────────────
    xgb_pred_idx  = int(_xgb.predict(features)[0])
    xgb_proba     = _xgb.predict_proba(features)[0]
    xgb_label     = _LABELS[xgb_pred_idx]
    xgb_confidence = float(xgb_proba[xgb_pred_idx])

    # ── CNN prediction ──────────────────────────────────────────────
    # CNN expects shape (samples, timesteps, features) — adjust if needed
    cnn_input = features.reshape(1, 1, features.shape[1])
    cnn_proba = _cnn.predict(cnn_input, verbose=0)[0]
    cnn_pred_idx = int(np.argmax(cnn_proba))
    cnn_label = _LABELS[cnn_pred_idx]
    cnn_confidence = float(cnn_proba[cnn_pred_idx])

    # ── Ensemble: average probabilities ─────────────────────────────
    ensemble_proba = (xgb_proba + cnn_proba) / 2
    final_idx = int(np.argmax(ensemble_proba))
    final_label = _LABELS[final_idx]
    final_confidence = float(ensemble_proba[final_idx])

    return {
        "prediction":  final_label,
        "confidence":  round(final_confidence, 4),
        "cnn_prediction":  cnn_label,
        "cnn_confidence":  round(cnn_confidence, 4),
        "xgb_prediction":  xgb_label,
        "xgb_confidence":  round(xgb_confidence, 4),
    }
