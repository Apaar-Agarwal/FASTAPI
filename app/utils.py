# app/utils.py
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import os

BASE = Path(__file__).resolve().parent.parent  # repo root (one level above app/)

# --------------------------
# Model paths
# --------------------------
CLS_MODEL_PATH = Path(os.getenv(
    "CLS_MODEL_PATH",
    str(BASE / "models" / "Randomforest_regression_rain.pkl")
))
REG_MODEL_PATH = Path(os.getenv(
    "REG_MODEL_PATH",
    str(BASE / "models" / "xgboost_regression_precip.pkl")
))

# --------------------------
# Feature list paths
# --------------------------
CLS_FEATS_PATH = Path(os.getenv(
    "CLS_FEATS_PATH",
    str(BASE / "models" / "feature_columns_classification.joblib")
))
REG_FEATS_PATH = Path(os.getenv(
    "REG_FEATS_PATH",
    str(BASE / "models" / "feature_columns_regression.joblib")
))

def _to_path(p) -> Path:
    """Convert str/Path-like to Path safely."""
    return p if isinstance(p, Path) else Path(p)

def _load_or_none(p: Path):
    """Load a joblib artifact if it exists; otherwise return None and log."""
    p = _to_path(p)
    if not p.exists():
        print(f"[utils] WARNING: artifact not found: {p}")
        return None
    try:
        print(f"[utils] Loading artifact: {p}")
        return joblib.load(p)
    except Exception as e:
        print(f"[utils] ERROR loading {p}: {e}")
        return None

# --------------------------
# Load models + feature lists
# --------------------------
clf_model     = _load_or_none(CLS_MODEL_PATH)
reg_model     = _load_or_none(REG_MODEL_PATH)
clf_features  = _load_or_none(CLS_FEATS_PATH)  # list[str]
reg_features  = _load_or_none(REG_FEATS_PATH)  # list[str]

if clf_features:
    print(f"[utils] Classifier feature list loaded: {len(clf_features)} features")
if reg_features:
    print(f"[utils] Regressor feature list loaded: {len(reg_features)} features")

def _align(df: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    """
    Reindex input to match training feature order exactly.
    - Keeps only expected columns.
    - Adds any missing as NaN.
    """
    out = pd.DataFrame(index=df.index)
    for c in expected_cols:
        out[c] = df[c] if c in df.columns else np.nan
    return out[expected_cols]

def classify_rain(features_df: pd.DataFrame, threshold: float = 0.5) -> dict:
    """
    features_df: single-row DataFrame with the SAME features used in training.
    """
    if clf_model is None:
        raise RuntimeError("Classification model artifact not found. Check CLS_MODEL_PATH.")
    if clf_features is None:
        raise RuntimeError("Classifier feature list not found. Check CLS_FEATS_PATH.")

    X = _align(features_df, clf_features)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if hasattr(clf_model, "predict_proba"):
        prob = float(clf_model.predict_proba(X)[:, 1][0])
        label = int(prob >= threshold)
    else:
        label = int(clf_model.predict(X)[0])
        prob = float(label)

    return {"will_rain": bool(label), "probability": prob, "threshold": threshold}

def regress_precip(features_df: pd.DataFrame) -> dict:
    """
    features_df: single-row DataFrame with the SAME features used in training.
    """
    if reg_model is None:
        raise RuntimeError("Regression model artifact not found. Check REG_MODEL_PATH.")
    if reg_features is None:
        raise RuntimeError("Regressor feature list not found. Check REG_FEATS_PATH.")

    X = _align(features_df, reg_features)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    yhat = float(reg_model.predict(X)[0])
    return {"precipitation_fall": max(0.0, yhat)}  # clamp negatives

