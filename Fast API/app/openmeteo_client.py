# app/openmeteo_client.py
from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
import math
import os
import requests
import pandas as pd
import numpy as np

# Import expected feature lists from utils (no circular import: utils doesn't import this file)
from app.utils import clf_features, reg_features

SYDNEY = {
    "latitude": -33.8678,
    "longitude": 151.2073,
    "timezone": "Australia/Sydney",
}

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/era5"

# ---- Daily params we can actually request as "daily" ----
DAILY_PARAMS = [
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "apparent_temperature_mean", "apparent_temperature_max", "apparent_temperature_min",
    "sunshine_duration", "precipitation_sum", "rain_sum",
    "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
    "shortwave_radiation_sum", "et0_fao_evapotranspiration",
    "pressure_msl_mean", "surface_pressure_mean",
]

# Map to your column names
DAILY_TO_COL = {
    "temperature_2m_mean": "temperature_2m_mean (°C)",
    "temperature_2m_max": "temperature_2m_max (°C)",
    "temperature_2m_min": "temperature_2m_min (°C)",
    "apparent_temperature_mean": "apparent_temperature_mean (°C)",
    "apparent_temperature_max": "apparent_temperature_max (°C)",
    "apparent_temperature_min": "apparent_temperature_min (°C)",
    "sunshine_duration": "sunshine_duration (s)",
    "precipitation_sum": "precipitation_sum (mm)",
    "rain_sum": "rain_sum (mm)",
    "wind_speed_10m_max": "wind_speed_10m_max (km/h)",
    "wind_gusts_10m_max": "wind_gusts_10m_max (km/h)",
    "wind_direction_10m_dominant": "wind_direction_10m_dominant (°)",
    "shortwave_radiation_sum": "shortwave_radiation_sum (MJ/m²)",
    "et0_fao_evapotranspiration": "et0_fao_evapotranspiration (mm)",
    "pressure_msl_mean": "pressure_msl (hPa)",
    "surface_pressure_mean": "surface_pressure (hPa)",
}

# Hourly params we will average to a daily mean
HOURLY_PARAMS = [
    "relative_humidity_2m", "dew_point_2m",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "temperature_2m",
]

HOURLY_TO_COL = {
    "relative_humidity_2m": "relative_humidity_2m (%)",
    "dew_point_2m": "dew_point_2m (°C)",
    "cloud_cover": "cloud_cover (%)",
    "cloud_cover_low": "cloud_cover_low (%)",
    "cloud_cover_mid": "cloud_cover_mid (%)",
    "cloud_cover_high": "cloud_cover_high (%)",
    "temperature_2m": "temperature_2m (°C)",
}

def _validate_date(date_str: str) -> datetime:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Invalid date format. Use YYYY-MM-DD.")

def _fetch_json(params: dict) -> dict:
    r = requests.get(ARCHIVE_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def _fetch_daily_df(start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude": SYDNEY["latitude"], "longitude": SYDNEY["longitude"],
        "start_date": start_date, "end_date": end_date,
        "daily": ",".join(DAILY_PARAMS),
        "timezone": SYDNEY["timezone"],
        "windspeed_unit": "kmh",
    }
    data = _fetch_json(params)
    if "daily" not in data or len(data["daily"].get("time", [])) == 0:
        raise RuntimeError("No daily data returned by Open-Meteo.")
    return pd.DataFrame(data["daily"])

def _fetch_hourly_df(start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude": SYDNEY["latitude"], "longitude": SYDNEY["longitude"],
        "start_date": start_date, "end_date": end_date,
        "hourly": ",".join(HOURLY_PARAMS),
        "timezone": SYDNEY["timezone"],
    }
    data = _fetch_json(params)
    if "hourly" not in data or len(data["hourly"].get("time", [])) == 0:
        raise RuntimeError("No hourly data returned by Open-Meteo.")
    return pd.DataFrame(data["hourly"])

def _hourly_means_by_day(hourly_df: pd.DataFrame) -> pd.DataFrame:
    df = hourly_df.copy()
    df["date"] = pd.to_datetime(df["time"]).dt.date
    means = df.groupby("date").agg({k: "mean" for k in HOURLY_PARAMS})
    return means

def fetch_features_raw(date_str: str) -> pd.Series:
    """
    Build a *raw* feature row for date t with:
      - daily fields for t
      - hourly means for t
      - lag-1 daily/hourly derived fields from t-1
      - engineered features used in notebooks (humidity_temp, is_weekend, cyclic)
    Returns a pandas Series (one row of features).
    """
    t = _validate_date(date_str)
    t_str = t.strftime("%Y-%m-%d")
    t_minus_1 = (t - timedelta(days=1)).strftime("%Y-%m-%d")

    # Daily for t-1..t
    daily = _fetch_daily_df(t_minus_1, t_str)
    if len(daily) < 2:
        raise RuntimeError("Daily API did not return both t-1 and t.")
    daily_t   = daily.iloc[1]
    daily_t_1 = daily.iloc[0]

    # Start with daily-mapped fields for t
    row = {}
    for k, out_col in DAILY_TO_COL.items():
        row[out_col] = daily_t.get(k, None)

    # Hourly means for t-1..t
    hourly = _fetch_hourly_df(t_minus_1, t_str)
    means = _hourly_means_by_day(hourly)

    # Map hourly means for t and t-1
    d_t   = means.loc[pd.to_datetime(t_str).date()].to_dict()
    d_t_1 = means.loc[pd.to_datetime(t_minus_1).date()].to_dict()

    for k, out_col in HOURLY_TO_COL.items():
        row[out_col] = float(d_t.get(k, np.nan))
        # store lag-1 also (we may or may not need them, keep raw for alignment)
        row[f"{out_col}_lag1_raw"] = float(d_t_1.get(k, np.nan))

    # Add calendar + cyclic
    row["year"]  = t.year
    row["month"] = t.month
    row["is_weekend"] = 1 if t.weekday() >= 5 else 0
    dayofyear = int(t.strftime("%j"))
    row["dayofyear"]     = dayofyear
    row["dayofyear_sin"] = math.sin(2 * math.pi * dayofyear / 365.0)
    row["dayofyear_cos"] = math.cos(2 * math.pi * dayofyear / 365.0)

    # Engineered features used in notebooks (derive from raw columns)
    # Lags (examples used earlier)
    row["humidity_lag1"]     = row.get("relative_humidity_2m (%)_lag1_raw")
    row["pressure_msl_lag1"] = daily_t_1.get("pressure_msl_mean", np.nan)  # daily lag from t-1

    # Interaction
    rh  = row.get("relative_humidity_2m (%)", np.nan)
    t2m = row.get("temperature_2m (°C)", np.nan)
    row["humidity_temp"] = (rh * t2m) if (pd.notna(rh) and pd.notna(t2m)) else np.nan

    # Return as Series
    return pd.Series(row, name=t_str)

def _align_to_expected(raw_row: pd.Series, expected_cols: list[str]) -> pd.DataFrame:
    """
    Keep exactly expected columns; add missing with NaN; drop extras.
    Returns a 1-row DataFrame.
    """
    df = pd.DataFrame([raw_row])
    out = pd.DataFrame(index=df.index)
    for c in expected_cols:
        out[c] = df[c] if c in df.columns else np.nan
    # final cleaning
    out = out.replace([np.inf, -np.inf], np.nan)
    return out

def features_for_classification(date_str: str) -> pd.DataFrame:
    """
    Build a 1-row DataFrame aligned to the classifier's expected features (from joblib list).
    """
    if clf_features is None:
        raise RuntimeError("Classifier feature list is not loaded; check CLS_FEATS_PATH in utils.")
    raw = fetch_features_raw(date_str)
    X = _align_to_expected(raw, clf_features)
    return X

def features_for_regression(date_str: str) -> pd.DataFrame:
    """
    Build a 1-row DataFrame aligned to the regressor's expected features (from joblib list).
    """
    if reg_features is None:
        raise RuntimeError("Regressor feature list is not loaded; check REG_FEATS_PATH in utils.")
    raw = fetch_features_raw(date_str)
    X = _align_to_expected(raw, reg_features)
    return X
