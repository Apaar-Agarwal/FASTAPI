# app/main.py
from __future__ import annotations
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta

# ⬇️ NEW: import the aligned feature builders
from app.openmeteo_client import (
    features_for_classification,
    features_for_regression,
)

from app.utils import classify_rain, regress_precip, clf_model, reg_model

GITHUB_LINK = "https://github.com/Apaar-Agarwal/FASTAPI.git"  

app = FastAPI(
    title="Open Meteo — ML Forecast API",
    description="Rain-in-7d classification and 3-day precipitation regression for Sydney.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "project": "Open Meteo — ML Forecast API (Sydney)",
        "objectives": [
            "Predict if it will rain exactly 7 days from a given date (binary).",
            "Predict cumulative precipitation in the next 3 days (mm)."
        ],
        "endpoints": {
            "/": "GET — this info",
            "/health/": "GET — service health",
            "/predict/rain/": "GET — params: date=YYYY-MM-DD; returns {input_date, prediction: {date, will_rain}}",
            "/predict/precipitation/fall/": "GET — params: date=YYYY-MM-DD; returns {input_date, prediction: {start_date, end_date, precipitation_fall}}"
        },
        "expected_inputs": {"date": "YYYY-MM-DD (Sydney local time)"},
        "outputs": {
            "/predict/rain/": {
                "input_date": "YYYY-MM-DD",
                "prediction": {"date": "YYYY-MM-DD", "will_rain": True}
            },
            "/predict/precipitation/fall/": {
                "input_date": "YYYY-MM-DD",
                "prediction": {
                    "start_date": "YYYY-MM-DD",
                    "end_date": "YYYY-MM-DD",
                    "precipitation_fall": "float mm"
                }
            }
        },
        "github": GITHUB_LINK or "see github.txt at repo root"
    }

@app.get("/health/")
def health():
    ok = (clf_model is not None) and (reg_model is not None)
    return {
        "status": "ok" if ok else "degraded",
        "classifier_loaded": clf_model is not None,
        "regressor_loaded": reg_model is not None
    }

@app.get("/predict/rain/")
def predict_rain(date: str = Query(..., description="YYYY-MM-DD")):
    """
    Returns prediction for 'will it rain exactly +7 days' from input date.
    """
    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    try:
        # NEW: build features aligned to classifier schema
        feats_df = features_for_classification(date)
        res = classify_rain(feats_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    pred_date = (dt + timedelta(days=7)).strftime("%Y-%m-%d")
    return {
        "input_date": date,
        "prediction": {"date": pred_date, "will_rain": bool(res["will_rain"])}
    }

@app.get("/predict/precipitation/fall/")
def predict_precipitation_fall(date: str = Query(..., description="YYYY-MM-DD")):
    """
    Returns predicted cumulative precipitation for t+1..t+3 (mm).
    """
    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    try:
        # NEW: build features aligned to regressor schema
        feats_df = features_for_regression(date)
        res = regress_precip(feats_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    start_date = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
    end_date   = (dt + timedelta(days=3)).strftime("%Y-%m-%d")
    return {
        "input_date": date,
        "prediction": {
            "start_date": start_date,
            "end_date": end_date,
            "precipitation_fall": round(res["precipitation_fall"], 3)
        }
    }
