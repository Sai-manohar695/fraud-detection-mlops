import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
import uvicorn
import json
import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# ── App Setup ─────────────────────────────────────────────
app = FastAPI(
    title       = "Fraud Detection API",
    description = "Real-time fraud detection using XGBoost trained on IEEE-CIS dataset",
    version     = "1.0.0"
)

# ── Config ────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "../mlflow/mlflow.db")
MODEL_NAME          = "fraud-detection-xgboost"
MODEL_VERSION       = "1"
THRESHOLD           = 0.8348

# ── Load MLflow Model ─────────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

try:
    model = mlflow.xgboost.load_model(
        f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    )
    print(f"Model loaded: {MODEL_NAME} v{MODEL_VERSION}")
except Exception as e:
    print(f"Model loading failed: {e}")
    model = None

# ── Load Feature Columns & Medians ────────────────────────
with open("feature_columns.json", "r") as f:
    FEATURE_COLUMNS = json.load(f)

with open("median_values.json", "r") as f:
    MEDIAN_VALUES = json.load(f)

print(f"Feature columns loaded: {len(FEATURE_COLUMNS)}")


# ── Request Schema ────────────────────────────────────────
class TransactionRequest(BaseModel):
    # Core transaction features
    TransactionAmt      : float         = Field(..., example=980.0)
    ProductCD           : Optional[int] = Field(None, example=0)
    card1               : Optional[int] = Field(None, example=1)
    card2               : Optional[int] = Field(None, example=1)
    card3               : Optional[int] = Field(None, example=1)
    card4               : Optional[int] = Field(None, example=1)
    card5               : Optional[int] = Field(None, example=1)
    card6               : Optional[int] = Field(None, example=1)
    addr1               : Optional[int] = Field(None, example=1)
    addr2               : Optional[int] = Field(None, example=1)
    P_emaildomain       : Optional[int] = Field(None, example=0)
    R_emaildomain       : Optional[int] = Field(None, example=0)
    # Engineered features
    hour                : Optional[int]   = Field(None, example=23)
    day_of_week         : Optional[int]   = Field(None, example=6)
    day_of_month        : Optional[int]   = Field(None, example=15)
    week                : Optional[int]   = Field(None, example=10)
    has_identity        : Optional[int]   = Field(None, example=1)
    TransactionAmt_log  : Optional[float] = Field(None, example=6.89)


# ── Response Schema ───────────────────────────────────────
class PredictionResponse(BaseModel):
    is_fraud            : bool
    fraud_probability   : float
    risk_level          : str
    threshold_used      : float
    inference_time_ms   : float


# ── Helper — Build Full Feature Vector ───────────────────
def build_feature_vector(transaction: TransactionRequest) -> pd.DataFrame:
    # Start with median values for all features
    row = MEDIAN_VALUES.copy()

    # Override with provided values
    provided = transaction.dict(exclude_none=False)
    for key, value in provided.items():
        if key in row and value is not None:
            row[key] = value

    # Auto-compute log transform if not provided
    if transaction.TransactionAmt_log is None:
        row['TransactionAmt_log'] = float(np.log1p(transaction.TransactionAmt))

    # Build DataFrame in correct column order
    df = pd.DataFrame([row])[FEATURE_COLUMNS]
    return df


# ── Health Check ──────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status"        : "ok",
        "model_loaded"  : model is not None,
        "model_name"    : MODEL_NAME,
        "model_version" : MODEL_VERSION,
        "threshold"     : THRESHOLD,
        "total_features": len(FEATURE_COLUMNS)
    }


# ── Prediction Endpoint ───────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check MLflow connection."
        )

    start = time.time()

    # Build full feature vector
    input_df = build_feature_vector(transaction)

    # Predict probability
    fraud_probability = float(model.predict_proba(input_df)[:, 1][0])

    # Apply optimal threshold
    is_fraud = fraud_probability >= THRESHOLD

    # Risk level
    if fraud_probability >= 0.85:
        risk_level = "CRITICAL"
    elif fraud_probability >= 0.70:
        risk_level = "HIGH"
    elif fraud_probability >= 0.50:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    inference_time = round((time.time() - start) * 1000, 2)

    return PredictionResponse(
        is_fraud          = is_fraud,
        fraud_probability = round(fraud_probability, 4),
        risk_level        = risk_level,
        threshold_used    = THRESHOLD,
        inference_time_ms = inference_time
    )


# ── Model Info Endpoint ───────────────────────────────────
@app.get("/model/info")
def model_info():
    return {
        "model_name"    : MODEL_NAME,
        "model_version" : MODEL_VERSION,
        "threshold"     : THRESHOLD,
        "algorithm"     : "XGBoost",
        "dataset"       : "IEEE-CIS Fraud Detection",
        "total_features": len(FEATURE_COLUMNS),
        "metrics"       : {
            "f1_score"  : 0.6618,
            "roc_auc"   : 0.9485,
            "precision" : 0.7509,
            "recall"    : 0.5916
        }
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)