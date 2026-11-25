"""NorthStar churn prediction API."""
import json
import os
from pathlib import Path
from typing import List, Literal, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models"))
MODEL_PATH = MODEL_DIR / "churn_model.joblib"
META_PATH = MODEL_DIR / "metadata.json"

if not MODEL_PATH.exists() or not META_PATH.exists():
    raise RuntimeError(f"Model artifacts not found in {MODEL_DIR.resolve()} "
                       f"(expected churn_model.joblib and metadata.json). Check MODEL_DIR.")

model = joblib.load(MODEL_PATH)
metadata = json.loads(META_PATH.read_text())
THRESHOLD = float(metadata["decision_threshold"])
FEATURE_ORDER = metadata["feature_order"]
NUMERIC = metadata["features"]["numeric"]
CATEGORICAL = metadata["features"]["categorical"]
SKLEARN_MATCH = sklearn.__version__ == metadata["library_versions"]["sklearn"]


class Customer(BaseModel):
    customer_id: Optional[str] = None
    tenure_months: int = Field(..., ge=0, le=120)
    monthly_charges: float = Field(..., gt=0, le=500)
    total_charges: Optional[float] = Field(None, ge=0)
    support_calls: int = Field(..., ge=0, le=50)
    senior_citizen: int = Field(..., ge=0, le=1)
    num_products: int = Field(..., ge=1, le=10)
    last_login_days: int = Field(..., ge=0, le=365)
    contract: Literal["Month-to-month", "One year", "Two year"]
    internet_service: Literal["DSL", "Fiber optic", "None"]
    payment_method: Optional[str] = None


class PredictRequest(BaseModel):
    customers: List[Customer] = Field(..., min_length=1, max_length=1000)


app = FastAPI(title="NorthStar Churn API", version=metadata["model_version"])


def risk_band(p: float) -> str:
    if p >= 0.5:
        return "high"
    if p >= THRESHOLD:
        return "medium"
    return "low"


@app.get("/health")
def health():
    return {"status": "ok", "model_version": metadata["model_version"],
            "sklearn_runtime": sklearn.__version__,
            "sklearn_trained": metadata["library_versions"]["sklearn"],
            "sklearn_match": SKLEARN_MATCH}


@app.get("/model-info")
def model_info():
    keys = ["model_name", "model_version", "created_utc", "algorithm",
            "decision_threshold", "feature_order", "test_metrics"]
    return {k: metadata[k] for k in keys}


@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame([c.model_dump() for c in req.customers])
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL:
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
    try:
        proba = model.predict_proba(df[FEATURE_ORDER])[:, 1]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"prediction failed: {exc}")
    preds = []
    for cid, p in zip(df["customer_id"], proba):
        p = float(p)
        preds.append({"customer_id": cid, "churn_probability": round(p, 4),
                      "contact_customer": bool(p >= THRESHOLD), "risk_band": risk_band(p)})
    return {"model_version": metadata["model_version"], "threshold": THRESHOLD,
            "predictions": preds}
