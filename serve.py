"""
FastAPI inference service for the AML re-score model.

The backend computes a named feature map for each rule candidate and POSTs it to
/predict; this service returns the fraud probability. The backend applies the
decision threshold (config) and suppresses candidates scored as normal.

Run:  uvicorn serve:app --host 0.0.0.0 --port 8000
Env:  MODEL_DIR (default ".") — dir containing model.json + feature_names.json
"""
from __future__ import annotations

import json
import os
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from xgboost import XGBClassifier

MODEL_DIR = os.environ.get("MODEL_DIR", ".")

app = FastAPI(title="AML re-score model")

_model: Optional[XGBClassifier] = None
_features: list = []


def _load():
    global _model, _features
    model_path = os.path.join(MODEL_DIR, "model.json")
    feat_path = os.path.join(MODEL_DIR, "feature_names.json")
    if os.path.exists(model_path) and os.path.exists(feat_path):
        m = XGBClassifier()
        m.load_model(model_path)
        with open(feat_path) as f:
            _features = json.load(f)
        _model = m


@app.on_event("startup")
def startup():
    _load()


class PredictRequest(BaseModel):
    features: dict[str, float]


class PredictResponse(BaseModel):
    probability: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "n_features": len(_features)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # No model yet → return 1.0 so the backend fallback keeps the candidate (fail-open).
    if _model is None:
        return PredictResponse(probability=1.0)
    row = [[float(req.features.get(name, 0.0)) for name in _features]]
    prob = float(_model.predict_proba(row)[0][1])
    return PredictResponse(probability=prob)
