"""
=============================================================
  BANK CUSTOMER CHURN PREDICTION — FastAPI REST API
  Run: uvicorn api.main:app --reload --port 8000
  Docs: http://localhost:8000/docs
=============================================================
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import joblib, json, os
import numpy as np
import pandas as pd

# ─── Paths ────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "models")

# ─── Load models at startup ──────────────────────────────
rf       = joblib.load(os.path.join(MODEL_DIR, "random_forest_tuned.pkl"))
scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
FEATURES = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
with open(os.path.join(MODEL_DIR, "model_meta.json")) as f:
    META = json.load(f)

# ─── Feature engineering (mirrors train_model.py) ────────
def engineer(d: dict) -> np.ndarray:
    d["balance_salary_ratio"] = d["balance"] / (d["estimated_salary"] + 1)
    d["products_per_year"]    = d["products_number"] / (d["tenure"] + 1)
    d["is_zero_balance"]      = int(d["balance"] == 0)
    d["age_group"]            = int(pd.cut([d["age"]], bins=[0,30,45,60,100], labels=[0,1,2,3])[0])
    d["credit_score_band"]    = int(pd.cut([d["credit_score"]], bins=[0,500,600,700,850], labels=[0,1,2,3])[0])
    d["wealth_score"]         = (d["balance"] + d["estimated_salary"]) / 2
    d["active_credit_card"]   = d["active_member"] * d["credit_card"]
    return np.array([[d[f] for f in FEATURES]])

# ─── App setup ───────────────────────────────────────────
app = FastAPI(
    title="🏦 Bank Churn Prediction API",
    description="""
    ## Bank Customer Churn Prediction

    Predict whether a bank customer will churn using a tuned Random Forest model.

    ### Endpoints
    - **POST /predict** — Single customer churn prediction
    - **POST /predict/batch** — Batch prediction (up to 1000 customers)
    - **GET /model/info** — Model metadata and performance
    - **GET /health** — Health check

    ### Model Performance
    - **AUC-ROC**: 0.851
    - **Accuracy**: 86%
    - **F1 Score**: 0.56 (on imbalanced churn class)
    """,
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Request/Response schemas ────────────────────────────
COUNTRY_MAP = {"France": 0, "Germany": 1, "Spain": 2}
GENDER_MAP  = {"Female": 0, "Male": 1}

class CustomerInput(BaseModel):
    credit_score:     int   = Field(..., ge=300, le=850,    example=650,     description="Credit score (300–850)")
    country:          str   = Field(...,                    example="France", description="France | Germany | Spain")
    gender:           str   = Field(...,                    example="Male",   description="Male | Female")
    age:              int   = Field(..., ge=18, le=92,      example=38)
    tenure:           int   = Field(..., ge=0, le=10,       example=5,        description="Years with bank")
    balance:          float = Field(..., ge=0,              example=76000.0)
    products_number:  int   = Field(..., ge=1, le=4,        example=2)
    credit_card:      int   = Field(..., ge=0, le=1,        example=1,        description="1=Yes, 0=No")
    active_member:    int   = Field(..., ge=0, le=1,        example=1,        description="1=Active, 0=Inactive")
    estimated_salary: float = Field(..., ge=0,              example=100000.0)

    @validator("country")
    def valid_country(cls, v):
        if v not in COUNTRY_MAP:
            raise ValueError(f"country must be one of {list(COUNTRY_MAP)}")
        return v

    @validator("gender")
    def valid_gender(cls, v):
        if v not in GENDER_MAP:
            raise ValueError(f"gender must be one of {list(GENDER_MAP)}")
        return v

class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction:  int
    risk_level:        str
    confidence:        str
    recommendation:    str

class BatchRequest(BaseModel):
    customers: list[CustomerInput]

# ─── Prediction logic ────────────────────────────────────
def predict_customer(c: CustomerInput) -> dict:
    raw = c.dict()
    raw["country"] = COUNTRY_MAP[raw["country"]]
    raw["gender"]  = GENDER_MAP[raw["gender"]]
    X = engineer(raw)
    prob = float(rf.predict_proba(X)[0][1])
    pred = int(prob > 0.5)

    if prob >= 0.7:
        risk   = "HIGH"
        conf   = "Very High Confidence"
        rec    = "Immediate intervention required: assign dedicated RM, offer retention package"
    elif prob >= 0.5:
        risk   = "HIGH"
        conf   = "High Confidence"
        rec    = "Schedule callback within 48h, offer fee waiver or loyalty upgrade"
    elif prob >= 0.35:
        risk   = "MEDIUM"
        conf   = "Moderate Confidence"
        rec    = "Enroll in re-engagement campaign, personalized product recommendations"
    else:
        risk   = "LOW"
        conf   = "High Confidence"
        rec    = "Maintain standard relationship, monthly loyalty touchpoints"

    return {
        "churn_probability": round(prob, 4),
        "churn_prediction":  pred,
        "risk_level":        risk,
        "confidence":        conf,
        "recommendation":    rec,
    }

# ─── Routes ──────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "model": "Random Forest (Tuned)", "version": "1.0.0"}

@app.get("/model/info", tags=["Model"])
def model_info():
    return {
        "model_name":        META["best_model"],
        "best_params":       META["best_params"],
        "auc_roc":           META["auc"],
        "accuracy":          META["accuracy"],
        "f1_score":          META["f1_score"],
        "training_samples":  META["total_customers"],
        "churn_rate_pct":    META["churn_rate_pct"],
        "features_used":     META["features"],
        "business_impact": {
            "revenue_at_risk_cr": META["revenue_at_risk_cr"],
            "savings_10pct_cr":   META["savings_10pct_cr"],
            "savings_20pct_cr":   META["savings_20pct_cr"],
        }
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerInput):
    """
    Predict churn risk for a single customer.

    Returns:
    - `churn_probability`: 0.0 to 1.0
    - `churn_prediction`: 1 = will churn, 0 = will stay
    - `risk_level`: LOW | MEDIUM | HIGH
    - `recommendation`: Actionable next step
    """
    try:
        return predict_customer(customer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(request: BatchRequest):
    """
    Batch prediction for up to 1,000 customers.
    Returns predictions for all customers with a summary.
    """
    if len(request.customers) > 1000:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 1000")

    predictions = [predict_customer(c) for c in request.customers]
    high_risk   = [i for i, p in enumerate(predictions) if p["risk_level"] == "HIGH"]
    med_risk    = [i for i, p in enumerate(predictions) if p["risk_level"] == "MEDIUM"]

    return {
        "total":            len(predictions),
        "high_risk_count":  len(high_risk),
        "medium_risk_count":len(med_risk),
        "low_risk_count":   len(predictions) - len(high_risk) - len(med_risk),
        "high_risk_indices":high_risk,
        "avg_churn_probability": round(
            sum(p["churn_probability"] for p in predictions) / len(predictions), 4
        ),
        "predictions": predictions,
    }

# ─── Run ─────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
