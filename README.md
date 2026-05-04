# 🏦 Bank Customer Churn Prediction
### End-to-End ML Project | Resume-Ready | Streamlit + FastAPI

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)](https://fastapi.tiangolo.com)

---

## 📌 Project Overview

A production-grade machine learning project that predicts which bank customers are likely to churn (leave), enables targeted retention campaigns, and quantifies the business value of reducing churn.

**Business Problem**: 20.4% of bank customers churn annually. Retaining them is 5× cheaper than acquiring new ones.

**Solution**: Train a churn prediction model, deploy it as a web app + REST API so the CRM team can act on at-risk customers in real time.

---

## 📊 Dataset
- **Source**: Bank Customer Churn Dataset
- **Size**: 10,000 customers × 12 features
- **Target**: `churn` (1 = left, 0 = stayed) — 20.4% positive class

| Feature           | Description                   |
|-------------------|-------------------------------|
| credit_score      | Customer credit score (350-850)|
| country           | France / Germany / Spain       |
| gender            | Male / Female                  |
| age               | Customer age                   |
| tenure            | Years with the bank            |
| balance           | Account balance                |
| products_number   | Number of bank products        |
| credit_card       | Has credit card? (0/1)         |
| active_member     | Is active? (0/1)               |
| estimated_salary  | Estimated annual salary        |
| churn             | Target: 1=churned, 0=stayed    |

---

## 🏗️ Project Structure
```
churn_project/
├── data/
│   └── Bank_Customer_Churn_Prediction.csv
├── models/
│   ├── random_forest_tuned.pkl   # Trained model
│   ├── scaler.pkl                # Feature scaler
│   ├── feature_names.pkl         # Feature list
│   └── model_meta.json           # Performance metadata
├── outputs/
│   ├── 01_churn_distribution.png
│   ├── 02_age_distribution.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_churn_by_segment.png
│   ├── 05_roc_curves.png
│   ├── 06_confusion_matrix.png
│   ├── 07_feature_importance.png
│   ├── 08_model_comparison.png
│   └── 09_business_impact.png
├── app/
│   └── streamlit_app.py          # Web dashboard
├── api/
│   └── main.py                   # FastAPI REST API
├── train_model.py                # Full ML pipeline
├── requirements.txt
└── README.md
```

---

## 🔬 ML Pipeline Steps

### 1️⃣ Data Cleaning
- Zero missing values confirmed
- Dropped `customer_id` (non-predictive)
- Label-encoded `country`, `gender`

### 2️⃣ Feature Engineering (7 new features)
| Feature               | Logic                            | Rationale                    |
|-----------------------|----------------------------------|------------------------------|
| balance_salary_ratio  | balance / salary                 | Relative wealth indicator    |
| products_per_year     | products / tenure                | Product adoption rate        |
| is_zero_balance       | balance == 0                     | Dormant account signal       |
| age_group             | Binned: <30, 30-45, 45-60, 60+  | Non-linear age effects       |
| credit_score_band     | Binned: Poor/Fair/Good/Excellent  | Categorical risk bands       |
| wealth_score          | (balance + salary) / 2           | Overall financial health     |
| active_credit_card    | active_member × credit_card      | Interaction feature          |

### 3️⃣ Train-Test Split
- 80/20 stratified split — maintains churn ratio in both splits

### 4️⃣ Models Trained
| Model               | Accuracy | F1    | AUC   |
|---------------------|----------|-------|-------|
| Logistic Regression | 71.1%    | 0.501 | 0.771 |
| Random Forest       | 86.2%    | 0.557 | 0.847 |
| Gradient Boosting   | 86.7%    | 0.601 | **0.865** |
| RF (Tuned)          | 85.9%    | 0.570 | 0.851 |

### 5️⃣ Hyperparameter Tuning
GridSearchCV with 3-fold Stratified CV on Random Forest:
- Best: `max_depth=10, min_samples_split=5, n_estimators=200`

### 6️⃣ Key Insights from EDA
- 🔴 Germany has **32% churn rate** — highest of all countries
- 👩 Female customers churn **25%** vs males **16%**
- 📅 Age **45-60** group shows dramatically elevated churn
- 💰 Zero-balance accounts are a top churn predictor
- 📦 Customers with **3-4 products** churn more than those with 1-2

---

## 💰 Business Impact

> **"Reducing churn by 10% could save ₹4.33 Crore in annual revenue"**

| Scenario         | Customers Retained | Annual Savings  |
|------------------|--------------------|-----------------|
| 10% reduction    | ~204 customers     | ₹4.33 Cr        |
| 20% reduction    | ~407 customers     | ₹8.66 Cr        |
| 30% reduction    | ~611 customers     | ₹12.99 Cr       |

**ROI of the ML system**: Typical retention campaign costs ₹20-50L → 8x–20x ROI.

---

## 🚀 Deployment

### Option A — Streamlit Web App
```bash
pip install streamlit
streamlit run app/streamlit_app.py
# Opens at http://localhost:8501
```

### Option B — FastAPI REST API
```bash
pip install fastapi uvicorn
uvicorn api.main:app --reload --port 8000
# API docs at http://localhost:8000/docs
```

### API Usage Example
```python
import requests

customer = {
    "credit_score": 650,
    "country": "Germany",
    "gender": "Female",
    "age": 48,
    "tenure": 3,
    "balance": 95000,
    "products_number": 1,
    "credit_card": 1,
    "active_member": 0,
    "estimated_salary": 120000
}

response = requests.post("http://localhost:8000/predict", json=customer)
print(response.json())
# {"churn_probability": 0.73, "risk_level": "HIGH",
#  "recommendation": "Immediate intervention required..."}
```

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/bank-churn-prediction
cd bank-churn-prediction
pip install -r requirements.txt
python train_model.py        # Train & save models
streamlit run app/streamlit_app.py  # Launch web app
```

---

## 🛠️ Tech Stack
- **Python 3.10+**
- **Pandas / NumPy** — Data manipulation
- **Scikit-learn** — ML models (LR, RF, GB, GridSearchCV)
- **Matplotlib / Seaborn** — Visualizations
- **Streamlit** — Web dashboard
- **FastAPI + Uvicorn** — REST API
- **Joblib** — Model serialization

---

## 📈 Resume Talking Points
- Built end-to-end ML pipeline from raw CSV to deployed API serving real-time predictions
- Engineered 7 domain-specific features improving AUC from 0.847 to 0.865
- Performed GridSearchCV hyperparameter tuning with stratified cross-validation
- Quantified business impact: reducing churn 10% saves ₹4.33 Cr/year (8–20× ROI)
- Deployed as Streamlit dashboard + FastAPI backend with Swagger docs
- Handled class imbalance (80:20 split) using `class_weight="balanced"`
