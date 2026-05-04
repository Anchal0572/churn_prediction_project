"""
=============================================================
  BANK CUSTOMER CHURN PREDICTION — Streamlit Web App
  Run: streamlit run app/streamlit_app.py
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="🏦 Churn Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0F1117; }
    .stApp { background-color: #0F1117; }
    h1,h2,h3 { color: #FAFAFA; }
    .metric-card {
        background: #1E1E2E;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2A2A3A;
    }
    .churn-high   { color: #E53935; font-size: 2rem; font-weight: bold; }
    .churn-low    { color: #4CAF50; font-size: 2rem; font-weight: bold; }
    .insight-box  {
        background: #1A237E;
        border-left: 4px solid #42A5F5;
        padding: 14px 18px;
        border-radius: 6px;
        margin: 8px 0;
        color: #E3F2FD;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1565C0, #6A1B9A);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 30px;
        font-size: 16px;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model artifacts ──────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "outputs")

@st.cache_resource
def load_artifacts():
    rf      = joblib.load(os.path.join(MODEL_DIR, "random_forest_tuned.pkl"))
    scaler  = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    features= joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    with open(os.path.join(MODEL_DIR, "model_meta.json")) as f:
        meta = json.load(f)
    return rf, scaler, features, meta

rf, scaler, FEATURES, meta = load_artifacts()

# ── Helper: Feature Engineering (must match train_model.py) ──
def engineer_features(raw: dict) -> pd.DataFrame:
    d = raw.copy()
    d["balance_salary_ratio"] = d["balance"] / (d["estimated_salary"] + 1)
    d["products_per_year"]    = d["products_number"] / (d["tenure"] + 1)
    d["is_zero_balance"]      = int(d["balance"] == 0)
    d["age_group"]            = int(pd.cut([d["age"]], bins=[0,30,45,60,100], labels=[0,1,2,3])[0])
    d["credit_score_band"]    = int(pd.cut([d["credit_score"]], bins=[0,500,600,700,850], labels=[0,1,2,3])[0])
    d["wealth_score"]         = (d["balance"] + d["estimated_salary"]) / 2
    d["active_credit_card"]   = d["active_member"] * d["credit_card"]
    return pd.DataFrame([d])[FEATURES]

# ════════════════════════════════════════════════════════
#  SIDEBAR — CUSTOMER INPUT
# ════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧾 Customer Profile")
    st.markdown("---")

    credit_score   = st.slider("Credit Score",    350, 850, 650)
    age            = st.slider("Age",              18, 92,  38)
    tenure         = st.slider("Tenure (years)",   0, 10,   5)
    balance        = st.number_input("Account Balance (₹ equiv.)", 0, 300000, 76000, step=1000)
    estimated_sal  = st.number_input("Estimated Salary (₹ equiv.)", 500, 200000, 100000, step=1000)
    products_num   = st.selectbox("Number of Products", [1,2,3,4])
    credit_card    = st.selectbox("Has Credit Card",    [1,0], format_func=lambda x: "Yes" if x else "No")
    active_member  = st.selectbox("Active Member",      [1,0], format_func=lambda x: "Yes" if x else "No")
    country        = st.selectbox("Country",            ["France","Germany","Spain"])
    gender         = st.selectbox("Gender",             ["Male","Female"])

    st.markdown("---")
    predict_btn    = st.button("🔮 Predict Churn Risk")

# Encode categoricals
country_map = {"France": 0, "Germany": 1, "Spain": 2}
gender_map  = {"Female": 0, "Male": 1}

raw_input = {
    "credit_score":     credit_score,
    "country":          country_map[country],
    "gender":           gender_map[gender],
    "age":              age,
    "tenure":           tenure,
    "balance":          balance,
    "products_number":  products_num,
    "credit_card":      credit_card,
    "active_member":    active_member,
    "estimated_salary": estimated_sal,
}

# ════════════════════════════════════════════════════════
#  MAIN CONTENT
# ════════════════════════════════════════════════════════
st.title("🏦 Bank Customer Churn Prediction")
st.markdown("##### AI-powered churn risk assessment with business impact analysis")
st.markdown("---")

# ── Top KPIs ─────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
metrics = [
    ("🎯 Model AUC",       f"{meta['auc']:.3f}",           "Gradient Boosting"),
    ("📊 Accuracy",        f"{meta['accuracy']*100:.1f}%",  "On test set"),
    ("👥 Total Customers", f"{meta['total_customers']:,}",  "In dataset"),
    ("💰 Revenue at Risk", f"₹{meta['revenue_at_risk_cr']:.1f} Cr", "From churners"),
]
for col, (label, val, sub) in zip([col1,col2,col3,col4], metrics):
    col.markdown(f"""
    <div class="metric-card">
        <div style="color:#9E9E9E; font-size:12px">{label}</div>
        <div style="color:#FAFAFA; font-size:24px; font-weight:bold; margin:4px 0">{val}</div>
        <div style="color:#616161; font-size:11px">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("")

# ── Tabs ─────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 EDA & Insights", "💰 Business Impact"])

# ─── TAB 1 — PREDICTION ───────────────────────────────────
with tab1:
    if predict_btn:
        X_input   = engineer_features(raw_input)
        prob      = rf.predict_proba(X_input.values)[0][1]
        pred      = int(prob > 0.5)
        risk_pct  = prob * 100

        st.markdown("### 🎯 Churn Risk Assessment")
        c1, c2, c3 = st.columns([1,1,1])

        risk_color = "#E53935" if prob > 0.6 else ("#FF9800" if prob > 0.35 else "#4CAF50")
        risk_label = "HIGH RISK 🔴" if prob > 0.6 else ("MEDIUM RISK 🟠" if prob > 0.35 else "LOW RISK 🟢")

        c1.markdown(f"""
        <div class="metric-card">
            <div style="color:#9E9E9E; font-size:13px">Churn Probability</div>
            <div style="color:{risk_color}; font-size:48px; font-weight:bold">{risk_pct:.1f}%</div>
            <div style="color:{risk_color}; font-size:16px; font-weight:bold">{risk_label}</div>
        </div>""", unsafe_allow_html=True)

        # Risk gauge (matplotlib)
        fig, ax = plt.subplots(figsize=(3.5, 2.2), facecolor="#1E1E2E")
        ax.set_facecolor("#1E1E2E")
        theta = np.linspace(0, np.pi, 200)
        for i, (start, end, color) in enumerate([(0,0.35,"#4CAF50"),(0.35,0.6,"#FF9800"),(0.6,1,"#E53935")]):
            mask = (theta/np.pi >= start) & (theta/np.pi <= end)
            ax.plot(np.cos(theta[mask]), np.sin(theta[mask]), lw=10, color=color, alpha=0.6, solid_capstyle="butt")
        angle = (1 - prob) * np.pi
        ax.annotate("", xy=(0.5*np.cos(angle), 0.5*np.sin(angle)), xytext=(0,0),
                    arrowprops=dict(arrowstyle="-|>", color="white", lw=2.5))
        ax.text(0, -0.35, f"{risk_pct:.0f}%", ha="center", va="center",
                color="white", fontsize=20, fontweight="bold")
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-0.5, 1.1); ax.axis("off")
        c2.pyplot(fig, use_container_width=True)
        plt.close()

        # Risk factors
        with c3:
            st.markdown("**Key Risk Factors**")
            if age > 45:
                st.error(f"⚠️ Age {age} — older customers churn more")
            if balance == 0:
                st.error("⚠️ Zero balance — high churn signal")
            if products_num >= 3:
                st.warning(f"⚠️ {products_num} products — may be over-served")
            if active_member == 0:
                st.warning("⚠️ Inactive member — engagement risk")
            if country == "Germany":
                st.warning("⚠️ Germany — 32% churn rate market")
            if credit_score < 500:
                st.info("ℹ️ Low credit score — monitor closely")
            if prob < 0.35:
                st.success("✅ No major risk factors detected")

        # Action recommendations
        st.markdown("### 💡 Recommended Actions")
        a1, a2, a3 = st.columns(3)
        if prob > 0.5:
            a1.markdown('<div class="insight-box">📞 <b>Priority Call</b><br>Assign a dedicated RM for immediate outreach</div>', unsafe_allow_html=True)
            a2.markdown('<div class="insight-box">🎁 <b>Retention Offer</b><br>Offer loyalty rewards or fee waivers</div>', unsafe_allow_html=True)
            a3.markdown('<div class="insight-box">📈 <b>Engagement Campaign</b><br>Enroll in premium banking benefits program</div>', unsafe_allow_html=True)
        else:
            a1.markdown('<div class="insight-box">📧 <b>Nurture Campaign</b><br>Monthly newsletter with personalized offers</div>', unsafe_allow_html=True)
            a2.markdown('<div class="insight-box">📱 <b>App Engagement</b><br>Push notifications for new features</div>', unsafe_allow_html=True)
            a3.markdown('<div class="insight-box">🏆 <b>Loyalty Points</b><br>Enroll in standard loyalty program</div>', unsafe_allow_html=True)

    else:
        st.info("👈 Fill in the customer profile in the sidebar and click **Predict Churn Risk**")
        st.markdown("""
        ### How it works
        1. **Enter** customer details in the sidebar
        2. **Click** Predict Churn Risk
        3. **Get** churn probability + risk factors + action recommendations
        
        **Model**: Random Forest (Tuned) | **AUC**: 0.851 | **Accuracy**: 86%
        """)

# ─── TAB 2 — EDA ──────────────────────────────────────────
with tab2:
    st.markdown("### 📊 Exploratory Data Analysis")
    charts = [
        ("01_churn_distribution.png", "Churn Distribution"),
        ("02_age_distribution.png",   "Age Distribution by Churn"),
        ("03_correlation_heatmap.png","Feature Correlation Heatmap"),
        ("04_churn_by_segment.png",   "Churn Rate by Segment"),
        ("05_roc_curves.png",         "ROC Curves — All Models"),
        ("06_confusion_matrix.png",   "Confusion Matrix"),
        ("07_feature_importance.png", "Feature Importance"),
        ("08_model_comparison.png",   "Model Comparison"),
    ]
    for i in range(0, len(charts), 2):
        cols = st.columns(2)
        for j, (fname, title) in enumerate(charts[i:i+2]):
            path = os.path.join(REPORT_DIR, fname)
            if os.path.exists(path):
                cols[j].markdown(f"**{title}**")
                cols[j].image(path, use_container_width=True)

    st.markdown("### 📌 Key Insights from EDA")
    insights = [
        "🔴 Overall churn rate is **20.4%** — 1 in 5 customers leaves",
        "📍 **Germany** has the highest churn rate (~32%) vs France (16%) and Spain (17%)",
        "👩 **Female customers** churn more (~25%) compared to males (~16%)",
        "📅 **Older customers (45-60)** are significantly more likely to churn",
        "💳 Customers with **3–4 products** have disproportionately high churn",
        "💰 **Zero-balance accounts** are a major churn predictor",
        "🏆 **Age** is the most important feature (Random Forest)",
    ]
    for ins in insights:
        st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)

# ─── TAB 3 — BUSINESS IMPACT ──────────────────────────────
with tab3:
    st.markdown("### 💰 Business Impact Analysis")
    bc1, bc2, bc3 = st.columns(3)
    bc1.metric("Revenue at Risk", f"₹{meta['revenue_at_risk_cr']:.1f} Cr", "Annual")
    bc2.metric("10% Reduction Saves", f"₹{meta['savings_10pct_cr']:.2f} Cr", "Conservative scenario")
    

    st.image(os.path.join(REPORT_DIR, "09_business_impact.png"), use_container_width=True)

    st.markdown("### 📋 Revenue Model Assumptions")
    st.markdown("""
    | Parameter | Value | Rationale |
    |---|---|---|
    | Avg Balance | ₹6.4 Lakh | Dataset average × INR rate |
    | Revenue/Balance | 2% p.a. | Net interest margin |
    | Revenue/Salary | 1% p.a. | Fee income proxy |
    | INR Rate | ₹84/$ | Current exchange rate |
    | Churn customers | 2,037 | Actual dataset count |
    """)

    st.markdown("### 🎯 ROI of Churn Prevention Model")
    st.markdown("""
    <div class="insight-box">
    💡 <b>If this model prevents just 10% of churners</b> through targeted interventions:
    <ul>
        <li>~204 customers retained</li>
        <li>₹4.33 Cr revenue saved annually</li>
        <li>Typical retention campaign costs: ₹20–50L</li>
        <li><b>ROI: 8x–20x on campaign spend</b></li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
