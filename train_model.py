"""
=============================================================
  BANK CUSTOMER CHURN PREDICTION  — Full ML Pipeline
  Author: [Your Name] | Dataset: 10,000 Bank Customers
=============================================================
  Steps:
    1. Data Loading & Cleaning
    2. Exploratory Data Analysis (EDA)
    3. Feature Engineering
    4. Train-Test Split
    5. Model Building (Logistic Regression + Random Forest)
    6. Model Evaluation (Accuracy, F1, ROC-AUC, Confusion Matrix)
    7. Model Tuning (GridSearchCV)
    8. Business Impact Calculation
    9. Save Model Artifacts
=============================================================
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data", "Bank_Customer_Churn_Prediction.csv")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
REPORT_DIR  = os.path.join(BASE_DIR, "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  PALETTE
# ─────────────────────────────────────────────
PALETTE   = {"stayed": "#4CAF50", "churned": "#E53935"}
BG_COLOR  = "#0F1117"
TEXT_COLOR = "#FAFAFA"
GRID_COLOR = "#2A2A3A"

sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)

# ─────────────────────────────────────────────
#  1. LOAD & CLEAN DATA
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 1 — DATA LOADING & CLEANING")
print("="*60)

df = pd.read_csv(DATA_PATH)
print(f"  ✔ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  ✔ Missing values: {df.isnull().sum().sum()}")
print(f"  ✔ Duplicates   : {df.duplicated().sum()}")

# Drop customer ID (not a feature)
df.drop(columns=["customer_id"], inplace=True)

# Encode categorical
le_gender  = LabelEncoder()
le_country = LabelEncoder()
df["gender"]  = le_gender.fit_transform(df["gender"])    # Male=1, Female=0
df["country"] = le_country.fit_transform(df["country"])  # France=0, Germany=1, Spain=2

print(f"\n  Churn rate: {df['churn'].mean()*100:.1f}%  ({df['churn'].sum():,} churned out of {len(df):,})")
print("  ✔ Cleaning complete.\n")

# ─────────────────────────────────────────────
#  2. EDA — VISUALIZATIONS
# ─────────────────────────────────────────────
print("="*60)
print("  STEP 2 — EXPLORATORY DATA ANALYSIS")
print("="*60)

# 2a — Churn distribution
fig, ax = plt.subplots(figsize=(6, 4), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)
churn_counts = df["churn"].value_counts()
bars = ax.bar(["Stayed", "Churned"], churn_counts.values,
              color=[PALETTE["stayed"], PALETTE["churned"]],
              edgecolor="white", linewidth=0.5, width=0.5)
for bar, val in zip(bars, churn_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 60,
            f"{val:,}\n({val/len(df)*100:.1f}%)", ha="center",
            color=TEXT_COLOR, fontsize=12, fontweight="bold")
ax.set_title("Customer Churn Distribution", color=TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
ax.set_ylabel("Count", color=TEXT_COLOR)
ax.tick_params(colors=TEXT_COLOR)
ax.spines[["top","right","left","bottom"]].set_color(GRID_COLOR)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "01_churn_distribution.png"), dpi=150, bbox_inches="tight",
            facecolor=BG_COLOR)
plt.close()
print("  ✔ Saved: 01_churn_distribution.png")

# 2b — Age distribution by churn
fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)
for label, color, name in [(0, PALETTE["stayed"], "Stayed"), (1, PALETTE["churned"], "Churned")]:
    subset = df[df["churn"] == label]["age"]
    ax.hist(subset, bins=30, alpha=0.7, color=color, label=name, edgecolor="none")
ax.set_title("Age Distribution by Churn", color=TEXT_COLOR, fontsize=14, fontweight="bold")
ax.set_xlabel("Age", color=TEXT_COLOR)
ax.set_ylabel("Count", color=TEXT_COLOR)
ax.tick_params(colors=TEXT_COLOR)
ax.legend(facecolor="#1E1E2E", labelcolor=TEXT_COLOR)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "02_age_distribution.png"), dpi=150, bbox_inches="tight",
            facecolor=BG_COLOR)
plt.close()
print("  ✔ Saved: 02_age_distribution.png")

# 2c — Correlation heatmap
fig, ax = plt.subplots(figsize=(9, 7), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8},
            annot_kws={"size": 9})
ax.set_title("Feature Correlation Matrix", color=TEXT_COLOR, fontsize=14, fontweight="bold", pad=15)
ax.tick_params(colors=TEXT_COLOR, labelsize=9)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "03_correlation_heatmap.png"), dpi=150, bbox_inches="tight",
            facecolor=BG_COLOR)
plt.close()
print("  ✔ Saved: 03_correlation_heatmap.png")

# 2d — Churn rate by country & gender
fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=BG_COLOR)
country_names = {v: k for k, v in zip(le_country.classes_, le_country.transform(le_country.classes_))}
df_plot = df.copy()
df_plot["country_name"]  = df_plot["country"].map(country_names)
df_plot["gender_name"]   = df_plot["gender"].map({0: "Female", 1: "Male"})

for ax, col, title in zip(axes, ["country_name", "gender_name"], ["by Country", "by Gender"]):
    ax.set_facecolor(BG_COLOR)
    churn_rate = df_plot.groupby(col)["churn"].mean() * 100
    bars = ax.bar(churn_rate.index, churn_rate.values,
                  color=[PALETTE["churned"] if v > 20 else PALETTE["stayed"] for v in churn_rate.values],
                  edgecolor="white", linewidth=0.5, width=0.5)
    for bar, val in zip(bars, churn_rate.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", color=TEXT_COLOR, fontsize=11, fontweight="bold")
    ax.set_title(f"Churn Rate {title}", color=TEXT_COLOR, fontsize=12, fontweight="bold")
    ax.set_ylabel("Churn Rate (%)", color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_ylim(0, churn_rate.max() + 10)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "04_churn_by_segment.png"), dpi=150, bbox_inches="tight",
            facecolor=BG_COLOR)
plt.close()
print("  ✔ Saved: 04_churn_by_segment.png")

# ─────────────────────────────────────────────
#  3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 3 — FEATURE ENGINEERING")
print("="*60)

df["balance_salary_ratio"]    = df["balance"] / (df["estimated_salary"] + 1)
df["products_per_year"]       = df["products_number"] / (df["tenure"] + 1)
df["is_zero_balance"]         = (df["balance"] == 0).astype(int)
df["age_group"]               = pd.cut(df["age"], bins=[0,30,45,60,100],
                                        labels=[0,1,2,3]).astype(int)
df["credit_score_band"]       = pd.cut(df["credit_score"], bins=[0,500,600,700,850],
                                        labels=[0,1,2,3]).astype(int)
df["wealth_score"]            = (df["balance"] + df["estimated_salary"]) / 2
df["active_credit_card"]      = df["active_member"] * df["credit_card"]

NEW_FEATURES = ["balance_salary_ratio","products_per_year","is_zero_balance",
                "age_group","credit_score_band","wealth_score","active_credit_card"]
print(f"  ✔ Created {len(NEW_FEATURES)} new features: {NEW_FEATURES}")

# ─────────────────────────────────────────────
#  4. TRAIN-TEST SPLIT
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 4 — TRAIN-TEST SPLIT")
print("="*60)

FEATURES = [c for c in df.columns if c != "churn"]
X = df[FEATURES]
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train):,} samples | Test: {len(X_test):,} samples")
print(f"  Train churn rate: {y_train.mean()*100:.1f}% | Test churn rate: {y_test.mean()*100:.1f}%")

# ─────────────────────────────────────────────
#  5. MODEL BUILDING
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 5 — MODEL BUILDING")
print("="*60)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, random_state=42),
}

results = {}
for name, model in models.items():
    X_tr = X_train_sc if name == "Logistic Regression" else X_train.values
    X_te = X_test_sc  if name == "Logistic Regression" else X_test.values
    model.fit(X_tr, y_train)
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    acc   = accuracy_score(y_test, y_pred)
    f1    = f1_score(y_test, y_pred)
    auc   = roc_auc_score(y_test, y_proba)
    results[name] = {"model": model, "y_pred": y_pred, "y_proba": y_proba,
                     "accuracy": acc, "f1": f1, "auc": auc}
    print(f"  {name:25s}  Acc={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}")

best_name  = max(results, key=lambda k: results[k]["auc"])
best       = results[best_name]
print(f"\n  ★ Best Model: {best_name}  (AUC={best['auc']:.4f})")

# ─────────────────────────────────────────────
#  6. MODEL EVALUATION — CHARTS
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 6 — MODEL EVALUATION")
print("="*60)

# 6a — ROC Curves (all models)
fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)
colors_roc = ["#42A5F5", "#AB47BC", "#FF7043"]
for (name, res), color in zip(results.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={res['auc']:.3f})")
ax.plot([0,1],[0,1],"--", color="gray", lw=1)
ax.set_title("ROC Curves — All Models", color=TEXT_COLOR, fontsize=13, fontweight="bold")
ax.set_xlabel("False Positive Rate", color=TEXT_COLOR)
ax.set_ylabel("True Positive Rate", color=TEXT_COLOR)
ax.tick_params(colors=TEXT_COLOR)
ax.legend(facecolor="#1E1E2E", labelcolor=TEXT_COLOR, fontsize=9)
ax.spines[["top","right"]].set_color(GRID_COLOR)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "05_roc_curves.png"), dpi=150, bbox_inches="tight",
            facecolor=BG_COLOR)
plt.close()
print("  ✔ Saved: 05_roc_curves.png")

# 6b — Confusion Matrix (best model)
fig, ax = plt.subplots(figsize=(5, 4), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)
cm = confusion_matrix(y_test, best["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Stayed","Churned"], yticklabels=["Stayed","Churned"],
            linewidths=1, linecolor=GRID_COLOR, cbar=False,
            annot_kws={"size": 14, "weight": "bold", "color": "white"})
ax.set_title(f"Confusion Matrix — {best_name}", color=TEXT_COLOR, fontsize=12, fontweight="bold")
ax.set_xlabel("Predicted", color=TEXT_COLOR)
ax.set_ylabel("Actual", color=TEXT_COLOR)
ax.tick_params(colors=TEXT_COLOR)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "06_confusion_matrix.png"), dpi=150, bbox_inches="tight",
            facecolor=BG_COLOR)
plt.close()
print("  ✔ Saved: 06_confusion_matrix.png")

# 6c — Feature Importance (Random Forest)
rf_model = results["Random Forest"]["model"]
fi = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values(ascending=True).tail(12)
fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)
colors_fi = ["#E53935" if v > fi.mean() else "#42A5F5" for v in fi.values]
bars = ax.barh(fi.index, fi.values, color=colors_fi, edgecolor="none", height=0.6)
ax.set_title("Feature Importance — Random Forest", color=TEXT_COLOR, fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score", color=TEXT_COLOR)
ax.tick_params(colors=TEXT_COLOR)
ax.spines[["top","right","bottom"]].set_color(GRID_COLOR)
for bar, val in zip(bars, fi.values):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", color=TEXT_COLOR, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "07_feature_importance.png"), dpi=150, bbox_inches="tight",
            facecolor=BG_COLOR)
plt.close()
print("  ✔ Saved: 07_feature_importance.png")

# 6d — Model Comparison Bar Chart
fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)
metrics_names = ["accuracy", "f1", "auc"]
x = np.arange(len(results))
width = 0.25
colors_bar = ["#42A5F5", "#AB47BC", "#FF7043"]
for i, (metric, color) in enumerate(zip(metrics_names, colors_bar)):
    vals = [results[m][metric] for m in results]
    bars = ax.bar(x + i*width, vals, width, label=metric.upper(), color=color, alpha=0.85)
ax.set_xticks(x + width)
ax.set_xticklabels(list(results.keys()), color=TEXT_COLOR, fontsize=9)
ax.set_title("Model Comparison", color=TEXT_COLOR, fontsize=13, fontweight="bold")
ax.set_ylabel("Score", color=TEXT_COLOR)
ax.set_ylim(0, 1.05)
ax.tick_params(colors=TEXT_COLOR)
ax.legend(facecolor="#1E1E2E", labelcolor=TEXT_COLOR)
ax.spines[["top","right"]].set_color(GRID_COLOR)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "08_model_comparison.png"), dpi=150, bbox_inches="tight",
            facecolor=BG_COLOR)
plt.close()
print("  ✔ Saved: 08_model_comparison.png")

# ─────────────────────────────────────────────
#  7. HYPERPARAMETER TUNING
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 7 — MODEL TUNING (GridSearchCV on Random Forest)")
print("="*60)

param_grid = {
    "n_estimators":      [100, 200],
    "max_depth":         [6, 10, None],
    "min_samples_split": [2, 5],
}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight="balanced"),
    param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0
)
grid_search.fit(X_train.values, y_train)
best_rf     = grid_search.best_estimator_
best_params = grid_search.best_params_
y_pred_tuned  = best_rf.predict(X_test.values)
y_proba_tuned = best_rf.predict_proba(X_test.values)[:, 1]
tuned_auc     = roc_auc_score(y_test, y_proba_tuned)
print(f"  ✔ Best params : {best_params}")
print(f"  ✔ Tuned AUC   : {tuned_auc:.4f}")

# Replace best model with tuned version
results["Random Forest (Tuned)"] = {
    "model": best_rf, "y_pred": y_pred_tuned, "y_proba": y_proba_tuned,
    "accuracy": accuracy_score(y_test, y_pred_tuned),
    "f1": f1_score(y_test, y_pred_tuned), "auc": tuned_auc
}

# ─────────────────────────────────────────────
#  8. BUSINESS IMPACT CALCULATION
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 8 — BUSINESS IMPACT (₹ Revenue Story)")
print("="*60)

total_customers    = len(df)
avg_balance        = df["balance"].mean()
avg_salary         = df["estimated_salary"].mean()
churn_rate         = df["churn"].mean()
total_churners     = int(df["churn"].sum())

# Avg annual revenue per customer (simplified: 2% of balance + 1% of salary)
avg_annual_revenue_per_customer = avg_balance * 0.02 + avg_salary * 0.01
# Convert to INR (1 USD ≈ 84 INR)
INR_RATE = 84
avg_revenue_inr = avg_annual_revenue_per_customer * INR_RATE

total_revenue_at_risk = total_churners * avg_revenue_inr

savings_10pct = total_revenue_at_risk * 0.10
savings_20pct = total_revenue_at_risk * 0.20
savings_30pct = total_revenue_at_risk * 0.30

print(f"\n  📊 Dataset: {total_customers:,} customers")
print(f"  📊 Avg Balance       : ₹{avg_balance*INR_RATE:,.0f}")
print(f"  📊 Churn Rate        : {churn_rate*100:.1f}%  ({total_churners:,} customers)")
print(f"  📊 Avg Annual Revenue/Customer: ₹{avg_revenue_inr:,.0f}")
print(f"\n  💰 Total Revenue at Risk: ₹{total_revenue_at_risk/1e7:.2f} Cr")
print(f"\n  ✅ Reducing churn by 10% → SAVES ₹{savings_10pct/1e7:.2f} Cr/year")
print(f"  ✅ Reducing churn by 20% → SAVES ₹{savings_20pct/1e7:.2f} Cr/year")
print(f"  ✅ Reducing churn by 30% → SAVES ₹{savings_30pct/1e7:.2f} Cr/year")

# Business impact chart
fig, ax = plt.subplots(figsize=(7, 4), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)
scenarios = ["10% Reduction\n(Conservative)", "20% Reduction\n(Moderate)", "30% Reduction\n(Aggressive)"]
savings   = [savings_10pct/1e7, savings_20pct/1e7, savings_30pct/1e7]
colors_bi = ["#42A5F5", "#AB47BC", "#E53935"]
bars = ax.bar(scenarios, savings, color=colors_bi, edgecolor="white", linewidth=0.5, width=0.55)
for bar, val in zip(bars, savings):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"₹{val:.2f} Cr", ha="center", color=TEXT_COLOR, fontsize=12, fontweight="bold")
ax.set_title("Business Impact: Churn Reduction Savings", color=TEXT_COLOR, fontsize=13, fontweight="bold")
ax.set_ylabel("Annual Savings (₹ Crore)", color=TEXT_COLOR)
ax.tick_params(colors=TEXT_COLOR)
ax.spines[["top","right"]].set_color(GRID_COLOR)
ax.set_ylim(0, max(savings) * 1.25)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "09_business_impact.png"), dpi=150, bbox_inches="tight",
            facecolor=BG_COLOR)
plt.close()
print("  ✔ Saved: 09_business_impact.png")

# ─────────────────────────────────────────────
#  9. SAVE MODEL ARTIFACTS
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 9 — SAVING MODEL ARTIFACTS")
print("="*60)

joblib.dump(best_rf,  os.path.join(MODEL_DIR, "random_forest_tuned.pkl"))
joblib.dump(scaler,   os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(FEATURES, os.path.join(MODEL_DIR, "feature_names.pkl"))

import json
meta = {
    "best_model":      "Random Forest (Tuned)",
    "best_params":     best_params,
    "auc":             round(tuned_auc, 4),
    "accuracy":        round(accuracy_score(y_test, y_pred_tuned), 4),
    "f1_score":        round(f1_score(y_test, y_pred_tuned), 4),
    "total_customers": total_customers,
    "churn_rate_pct":  round(churn_rate*100, 2),
    "revenue_at_risk_cr": round(total_revenue_at_risk/1e7, 2),
    "savings_10pct_cr":   round(savings_10pct/1e7, 2),
    "savings_20pct_cr":   round(savings_20pct/1e7, 2),
    "features":        FEATURES,
}
with open(os.path.join(MODEL_DIR, "model_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"  ✔ Saved: random_forest_tuned.pkl")
print(f"  ✔ Saved: scaler.pkl")
print(f"  ✔ Saved: model_meta.json")

# Print final classification report
print("\n" + "="*60)
print(f"  FINAL CLASSIFICATION REPORT — {best_name} (Tuned)")
print("="*60)
print(classification_report(y_test, y_pred_tuned, target_names=["Stayed","Churned"]))

print("\n✅ Pipeline complete! All charts saved to /outputs/  |  Models saved to /models/\n")
