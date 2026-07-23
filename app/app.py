"""Streamlit dashboard for the Telco Churn model.

Run with:  streamlit run app/app.py

A user fills in a single customer's attributes, and the app returns:
  1. the model's churn probability,
  2. a plain-English risk label,
  3. the top factors pushing the prediction up or down (from SHAP).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable when Streamlit runs this file directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

from src.data import get_feature_lists

MODEL_PATH = ROOT / "models" / "churn_model.joblib"


# --------------------------------------------------------------------------- #
# Caching: load model once per process; recomputing on every rerun is wasteful.
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reference_frame() -> pd.DataFrame:
    """A tiny reference frame we use to enumerate valid categorical values in
    the form. Loaded once, cached across reruns."""
    from src.data import clean_data, load_raw
    return clean_data(load_raw()).drop(columns=["Churn"])


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
)

st.title("Customer Churn Predictor")
st.caption(
    "Enter a customer's account details and get a churn-risk score, plus the "
    "top factors driving that score. Model: XGBoost trained on the IBM Telco "
    "Customer Churn dataset (7,043 customers, held-out ROC-AUC ≈ 0.84)."
)

pipe = load_pipeline()
ref = load_reference_frame()
numeric_cols, categorical_cols = get_feature_lists(ref)

# ----- Sidebar form ----- #
st.sidebar.header("Customer profile")

user_input: dict[str, object] = {}

with st.sidebar.form("customer_form"):
    st.markdown("**Account**")
    user_input["tenure"] = st.slider("Tenure (months)", 0, 72, 12)
    user_input["Contract"] = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    user_input["PaperlessBilling"] = st.selectbox("Paperless billing", ["Yes", "No"])
    user_input["PaymentMethod"] = st.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    user_input["MonthlyCharges"] = st.number_input("Monthly charges ($)", 0.0, 200.0, 70.0, step=5.0)
    user_input["TotalCharges"] = st.number_input("Total charges ($)", 0.0, 10000.0, 840.0, step=50.0)

    st.markdown("**Demographics**")
    user_input["gender"] = st.selectbox("Gender", ["Female", "Male"])
    user_input["SeniorCitizen"] = int(st.selectbox("Senior citizen", ["No", "Yes"]) == "Yes")
    user_input["Partner"] = st.selectbox("Has partner", ["No", "Yes"])
    user_input["Dependents"] = st.selectbox("Has dependents", ["No", "Yes"])

    st.markdown("**Services**")
    user_input["PhoneService"] = st.selectbox("Phone service", ["Yes", "No"])
    user_input["MultipleLines"] = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
    user_input["InternetService"] = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    for svc in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                "StreamingTV", "StreamingMovies"]:
        user_input[svc] = st.selectbox(svc, ["No", "Yes", "No internet service"])

    submitted = st.form_submit_button("Predict churn risk", use_container_width=True)


# ----- Prediction ----- #
if not submitted:
    st.info("Fill in the sidebar and click **Predict churn risk** to see a result.")
    st.stop()

# Build a single-row DataFrame in the exact column order the pipeline expects.
row = pd.DataFrame([user_input])[ref.columns]

proba = float(pipe.predict_proba(row)[0, 1])

# ----- Headline ----- #
col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Churn probability", f"{proba:.1%}")

if proba >= 0.70:
    band, colour = "High risk", "🔴"
elif proba >= 0.40:
    band, colour = "Medium risk", "🟡"
else:
    band, colour = "Low risk", "🟢"

with col2:
    st.markdown(f"### {colour} {band}")
    if band == "High risk":
        st.write("Recommend a proactive retention offer within the next billing cycle.")
    elif band == "Medium risk":
        st.write("Worth monitoring. Consider a light-touch check-in.")
    else:
        st.write("No action needed. Customer looks stable.")

st.progress(proba, text=f"Model confidence: {proba:.1%} churn")

# ----- Local SHAP explanation ----- #
st.subheader("What drove this prediction?")

preprocessor = pipe.named_steps["prep"]
model = pipe.named_steps["model"]

row_t = preprocessor.transform(row)
feature_names = list(preprocessor.get_feature_names_out())

# XGBoost native SHAP (avoids a version-compat bug in shap.TreeExplainer + XGB 3.x).
booster = model.get_booster()
contribs = booster.predict(xgb.DMatrix(row_t), pred_contribs=True)[0]
shap_row = contribs[:-1]  # last entry is the bias/base value


def _pretty(name: str) -> str:
    return name.replace("num__", "").replace("cat__", "").replace("_", " ")


ranked = sorted(zip(feature_names, shap_row), key=lambda x: abs(x[1]), reverse=True)[:8]

up = [(n, v) for n, v in ranked if v > 0]
down = [(n, v) for n, v in ranked if v < 0]

c1, c2 = st.columns(2)
with c1:
    st.markdown("**🔺 Pushing churn UP**")
    if up:
        st.dataframe(
            pd.DataFrame(
                [{"Factor": _pretty(n), "Impact": round(float(v), 3)} for n, v in up]
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.write("_None among the top factors._")
with c2:
    st.markdown("**🔻 Pushing churn DOWN**")
    if down:
        st.dataframe(
            pd.DataFrame(
                [{"Factor": _pretty(n), "Impact": round(float(v), 3)} for n, v in down]
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.write("_None among the top factors._")

st.caption(
    "Impact values are SHAP contributions in log-odds space. Larger absolute "
    "values mean the feature moved the prediction more, in the direction shown."
)

with st.expander("Global feature importance (whole dataset)"):
    shap_summary_path = ROOT / "models" / "shap_summary.png"
    if shap_summary_path.exists():
        st.image(str(shap_summary_path), caption="Top features by mean |SHAP| across the test set")
    else:
        st.write("Run `python -m src.explain` to generate the global SHAP plots.")
