# Project: Customer Churn Prediction

## Goal
End-to-end ML project: predict telco customer churn, deploy an interactive dashboard, explain predictions with SHAP.

## Stack
Python 3.12, pandas, scikit-learn, XGBoost, SHAP, Streamlit, matplotlib/seaborn.

## Conventions
- All data cleaning/feature code lives in `src/` as importable functions.
- Notebook is for EDA and storytelling only, not for defining core logic.
- Never fit any transformer on the full dataset before train/test split (avoid data leakage).
- Every model result must come from cross-validation, not a single split.
- Handle class imbalance explicitly.

## Commands
- Install: `pip install -r requirements.txt`
- Run dashboard: `streamlit run app/app.py`
