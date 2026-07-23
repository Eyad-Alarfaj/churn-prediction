"""Data loading and cleaning for the Telco Customer Churn dataset.

Kept intentionally small and pure: functions here take a path or DataFrame and
return a DataFrame, so they can be reused from the notebook, the training
script, and the Streamlit app.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "telco_churn.csv"
PROCESSED_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "telco_clean.csv"


def load_raw(path: Path | str = RAW_PATH) -> pd.DataFrame:
    """Load the raw Telco CSV as-is."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the Telco dataset.

    Steps:
      1. Drop the customerID column (identifier, not a feature).
      2. Coerce TotalCharges to numeric (it ships as string with blank rows
         for brand-new customers whose tenure is 0).
      3. Impute the resulting NaNs with 0 — these are tenure-0 customers who
         have not been billed yet, so 0 is the true value, not a guess.
      4. Map the binary target Churn to {0, 1}.
    """
    df = df.copy()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

    return df


def get_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_features, categorical_features), excluding the target."""
    features = [c for c in df.columns if c != "Churn"]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in features if c not in numeric]
    return numeric, categorical


if __name__ == "__main__":
    raw = load_raw()
    clean = clean_data(raw)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved cleaned data: {PROCESSED_PATH} shape={clean.shape}")
