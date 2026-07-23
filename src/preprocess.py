"""Preprocessing pipeline for the Telco Customer Churn dataset.

Design principles:
  * Everything is a scikit-learn ColumnTransformer wrapped in a Pipeline. This
    means the same object handles imputation, encoding, and scaling — and it
    can be pickled and reused inside the Streamlit app without re-implementing
    the transforms.
  * We NEVER fit on the full dataset. The `split_features_target` and
    `build_preprocessor` functions are separate on purpose so the caller must
    split first, then fit. See `if __name__ == '__main__'` below for the
    correct order.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import get_feature_lists


TARGET = "Churn"
RANDOM_STATE = 42


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate the feature matrix X from the target vector y."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Build (but do NOT fit) the preprocessing ColumnTransformer.

    Numeric columns → median imputation + standard scaling.
    Categorical columns → most-frequent imputation + one-hot encoding
    (dense output so it plays nicely with XGBoost and SHAP).

    The caller is responsible for calling `.fit()` on training data only.
    """
    numeric, categorical = get_feature_lists(df)

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ]
    )


def get_train_test_split(df: pd.DataFrame, test_size: float = 0.2):
    """Stratified train/test split. Stratification keeps the churn rate the
    same in both sets — important because the target is imbalanced (26.5%)."""
    X, y = split_features_target(df)
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )


if __name__ == "__main__":
    # Sanity check: demonstrate the correct split-then-fit order.
    from .data import load_raw, clean_data

    df = clean_data(load_raw())
    X_train, X_test, y_train, y_test = get_train_test_split(df)

    print(f"Train shape: {X_train.shape}, churn rate: {y_train.mean():.3f}")
    print(f"Test  shape: {X_test.shape}, churn rate: {y_test.mean():.3f}")

    preprocessor = build_preprocessor(X_train)   # unfitted
    preprocessor.fit(X_train)                    # FIT ONLY ON TRAIN
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)    # test is only transformed

    print(f"Transformed train: {X_train_t.shape}")
    print(f"Transformed test : {X_test_t.shape}")
    print("Preprocessor fitted on training data only — no leakage.")
