"""Train, compare, tune and save the churn-prediction model.

Why the choices in this file:
  * 3 models: Logistic Regression (transparent baseline), Random Forest
    (non-linear, robust), XGBoost (usual winner on tabular).
  * Stratified 5-fold CV — a single train/test split can be lucky/unlucky on
    an imbalanced target.
  * Class imbalance is handled with `class_weight='balanced'` (LR, RF) and
    `scale_pos_weight` (XGB), NOT SMOTE. Reasoning:
        - Class-weighting is done inside the CV fold, so it can't leak.
        - SMOTE would need to run inside every CV fold via imblearn.Pipeline;
          class-weighting achieves the same 'penalise missing a churner more'
          effect with less machinery and no synthetic rows.
  * Primary metric: **ROC-AUC** for ranking quality, plus **recall** because
    the business cost of missing a churner (losing them silently) is much
    higher than the cost of a false alarm (a wasted retention offer).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .data import clean_data, load_raw
from .preprocess import RANDOM_STATE, build_preprocessor, get_train_test_split

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(exist_ok=True)


def make_candidates(y_train: pd.Series) -> dict[str, Pipeline]:
    """Return three candidate model pipelines, all with class-imbalance handled."""
    # scale_pos_weight = (# negatives) / (# positives). This tells XGBoost to
    # weight the minority (churn) class inversely to its frequency.
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    candidates = {
        "logreg": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    return candidates


def cross_validate_all(X_train, y_train, preprocessor) -> pd.DataFrame:
    """Run stratified 5-fold CV over every candidate. Returns a summary DataFrame."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    rows = []
    for name, model in make_candidates(y_train).items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append({
            "model": name,
            **{m: scores[f"test_{m}"].mean() for m in scoring},
        })
    return pd.DataFrame(rows).set_index("model").round(4)


def tune_xgboost(X_train, y_train, preprocessor) -> Pipeline:
    """Small, sensible grid over XGBoost — good marginal wins, quick to run."""
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    base_pipe = Pipeline([
        ("prep", preprocessor),
        ("model", XGBClassifier(
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    grid = {
        "model__n_estimators": [200, 400, 600],
        "model__max_depth": [3, 4, 6],
        "model__learning_rate": [0.05, 0.1],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(base_pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0)
    search.fit(X_train, y_train)
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Best params    : {search.best_params_}")
    return search.best_estimator_


def evaluate_on_test(model: Pipeline, X_test, y_test) -> dict:
    """Final held-out evaluation — done ONCE at the very end."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
    }
    print("\n=== Held-out test set ===")
    for k, v in metrics.items():
        print(f"  {k:9s}: {v}")
    print("\n" + classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]))
    return metrics


def main():
    df = clean_data(load_raw())
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    preprocessor = build_preprocessor(X_train)

    print("=== Cross-validation results (5-fold stratified) ===")
    cv_scores = cross_validate_all(X_train, y_train, preprocessor)
    print(cv_scores)

    winner = cv_scores["roc_auc"].idxmax()
    print(f"\nBest-by-CV model: {winner}")

    print("\n=== Tuning XGBoost ===")
    best_model = tune_xgboost(X_train, y_train, preprocessor)

    test_metrics = evaluate_on_test(best_model, X_test, y_test)

    # Persist everything the app + Phase-4 SHAP notebook will need.
    joblib.dump(best_model, MODELS_DIR / "churn_model.joblib")
    (MODELS_DIR / "cv_results.csv").write_text(cv_scores.to_csv())
    (MODELS_DIR / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    print(f"\nSaved model to: {MODELS_DIR / 'churn_model.joblib'}")


if __name__ == "__main__":
    main()
