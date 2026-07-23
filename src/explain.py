"""SHAP explainability for the trained churn model.

Produces:
  * models/shap_summary.png       — global feature importance (bar)
  * models/shap_beeswarm.png      — global effect direction + spread
  * models/shap_local_churner.png — one high-risk customer explained
  * models/shap_local_stayer.png  — one low-risk customer explained
  * models/top_features.json      — top-10 feature ranking used by the app
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # headless — needed on Windows without a display server
import matplotlib.pyplot as plt
import numpy as np
import shap

from .data import clean_data, load_raw
from .preprocess import get_train_test_split

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def get_feature_names(fitted_preprocessor) -> list[str]:
    """Extract the post-transform feature names from a fitted ColumnTransformer."""
    return list(fitted_preprocessor.get_feature_names_out())


def main():
    pipe = joblib.load(MODELS_DIR / "churn_model.joblib")
    preprocessor = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]

    df = clean_data(load_raw())
    X_train, X_test, _, y_test = get_train_test_split(df)

    # SHAP wants the transformed matrix — the same one the tree model sees.
    X_test_t = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)

    # XGBoost has native SHAP via pred_contribs — avoids a shap/xgboost
    # version-compatibility bug in TreeExplainer with XGBoost 3.x.
    import xgboost as xgb
    booster = model.get_booster()
    contribs = booster.predict(xgb.DMatrix(X_test_t), pred_contribs=True)
    shap_values = contribs[:, :-1]           # last column is the bias
    expected_value = float(contribs[0, -1])   # same bias for every row
    explanation = shap.Explanation(
        values=shap_values,
        base_values=np.full(shap_values.shape[0], expected_value),
        data=X_test_t,
        feature_names=feature_names,
    )

    # ---- Global: bar summary ----
    plt.figure()
    shap.summary_plot(
        shap_values, X_test_t, feature_names=feature_names,
        plot_type="bar", show=False, max_display=15,
    )
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "shap_summary.png", dpi=140, bbox_inches="tight")
    plt.close()

    # ---- Global: beeswarm ----
    plt.figure()
    shap.summary_plot(
        shap_values, X_test_t, feature_names=feature_names,
        show=False, max_display=15,
    )
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "shap_beeswarm.png", dpi=140, bbox_inches="tight")
    plt.close()

    # ---- Local: one high-risk + one low-risk customer ----
    proba = model.predict_proba(X_test_t)[:, 1]
    high_idx = int(np.argmax(proba))
    low_idx = int(np.argmin(proba))

    for idx, tag in [(high_idx, "churner"), (low_idx, "stayer")]:
        plt.figure()
        shap.plots.waterfall(explanation[idx], max_display=12, show=False)
        plt.tight_layout()
        plt.savefig(MODELS_DIR / f"shap_local_{tag}.png", dpi=140, bbox_inches="tight")
        plt.close()
        print(f"{tag}: predicted churn probability = {proba[idx]:.3f}")

    # ---- Top-10 features by mean |SHAP| — used by the Streamlit app ----
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    top = [
        {"feature": feature_names[i], "mean_abs_shap": float(mean_abs[i])}
        for i in order[:10]
    ]
    (MODELS_DIR / "top_features.json").write_text(json.dumps(top, indent=2))

    print("\nTop 10 features driving churn (global):")
    for row in top:
        print(f"  {row['feature']:40s}  {row['mean_abs_shap']:.4f}")


if __name__ == "__main__":
    main()
