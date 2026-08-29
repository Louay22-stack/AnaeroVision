# ============================================================
# TASK 1 — XGBoost Hyperparameter Tuning
# Uses finalized preprocessing outputs
# Target: Methane_Yield_mL_CH4_gVS
# ============================================================

import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ------------------------------------------------------------
# 0. Paths
# ------------------------------------------------------------

script_dir = Path(__file__).resolve().parent

preprocessing_dir = script_dir / "task1_preprocessing_outputs"
output_dir = script_dir / "task1_xgboost_tuning_outputs"
output_dir.mkdir(exist_ok=True)

if not preprocessing_dir.exists():
    raise FileNotFoundError(
        "Preprocessing folder not found. Run task1_finalize_preprocessing.py first."
    )

print("Loading preprocessed data from:")
print(preprocessing_dir.resolve())

print("\nXGBoost tuning outputs will be saved in:")
print(output_dir.resolve())

# ------------------------------------------------------------
# 1. Load preprocessed train/test data
# ------------------------------------------------------------

X_train = pd.read_csv(preprocessing_dir / "X_train_task1_preprocessed.csv")
X_test = pd.read_csv(preprocessing_dir / "X_test_task1_preprocessed.csv")

y_train = pd.read_csv(preprocessing_dir / "y_train_task1_processed.csv").squeeze()
y_test = pd.read_csv(preprocessing_dir / "y_test_task1_processed.csv").squeeze()

y_train_original = pd.read_csv(preprocessing_dir / "y_train_task1_original.csv").squeeze()
y_test_original = pd.read_csv(preprocessing_dir / "y_test_task1_original.csv").squeeze()

with open(preprocessing_dir / "task1_preprocessing_summary.json", "r", encoding="utf-8") as f:
    preprocessing_summary = json.load(f)

LOG_TRANSFORM_APPLIED = preprocessing_summary["log_transform_applied"]

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("\nFeature columns:")
print(X_train.columns.tolist())

# ------------------------------------------------------------
# 2. Define base XGBoost model
# ------------------------------------------------------------

base_xgb = XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=42,
    tree_method="hist",

    # Important:
    # n_jobs=1 here because RandomizedSearchCV will parallelize the search.
    # This avoids too much CPU oversubscription on Windows.
    n_jobs=1
)

# ------------------------------------------------------------
# 3. Hyperparameter search space
# ------------------------------------------------------------

param_distributions = {
    "n_estimators": [200, 300, 500, 700, 1000],
    "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1],
    "max_depth": [2, 3, 4, 5, 6],
    "min_child_weight": [1, 2, 3, 5, 10],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "gamma": [0, 0.01, 0.05, 0.1, 0.2],
    "reg_alpha": [0, 0.001, 0.01, 0.05, 0.1],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0]
}

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

search = RandomizedSearchCV(
    estimator=base_xgb,
    param_distributions=param_distributions,
    n_iter=30,
    scoring="r2",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=2,
    return_train_score=True
)

# ------------------------------------------------------------
# 4. Run tuning
# ------------------------------------------------------------

print("\nStarting XGBoost hyperparameter tuning...")
search.fit(X_train, y_train)

print("\nXGBoost tuning completed.")

print("\nBest CV R²:")
print(search.best_score_)

print("\nBest parameters:")
print(search.best_params_)

cv_results_df = pd.DataFrame(search.cv_results_)
cv_results_df.to_csv(output_dir / "xgboost_tuning_cv_results.csv", index=False)

# ------------------------------------------------------------
# ------------------------------------------------------------
# 5. Evaluate best model on train and test sets
# ------------------------------------------------------------

best_xgb_model = search.best_estimator_

y_train_pred = best_xgb_model.predict(X_train)
y_test_pred = best_xgb_model.predict(X_test)

# Back-transform if log transform was applied earlier
if LOG_TRANSFORM_APPLIED:
    y_train_pred_eval = np.expm1(y_train_pred)
    y_test_pred_eval = np.expm1(y_test_pred)

    y_train_eval = y_train_original
    y_test_eval = y_test_original
else:
    y_train_pred_eval = y_train_pred
    y_test_pred_eval = y_test_pred

    y_train_eval = y_train
    y_test_eval = y_test

train_r2 = r2_score(y_train_eval, y_train_pred_eval)
test_r2 = r2_score(y_test_eval, y_test_pred_eval)

train_rmse = np.sqrt(mean_squared_error(y_train_eval, y_train_pred_eval))
test_rmse = np.sqrt(mean_squared_error(y_test_eval, y_test_pred_eval))

train_mae = mean_absolute_error(y_train_eval, y_train_pred_eval)
test_mae = mean_absolute_error(y_test_eval, y_test_pred_eval)

print("\nTuned XGBoost Regression Metrics:")
print(f"Train R²:   {train_r2:.4f}")
print(f"Test R²:    {test_r2:.4f}")
print(f"Train RMSE: {train_rmse:.4f} mL CH4/gVS")
print(f"Test RMSE:  {test_rmse:.4f} mL CH4/gVS")
print(f"Train MAE:  {train_mae:.4f} mL CH4/gVS")
print(f"Test MAE:   {test_mae:.4f} mL CH4/gVS")

# ------------------------------------------------------------
# 6. Save metrics
# ------------------------------------------------------------

metrics = {
    "model": "Tuned XGBoost Regressor",
    "best_cv_r2": float(search.best_score_),
    "best_params": search.best_params_,
    "train_r2": float(train_r2),
    "test_r2": float(test_r2),
    "train_rmse": float(train_rmse),
    "test_rmse": float(test_rmse),
    "train_mae": float(train_mae),
    "test_mae": float(test_mae),
    "log_transform_applied": LOG_TRANSFORM_APPLIED
}

with open(output_dir / "xgboost_tuned_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

# ------------------------------------------------------------
# 7. Save predictions
# ------------------------------------------------------------

test_predictions = pd.DataFrame({
    "Actual_Methane_Yield_mL_CH4_gVS": y_test_eval,
    "Predicted_Methane_Yield_mL_CH4_gVS": y_test_pred_eval,
    "Residual": y_test_eval - y_test_pred_eval
})

test_predictions.to_csv(
    output_dir / "xgboost_tuned_test_predictions.csv",
    index=False
)

train_predictions = pd.DataFrame({
    "Actual_Methane_Yield_mL_CH4_gVS": y_train_eval,
    "Predicted_Methane_Yield_mL_CH4_gVS": y_train_pred_eval,
    "Residual": y_train_eval - y_train_pred_eval
})

train_predictions.to_csv(
    output_dir / "xgboost_tuned_train_predictions.csv",
    index=False
)

# ------------------------------------------------------------
# 8. Feature importance
# ------------------------------------------------------------

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": best_xgb_model.feature_importances_
}).sort_values("Importance", ascending=False)

feature_importance.to_csv(
    output_dir / "xgboost_tuned_feature_importance.csv",
    index=False
)

print("\nTop 15 tuned XGBoost feature importances:")
print(feature_importance.head(15))

top_features = feature_importance.head(15).sort_values("Importance")

plt.figure(figsize=(9, 7))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.xlabel("Feature importance")
plt.title("Tuned XGBoost Feature Importance — Top 15")
plt.tight_layout()
plt.savefig(output_dir / "xgboost_tuned_feature_importance_top15.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 9. Predicted vs actual plot
# ------------------------------------------------------------

plt.figure(figsize=(7, 7))
plt.scatter(y_test_eval, y_test_pred_eval, alpha=0.7)

min_value = min(y_test_eval.min(), y_test_pred_eval.min())
max_value = max(y_test_eval.max(), y_test_pred_eval.max())

plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")

plt.xlabel("Actual methane yield (mL CH4/gVS)")
plt.ylabel("Predicted methane yield (mL CH4/gVS)")
plt.title("Tuned XGBoost — Predicted vs Actual")
plt.tight_layout()
plt.savefig(output_dir / "xgboost_tuned_predicted_vs_actual.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 10. Residual plot
# ------------------------------------------------------------

residuals = y_test_eval - y_test_pred_eval

plt.figure(figsize=(8, 6))
plt.scatter(y_test_pred_eval, residuals, alpha=0.7)
plt.axhline(0, linestyle="--")

plt.xlabel("Predicted methane yield (mL CH4/gVS)")
plt.ylabel("Residuals")
plt.title("Tuned XGBoost — Residuals vs Predicted")
plt.tight_layout()
plt.savefig(output_dir / "xgboost_tuned_residuals_vs_predicted.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 11. Save tuned model
# ------------------------------------------------------------

joblib.dump(best_xgb_model, output_dir / "xgboost_tuned_task1_model.pkl")

# ------------------------------------------------------------
# 12. Final message
# ------------------------------------------------------------

print("\n============================================================")
print("XGBOOST HYPERPARAMETER TUNING COMPLETED SUCCESSFULLY")
print("============================================================")
print("Saved outputs in:")
print(output_dir.resolve())

print("\nFiles created:")
for file in sorted(output_dir.iterdir()):
    print("-", file.name)