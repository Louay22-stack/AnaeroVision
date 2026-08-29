# ============================================================
# TASK 1 — ANN / MLPRegressor Hyperparameter Tuning
# Uses finalized preprocessing outputs
# Target: Methane_Yield_mL_CH4_gVS
# ============================================================

import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ------------------------------------------------------------
# 0. Paths
# ------------------------------------------------------------

script_dir = Path(__file__).resolve().parent

preprocessing_dir = script_dir / "task1_preprocessing_outputs"
output_dir = script_dir / "task1_ann_tuning_outputs"
output_dir.mkdir(exist_ok=True)

if not preprocessing_dir.exists():
    raise FileNotFoundError(
        "Preprocessing folder not found. Run task1_finalize_preprocessing.py first."
    )

print("Loading preprocessed data from:")
print(preprocessing_dir.resolve())

print("\nANN tuning outputs will be saved in:")
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
# 2. Define ANN model
# ------------------------------------------------------------

base_ann = MLPRegressor(
    solver="adam",
    max_iter=2000,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=30,
    random_state=42
)

# Scale target y inside the model
ann_model = TransformedTargetRegressor(
    regressor=base_ann,
    transformer=StandardScaler()
)

# ------------------------------------------------------------
# 3. Hyperparameter search space
# ------------------------------------------------------------

param_distributions = {
    "regressor__hidden_layer_sizes": [
        (64, 32),
        (128, 64),
        (128, 64, 32),
        (256, 128),
        (256, 128, 64)
    ],
    "regressor__activation": ["relu", "tanh"],
    "regressor__alpha": [0.0001, 0.001, 0.01, 0.05],
    "regressor__learning_rate_init": [0.0005, 0.001, 0.005],
    "regressor__batch_size": [16, 32, 64]
}

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

search = RandomizedSearchCV(
    estimator=ann_model,
    param_distributions=param_distributions,
    n_iter=20,
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

print("\nStarting ANN hyperparameter tuning...")
search.fit(X_train, y_train)

print("\nANN tuning completed.")

print("\nBest CV R²:")
print(search.best_score_)

print("\nBest parameters:")
print(search.best_params_)

# Save full CV results
cv_results_df = pd.DataFrame(search.cv_results_)
cv_results_df.to_csv(output_dir / "ann_tuning_cv_results.csv", index=False)

# ------------------------------------------------------------
# 5. Evaluate best model on train and test sets
# ------------------------------------------------------------

best_ann_model = search.best_estimator_

y_train_pred = best_ann_model.predict(X_train)
y_test_pred = best_ann_model.predict(X_test)

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

print("\nTuned ANN Regression Metrics:")
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
    "model": "Tuned ANN / MLPRegressor",
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

with open(output_dir / "ann_tuned_metrics.json", "w", encoding="utf-8") as f:
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
    output_dir / "ann_tuned_test_predictions.csv",
    index=False
)

train_predictions = pd.DataFrame({
    "Actual_Methane_Yield_mL_CH4_gVS": y_train_eval,
    "Predicted_Methane_Yield_mL_CH4_gVS": y_train_pred_eval,
    "Residual": y_train_eval - y_train_pred_eval
})

train_predictions.to_csv(
    output_dir / "ann_tuned_train_predictions.csv",
    index=False
)

# ------------------------------------------------------------
# 8. Extract final ANN details
# ------------------------------------------------------------

final_ann = best_ann_model.regressor_

ann_details = {
    "n_iter": int(final_ann.n_iter_),
    "final_loss": float(final_ann.loss_),
    "hidden_layer_sizes": str(final_ann.hidden_layer_sizes),
    "activation": final_ann.activation,
    "alpha": float(final_ann.alpha),
    "learning_rate_init": float(final_ann.learning_rate_init),
    "batch_size": str(final_ann.batch_size)
}

with open(output_dir / "ann_tuned_training_details.json", "w", encoding="utf-8") as f:
    json.dump(ann_details, f, indent=4)

print("\nFinal tuned ANN details:")
print(ann_details)

# ------------------------------------------------------------
# 9. Loss curve of best ANN
# ------------------------------------------------------------

if hasattr(final_ann, "loss_curve_"):
    plt.figure(figsize=(8, 6))
    plt.plot(final_ann.loss_curve_)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Tuned ANN Training Loss Curve")
    plt.tight_layout()
    plt.savefig(output_dir / "ann_tuned_loss_curve.png", dpi=300)
    plt.show()

# ------------------------------------------------------------
# 10. Predicted vs actual plot
# ------------------------------------------------------------

plt.figure(figsize=(7, 7))
plt.scatter(y_test_eval, y_test_pred_eval, alpha=0.7)

min_value = min(y_test_eval.min(), y_test_pred_eval.min())
max_value = max(y_test_eval.max(), y_test_pred_eval.max())

plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")

plt.xlabel("Actual methane yield (mL CH4/gVS)")
plt.ylabel("Predicted methane yield (mL CH4/gVS)")
plt.title("Tuned ANN — Predicted vs Actual")
plt.tight_layout()
plt.savefig(output_dir / "ann_tuned_predicted_vs_actual.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 11. Residual plot
# ------------------------------------------------------------

residuals = y_test_eval - y_test_pred_eval

plt.figure(figsize=(8, 6))
plt.scatter(y_test_pred_eval, residuals, alpha=0.7)
plt.axhline(0, linestyle="--")

plt.xlabel("Predicted methane yield (mL CH4/gVS)")
plt.ylabel("Residuals")
plt.title("Tuned ANN — Residuals vs Predicted")
plt.tight_layout()
plt.savefig(output_dir / "ann_tuned_residuals_vs_predicted.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 12. Save tuned model
# ------------------------------------------------------------

joblib.dump(best_ann_model, output_dir / "ann_tuned_task1_model.pkl")

print("\n============================================================")
print("ANN HYPERPARAMETER TUNING COMPLETED SUCCESSFULLY")
print("============================================================")
print("Saved outputs in:")
print(output_dir.resolve())

print("\nFiles created:")
for file in sorted(output_dir.iterdir()):
    print("-", file.name)