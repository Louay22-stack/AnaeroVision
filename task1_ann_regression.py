# ============================================================
# TASK 1 — ANN / MLP REGRESSION
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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ------------------------------------------------------------
# 0. Paths
# ------------------------------------------------------------

script_dir = Path(__file__).resolve().parent

preprocessing_dir = script_dir / "task1_preprocessing_outputs"
output_dir = script_dir / "task1_ann_outputs"
output_dir.mkdir(exist_ok=True)

if not preprocessing_dir.exists():
    raise FileNotFoundError(
        "Preprocessing folder not found. Run task1_finalize_preprocessing.py first."
    )

print("Loading preprocessed data from:")
print(preprocessing_dir.resolve())

print("\nANN outputs will be saved in:")
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
# 2. Scale target y for ANN stability
# ------------------------------------------------------------

y_scaler = StandardScaler()

y_train_scaled = y_scaler.fit_transform(
    y_train.values.reshape(-1, 1)
).ravel()

joblib.dump(y_scaler, output_dir / "ann_y_scaler_task1.pkl")

print("\nTarget scaling completed for ANN.")

# ------------------------------------------------------------
# 3. Train ANN / MLPRegressor
# ------------------------------------------------------------

ann_reg = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    alpha=0.001,
    batch_size=32,
    learning_rate_init=0.001,
    max_iter=2000,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=30,
    random_state=42
)

ann_reg.fit(X_train, y_train_scaled)

print("\nANN training completed.")
print("Iterations:", ann_reg.n_iter_)
print("Final loss:", ann_reg.loss_)

# ------------------------------------------------------------
# 4. Predictions
# ------------------------------------------------------------

y_train_pred_scaled = ann_reg.predict(X_train)
y_test_pred_scaled = ann_reg.predict(X_test)

# Convert ANN predictions back from scaled target units
y_train_pred = y_scaler.inverse_transform(
    y_train_pred_scaled.reshape(-1, 1)
).ravel()

y_test_pred = y_scaler.inverse_transform(
    y_test_pred_scaled.reshape(-1, 1)
).ravel()

# Back-transform log target if log transform had been applied
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

# ------------------------------------------------------------
# 5. Evaluation metrics
# ------------------------------------------------------------

train_r2 = r2_score(y_train_eval, y_train_pred_eval)
test_r2 = r2_score(y_test_eval, y_test_pred_eval)

train_rmse = np.sqrt(mean_squared_error(y_train_eval, y_train_pred_eval))
test_rmse = np.sqrt(mean_squared_error(y_test_eval, y_test_pred_eval))

train_mae = mean_absolute_error(y_train_eval, y_train_pred_eval)
test_mae = mean_absolute_error(y_test_eval, y_test_pred_eval)

metrics = {
    "model": "ANN / MLPRegressor",
    "hidden_layer_sizes": [128, 64],
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.001,
    "batch_size": 32,
    "learning_rate_init": 0.001,
    "max_iter": 2000,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "n_iter_no_change": 30,
    "n_iter_actual": ann_reg.n_iter_,
    "final_loss": float(ann_reg.loss_),
    "train_r2": train_r2,
    "test_r2": test_r2,
    "train_rmse": train_rmse,
    "test_rmse": test_rmse,
    "train_mae": train_mae,
    "test_mae": test_mae,
    "log_transform_applied": LOG_TRANSFORM_APPLIED
}

print("\nANN Regression Metrics:")
print(f"Train R²:   {train_r2:.4f}")
print(f"Test R²:    {test_r2:.4f}")
print(f"Train RMSE: {train_rmse:.4f} mL CH4/gVS")
print(f"Test RMSE:  {test_rmse:.4f} mL CH4/gVS")
print(f"Train MAE:  {train_mae:.4f} mL CH4/gVS")
print(f"Test MAE:   {test_mae:.4f} mL CH4/gVS")

with open(output_dir / "ann_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

# ------------------------------------------------------------
# 6. Save predictions
# ------------------------------------------------------------

test_predictions = pd.DataFrame({
    "Actual_Methane_Yield_mL_CH4_gVS": y_test_eval,
    "Predicted_Methane_Yield_mL_CH4_gVS": y_test_pred_eval,
    "Residual": y_test_eval - y_test_pred_eval
})

test_predictions.to_csv(
    output_dir / "ann_test_predictions.csv",
    index=False
)

train_predictions = pd.DataFrame({
    "Actual_Methane_Yield_mL_CH4_gVS": y_train_eval,
    "Predicted_Methane_Yield_mL_CH4_gVS": y_train_pred_eval,
    "Residual": y_train_eval - y_train_pred_eval
})

train_predictions.to_csv(
    output_dir / "ann_train_predictions.csv",
    index=False
)

# ------------------------------------------------------------
# 7. Loss curve
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))
plt.plot(ann_reg.loss_curve_)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("ANN Training Loss Curve")
plt.tight_layout()
plt.savefig(output_dir / "ann_loss_curve.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 8. Predicted vs actual plot
# ------------------------------------------------------------

plt.figure(figsize=(7, 7))
plt.scatter(y_test_eval, y_test_pred_eval, alpha=0.7)

min_value = min(y_test_eval.min(), y_test_pred_eval.min())
max_value = max(y_test_eval.max(), y_test_pred_eval.max())

plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")

plt.xlabel("Actual methane yield (mL CH4/gVS)")
plt.ylabel("Predicted methane yield (mL CH4/gVS)")
plt.title("ANN — Predicted vs Actual")
plt.tight_layout()
plt.savefig(output_dir / "ann_predicted_vs_actual.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 9. Residual plot
# ------------------------------------------------------------

residuals = y_test_eval - y_test_pred_eval

plt.figure(figsize=(8, 6))
plt.scatter(y_test_pred_eval, residuals, alpha=0.7)
plt.axhline(0, linestyle="--")

plt.xlabel("Predicted methane yield (mL CH4/gVS)")
plt.ylabel("Residuals")
plt.title("ANN — Residuals vs Predicted")
plt.tight_layout()
plt.savefig(output_dir / "ann_residuals_vs_predicted.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 10. Save model
# ------------------------------------------------------------

joblib.dump(ann_reg, output_dir / "ann_task1_model.pkl")

print("\n============================================================")
print("ANN REGRESSION COMPLETED SUCCESSFULLY")
print("============================================================")
print("Saved outputs in:")
print(output_dir.resolve())

print("\nFiles created:")
for file in sorted(output_dir.iterdir()):
    print("-", file.name)