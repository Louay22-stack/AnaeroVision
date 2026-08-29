# ============================================================
# TASK 2 — XGBOOST CLASSIFIER
# Inhibition Risk Classification
# Target: Inhibition_Label_Numeric
# ============================================================

import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

# ------------------------------------------------------------
# 0. Paths
# ------------------------------------------------------------

script_dir = Path(__file__).resolve().parent

preprocessing_dir = script_dir / "task2_preprocessing_outputs"

output_dir = script_dir / "task2_xgboost_outputs"
output_dir.mkdir(exist_ok=True)

if not preprocessing_dir.exists():
    raise FileNotFoundError("Missing folder: task2_preprocessing_outputs")

# ------------------------------------------------------------
# 1. Load preprocessed Task 2 data
# ------------------------------------------------------------

X_train = pd.read_csv(preprocessing_dir / "X_train_task2_preprocessed.csv")
X_test = pd.read_csv(preprocessing_dir / "X_test_task2_preprocessed.csv")

y_train = pd.read_csv(preprocessing_dir / "y_train_task2.csv").squeeze()
y_test = pd.read_csv(preprocessing_dir / "y_test_task2.csv").squeeze()

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("\nTrain target distribution:")
print(y_train.value_counts().sort_index())

print("\nTest target distribution:")
print(y_test.value_counts().sort_index())

# ------------------------------------------------------------
# 2. Load class imbalance information
# ------------------------------------------------------------

with open(preprocessing_dir / "task2_class_imbalance_info.json", "r", encoding="utf-8") as f:
    imbalance_info = json.load(f)

scale_pos_weight = float(imbalance_info["scale_pos_weight_for_xgboost"])

print("\nscale_pos_weight used for XGBoost:")
print(scale_pos_weight)

# ------------------------------------------------------------
# 3. Train XGBoost Classifier
# ------------------------------------------------------------

xgb_clf = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.0,
    reg_alpha=0.01,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)

xgb_clf.fit(X_train, y_train)

# ------------------------------------------------------------
# 4. Predictions
# ------------------------------------------------------------

y_train_pred = xgb_clf.predict(X_train)
y_test_pred = xgb_clf.predict(X_test)

y_train_proba = xgb_clf.predict_proba(X_train)[:, 1]
y_test_proba = xgb_clf.predict_proba(X_test)[:, 1]

# ------------------------------------------------------------
# 5. Metrics
# ------------------------------------------------------------

metrics = {
    "train_accuracy": accuracy_score(y_train, y_train_pred),
    "test_accuracy": accuracy_score(y_test, y_test_pred),

    "train_precision_class_1": precision_score(y_train, y_train_pred, pos_label=1, zero_division=0),
    "test_precision_class_1": precision_score(y_test, y_test_pred, pos_label=1, zero_division=0),

    "train_recall_class_1": recall_score(y_train, y_train_pred, pos_label=1, zero_division=0),
    "test_recall_class_1": recall_score(y_test, y_test_pred, pos_label=1, zero_division=0),

    "train_f1_class_1": f1_score(y_train, y_train_pred, pos_label=1, zero_division=0),
    "test_f1_class_1": f1_score(y_test, y_test_pred, pos_label=1, zero_division=0),

    "train_roc_auc": roc_auc_score(y_train, y_train_proba),
    "test_roc_auc": roc_auc_score(y_test, y_test_proba),

    "train_pr_auc": average_precision_score(y_train, y_train_proba),
    "test_pr_auc": average_precision_score(y_test, y_test_proba)
}

print("\n============================================================")
print("XGBOOST CLASSIFIER — TASK 2 RESULTS")
print("============================================================")

for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

# ------------------------------------------------------------
# 6. Classification reports
# ------------------------------------------------------------

target_names = ["No inhibition", "Inhibition"]

train_report = classification_report(
    y_train,
    y_train_pred,
    target_names=target_names,
    zero_division=0
)

test_report = classification_report(
    y_test,
    y_test_pred,
    target_names=target_names,
    zero_division=0
)

print("\nTrain classification report:")
print(train_report)

print("\nTest classification report:")
print(test_report)

with open(output_dir / "xgb_task2_train_classification_report.txt", "w", encoding="utf-8") as f:
    f.write(train_report)

with open(output_dir / "xgb_task2_test_classification_report.txt", "w", encoding="utf-8") as f:
    f.write(test_report)

# ------------------------------------------------------------
# 7. Confusion matrix
# ------------------------------------------------------------

cm = confusion_matrix(y_test, y_test_pred)

print("\nTest confusion matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=target_names
)

fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, values_format="d")
plt.title("XGBoost — Test Confusion Matrix")
plt.tight_layout()
plt.savefig(output_dir / "xgb_task2_confusion_matrix.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 8. ROC curve
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_predictions(
    y_test,
    y_test_proba,
    ax=ax
)
plt.title("XGBoost — ROC Curve")
plt.tight_layout()
plt.savefig(output_dir / "xgb_task2_roc_curve.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 9. Precision-Recall curve
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 5))
PrecisionRecallDisplay.from_predictions(
    y_test,
    y_test_proba,
    ax=ax
)
plt.title("XGBoost — Precision-Recall Curve")
plt.tight_layout()
plt.savefig(output_dir / "xgb_task2_precision_recall_curve.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 10. Feature importance
# ------------------------------------------------------------

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": xgb_clf.feature_importances_
}).sort_values(by="Importance", ascending=False)

feature_importance.to_csv(
    output_dir / "xgb_task2_feature_importance.csv",
    index=False
)

print("\nTop feature importances:")
print(feature_importance.head(15))

plt.figure(figsize=(9, 6))
plt.barh(
    feature_importance["Feature"].head(15)[::-1],
    feature_importance["Importance"].head(15)[::-1]
)
plt.xlabel("Feature importance")
plt.title("XGBoost — Top 15 Feature Importances")
plt.tight_layout()
plt.savefig(output_dir / "xgb_task2_feature_importance_top15.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 11. Save predictions
# ------------------------------------------------------------

test_predictions = pd.DataFrame({
    "y_test_actual": y_test.values,
    "y_test_predicted": y_test_pred,
    "probability_inhibition": y_test_proba
})

test_predictions.to_csv(
    output_dir / "xgb_task2_test_predictions.csv",
    index=False
)

# ------------------------------------------------------------
# 12. Save model and metrics
# ------------------------------------------------------------

joblib.dump(xgb_clf, output_dir / "xgb_task2_model.pkl")

with open(output_dir / "xgb_task2_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

# ------------------------------------------------------------
# 13. Final message
# ------------------------------------------------------------

print("\n============================================================")
print("TASK 2 XGBOOST CLASSIFIER COMPLETED SUCCESSFULLY")
print("============================================================")
print("Saved outputs in:")
print(output_dir.resolve())

print("\nFiles created:")
for file in sorted(output_dir.iterdir()):
    print("-", file.name)