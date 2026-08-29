# ============================================================
# TASK 2 — LOGISTIC REGRESSION CLASSIFIER
# Inhibition Risk Classification
# Target: Inhibition_Label_Numeric
# ============================================================

import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.linear_model import LogisticRegression
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

output_dir = script_dir / "task2_logistic_regression_outputs"
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
# 2. Load class weights
# ------------------------------------------------------------

with open(preprocessing_dir / "task2_class_imbalance_info.json", "r", encoding="utf-8") as f:
    imbalance_info = json.load(f)

class_weight_dict = {
    int(k): float(v)
    for k, v in imbalance_info["class_weight_dict"].items()
}

print("\nClass weights used:")
print(class_weight_dict)

# ------------------------------------------------------------
# 3. Train Logistic Regression Classifier
# ------------------------------------------------------------

logreg_clf = LogisticRegression(
    penalty="l2",
    C=1.0,
    solver="lbfgs",
    max_iter=5000,
    class_weight=class_weight_dict,
    random_state=42
)

logreg_clf.fit(X_train, y_train)

# ------------------------------------------------------------
# 4. Predictions
# ------------------------------------------------------------

y_train_pred = logreg_clf.predict(X_train)
y_test_pred = logreg_clf.predict(X_test)

y_train_proba = logreg_clf.predict_proba(X_train)[:, 1]
y_test_proba = logreg_clf.predict_proba(X_test)[:, 1]

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
print("LOGISTIC REGRESSION CLASSIFIER — TASK 2 RESULTS")
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

with open(output_dir / "logreg_task2_train_classification_report.txt", "w", encoding="utf-8") as f:
    f.write(train_report)

with open(output_dir / "logreg_task2_test_classification_report.txt", "w", encoding="utf-8") as f:
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
plt.title("Logistic Regression — Test Confusion Matrix")
plt.tight_layout()
plt.savefig(output_dir / "logreg_task2_confusion_matrix.png", dpi=300)
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
plt.title("Logistic Regression — ROC Curve")
plt.tight_layout()
plt.savefig(output_dir / "logreg_task2_roc_curve.png", dpi=300)
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
plt.title("Logistic Regression — Precision-Recall Curve")
plt.tight_layout()
plt.savefig(output_dir / "logreg_task2_precision_recall_curve.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 10. Coefficient importance
# ------------------------------------------------------------

coef_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": logreg_clf.coef_[0],
    "Absolute_Coefficient": abs(logreg_clf.coef_[0])
}).sort_values(by="Absolute_Coefficient", ascending=False)

coef_importance.to_csv(
    output_dir / "logreg_task2_coefficients.csv",
    index=False
)

print("\nTop logistic regression coefficients:")
print(coef_importance.head(15))

plt.figure(figsize=(9, 6))
plt.barh(
    coef_importance["Feature"].head(15)[::-1],
    coef_importance["Coefficient"].head(15)[::-1]
)
plt.xlabel("Coefficient")
plt.title("Logistic Regression — Top 15 Coefficients")
plt.tight_layout()
plt.savefig(output_dir / "logreg_task2_coefficients_top15.png", dpi=300)
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
    output_dir / "logreg_task2_test_predictions.csv",
    index=False
)

# ------------------------------------------------------------
# 12. Save model and metrics
# ------------------------------------------------------------

joblib.dump(logreg_clf, output_dir / "logreg_task2_model.pkl")

with open(output_dir / "logreg_task2_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

# ------------------------------------------------------------
# 13. Final message
# ------------------------------------------------------------

print("\n============================================================")
print("TASK 2 LOGISTIC REGRESSION CLASSIFIER COMPLETED SUCCESSFULLY")
print("============================================================")
print("Saved outputs in:")
print(output_dir.resolve())

print("\nFiles created:")
for file in sorted(output_dir.iterdir()):
    print("-", file.name)