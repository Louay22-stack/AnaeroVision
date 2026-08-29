# ============================================================
# TASK 2 — ANN / MLP CLASSIFIER
# Inhibition Risk Classification
# Target: Inhibition_Label_Numeric
# Uses balanced upsampling for minority inhibition class
# ============================================================

import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
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

output_dir = script_dir / "task2_ann_outputs"
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

print("\nOriginal train target distribution:")
print(y_train.value_counts().sort_index())

print("\nTest target distribution:")
print(y_test.value_counts().sort_index())

# ------------------------------------------------------------
# 2. Balance training data by upsampling minority class
# ------------------------------------------------------------

train_data = X_train.copy()
train_data["target"] = y_train.values

class_0 = train_data[train_data["target"] == 0]
class_1 = train_data[train_data["target"] == 1]

class_1_upsampled = resample(
    class_1,
    replace=True,
    n_samples=len(class_0),
    random_state=42
)

train_balanced = pd.concat([class_0, class_1_upsampled], axis=0)
train_balanced = train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

X_train_balanced = train_balanced.drop(columns=["target"])
y_train_balanced = train_balanced["target"]

print("\nBalanced train target distribution:")
print(y_train_balanced.value_counts().sort_index())

# ------------------------------------------------------------
# 3. Train ANN / MLPClassifier
# ------------------------------------------------------------

ann_clf = MLPClassifier(
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

ann_clf.fit(X_train_balanced, y_train_balanced)

print("\nANN training completed.")
print("Number of iterations:", ann_clf.n_iter_)
print("Final loss:", ann_clf.loss_)

# ------------------------------------------------------------
# 4. Predictions
# ------------------------------------------------------------

y_train_pred = ann_clf.predict(X_train)
y_test_pred = ann_clf.predict(X_test)

y_train_proba = ann_clf.predict_proba(X_train)[:, 1]
y_test_proba = ann_clf.predict_proba(X_test)[:, 1]

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
    "test_pr_auc": average_precision_score(y_test, y_test_proba),

    "n_iter": ann_clf.n_iter_,
    "final_loss": ann_clf.loss_
}

print("\n============================================================")
print("ANN / MLP CLASSIFIER — TASK 2 RESULTS")
print("============================================================")

for key, value in metrics.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

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

with open(output_dir / "ann_task2_train_classification_report.txt", "w", encoding="utf-8") as f:
    f.write(train_report)

with open(output_dir / "ann_task2_test_classification_report.txt", "w", encoding="utf-8") as f:
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
plt.title("ANN / MLP — Test Confusion Matrix")
plt.tight_layout()
plt.savefig(output_dir / "ann_task2_confusion_matrix.png", dpi=300)
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
plt.title("ANN / MLP — ROC Curve")
plt.tight_layout()
plt.savefig(output_dir / "ann_task2_roc_curve.png", dpi=300)
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
plt.title("ANN / MLP — Precision-Recall Curve")
plt.tight_layout()
plt.savefig(output_dir / "ann_task2_precision_recall_curve.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 10. Loss curve
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))
plt.plot(ann_clf.loss_curve_)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("ANN / MLP — Training Loss Curve")
plt.tight_layout()
plt.savefig(output_dir / "ann_task2_loss_curve.png", dpi=300)
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
    output_dir / "ann_task2_test_predictions.csv",
    index=False
)

# ------------------------------------------------------------
# 12. Save model and metrics
# ------------------------------------------------------------

joblib.dump(ann_clf, output_dir / "ann_task2_model.pkl")

with open(output_dir / "ann_task2_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

# ------------------------------------------------------------
# 13. Final message
# ------------------------------------------------------------

print("\n============================================================")
print("TASK 2 ANN / MLP CLASSIFIER COMPLETED SUCCESSFULLY")
print("============================================================")
print("Saved outputs in:")
print(output_dir.resolve())

print("\nFiles created:")
for file in sorted(output_dir.iterdir()):
    print("-", file.name)