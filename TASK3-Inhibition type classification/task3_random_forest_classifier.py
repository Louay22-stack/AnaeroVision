# ============================================================
# TASK 3 — RANDOM FOREST MULTICLASS CLASSIFIER
# Known Inhibition Type Classification
# Target: Inhibition_Type_Final
# ============================================================

import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)

# ------------------------------------------------------------
# 0. Paths
# ------------------------------------------------------------

script_dir = Path(__file__).resolve().parent

preprocessing_dir = script_dir / "task3_preprocessing_known_types_outputs"

output_dir = script_dir / "task3_random_forest_outputs"
output_dir.mkdir(exist_ok=True)

if not preprocessing_dir.exists():
    raise FileNotFoundError(
        "Missing folder: task3_preprocessing_known_types_outputs"
    )

print("Loading Task 3 preprocessed data from:")
print(preprocessing_dir.resolve())

print("\nRandom Forest outputs will be saved in:")
print(output_dir.resolve())

# ------------------------------------------------------------
# 1. Load preprocessed data
# ------------------------------------------------------------

X_train = pd.read_csv(preprocessing_dir / "X_train_task3_preprocessed.csv")
X_test = pd.read_csv(preprocessing_dir / "X_test_task3_preprocessed.csv")

y_train = pd.read_csv(preprocessing_dir / "y_train_task3_encoded.csv").squeeze()
y_test = pd.read_csv(preprocessing_dir / "y_test_task3_encoded.csv").squeeze()

y_train_labels = pd.read_csv(preprocessing_dir / "y_train_task3_labels.csv").squeeze()
y_test_labels = pd.read_csv(preprocessing_dir / "y_test_task3_labels.csv").squeeze()

with open(preprocessing_dir / "task3_label_mapping.json", "r", encoding="utf-8") as f:
    label_mapping_raw = json.load(f)

label_mapping = {
    int(k): v
    for k, v in label_mapping_raw.items()
}

target_names = [
    label_mapping[i]
    for i in sorted(label_mapping.keys())
]

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("\nTarget classes:")
print(label_mapping)

print("\nTrain target distribution:")
print(y_train_labels.value_counts())

print("\nTest target distribution:")
print(y_test_labels.value_counts())

# ------------------------------------------------------------
# 2. Load class weights
# ------------------------------------------------------------

with open(preprocessing_dir / "task3_class_weights.json", "r", encoding="utf-8") as f:
    class_weight_raw = json.load(f)

class_weight_dict = {
    int(k): float(v)
    for k, v in class_weight_raw.items()
}

print("\nClass weights used:")
print(class_weight_dict)

# ------------------------------------------------------------
# 3. Train Random Forest Classifier
# ------------------------------------------------------------

rf_clf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight=class_weight_dict,
    random_state=42,
    n_jobs=-1
)

rf_clf.fit(X_train, y_train)

print("\nRandom Forest training completed.")

# ------------------------------------------------------------
# 4. Predictions
# ------------------------------------------------------------

y_train_pred = rf_clf.predict(X_train)
y_test_pred = rf_clf.predict(X_test)

y_train_proba = rf_clf.predict_proba(X_train)
y_test_proba = rf_clf.predict_proba(X_test)

# ------------------------------------------------------------
# 5. Metrics
# ------------------------------------------------------------

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

train_balanced_accuracy = balanced_accuracy_score(y_train, y_train_pred)
test_balanced_accuracy = balanced_accuracy_score(y_test, y_test_pred)

train_macro_precision = precision_score(
    y_train, y_train_pred, average="macro", zero_division=0
)
test_macro_precision = precision_score(
    y_test, y_test_pred, average="macro", zero_division=0
)

train_macro_recall = recall_score(
    y_train, y_train_pred, average="macro", zero_division=0
)
test_macro_recall = recall_score(
    y_test, y_test_pred, average="macro", zero_division=0
)

train_macro_f1 = f1_score(
    y_train, y_train_pred, average="macro", zero_division=0
)
test_macro_f1 = f1_score(
    y_test, y_test_pred, average="macro", zero_division=0
)

train_weighted_f1 = f1_score(
    y_train, y_train_pred, average="weighted", zero_division=0
)
test_weighted_f1 = f1_score(
    y_test, y_test_pred, average="weighted", zero_division=0
)

# Multiclass ROC-AUC
try:
    train_roc_auc_ovr = roc_auc_score(
        y_train,
        y_train_proba,
        multi_class="ovr",
        average="macro"
    )

    test_roc_auc_ovr = roc_auc_score(
        y_test,
        y_test_proba,
        multi_class="ovr",
        average="macro"
    )

except ValueError:
    train_roc_auc_ovr = np.nan
    test_roc_auc_ovr = np.nan

metrics = {
    "model": "Random Forest Classifier",
    "train_accuracy": train_accuracy,
    "test_accuracy": test_accuracy,
    "train_balanced_accuracy": train_balanced_accuracy,
    "test_balanced_accuracy": test_balanced_accuracy,
    "train_macro_precision": train_macro_precision,
    "test_macro_precision": test_macro_precision,
    "train_macro_recall": train_macro_recall,
    "test_macro_recall": test_macro_recall,
    "train_macro_f1": train_macro_f1,
    "test_macro_f1": test_macro_f1,
    "train_weighted_f1": train_weighted_f1,
    "test_weighted_f1": test_weighted_f1,
    "train_roc_auc_ovr_macro": train_roc_auc_ovr,
    "test_roc_auc_ovr_macro": test_roc_auc_ovr
}

print("\n============================================================")
print("RANDOM FOREST — TASK 3 RESULTS")
print("============================================================")

for key, value in metrics.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

# ------------------------------------------------------------
# 6. Classification reports
# ------------------------------------------------------------

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

with open(output_dir / "rf_task3_train_classification_report.txt", "w", encoding="utf-8") as f:
    f.write(train_report)

with open(output_dir / "rf_task3_test_classification_report.txt", "w", encoding="utf-8") as f:
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

fig, ax = plt.subplots(figsize=(9, 7))
disp.plot(ax=ax, values_format="d", xticks_rotation=45)
plt.title("Random Forest — Task 3 Test Confusion Matrix")
plt.tight_layout()
plt.savefig(output_dir / "rf_task3_confusion_matrix.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 8. Feature importance
# ------------------------------------------------------------

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf_clf.feature_importances_
}).sort_values(by="Importance", ascending=False)

feature_importance.to_csv(
    output_dir / "rf_task3_feature_importance.csv",
    index=False
)

print("\nTop feature importances:")
print(feature_importance.head(15))

plt.figure(figsize=(10, 7))
plt.barh(
    feature_importance["Feature"].head(15)[::-1],
    feature_importance["Importance"].head(15)[::-1]
)
plt.xlabel("Feature importance")
plt.title("Random Forest — Task 3 Top 15 Feature Importances")
plt.tight_layout()
plt.savefig(output_dir / "rf_task3_feature_importance_top15.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 9. Save predictions
# ------------------------------------------------------------

test_predictions = pd.DataFrame({
    "Actual_Label_Encoded": y_test.values,
    "Predicted_Label_Encoded": y_test_pred,
    "Actual_Inhibition_Type": y_test_labels.values,
    "Predicted_Inhibition_Type": [
        label_mapping[int(label)]
        for label in y_test_pred
    ]
})

# Add predicted probabilities for each class
for class_index, class_name in label_mapping.items():
    test_predictions[f"Probability_{class_name}"] = y_test_proba[:, int(class_index)]

test_predictions.to_csv(
    output_dir / "rf_task3_test_predictions.csv",
    index=False
)

# ------------------------------------------------------------
# 10. Save model and metrics
# ------------------------------------------------------------

joblib.dump(rf_clf, output_dir / "rf_task3_model.pkl")

with open(output_dir / "rf_task3_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

# ------------------------------------------------------------
# 11. Final message
# ------------------------------------------------------------

print("\n============================================================")
print("TASK 3 RANDOM FOREST CLASSIFIER COMPLETED SUCCESSFULLY")
print("============================================================")
print("Saved outputs in:")
print(output_dir.resolve())

print("\nFiles created:")
for file in sorted(output_dir.iterdir()):
    print("-", file.name)