# ============================================================
# TASK 3B — MECHANISTIC RANDOM FOREST CLASSIFIER
# Known Inhibition Type Classification
#
# Goal:
# Predict inhibition type using only operational/mechanistic
# inhibition-related variables, without Substrate encoding.
# ============================================================

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
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

input_file = (
    script_dir
    / "task3_cleaned_inhibition_type_outputs"
    / "task3_inhibited_only_cleaned_dataset.csv"
)

if not input_file.exists():
    raise FileNotFoundError(
        "Missing cleaned Task 3 dataset. Run task3_clean_inhibition_type_labels.py first."
    )

output_dir = script_dir / "task3_mechanistic_random_forest_outputs"
output_dir.mkdir(exist_ok=True)

print("Loading cleaned Task 3 dataset from:")
print(input_file)

print("\nOutputs will be saved in:")
print(output_dir.resolve())

# ------------------------------------------------------------
# 1. Load cleaned inhibited-only dataset
# ------------------------------------------------------------

df = pd.read_csv(input_file)

print("\nLoaded dataset shape:", df.shape)

# ------------------------------------------------------------
# 2. Remove Unknown inhibition type
# ------------------------------------------------------------

df_task3 = df[df["Inhibition_Type_Final"] != "Unknown"].copy()

print("\nRows after removing Unknown:", df_task3.shape[0])

print("\nKnown inhibition-type distribution:")
print(df_task3["Inhibition_Type_Final"].value_counts())

# ------------------------------------------------------------
# 3. Define mechanistic feature set
# ------------------------------------------------------------

target_col = "Inhibition_Type_Final"

feature_cols = [
    "Temperature_C",
    "pH",
    "HRT_days",
    "OLR_gVS_L_day",
    "ISR",
    "TAN_mg_L",
    "VFA_mg_L"
]

missing_cols = [col for col in feature_cols + [target_col] if col not in df_task3.columns]

if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

X = df_task3[feature_cols].copy()
y_text = df_task3[target_col].copy()

print("\nMechanistic feature set:")
print(feature_cols)

print("\nInitial X shape:", X.shape)

# ------------------------------------------------------------
# 4. Encode target labels
# ------------------------------------------------------------

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)

label_mapping = {
    int(i): str(label)
    for i, label in enumerate(label_encoder.classes_)
}

target_names = [
    label_mapping[i]
    for i in sorted(label_mapping.keys())
]

print("\nLabel mapping:")
print(label_mapping)

with open(output_dir / "task3_mechanistic_label_mapping.json", "w", encoding="utf-8") as f:
    json.dump(label_mapping, f, indent=4, ensure_ascii=False)

joblib.dump(label_encoder, output_dir / "label_encoder_task3_mechanistic.pkl")

# ------------------------------------------------------------
# 5. Train/test split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test, y_train_text, y_test_text = train_test_split(
    X,
    y,
    y_text,
    test_size=0.2,
    random_state=42,
    stratify=y_text
)

print("\nTrain shape before preprocessing:", X_train.shape)
print("Test shape before preprocessing:", X_test.shape)

print("\nTrain target distribution:")
print(pd.Series(y_train_text).value_counts())

print("\nTest target distribution:")
print(pd.Series(y_test_text).value_counts())

# ------------------------------------------------------------
# 6. Median imputation
# ------------------------------------------------------------

imputer = SimpleImputer(strategy="median")

X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test_imputed = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

joblib.dump(imputer, output_dir / "median_imputer_task3_mechanistic.pkl")

imputation_values = {
    col: float(value)
    for col, value in zip(X_train.columns, imputer.statistics_)
}

print("\nMedian imputation completed.")
print("Imputation values:")
print(imputation_values)

with open(output_dir / "imputation_values_task3_mechanistic.json", "w", encoding="utf-8") as f:
    json.dump(imputation_values, f, indent=4)

# ------------------------------------------------------------
# 7. Standard scaling
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_imputed),
    columns=X_train_imputed.columns,
    index=X_train_imputed.index
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_imputed),
    columns=X_test_imputed.columns,
    index=X_test_imputed.index
)

joblib.dump(scaler, output_dir / "standard_scaler_task3_mechanistic.pkl")

print("\nStandard scaling completed.")

# ------------------------------------------------------------
# 8. Class weights
# ------------------------------------------------------------

class_values = np.unique(y_train)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=class_values,
    y=y_train
)

class_weight_dict = {
    int(cls): float(weight)
    for cls, weight in zip(class_values, class_weights)
}

class_weight_named = {
    label_mapping[int(cls)]: float(weight)
    for cls, weight in zip(class_values, class_weights)
}

print("\nClass weights:")
print(class_weight_named)

with open(output_dir / "class_weights_task3_mechanistic.json", "w", encoding="utf-8") as f:
    json.dump(class_weight_dict, f, indent=4)

# ------------------------------------------------------------
# 9. Train Random Forest
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

rf_clf.fit(X_train_scaled, y_train)

print("\nMechanistic Random Forest training completed.")

# ------------------------------------------------------------
# 10. Predictions
# ------------------------------------------------------------

y_train_pred = rf_clf.predict(X_train_scaled)
y_test_pred = rf_clf.predict(X_test_scaled)

y_train_proba = rf_clf.predict_proba(X_train_scaled)
y_test_proba = rf_clf.predict_proba(X_test_scaled)

# ------------------------------------------------------------
# 11. Metrics
# ------------------------------------------------------------

metrics = {
    "model": "Mechanistic Random Forest Classifier",
    "features_used": feature_cols,
    "train_accuracy": accuracy_score(y_train, y_train_pred),
    "test_accuracy": accuracy_score(y_test, y_test_pred),
    "train_balanced_accuracy": balanced_accuracy_score(y_train, y_train_pred),
    "test_balanced_accuracy": balanced_accuracy_score(y_test, y_test_pred),
    "train_macro_precision": precision_score(y_train, y_train_pred, average="macro", zero_division=0),
    "test_macro_precision": precision_score(y_test, y_test_pred, average="macro", zero_division=0),
    "train_macro_recall": recall_score(y_train, y_train_pred, average="macro", zero_division=0),
    "test_macro_recall": recall_score(y_test, y_test_pred, average="macro", zero_division=0),
    "train_macro_f1": f1_score(y_train, y_train_pred, average="macro", zero_division=0),
    "test_macro_f1": f1_score(y_test, y_test_pred, average="macro", zero_division=0),
    "train_weighted_f1": f1_score(y_train, y_train_pred, average="weighted", zero_division=0),
    "test_weighted_f1": f1_score(y_test, y_test_pred, average="weighted", zero_division=0)
}

try:
    metrics["train_roc_auc_ovr_macro"] = roc_auc_score(
        y_train,
        y_train_proba,
        multi_class="ovr",
        average="macro"
    )

    metrics["test_roc_auc_ovr_macro"] = roc_auc_score(
        y_test,
        y_test_proba,
        multi_class="ovr",
        average="macro"
    )

except ValueError:
    metrics["train_roc_auc_ovr_macro"] = np.nan
    metrics["test_roc_auc_ovr_macro"] = np.nan

print("\n============================================================")
print("TASK 3B — MECHANISTIC RANDOM FOREST RESULTS")
print("============================================================")

for key, value in metrics.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

# ------------------------------------------------------------
# 12. Classification reports
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

with open(output_dir / "mechanistic_rf_task3_train_classification_report.txt", "w", encoding="utf-8") as f:
    f.write(train_report)

with open(output_dir / "mechanistic_rf_task3_test_classification_report.txt", "w", encoding="utf-8") as f:
    f.write(test_report)

# ------------------------------------------------------------
# 13. Confusion matrix
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
plt.title("Mechanistic Random Forest — Task 3 Test Confusion Matrix")
plt.tight_layout()
plt.savefig(output_dir / "mechanistic_rf_task3_confusion_matrix.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 14. Feature importance
# ------------------------------------------------------------

feature_importance = pd.DataFrame({
    "Feature": X_train_scaled.columns,
    "Importance": rf_clf.feature_importances_
}).sort_values(by="Importance", ascending=False)

feature_importance.to_csv(
    output_dir / "mechanistic_rf_task3_feature_importance.csv",
    index=False
)

print("\nFeature importances:")
print(feature_importance)

plt.figure(figsize=(9, 6))
plt.barh(
    feature_importance["Feature"][::-1],
    feature_importance["Importance"][::-1]
)
plt.xlabel("Feature importance")
plt.title("Mechanistic Random Forest — Task 3 Feature Importances")
plt.tight_layout()
plt.savefig(output_dir / "mechanistic_rf_task3_feature_importance.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 15. Save predictions, model, and summary
# ------------------------------------------------------------

test_predictions = pd.DataFrame({
    "Actual_Label_Encoded": y_test,
    "Predicted_Label_Encoded": y_test_pred,
    "Actual_Inhibition_Type": y_test_text.values,
    "Predicted_Inhibition_Type": [
        label_mapping[int(label)]
        for label in y_test_pred
    ]
})

for class_index, class_name in label_mapping.items():
    test_predictions[f"Probability_{class_name}"] = y_test_proba[:, int(class_index)]

test_predictions.to_csv(
    output_dir / "mechanistic_rf_task3_test_predictions.csv",
    index=False
)

joblib.dump(rf_clf, output_dir / "mechanistic_rf_task3_model.pkl")

with open(output_dir / "mechanistic_rf_task3_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

summary = {
    "task": "Task 3B - Mechanistic inhibition type classification",
    "unknown_removed": True,
    "rows_used": int(df_task3.shape[0]),
    "train_samples": int(X_train_scaled.shape[0]),
    "test_samples": int(X_test_scaled.shape[0]),
    "features_used": feature_cols,
    "label_mapping": label_mapping,
    "class_weights": class_weight_named,
    "metrics": metrics
}

with open(output_dir / "mechanistic_rf_task3_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4, ensure_ascii=False)

# ------------------------------------------------------------
# 16. Final message
# ------------------------------------------------------------

print("\n============================================================")
print("TASK 3B MECHANISTIC RANDOM FOREST COMPLETED SUCCESSFULLY")
print("============================================================")
print("Saved outputs in:")
print(output_dir.resolve())

print("\nFiles created:")
for file in sorted(output_dir.iterdir()):
    print("-", file.name)