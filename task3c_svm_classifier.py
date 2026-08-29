# ============================================================
# TASK 3C — SVM CLASSIFIER
# Interpretable Hybrid Inhibition-Type Classification
#
# Features:
# - Mechanistic operational variables
# - Substrate category
# - Reactor type
#
# Model:
# - Support Vector Machine with RBF kernel
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
from sklearn.svm import SVC

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

output_dir = script_dir / "task3c_svm_outputs"
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
# 2. Detect substrate category column
# ------------------------------------------------------------

possible_category_cols = [
    "Substrate_Composition_Category",
    "Substrate_Category",
    "Substrate category",
    "substrate_category"
]

category_col = None

for col in possible_category_cols:
    if col in df.columns:
        category_col = col
        break

if category_col is None:
    raise ValueError(
        "No substrate category column found. Expected one of: "
        f"{possible_category_cols}"
    )

print("\nDetected substrate category column:")
print(category_col)

# ------------------------------------------------------------
# 3. Remove Unknown inhibition type
# ------------------------------------------------------------

df_task3 = df[df["Inhibition_Type_Final"] != "Unknown"].copy()

print("\nRows after removing Unknown:", df_task3.shape[0])

print("\nKnown inhibition-type distribution:")
print(df_task3["Inhibition_Type_Final"].value_counts())

# ------------------------------------------------------------
# 4. Define interpretable hybrid feature set
# ------------------------------------------------------------

target_col = "Inhibition_Type_Final"

numeric_features = [
    "Temperature_C",
    "pH",
    "HRT_days",
    "OLR_gVS_L_day",
    "ISR",
    "TAN_mg_L",
    "VFA_mg_L"
]

categorical_features = [
    category_col,
    "Reactor_Type"
]

feature_cols = numeric_features + categorical_features

missing_cols = [col for col in feature_cols + [target_col] if col not in df_task3.columns]

if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

X = df_task3[feature_cols].copy()
y_text = df_task3[target_col].copy()

print("\nFeature set used:")
print(feature_cols)

# ------------------------------------------------------------
# 5. Clean categorical variables
# ------------------------------------------------------------

X[category_col] = X[category_col].fillna("Other/unknown").astype(str)

reactor_mapping = {
    "Batch": "Batch",
    "Batch (BMP)": "Batch",
    "Batch serum bottles": "Batch",

    "Thermophilic Batch": "Thermophilic_Batch",
    "Thermophilic AD": "Thermophilic_Batch",

    "CSTR": "CSTR",
    "CSTR (fed-batch)": "CSTR",
    "CSTR (mesophilic)": "CSTR",
    "Two-stage CSTR": "CSTR",
    "Serial CSTR": "CSTR",

    "Horizontal flow reactor": "Other",
    "Horizontal flow": "Other"
}

X["Reactor_Type"] = X["Reactor_Type"].map(reactor_mapping)
X["Reactor_Type"] = X["Reactor_Type"].fillna("Other")

print("\nReactor type distribution after consolidation:")
print(X["Reactor_Type"].value_counts())

# ------------------------------------------------------------
# 6. Encode target labels
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

with open(output_dir / "task3c_svm_label_mapping.json", "w", encoding="utf-8") as f:
    json.dump(label_mapping, f, indent=4, ensure_ascii=False)

joblib.dump(label_encoder, output_dir / "label_encoder_task3c_svm.pkl")

# ------------------------------------------------------------
# 7. Train/test split
# Same split strategy as Random Forest
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
# 8. Median imputation
# ------------------------------------------------------------

imputer = SimpleImputer(strategy="median")

X_train[numeric_features] = imputer.fit_transform(X_train[numeric_features])
X_test[numeric_features] = imputer.transform(X_test[numeric_features])

joblib.dump(imputer, output_dir / "median_imputer_task3c_svm.pkl")

imputation_values = {
    col: float(value)
    for col, value in zip(numeric_features, imputer.statistics_)
}

print("\nMedian imputation completed.")
print("Imputation values:")
print(imputation_values)

with open(output_dir / "imputation_values_task3c_svm.json", "w", encoding="utf-8") as f:
    json.dump(imputation_values, f, indent=4)

# ------------------------------------------------------------
# 9. Standard scaling
# ------------------------------------------------------------

scaler = StandardScaler()

X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

joblib.dump(scaler, output_dir / "standard_scaler_task3c_svm.pkl")

print("\nStandard scaling completed.")

# ------------------------------------------------------------
# 10. One-hot encoding
# ------------------------------------------------------------

X_train = pd.get_dummies(
    X_train,
    columns=categorical_features,
    drop_first=False
)

X_test = pd.get_dummies(
    X_test,
    columns=categorical_features,
    drop_first=False
)

X_train, X_test = X_train.align(
    X_test,
    join="left",
    axis=1,
    fill_value=0
)

feature_names = X_train.columns.tolist()

with open(output_dir / "task3c_svm_feature_names.json", "w", encoding="utf-8") as f:
    json.dump(feature_names, f, indent=4)

print("\nFinal X_train shape:", X_train.shape)
print("Final X_test shape:", X_test.shape)

print("\nMissing values in X_train:", int(X_train.isna().sum().sum()))
print("Missing values in X_test:", int(X_test.isna().sum().sum()))

if X_train.isna().sum().sum() > 0 or X_test.isna().sum().sum() > 0:
    raise ValueError("Missing values still present after preprocessing.")

# ------------------------------------------------------------
# 11. Class weights
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

with open(output_dir / "class_weights_task3c_svm.json", "w", encoding="utf-8") as f:
    json.dump(class_weight_dict, f, indent=4)

with open(output_dir / "class_weights_task3c_svm_named.json", "w", encoding="utf-8") as f:
    json.dump(class_weight_named, f, indent=4, ensure_ascii=False)

# ------------------------------------------------------------
# 12. Train SVM classifier
# ------------------------------------------------------------

svm_clf = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    class_weight=class_weight_dict,
    probability=True,
    random_state=42
)

svm_clf.fit(X_train, y_train)

print("\nSVM RBF training completed.")

# ------------------------------------------------------------
# 13. Predictions
# ------------------------------------------------------------

y_train_pred = svm_clf.predict(X_train)
y_test_pred = svm_clf.predict(X_test)

y_train_proba = svm_clf.predict_proba(X_train)
y_test_proba = svm_clf.predict_proba(X_test)

# ------------------------------------------------------------
# 14. Metrics
# ------------------------------------------------------------

metrics = {
    "model": "SVM RBF Classifier",
    "features_used": feature_cols,
    "category_column_used": category_col,
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
print("TASK 3C — SVM RBF RESULTS")
print("============================================================")

for key, value in metrics.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

# ------------------------------------------------------------
# 15. Classification reports
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

with open(output_dir / "svm_task3c_train_classification_report.txt", "w", encoding="utf-8") as f:
    f.write(train_report)

with open(output_dir / "svm_task3c_test_classification_report.txt", "w", encoding="utf-8") as f:
    f.write(test_report)

# ------------------------------------------------------------
# 16. Confusion matrix
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
plt.title("SVM RBF — Task 3C Test Confusion Matrix")
plt.tight_layout()
plt.savefig(output_dir / "svm_task3c_confusion_matrix.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 17. Save predictions
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
    output_dir / "svm_task3c_test_predictions.csv",
    index=False
)

# ------------------------------------------------------------
# 18. Save model and summary
# ------------------------------------------------------------

joblib.dump(svm_clf, output_dir / "svm_task3c_model.pkl")

with open(output_dir / "svm_task3c_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

summary = {
    "task": "Task 3C - SVM RBF inhibition type classification",
    "unknown_removed": True,
    "rows_used": int(df_task3.shape[0]),
    "train_samples": int(X_train.shape[0]),
    "test_samples": int(X_test.shape[0]),
    "features_used": feature_cols,
    "category_column_used": category_col,
    "label_mapping": label_mapping,
    "class_weights": class_weight_named,
    "metrics": metrics
}

with open(output_dir / "svm_task3c_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4, ensure_ascii=False)

# ------------------------------------------------------------
# 19. Final message
# ------------------------------------------------------------

print("\n============================================================")
print("TASK 3C SVM RBF CLASSIFIER COMPLETED")
print("============================================================")
print("Saved outputs in:")
print(output_dir.resolve())

print("\nFiles created:")
for file in sorted(output_dir.iterdir()):
    print("-", file.name)