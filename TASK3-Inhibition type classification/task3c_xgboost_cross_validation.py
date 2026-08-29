# ============================================================
# TASK 3C — XGBOOST STRATIFIED CROSS-VALIDATION
# Interpretable Hybrid Inhibition-Type Classification
#
# Features:
# - Mechanistic operational variables
# - Substrate category
# - Reactor type
#
# Purpose:
# Validate XGBoost stability across folds before tuning.
# ============================================================

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

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

from xgboost import XGBClassifier

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

output_dir = script_dir / "task3c_xgb_cross_validation_outputs"
output_dir.mkdir(exist_ok=True)

print("Loading cleaned Task 3 dataset from:")
print(input_file)

print("\nOutputs will be saved in:")
print(output_dir.resolve())

# ------------------------------------------------------------
# 1. Load dataset
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
# 4. Define Task 3C features
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

print("\nFeature set used:")
print(feature_cols)

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

with open(output_dir / "task3c_xgb_cv_label_mapping.json", "w", encoding="utf-8") as f:
    json.dump(label_mapping, f, indent=4, ensure_ascii=False)

joblib.dump(label_encoder, output_dir / "label_encoder_task3c_xgb_cv.pkl")

# ------------------------------------------------------------
# 7. Preprocessing pipeline
# ------------------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ],
    remainder="drop"
)

# ------------------------------------------------------------
# 8. XGBoost model
# ------------------------------------------------------------

xgb_model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(label_encoder.classes_),
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=1,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.0,
    reg_lambda=1.0,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", xgb_model)
    ]
)

# ------------------------------------------------------------
# 9. Stratified cross-validation
# ------------------------------------------------------------

n_splits = 5

skf = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=42
)

fold_metrics = []
all_y_true = []
all_y_pred = []
all_y_proba = []

print("\n============================================================")
print("STARTING 5-FOLD STRATIFIED CROSS-VALIDATION")
print("============================================================")

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):

    print(f"\n---------------- Fold {fold_idx} / {n_splits} ----------------")

    X_train_fold = X.iloc[train_idx].copy()
    X_test_fold = X.iloc[test_idx].copy()

    y_train_fold = y[train_idx]
    y_test_fold = y[test_idx]

    class_values = np.unique(y_train_fold)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=class_values,
        y=y_train_fold
    )

    class_weight_dict = {
        int(cls): float(weight)
        for cls, weight in zip(class_values, class_weights)
    }

    sample_weights_train = np.array([
        class_weight_dict[int(label)]
        for label in y_train_fold
    ])

    pipeline.fit(
        X_train_fold,
        y_train_fold,
        model__sample_weight=sample_weights_train
    )

    y_pred_fold = pipeline.predict(X_test_fold)
    y_proba_fold = pipeline.predict_proba(X_test_fold)

    fold_accuracy = accuracy_score(y_test_fold, y_pred_fold)
    fold_balanced_accuracy = balanced_accuracy_score(y_test_fold, y_pred_fold)
    fold_macro_precision = precision_score(
        y_test_fold, y_pred_fold, average="macro", zero_division=0
    )
    fold_macro_recall = recall_score(
        y_test_fold, y_pred_fold, average="macro", zero_division=0
    )
    fold_macro_f1 = f1_score(
        y_test_fold, y_pred_fold, average="macro", zero_division=0
    )
    fold_weighted_f1 = f1_score(
        y_test_fold, y_pred_fold, average="weighted", zero_division=0
    )

    try:
        fold_roc_auc = roc_auc_score(
            y_test_fold,
            y_proba_fold,
            multi_class="ovr",
            average="macro"
        )
    except ValueError:
        fold_roc_auc = np.nan

    metrics = {
        "fold": fold_idx,
        "accuracy": fold_accuracy,
        "balanced_accuracy": fold_balanced_accuracy,
        "macro_precision": fold_macro_precision,
        "macro_recall": fold_macro_recall,
        "macro_f1": fold_macro_f1,
        "weighted_f1": fold_weighted_f1,
        "roc_auc_ovr_macro": fold_roc_auc
    }

    fold_metrics.append(metrics)

    print(f"Accuracy: {fold_accuracy:.4f}")
    print(f"Balanced accuracy: {fold_balanced_accuracy:.4f}")
    print(f"Macro F1: {fold_macro_f1:.4f}")
    print(f"Weighted F1: {fold_weighted_f1:.4f}")
    print(f"ROC-AUC OvR macro: {fold_roc_auc:.4f}")

    report = classification_report(
        y_test_fold,
        y_pred_fold,
        target_names=target_names,
        zero_division=0
    )

    print("\nClassification report:")
    print(report)

    with open(output_dir / f"xgb_cv_fold_{fold_idx}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    cm_fold = confusion_matrix(y_test_fold, y_pred_fold)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_fold,
        display_labels=target_names
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    disp.plot(ax=ax, values_format="d", xticks_rotation=45)
    plt.title(f"XGBoost — Task 3C CV Fold {fold_idx} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / f"xgb_cv_fold_{fold_idx}_confusion_matrix.png", dpi=300)
    plt.close()

    all_y_true.extend(y_test_fold.tolist())
    all_y_pred.extend(y_pred_fold.tolist())
    all_y_proba.extend(y_proba_fold.tolist())

# ------------------------------------------------------------
# 10. Save fold metrics
# ------------------------------------------------------------

fold_metrics_df = pd.DataFrame(fold_metrics)

fold_metrics_df.to_csv(
    output_dir / "xgb_task3c_cv_fold_metrics.csv",
    index=False
)

print("\n============================================================")
print("XGBOOST CROSS-VALIDATION SUMMARY")
print("============================================================")

summary_metrics = {}

for metric in [
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "roc_auc_ovr_macro"
]:
    mean_value = fold_metrics_df[metric].mean()
    std_value = fold_metrics_df[metric].std()

    summary_metrics[f"{metric}_mean"] = float(mean_value)
    summary_metrics[f"{metric}_std"] = float(std_value)

    print(f"{metric}: {mean_value:.4f} ± {std_value:.4f}")

# ------------------------------------------------------------
# 11. Overall out-of-fold metrics
# ------------------------------------------------------------

all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)
all_y_proba = np.array(all_y_proba)

overall_metrics = {
    "overall_accuracy": accuracy_score(all_y_true, all_y_pred),
    "overall_balanced_accuracy": balanced_accuracy_score(all_y_true, all_y_pred),
    "overall_macro_precision": precision_score(
        all_y_true, all_y_pred, average="macro", zero_division=0
    ),
    "overall_macro_recall": recall_score(
        all_y_true, all_y_pred, average="macro", zero_division=0
    ),
    "overall_macro_f1": f1_score(
        all_y_true, all_y_pred, average="macro", zero_division=0
    ),
    "overall_weighted_f1": f1_score(
        all_y_true, all_y_pred, average="weighted", zero_division=0
    )
}

try:
    overall_metrics["overall_roc_auc_ovr_macro"] = roc_auc_score(
        all_y_true,
        all_y_proba,
        multi_class="ovr",
        average="macro"
    )
except ValueError:
    overall_metrics["overall_roc_auc_ovr_macro"] = np.nan

print("\nOverall out-of-fold metrics:")
for key, value in overall_metrics.items():
    print(f"{key}: {value:.4f}")

# ------------------------------------------------------------
# 12. Overall classification report and confusion matrix
# ------------------------------------------------------------

overall_report = classification_report(
    all_y_true,
    all_y_pred,
    target_names=target_names,
    zero_division=0
)

print("\nOverall out-of-fold classification report:")
print(overall_report)

with open(output_dir / "xgb_task3c_cv_overall_classification_report.txt", "w", encoding="utf-8") as f:
    f.write(overall_report)

overall_cm = confusion_matrix(all_y_true, all_y_pred)

print("\nOverall out-of-fold confusion matrix:")
print(overall_cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=overall_cm,
    display_labels=target_names
)

fig, ax = plt.subplots(figsize=(9, 7))
disp.plot(ax=ax, values_format="d", xticks_rotation=45)
plt.title("XGBoost — Task 3C Overall CV Confusion Matrix")
plt.tight_layout()
plt.savefig(output_dir / "xgb_task3c_cv_overall_confusion_matrix.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 13. Save summary
# ------------------------------------------------------------

cv_summary = {
    "task": "Task 3C - XGBoost stratified cross-validation",
    "n_splits": n_splits,
    "rows_used": int(df_task3.shape[0]),
    "features_used": feature_cols,
    "category_column_used": category_col,
    "label_mapping": label_mapping,
    "fold_metrics": fold_metrics,
    "summary_metrics": summary_metrics,
    "overall_metrics": overall_metrics
}

with open(output_dir / "xgb_task3c_cv_summary.json", "w", encoding="utf-8") as f:
    json.dump(cv_summary, f, indent=4, ensure_ascii=False)

# ------------------------------------------------------------
# 14. Train final XGBoost on all data after CV
# ------------------------------------------------------------

class_values_all = np.unique(y)

class_weights_all = compute_class_weight(
    class_weight="balanced",
    classes=class_values_all,
    y=y
)

class_weight_dict_all = {
    int(cls): float(weight)
    for cls, weight in zip(class_values_all, class_weights_all)
}

sample_weights_all = np.array([
    class_weight_dict_all[int(label)]
    for label in y
])

pipeline.fit(
    X,
    y,
    model__sample_weight=sample_weights_all
)

joblib.dump(
    pipeline,
    output_dir / "xgb_task3c_cv_trained_on_all_known_samples.pkl"
)

print("\nFinal XGBoost pipeline retrained on all 220 known samples and saved.")

# ------------------------------------------------------------
# 15. Final message
# ------------------------------------------------------------

print("\n============================================================")
print("TASK 3C XGBOOST CROSS-VALIDATION COMPLETED")
print("============================================================")
print("Saved outputs in:")
print(output_dir.resolve())

print("\nFiles created:")
for file in sorted(output_dir.iterdir()):
    print("-", file.name)