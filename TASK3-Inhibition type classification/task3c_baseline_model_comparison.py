# ============================================================
# TASK 3C — BASELINE MODEL COMPARISON
# Interpretable Hybrid Inhibition-Type Classification
#
# Features:
# - Mechanistic operational variables
# - Substrate category
# - Reactor type
#
# Models:
# - Random Forest
# - XGBoost
# - SVM
# - Logistic Regression
# - ANN / MLPClassifier
# ============================================================

import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

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

warnings.filterwarnings("ignore")

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

output_dir = script_dir / "task3c_baseline_model_comparison_outputs"
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

with open(output_dir / "task3c_label_mapping.json", "w", encoding="utf-8") as f:
    json.dump(label_mapping, f, indent=4, ensure_ascii=False)

joblib.dump(label_encoder, output_dir / "label_encoder_task3c.pkl")

# ------------------------------------------------------------
# 7. Train/test split before fitted preprocessing
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

joblib.dump(imputer, output_dir / "median_imputer_task3c.pkl")

imputation_values = {
    col: float(value)
    for col, value in zip(numeric_features, imputer.statistics_)
}

with open(output_dir / "imputation_values_task3c.json", "w", encoding="utf-8") as f:
    json.dump(imputation_values, f, indent=4)

print("\nMedian imputation completed.")

# ------------------------------------------------------------
# 9. Standard scaling
# ------------------------------------------------------------

scaler = StandardScaler()

X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

joblib.dump(scaler, output_dir / "standard_scaler_task3c.pkl")

print("Standard scaling completed.")

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

with open(output_dir / "task3c_feature_names.json", "w", encoding="utf-8") as f:
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

with open(output_dir / "class_weights_task3c.json", "w", encoding="utf-8") as f:
    json.dump(class_weight_dict, f, indent=4)

with open(output_dir / "class_weights_task3c_named.json", "w", encoding="utf-8") as f:
    json.dump(class_weight_named, f, indent=4, ensure_ascii=False)

# ------------------------------------------------------------
# 12. Balanced training set for ANN
# ------------------------------------------------------------

def create_balanced_training_set(X_train_df, y_train_array, random_state=42):
    train_df = X_train_df.copy()
    train_df["target"] = y_train_array

    max_count = train_df["target"].value_counts().max()

    balanced_parts = []

    for cls in sorted(train_df["target"].unique()):
        cls_df = train_df[train_df["target"] == cls]

        cls_upsampled = resample(
            cls_df,
            replace=True,
            n_samples=max_count,
            random_state=random_state
        )

        balanced_parts.append(cls_upsampled)

    balanced_df = pd.concat(balanced_parts, axis=0)
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    y_balanced = balanced_df["target"].values
    X_balanced = balanced_df.drop(columns=["target"])

    return X_balanced, y_balanced

X_train_balanced, y_train_balanced = create_balanced_training_set(
    X_train,
    y_train,
    random_state=42
)

print("\nBalanced ANN training distribution:")
print(pd.Series(y_train_balanced).value_counts().sort_index())

# ------------------------------------------------------------
# 13. Define models
# ------------------------------------------------------------

models = {}

models["Random Forest"] = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight=class_weight_dict,
    random_state=42,
    n_jobs=-1
)

models["SVM RBF"] = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    class_weight=class_weight_dict,
    probability=True,
    random_state=42
)

models["Logistic Regression"] = LogisticRegression(
    max_iter=3000,
    class_weight=class_weight_dict,
    solver="lbfgs",
    random_state=42
)

models["ANN MLP"] = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    alpha=0.001,
    batch_size=32,
    learning_rate_init=0.001,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=30,
    max_iter=1000,
    random_state=42
)

# Optional XGBoost
xgboost_available = True

try:
    from xgboost import XGBClassifier

    sample_weights_xgb = np.array([
        class_weight_dict[int(label)]
        for label in y_train
    ])

    models["XGBoost"] = XGBClassifier(
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )

except Exception as e:
    xgboost_available = False
    print("\nXGBoost not available or failed to import.")
    print("Reason:", e)

# ------------------------------------------------------------
# 14. Training and evaluation function
# ------------------------------------------------------------

def evaluate_model(model_name, model, X_train_df, y_train_array, X_test_df, y_test_array):
    print("\n============================================================")
    print(f"TRAINING MODEL: {model_name}")
    print("============================================================")

    if model_name == "ANN MLP":
        model.fit(X_train_balanced, y_train_balanced)

    elif model_name == "XGBoost":
        model.fit(
            X_train_df,
            y_train_array,
            sample_weight=sample_weights_xgb
        )

    else:
        model.fit(X_train_df, y_train_array)

    y_train_pred = model.predict(X_train_df)
    y_test_pred = model.predict(X_test_df)

    y_train_proba = model.predict_proba(X_train_df)
    y_test_proba = model.predict_proba(X_test_df)

    metrics = {
        "model": model_name,
        "train_accuracy": accuracy_score(y_train_array, y_train_pred),
        "test_accuracy": accuracy_score(y_test_array, y_test_pred),
        "train_balanced_accuracy": balanced_accuracy_score(y_train_array, y_train_pred),
        "test_balanced_accuracy": balanced_accuracy_score(y_test_array, y_test_pred),
        "train_macro_precision": precision_score(y_train_array, y_train_pred, average="macro", zero_division=0),
        "test_macro_precision": precision_score(y_test_array, y_test_pred, average="macro", zero_division=0),
        "train_macro_recall": recall_score(y_train_array, y_train_pred, average="macro", zero_division=0),
        "test_macro_recall": recall_score(y_test_array, y_test_pred, average="macro", zero_division=0),
        "train_macro_f1": f1_score(y_train_array, y_train_pred, average="macro", zero_division=0),
        "test_macro_f1": f1_score(y_test_array, y_test_pred, average="macro", zero_division=0),
        "train_weighted_f1": f1_score(y_train_array, y_train_pred, average="weighted", zero_division=0),
        "test_weighted_f1": f1_score(y_test_array, y_test_pred, average="weighted", zero_division=0)
    }

    try:
        metrics["train_roc_auc_ovr_macro"] = roc_auc_score(
            y_train_array,
            y_train_proba,
            multi_class="ovr",
            average="macro"
        )

        metrics["test_roc_auc_ovr_macro"] = roc_auc_score(
            y_test_array,
            y_test_proba,
            multi_class="ovr",
            average="macro"
        )

    except ValueError:
        metrics["train_roc_auc_ovr_macro"] = np.nan
        metrics["test_roc_auc_ovr_macro"] = np.nan

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    train_report = classification_report(
        y_train_array,
        y_train_pred,
        target_names=target_names,
        zero_division=0
    )

    test_report = classification_report(
        y_test_array,
        y_test_pred,
        target_names=target_names,
        zero_division=0
    )

    print("\nTest classification report:")
    print(test_report)

    safe_model_name = model_name.lower().replace(" ", "_").replace("/", "_")

    with open(output_dir / f"{safe_model_name}_train_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(train_report)

    with open(output_dir / f"{safe_model_name}_test_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(test_report)

    cm = confusion_matrix(y_test_array, y_test_pred)

    print("\nTest confusion matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=target_names
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    disp.plot(ax=ax, values_format="d", xticks_rotation=45)
    plt.title(f"{model_name} — Task 3C Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / f"{safe_model_name}_confusion_matrix.png", dpi=300)
    plt.close()

    test_predictions = pd.DataFrame({
        "Actual_Label_Encoded": y_test_array,
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
        output_dir / f"{safe_model_name}_test_predictions.csv",
        index=False
    )

    joblib.dump(model, output_dir / f"{safe_model_name}_model.pkl")

    with open(output_dir / f"{safe_model_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    return metrics

# ------------------------------------------------------------
# 15. Train and compare models
# ------------------------------------------------------------

all_metrics = []

for model_name, model in models.items():
    metrics = evaluate_model(
        model_name=model_name,
        model=model,
        X_train_df=X_train,
        y_train_array=y_train,
        X_test_df=X_test,
        y_test_array=y_test
    )

    all_metrics.append(metrics)

comparison_df = pd.DataFrame(all_metrics)

comparison_df = comparison_df.sort_values(
    by=["test_macro_f1", "test_balanced_accuracy", "test_accuracy"],
    ascending=False
)

comparison_df.to_csv(
    output_dir / "task3c_baseline_model_comparison.csv",
    index=False
)

print("\n============================================================")
print("TASK 3C BASELINE MODEL COMPARISON")
print("============================================================")

print(
    comparison_df[
        [
            "model",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_macro_precision",
            "test_macro_recall",
            "test_macro_f1",
            "test_weighted_f1",
            "test_roc_auc_ovr_macro"
        ]
    ]
)

# ------------------------------------------------------------
# 16. Select best two models
# ------------------------------------------------------------

top_two_models = comparison_df["model"].head(2).tolist()

selection_summary = {
    "selection_metric": "test_macro_f1",
    "secondary_metrics": [
        "test_balanced_accuracy",
        "test_accuracy",
        "test_roc_auc_ovr_macro"
    ],
    "top_two_baseline_models": top_two_models,
    "comparison_table": comparison_df.to_dict(orient="records")
}

with open(output_dir / "task3c_baseline_selection_summary.json", "w", encoding="utf-8") as f:
    json.dump(selection_summary, f, indent=4)

print("\nTop two baseline models selected for cross-validation and tuning:")
print(top_two_models)

# ------------------------------------------------------------
# 17. Final message
# ------------------------------------------------------------

print("\n============================================================")
print("TASK 3C BASELINE MODEL COMPARISON COMPLETED")
print("============================================================")
print("Saved outputs in:")
print(output_dir.resolve())

print("\nFiles created:")
for file in sorted(output_dir.iterdir()):
    print("-", file.name)