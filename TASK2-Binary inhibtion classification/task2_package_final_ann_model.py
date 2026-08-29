# ============================================================
# TASK 2 — PACKAGE FINAL INHIBITION RISK CLASSIFICATION MODEL
# Final model: ANN / MLPClassifier
# Final decision threshold: 0.14
# Trained on full dataset for deployment
# ============================================================

import json
import joblib
import pandas as pd

from pathlib import Path

from category_encoders import TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample

# ------------------------------------------------------------
# 0. Paths
# ------------------------------------------------------------

script_dir = Path(__file__).resolve().parent

file_path = script_dir / "Operational-dataset.xlsx"

if not file_path.exists():
    file_path = script_dir / "Operational-dataset(1).xlsx"

if not file_path.exists():
    raise FileNotFoundError(
        "Dataset not found. Put 'Operational-dataset.xlsx' or "
        "'Operational-dataset(1).xlsx' in the same folder as this script."
    )

output_dir = script_dir / "final_task2_inhibition_risk_model"
output_dir.mkdir(exist_ok=True)

print("Dataset loaded from:")
print(file_path)

print("\nFinal Task 2 model package will be saved in:")
print(output_dir.resolve())

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

df = pd.read_excel(file_path)

print("\nOriginal dataset shape:", df.shape)

target_col = "Inhibition_Label_Numeric"

# ------------------------------------------------------------
# 2. Save substrate-category lookup if available
# ------------------------------------------------------------

if "Substrate_Category" in df.columns:
    substrate_category_map = (
        df[["Substrate", "Substrate_Category"]]
        .dropna()
        .drop_duplicates(subset=["Substrate"])
        .set_index("Substrate")["Substrate_Category"]
        .to_dict()
    )

    with open(output_dir / "substrate_category_map_task2_final.json", "w", encoding="utf-8") as f:
        json.dump(substrate_category_map, f, indent=4, ensure_ascii=False)

    print("\nSubstrate-category lookup saved.")
    print("Number of substrates in lookup:", len(substrate_category_map))

# ------------------------------------------------------------
# 3. Drop non-model columns
# ------------------------------------------------------------

cols_to_drop = [
    # Metadata
    "Experiment_ID",
    "Reference",
    "DOI",
    "Source",

    # Lookup output, not model input
    "Substrate_Category",

    # Post-process output, not predictor
    "CH4_percent",

    # Redundant or non-task labels
    "Inhibition_Status",
    "Original_Inhibition_Label",

    # Task 3 target
    "Inhibition_Type",

    # Task 1 target
    "Methane_Yield_mL_CH4_gVS",

    # Removed by decision
    "TAN_was_missing",
    "VFA_was_missing"
]

df_task2 = df.drop(columns=[col for col in cols_to_drop if col in df.columns]).copy()

df_task2 = df_task2.dropna(subset=[target_col]).copy()
df_task2[target_col] = df_task2[target_col].astype(int)

print("\nTask 2 final dataset shape:", df_task2.shape)

print("\nTarget distribution:")
print(df_task2[target_col].value_counts().sort_index())

print("\nTarget distribution (%):")
print((df_task2[target_col].value_counts(normalize=True).sort_index() * 100).round(2))

# ------------------------------------------------------------
# 4. Consolidate Reactor_Type
# ------------------------------------------------------------

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

df_task2["Reactor_Type"] = df_task2["Reactor_Type"].map(reactor_mapping)
df_task2["Reactor_Type"] = df_task2["Reactor_Type"].fillna("Other")

with open(output_dir / "reactor_mapping_task2_final.json", "w", encoding="utf-8") as f:
    json.dump(reactor_mapping, f, indent=4, ensure_ascii=False)

print("\nReactor type distribution after consolidation:")
print(df_task2["Reactor_Type"].value_counts())

# ------------------------------------------------------------
# 5. Define X and y
# ------------------------------------------------------------

X = df_task2.drop(columns=[target_col]).copy()
y = df_task2[target_col].copy()

print("\nInitial X shape:", X.shape)
print("Initial y shape:", y.shape)

# ------------------------------------------------------------
# 6. Target encode Substrate on full dataset
# ------------------------------------------------------------

target_encoder = TargetEncoder(cols=["Substrate"])

X["Substrate"] = target_encoder.fit_transform(
    X["Substrate"],
    y
)

joblib.dump(target_encoder, output_dir / "target_encoder_task2_final.pkl")

print("\nTarget encoding completed for Substrate.")

# ------------------------------------------------------------
# 7. One-hot encode Reactor_Type
# ------------------------------------------------------------

X = pd.get_dummies(
    X,
    columns=["Reactor_Type"],
    drop_first=False
)

print("\nOne-hot encoding completed for Reactor_Type.")

# ------------------------------------------------------------
# 8. Median imputation
# ------------------------------------------------------------

cols_to_impute = [
    "HRT_days",
    "OLR_gVS_L_day",
    "ISR"
]

cols_to_impute = [col for col in cols_to_impute if col in X.columns]

imputer = SimpleImputer(strategy="median")

X[cols_to_impute] = imputer.fit_transform(X[cols_to_impute])

joblib.dump(imputer, output_dir / "median_imputer_task2_final.pkl")

imputation_values = {
    col: float(value)
    for col, value in zip(cols_to_impute, imputer.statistics_)
}

with open(output_dir / "imputation_values_task2_final.json", "w", encoding="utf-8") as f:
    json.dump(imputation_values, f, indent=4)

print("\nMedian imputation completed.")
print("Imputation values:")
print(imputation_values)

# ------------------------------------------------------------
# 9. Standard scaling
# ------------------------------------------------------------

numeric_cols = [
    "Temperature_C",
    "pH",
    "HRT_days",
    "OLR_gVS_L_day",
    "ISR",
    "TAN_mg_L",
    "VFA_mg_L",
    "Substrate"
]

numeric_cols = [col for col in numeric_cols if col in X.columns]

scaler = StandardScaler()

X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

joblib.dump(scaler, output_dir / "standard_scaler_task2_final.pkl")

print("\nStandard scaling completed for numeric columns:")
print(numeric_cols)

# ------------------------------------------------------------
# 10. Final feature check
# ------------------------------------------------------------

feature_names = X.columns.tolist()

with open(output_dir / "task2_final_feature_names.json", "w", encoding="utf-8") as f:
    json.dump(feature_names, f, indent=4)

print("\nFinal feature columns:")
print(feature_names)

print("\nFinal X shape:", X.shape)
print("Missing values in X:", int(X.isna().sum().sum()))

if X.isna().sum().sum() > 0:
    raise ValueError("Missing values still present after preprocessing.")

# ------------------------------------------------------------
# 11. Balance full training dataset by upsampling inhibition class
# ------------------------------------------------------------

train_data = X.copy()
train_data["target"] = y.values

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

X_final_train = train_balanced.drop(columns=["target"])
y_final_train = train_balanced["target"]

print("\nBalanced final training distribution:")
print(y_final_train.value_counts().sort_index())

# ------------------------------------------------------------
# 12. Train final ANN / MLPClassifier
# ------------------------------------------------------------

final_ann = MLPClassifier(
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

final_ann.fit(X_final_train, y_final_train)

print("\nFinal ANN training completed.")
print("Number of iterations:", final_ann.n_iter_)
print("Final loss:", final_ann.loss_)

# ------------------------------------------------------------
# 13. Save final model and metadata
# ------------------------------------------------------------

joblib.dump(final_ann, output_dir / "ann_mlp_task2_final_model.pkl")

final_threshold = 0.14

model_summary = {
    "task": "Task 2 - Inhibition risk classification",
    "final_model": "ANN / MLPClassifier",
    "target": target_col,
    "decision_threshold": final_threshold,
    "validation_method_used_for_model_selection": "Substrate-based group split",
    "reported_validation_performance": {
        "accuracy": 0.9122,
        "precision_class_1": 0.8333,
        "recall_class_1": 0.8621,
        "f1_class_1": 0.8475,
        "false_positives": 10,
        "false_negatives": 8,
        "true_positives": 50,
        "true_negatives": 137
    },
    "training_for_final_package": "Final model trained on full 1500-row dataset after model selection.",
    "ann_parameters": {
        "hidden_layer_sizes": [128, 64],
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.001,
        "batch_size": 32,
        "learning_rate_init": 0.001,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 30,
        "random_state": 42
    },
    "original_dataset_shape": list(df.shape),
    "final_preprocessed_shape": list(X.shape),
    "balanced_training_shape": list(X_final_train.shape),
    "final_feature_names": feature_names,
    "numeric_scaled_columns": numeric_cols,
    "imputed_columns": cols_to_impute,
    "imputation_values": imputation_values
}

with open(output_dir / "task2_final_model_summary.json", "w", encoding="utf-8") as f:
    json.dump(model_summary, f, indent=4)

# Save final preprocessed full dataset
full_preprocessed = X.copy()
full_preprocessed[target_col] = y.values

full_preprocessed.to_csv(
    output_dir / "task2_full_dataset_preprocessed_with_target.csv",
    index=False
)

# ------------------------------------------------------------
# 14. Final message
# ------------------------------------------------------------

print("\n============================================================")
print("FINAL TASK 2 MODEL PACKAGE CREATED SUCCESSFULLY")
print("============================================================")

print("\nFinal model:")
print("ANN / MLPClassifier")

print("\nDecision threshold:")
print(final_threshold)

print("\nReported validation performance from group-split threshold tuning:")
print("Accuracy: 0.9122")
print("Precision class 1: 0.8333")
print("Recall class 1: 0.8621")
print("F1 class 1: 0.8475")

print("\nSaved outputs in:")
print(output_dir.resolve())

print("\nFiles created:")
for file in sorted(output_dir.iterdir()):
    print("-", file.name)