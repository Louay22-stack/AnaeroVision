# ============================================================
# TASK 2 — FINALIZE PREPROCESSING FOR INHIBITION RISK CLASSIFICATION
# Target: Inhibition_Label_Numeric
# No model training in this script
# Excludes: TAN_was_missing and VFA_was_missing
# ============================================================

import json
import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from category_encoders import TargetEncoder

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

output_dir = script_dir / "task2_preprocessing_outputs"
output_dir.mkdir(exist_ok=True)

print("Dataset loaded from:")
print(file_path)

print("\nTask 2 preprocessing outputs will be saved in:")
print(output_dir.resolve())

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

df = pd.read_excel(file_path)

print("\nOriginal dataset shape:", df.shape)

# ------------------------------------------------------------
# 2. Define target and check required columns
# ------------------------------------------------------------

target_col = "Inhibition_Label_Numeric"

required_cols = [
    "Substrate",
    "Reactor_Type",
    "Temperature_C",
    "pH",
    "HRT_days",
    "OLR_gVS_L_day",
    "ISR",
    "TAN_mg_L",
    "VFA_mg_L",
    target_col
]

missing_required_cols = [col for col in required_cols if col not in df.columns]

if missing_required_cols:
    raise ValueError(f"Missing required columns: {missing_required_cols}")

# ------------------------------------------------------------
# 3. Drop metadata, leakage, non-selected, and non-task columns
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

    # Task 1 target — not used as raw input for Task 2
    "Methane_Yield_mL_CH4_gVS",

    # User decision: remove completely from model inputs
    "TAN_was_missing",
    "VFA_was_missing"
]

df_task2 = df.drop(columns=[col for col in cols_to_drop if col in df.columns]).copy()

print("\nShape after dropping non-Task-2 columns:", df_task2.shape)
print("\nRemaining columns:")
print(df_task2.columns.tolist())

# ------------------------------------------------------------
# 4. Remove rows with missing classification target
# ------------------------------------------------------------

before_drop_target = df_task2.shape[0]

df_task2 = df_task2.dropna(subset=[target_col]).copy()

after_drop_target = df_task2.shape[0]

print("\nRows before dropping missing target:", before_drop_target)
print("Rows after dropping missing target:", after_drop_target)
print("Rows removed because target was missing:", before_drop_target - after_drop_target)

# Ensure target is integer 0/1
df_task2[target_col] = df_task2[target_col].astype(int)

print("\nFull dataset target distribution:")
print(df_task2[target_col].value_counts().sort_index())

print("\nFull dataset target distribution (%):")
print((df_task2[target_col].value_counts(normalize=True).sort_index() * 100).round(2))

# ------------------------------------------------------------
# 5. Consolidate Reactor_Type
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

print("\nReactor type distribution after consolidation:")
print(df_task2["Reactor_Type"].value_counts())

with open(output_dir / "reactor_mapping_task2.json", "w", encoding="utf-8") as f:
    json.dump(reactor_mapping, f, indent=4, ensure_ascii=False)

# ------------------------------------------------------------
# 6. Define X and y
# ------------------------------------------------------------

X = df_task2.drop(columns=[target_col]).copy()
y = df_task2[target_col].copy()

print("\nInitial X shape:", X.shape)
print("Initial y shape:", y.shape)

print("\nInitial feature columns:")
print(X.columns.tolist())

# ------------------------------------------------------------
# 7. Train/test split BEFORE fitted preprocessing
# Stratify is used here to preserve inhibition/no-inhibition ratio
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain shape before preprocessing:", X_train.shape)
print("Test shape before preprocessing:", X_test.shape)

print("\nTrain target distribution:")
print(y_train.value_counts().sort_index())

print("\nTrain target distribution (%):")
print((y_train.value_counts(normalize=True).sort_index() * 100).round(2))

print("\nTest target distribution:")
print(y_test.value_counts().sort_index())

print("\nTest target distribution (%):")
print((y_test.value_counts(normalize=True).sort_index() * 100).round(2))

# ------------------------------------------------------------
# 8. Target encoding for Substrate
# Fit only on training data
# For Task 2, encode against Inhibition_Label_Numeric
# ------------------------------------------------------------

target_encoder = TargetEncoder(cols=["Substrate"])

X_train["Substrate"] = target_encoder.fit_transform(
    X_train["Substrate"],
    y_train
)

X_test["Substrate"] = target_encoder.transform(
    X_test["Substrate"]
)

joblib.dump(target_encoder, output_dir / "target_encoder_task2.pkl")

print("\nTarget encoding completed for Substrate.")

# ------------------------------------------------------------
# 9. One-hot encode Reactor_Type
# ------------------------------------------------------------

X_train = pd.get_dummies(
    X_train,
    columns=["Reactor_Type"],
    drop_first=False
)

X_test = pd.get_dummies(
    X_test,
    columns=["Reactor_Type"],
    drop_first=False
)

X_train, X_test = X_train.align(
    X_test,
    join="left",
    axis=1,
    fill_value=0
)

print("\nOne-hot encoding completed for Reactor_Type.")

# ------------------------------------------------------------
# 10. Median imputation for missing numeric process variables
# Fit only on training data
# ------------------------------------------------------------

cols_to_impute = [
    "HRT_days",
    "OLR_gVS_L_day",
    "ISR"
]

cols_to_impute = [col for col in cols_to_impute if col in X_train.columns]

imputer = SimpleImputer(strategy="median")

X_train[cols_to_impute] = imputer.fit_transform(X_train[cols_to_impute])
X_test[cols_to_impute] = imputer.transform(X_test[cols_to_impute])

joblib.dump(imputer, output_dir / "median_imputer_task2.pkl")

imputation_values = {
    col: float(value)
    for col, value in zip(cols_to_impute, imputer.statistics_)
}

with open(output_dir / "imputation_values_task2.json", "w", encoding="utf-8") as f:
    json.dump(imputation_values, f, indent=4)

print("\nMedian imputation completed.")
print("Imputation values:")
print(imputation_values)

# ------------------------------------------------------------
# 11. Scale numeric features only
# Do NOT scale one-hot Reactor_Type columns
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

numeric_cols = [col for col in numeric_cols if col in X_train.columns]

scaler = StandardScaler()

X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

joblib.dump(scaler, output_dir / "standard_scaler_task2.pkl")

print("\nStandard scaling completed for numeric columns:")
print(numeric_cols)

# ------------------------------------------------------------
# 12. Compute class weights for imbalanced classification
# ------------------------------------------------------------

classes = np.array([0, 1])

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weight_dict = {
    int(cls): float(weight)
    for cls, weight in zip(classes, class_weights)
}

n_class_0 = int((y_train == 0).sum())
n_class_1 = int((y_train == 1).sum())

scale_pos_weight = n_class_0 / n_class_1 if n_class_1 > 0 else None

imbalance_info = {
    "train_class_distribution": {
        str(int(k)): int(v)
        for k, v in y_train.value_counts().sort_index().items()
    },
    "test_class_distribution": {
        str(int(k)): int(v)
        for k, v in y_test.value_counts().sort_index().items()
    },
    "class_weight_dict": {
        str(k): v
        for k, v in class_weight_dict.items()
    },
    "scale_pos_weight_for_xgboost": scale_pos_weight
}

with open(output_dir / "task2_class_imbalance_info.json", "w", encoding="utf-8") as f:
    json.dump(imbalance_info, f, indent=4)

print("\nClass imbalance information:")
print(imbalance_info)

# ------------------------------------------------------------
# 13. Final checks
# ------------------------------------------------------------

print("\nFinal X_train shape:", X_train.shape)
print("Final X_test shape:", X_test.shape)

print("\nFinal feature columns:")
print(X_train.columns.tolist())

print("\nMissing values in X_train:", int(X_train.isna().sum().sum()))
print("Missing values in X_test:", int(X_test.isna().sum().sum()))

forbidden_cols = [
    "TAN_was_missing",
    "VFA_was_missing",
    "Methane_Yield_mL_CH4_gVS",
    "Inhibition_Status",
    "Original_Inhibition_Label",
    "Inhibition_Type",
    "CH4_percent",
    "Substrate_Category"
]

still_present = [col for col in forbidden_cols if col in X_train.columns]

if still_present:
    raise ValueError(f"Forbidden columns still present in X_train: {still_present}")
else:
    print("\nForbidden columns check passed.")

# ------------------------------------------------------------
# 14. Save preprocessed outputs
# ------------------------------------------------------------

X_train.to_csv(output_dir / "X_train_task2_preprocessed.csv", index=False)
X_test.to_csv(output_dir / "X_test_task2_preprocessed.csv", index=False)

y_train.to_csv(output_dir / "y_train_task2.csv", index=False)
y_test.to_csv(output_dir / "y_test_task2.csv", index=False)

train_table = X_train.copy()
train_table[target_col] = y_train.values

test_table = X_test.copy()
test_table[target_col] = y_test.values

train_table.to_csv(output_dir / "train_task2_preprocessed_with_target.csv", index=False)
test_table.to_csv(output_dir / "test_task2_preprocessed_with_target.csv", index=False)

feature_names = X_train.columns.tolist()

with open(output_dir / "task2_feature_names.json", "w", encoding="utf-8") as f:
    json.dump(feature_names, f, indent=4)

preprocessing_summary = {
    "target": target_col,
    "original_dataset_shape": list(df.shape),
    "task2_dataset_shape_after_drops": list(df_task2.shape),
    "train_shape": list(X_train.shape),
    "test_shape": list(X_test.shape),
    "excluded_from_model_input": [
        "TAN_was_missing",
        "VFA_was_missing",
        "Methane_Yield_mL_CH4_gVS"
    ],
    "dropped_columns": [col for col in cols_to_drop if col in df.columns],
    "numeric_scaled_columns": numeric_cols,
    "imputed_columns": cols_to_impute,
    "imputation_values": imputation_values,
    "class_imbalance_info": imbalance_info,
    "final_feature_names": feature_names
}

with open(output_dir / "task2_preprocessing_summary.json", "w", encoding="utf-8") as f:
    json.dump(preprocessing_summary, f, indent=4)

# ------------------------------------------------------------
# 15. Final message
# ------------------------------------------------------------

print("\n============================================================")
print("TASK 2 PREPROCESSING COMPLETED SUCCESSFULLY")
print("============================================================")
print("Saved outputs in:")
print(output_dir.resolve())

print("\nFiles created:")
for file in sorted(output_dir.iterdir()):
    print("-", file.name)