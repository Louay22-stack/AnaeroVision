# ============================================================
# TASK 1 — FINALIZE PREPROCESSING FOR METHANE YIELD REGRESSION
# No model training in this script
# Target: Methane_Yield_mL_CH4_gVS
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
from category_encoders import TargetEncoder

# ------------------------------------------------------------
# 0. Paths
# ------------------------------------------------------------

script_dir = Path(__file__).resolve().parent

file_path = script_dir / "Operational-dataset.xlsx"

# Backup option in case your file has this name
if not file_path.exists():
    file_path = script_dir / "Operational-dataset(1).xlsx"

if not file_path.exists():
    raise FileNotFoundError(
        "Dataset not found. Put 'Operational-dataset.xlsx' or "
        "'Operational-dataset(1).xlsx' in the same folder as this script."
    )

output_dir = script_dir / "task1_preprocessing_outputs"
output_dir.mkdir(exist_ok=True)

print("Dataset loaded from:")
print(file_path)

print("\nPreprocessing outputs will be saved in:")
print(output_dir.resolve())

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

df = pd.read_excel(file_path)

print("\nOriginal dataset shape:", df.shape)

# ------------------------------------------------------------
# 2. Basic column checks
# ------------------------------------------------------------

target_col = "Methane_Yield_mL_CH4_gVS"

required_cols = [
    "Substrate",
    "Substrate_Category",
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
# 3. Build substrate category lookup before dropping category
# ------------------------------------------------------------

substrate_category_df = (
    df[["Substrate", "Substrate_Category"]]
    .dropna(subset=["Substrate"])
    .drop_duplicates()
)

# Check whether one substrate appears with multiple categories
category_counts = (
    substrate_category_df
    .groupby("Substrate")["Substrate_Category"]
    .nunique()
    .reset_index(name="n_categories")
)

conflicting_substrates = category_counts[category_counts["n_categories"] > 1]

if not conflicting_substrates.empty:
    conflicting_details = substrate_category_df[
        substrate_category_df["Substrate"].isin(conflicting_substrates["Substrate"])
    ].sort_values("Substrate")

    conflicting_details.to_csv(
        output_dir / "substrate_category_conflicts.csv",
        index=False
    )

    print("\nWarning: Some substrates have multiple categories.")
    print("A conflict report was saved as substrate_category_conflicts.csv")

# Build lookup using the first category found for each substrate
substrate_category_map = (
    substrate_category_df
    .drop_duplicates(subset=["Substrate"], keep="first")
    .set_index("Substrate")["Substrate_Category"]
    .to_dict()
)

with open(output_dir / "substrate_category_map.json", "w", encoding="utf-8") as f:
    json.dump(substrate_category_map, f, indent=4, ensure_ascii=False)

print("\nSubstrate category lookup saved.")
print("Number of substrates in lookup:", len(substrate_category_map))

# ------------------------------------------------------------
# 4. Drop metadata, leakage, non-selected, and non-task columns
# ------------------------------------------------------------

cols_to_drop = [
    # Metadata
    "Experiment_ID",
    "Reference",
    "DOI",
    "Source",

    # Not used as predictor
    "Substrate_Category",

    # Leakage / non-input output
    "CH4_percent",

    # Classification targets / labels not used in Task 1 regression
    "Inhibition_Status",
    "Original_Inhibition_Label",
    "Inhibition_Type",
    "Inhibition_Label_Numeric",

    # User decision: remove completely from EDA and model input
    "TAN_was_missing",
    "VFA_was_missing"
]

df_task1 = df.drop(columns=[col for col in cols_to_drop if col in df.columns]).copy()

print("\nShape after dropping non-regression columns:", df_task1.shape)
print("\nRemaining columns:")
print(df_task1.columns.tolist())

# ------------------------------------------------------------
# 5. Remove rows with missing regression target
# ------------------------------------------------------------

before_drop_target = df_task1.shape[0]

df_task1 = df_task1.dropna(subset=[target_col]).copy()

after_drop_target = df_task1.shape[0]

print("\nRows before dropping missing target:", before_drop_target)
print("Rows after dropping missing target:", after_drop_target)
print("Rows removed because target was missing:", before_drop_target - after_drop_target)

# ------------------------------------------------------------
# 6. Consolidate Reactor_Type
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

df_task1["Reactor_Type"] = df_task1["Reactor_Type"].map(reactor_mapping)

# Any reactor type not included in mapping becomes Other
df_task1["Reactor_Type"] = df_task1["Reactor_Type"].fillna("Other")

print("\nReactor type distribution after consolidation:")
print(df_task1["Reactor_Type"].value_counts())

# Save reactor mapping
with open(output_dir / "reactor_mapping.json", "w", encoding="utf-8") as f:
    json.dump(reactor_mapping, f, indent=4, ensure_ascii=False)

# ------------------------------------------------------------
# 7. Define X and y
# ------------------------------------------------------------

X = df_task1.drop(columns=[target_col]).copy()
y = df_task1[target_col].copy()

print("\nInitial X shape:", X.shape)
print("Initial y shape:", y.shape)

print("\nInitial feature columns:")
print(X.columns.tolist())

# ------------------------------------------------------------
# 8. Train/test split BEFORE fitted preprocessing
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain shape before preprocessing:", X_train.shape)
print("Test shape before preprocessing:", X_test.shape)

# ------------------------------------------------------------
# 9. Target encoding for Substrate
# Fit only on training data
# ------------------------------------------------------------

target_encoder = TargetEncoder(cols=["Substrate"])

X_train["Substrate"] = target_encoder.fit_transform(
    X_train["Substrate"],
    y_train
)

X_test["Substrate"] = target_encoder.transform(
    X_test["Substrate"]
)

joblib.dump(target_encoder, output_dir / "target_encoder_task1.pkl")

print("\nTarget encoding completed for Substrate.")

# ------------------------------------------------------------
# 10. One-hot encode Reactor_Type
# Fit structure using training data, then align test data
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

# Ensure train and test have exactly the same columns
X_train, X_test = X_train.align(
    X_test,
    join="left",
    axis=1,
    fill_value=0
)

print("\nOne-hot encoding completed for Reactor_Type.")

# ------------------------------------------------------------
# 11. Median imputation for missing numeric process variables
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

joblib.dump(imputer, output_dir / "median_imputer_task1.pkl")

imputation_values = dict(zip(cols_to_impute, imputer.statistics_))

with open(output_dir / "imputation_values_task1.json", "w", encoding="utf-8") as f:
    json.dump(imputation_values, f, indent=4)

print("\nMedian imputation completed.")
print("Imputation values:")
print(imputation_values)

# ------------------------------------------------------------
# 12. Scale numeric features only
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

joblib.dump(scaler, output_dir / "standard_scaler_task1.pkl")

print("\nStandard scaling completed for numeric columns:")
print(numeric_cols)

# ------------------------------------------------------------
# 13. Check target skewness and log-transform decision
# No transform applied here unless skewness > 1.0
# ------------------------------------------------------------

target_skewness = y_train.skew()

LOG_TRANSFORM_APPLIED = False

if target_skewness > 1.0:
    y_train_processed = np.log1p(y_train)
    y_test_processed = np.log1p(y_test)
    LOG_TRANSFORM_APPLIED = True
else:
    y_train_processed = y_train.copy()
    y_test_processed = y_test.copy()

print("\nMethane yield skewness in training set:", round(target_skewness, 4))
print("Log transform applied:", LOG_TRANSFORM_APPLIED)

# ------------------------------------------------------------
# 14. Final checks
# ------------------------------------------------------------

print("\nFinal X_train shape:", X_train.shape)
print("Final X_test shape:", X_test.shape)

print("\nFinal feature columns:")
print(X_train.columns.tolist())

print("\nMissing values in X_train:", int(X_train.isna().sum().sum()))
print("Missing values in X_test:", int(X_test.isna().sum().sum()))

# Safety check: make sure removed columns are not present
forbidden_cols = [
    "TAN_was_missing",
    "VFA_was_missing",
    "Inhibition_Label_Numeric",
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
# 15. Save preprocessed outputs
# ------------------------------------------------------------

X_train.to_csv(output_dir / "X_train_task1_preprocessed.csv", index=False)
X_test.to_csv(output_dir / "X_test_task1_preprocessed.csv", index=False)

y_train_processed.to_csv(output_dir / "y_train_task1_processed.csv", index=False)
y_test_processed.to_csv(output_dir / "y_test_task1_processed.csv", index=False)

# Save untransformed targets too
y_train.to_csv(output_dir / "y_train_task1_original.csv", index=False)
y_test.to_csv(output_dir / "y_test_task1_original.csv", index=False)

# Save full train/test tables with target
train_table = X_train.copy()
train_table[target_col] = y_train_processed.values

test_table = X_test.copy()
test_table[target_col] = y_test_processed.values

train_table.to_csv(output_dir / "train_task1_preprocessed_with_target.csv", index=False)
test_table.to_csv(output_dir / "test_task1_preprocessed_with_target.csv", index=False)

# Save feature names
feature_names = X_train.columns.tolist()

with open(output_dir / "task1_feature_names.json", "w", encoding="utf-8") as f:
    json.dump(feature_names, f, indent=4)

# Save preprocessing summary
preprocessing_summary = {
    "target": target_col,
    "original_dataset_shape": list(df.shape),
    "task1_dataset_shape_after_drops": list(df_task1.shape),
    "train_shape": list(X_train.shape),
    "test_shape": list(X_test.shape),
    "excluded_from_model_input": [
        "TAN_was_missing",
        "VFA_was_missing"
    ],
    "dropped_columns": [col for col in cols_to_drop if col in df.columns],
    "numeric_scaled_columns": numeric_cols,
    "imputed_columns": cols_to_impute,
    "imputation_values": imputation_values,
    "target_skewness_train": float(target_skewness),
    "log_transform_applied": LOG_TRANSFORM_APPLIED,
    "final_feature_names": feature_names
}

with open(output_dir / "task1_preprocessing_summary.json", "w", encoding="utf-8") as f:
    json.dump(preprocessing_summary, f, indent=4)

# ------------------------------------------------------------
# 16. Final message
# ------------------------------------------------------------

print("\n============================================================")
print("TASK 1 PREPROCESSING COMPLETED SUCCESSFULLY")
print("============================================================")
print("Saved outputs in:")
print(output_dir.resolve())

print("\nFiles created:")
for file in sorted(output_dir.iterdir()):
    print("-", file.name)