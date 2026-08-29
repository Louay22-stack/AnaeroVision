# ============================================================
# TASK 3 — PREPROCESSING FOR KNOWN INHIBITION TYPE CLASSIFICATION
# Target: Inhibition_Type_Final
#
# Strategy:
# - Use inhibited samples only
# - Remove "Unknown" from model training
# - Classify known inhibition mechanisms only
# - Use confidence threshold later in the interface for "Unknown/uncertain"
# ============================================================

import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight

# ------------------------------------------------------------
# 0. Paths
# ------------------------------------------------------------

script_dir = Path(__file__).resolve().parent

input_dir = script_dir / "task3_cleaned_inhibition_type_outputs"
input_file = input_dir / "task3_inhibited_only_cleaned_dataset.csv"

if not input_file.exists():
    raise FileNotFoundError(
        "Missing cleaned Task 3 dataset. Run task3_clean_inhibition_type_labels.py first."
    )

output_dir = script_dir / "task3_preprocessing_known_types_outputs"
output_dir.mkdir(exist_ok=True)

print("Cleaned Task 3 dataset loaded from:")
print(input_file)

print("\nTask 3 preprocessing outputs will be saved in:")
print(output_dir.resolve())

# ------------------------------------------------------------
# 1. Load cleaned inhibited-only dataset
# ------------------------------------------------------------

df = pd.read_csv(input_file)

print("\nLoaded cleaned inhibited-only dataset shape:", df.shape)

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
    "Inhibition_Type_Final"
]

missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# ------------------------------------------------------------
# 2. Remove Unknown inhibition type
# ------------------------------------------------------------

before_remove_unknown = df.shape[0]

df_task3 = df[df["Inhibition_Type_Final"] != "Unknown"].copy()

after_remove_unknown = df_task3.shape[0]

print("\nRows before removing Unknown:", before_remove_unknown)
print("Rows after removing Unknown:", after_remove_unknown)
print("Rows removed as Unknown:", before_remove_unknown - after_remove_unknown)

print("\nKnown inhibition-type distribution:")
print(df_task3["Inhibition_Type_Final"].value_counts())

print("\nKnown inhibition-type distribution (%):")
print((df_task3["Inhibition_Type_Final"].value_counts(normalize=True) * 100).round(2))

# ------------------------------------------------------------
# 3. Keep selected input features and target
# ------------------------------------------------------------

target_col = "Inhibition_Type_Final"

feature_cols = [
    "Substrate",
    "Reactor_Type",
    "Temperature_C",
    "pH",
    "HRT_days",
    "OLR_gVS_L_day",
    "ISR",
    "TAN_mg_L",
    "VFA_mg_L"
]

X = df_task3[feature_cols].copy()
y_text = df_task3[target_col].copy()

print("\nInitial X shape:", X.shape)
print("Initial target shape:", y_text.shape)

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

X["Reactor_Type"] = X["Reactor_Type"].map(reactor_mapping)
X["Reactor_Type"] = X["Reactor_Type"].fillna("Other")

print("\nReactor type distribution after consolidation:")
print(X["Reactor_Type"].value_counts())

with open(output_dir / "reactor_mapping_task3.json", "w", encoding="utf-8") as f:
    json.dump(reactor_mapping, f, indent=4, ensure_ascii=False)

# ------------------------------------------------------------
# 5. Encode target labels
# ------------------------------------------------------------

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)

label_mapping = {
    int(i): str(label)
    for i, label in enumerate(label_encoder.classes_)
}

inverse_label_mapping = {
    str(label): int(i)
    for i, label in enumerate(label_encoder.classes_)
}

print("\nLabel mapping:")
print(label_mapping)

with open(output_dir / "task3_label_mapping.json", "w", encoding="utf-8") as f:
    json.dump(label_mapping, f, indent=4, ensure_ascii=False)

with open(output_dir / "task3_inverse_label_mapping.json", "w", encoding="utf-8") as f:
    json.dump(inverse_label_mapping, f, indent=4, ensure_ascii=False)

joblib.dump(label_encoder, output_dir / "label_encoder_task3.pkl")

# ------------------------------------------------------------
# 6. Train/test split before fitted preprocessing
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
# 7. Multiclass target encoding for Substrate
# This avoids ordinal bias from simple numeric target encoding.
# For each class, one substrate probability feature is created.
# ------------------------------------------------------------

def safe_name(text):
    return (
        str(text)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )

def fit_multiclass_target_encoder(X_train_df, y_train_labels, col_name, classes, smoothing=10):
    encoder_info = {
        "column": col_name,
        "smoothing": smoothing,
        "classes": list(classes),
        "global_priors": {},
        "mappings": {}
    }

    temp = X_train_df[[col_name]].copy()
    temp["target"] = y_train_labels.values

    counts = temp.groupby(col_name).size()

    for cls in classes:
        binary_target = (temp["target"] == cls).astype(int)
        global_prior = float(binary_target.mean())

        cls_means = temp.assign(binary_target=binary_target).groupby(col_name)["binary_target"].mean()

        smoothed_values = (
            (counts * cls_means + smoothing * global_prior) /
            (counts + smoothing)
        )

        encoder_info["global_priors"][cls] = global_prior
        encoder_info["mappings"][cls] = {
            str(k): float(v)
            for k, v in smoothed_values.to_dict().items()
        }

    return encoder_info

def transform_multiclass_target_encoder(X_df, encoder_info):
    X_encoded = X_df.copy()
    col_name = encoder_info["column"]

    for cls in encoder_info["classes"]:
        safe_cls = safe_name(cls)
        new_col = f"{col_name}_TE_{safe_cls}"

        mapping = encoder_info["mappings"][cls]
        global_prior = encoder_info["global_priors"][cls]

        X_encoded[new_col] = (
            X_encoded[col_name]
            .astype(str)
            .map(mapping)
            .fillna(global_prior)
        )

    X_encoded = X_encoded.drop(columns=[col_name])

    return X_encoded

classes = list(label_encoder.classes_)

substrate_encoder_info = fit_multiclass_target_encoder(
    X_train_df=X_train,
    y_train_labels=pd.Series(y_train_text),
    col_name="Substrate",
    classes=classes,
    smoothing=10
)

X_train = transform_multiclass_target_encoder(X_train, substrate_encoder_info)
X_test = transform_multiclass_target_encoder(X_test, substrate_encoder_info)

with open(output_dir / "substrate_multiclass_target_encoder_task3.json", "w", encoding="utf-8") as f:
    json.dump(substrate_encoder_info, f, indent=4, ensure_ascii=False)

print("\nMulticlass target encoding completed for Substrate.")

# ------------------------------------------------------------
# 8. One-hot encode Reactor_Type
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
# 9. Median imputation
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

joblib.dump(imputer, output_dir / "median_imputer_task3.pkl")

imputation_values = {
    col: float(value)
    for col, value in zip(cols_to_impute, imputer.statistics_)
}

with open(output_dir / "imputation_values_task3.json", "w", encoding="utf-8") as f:
    json.dump(imputation_values, f, indent=4)

print("\nMedian imputation completed.")
print("Imputation values:")
print(imputation_values)

# ------------------------------------------------------------
# 10. Standard scaling
# ------------------------------------------------------------

substrate_te_cols = [
    col for col in X_train.columns
    if col.startswith("Substrate_TE_")
]

numeric_cols = [
    "Temperature_C",
    "pH",
    "HRT_days",
    "OLR_gVS_L_day",
    "ISR",
    "TAN_mg_L",
    "VFA_mg_L"
] + substrate_te_cols

numeric_cols = [col for col in numeric_cols if col in X_train.columns]

scaler = StandardScaler()

X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

joblib.dump(scaler, output_dir / "standard_scaler_task3.pkl")

print("\nStandard scaling completed for numeric columns:")
print(numeric_cols)

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

with open(output_dir / "task3_class_weights.json", "w", encoding="utf-8") as f:
    json.dump(class_weight_dict, f, indent=4)

with open(output_dir / "task3_class_weights_named.json", "w", encoding="utf-8") as f:
    json.dump(class_weight_named, f, indent=4, ensure_ascii=False)

# ------------------------------------------------------------
# 12. Final checks
# ------------------------------------------------------------

print("\nFinal X_train shape:", X_train.shape)
print("Final X_test shape:", X_test.shape)

print("\nFinal feature columns:")
print(X_train.columns.tolist())

print("\nMissing values in X_train:", int(X_train.isna().sum().sum()))
print("Missing values in X_test:", int(X_test.isna().sum().sum()))

if X_train.isna().sum().sum() > 0 or X_test.isna().sum().sum() > 0:
    raise ValueError("Missing values still present after preprocessing.")

# ------------------------------------------------------------
# 13. Save outputs
# ------------------------------------------------------------

X_train.to_csv(output_dir / "X_train_task3_preprocessed.csv", index=False)
X_test.to_csv(output_dir / "X_test_task3_preprocessed.csv", index=False)

pd.Series(y_train, name="Inhibition_Type_Encoded").to_csv(
    output_dir / "y_train_task3_encoded.csv",
    index=False
)

pd.Series(y_test, name="Inhibition_Type_Encoded").to_csv(
    output_dir / "y_test_task3_encoded.csv",
    index=False
)

pd.Series(y_train_text.values, name="Inhibition_Type_Final").to_csv(
    output_dir / "y_train_task3_labels.csv",
    index=False
)

pd.Series(y_test_text.values, name="Inhibition_Type_Final").to_csv(
    output_dir / "y_test_task3_labels.csv",
    index=False
)

train_table = X_train.copy()
train_table["Inhibition_Type_Encoded"] = y_train
train_table["Inhibition_Type_Final"] = y_train_text.values

test_table = X_test.copy()
test_table["Inhibition_Type_Encoded"] = y_test
test_table["Inhibition_Type_Final"] = y_test_text.values

train_table.to_csv(
    output_dir / "train_task3_preprocessed_with_target.csv",
    index=False
)

test_table.to_csv(
    output_dir / "test_task3_preprocessed_with_target.csv",
    index=False
)

feature_names = X_train.columns.tolist()

with open(output_dir / "task3_feature_names.json", "w", encoding="utf-8") as f:
    json.dump(feature_names, f, indent=4)

preprocessing_summary = {
    "task": "Task 3 - Known inhibition type classification",
    "target": "Inhibition_Type_Final",
    "unknown_removed": True,
    "rows_before_removing_unknown": int(before_remove_unknown),
    "rows_after_removing_unknown": int(after_remove_unknown),
    "known_classes": list(label_encoder.classes_),
    "label_mapping": label_mapping,
    "train_shape": list(X_train.shape),
    "test_shape": list(X_test.shape),
    "train_target_distribution": {
        str(k): int(v)
        for k, v in pd.Series(y_train_text).value_counts().items()
    },
    "test_target_distribution": {
        str(k): int(v)
        for k, v in pd.Series(y_test_text).value_counts().items()
    },
    "numeric_scaled_columns": numeric_cols,
    "imputed_columns": cols_to_impute,
    "imputation_values": imputation_values,
    "class_weights_named": class_weight_named,
    "feature_names": feature_names
}

with open(output_dir / "task3_preprocessing_summary.json", "w", encoding="utf-8") as f:
    json.dump(preprocessing_summary, f, indent=4, ensure_ascii=False)

# ------------------------------------------------------------
# 14. Final message
# ------------------------------------------------------------

print("\n============================================================")
print("TASK 3 PREPROCESSING FOR KNOWN TYPES COMPLETED SUCCESSFULLY")
print("============================================================")
print("Saved outputs in:")
print(output_dir.resolve())

print("\nFiles created:")
for file in sorted(output_dir.iterdir()):
    print("-", file.name)