# ============================================================
# TASK 2 — GROUP SPLIT SANITY CHECK BY SUBSTRATE
# Inhibition Risk Classification
#
# Purpose:
# Check whether high performance is caused by substrate overlap
# between train and test sets.
#
# Important:
# This script starts again from the RAW dataset.
# Do NOT use the already-preprocessed Task 2 CSV files here.
# ============================================================

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from category_encoders import TargetEncoder

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")

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

output_dir = script_dir / "task2_group_split_sanity_check_outputs"
output_dir.mkdir(exist_ok=True)

print("Dataset loaded from:")
print(file_path)

print("\nOutputs will be saved in:")
print(output_dir.resolve())

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

df = pd.read_excel(file_path)

print("\nOriginal dataset shape:", df.shape)

target_col = "Inhibition_Label_Numeric"
group_col = "Substrate"

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
# 2. Drop non-model columns
# ------------------------------------------------------------

cols_to_drop = [
    "Experiment_ID",
    "Reference",
    "DOI",
    "Source",
    "Substrate_Category",
    "CH4_percent",
    "Inhibition_Status",
    "Original_Inhibition_Label",
    "Inhibition_Type",
    "Methane_Yield_mL_CH4_gVS",
    "TAN_was_missing",
    "VFA_was_missing"
]

df_task2 = df.drop(columns=[col for col in cols_to_drop if col in df.columns]).copy()

df_task2 = df_task2.dropna(subset=[target_col]).copy()
df_task2[target_col] = df_task2[target_col].astype(int)

print("\nTask 2 dataset shape after column drops:", df_task2.shape)

print("\nFull target distribution:")
print(df_task2[target_col].value_counts().sort_index())

print("\nFull target distribution (%):")
print((df_task2[target_col].value_counts(normalize=True).sort_index() * 100).round(2))

# ------------------------------------------------------------
# 3. Consolidate Reactor_Type
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

# ------------------------------------------------------------
# 4. Define X, y, groups
# ------------------------------------------------------------

X = df_task2.drop(columns=[target_col]).copy()
y = df_task2[target_col].copy()
groups = df_task2[group_col].copy()

print("\nNumber of unique substrates:", groups.nunique())

# ------------------------------------------------------------
# 5. Group split by Substrate
# Find a valid split where both train and test contain both classes
# ------------------------------------------------------------

selected_split = None

for seed in range(42, 200):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=seed
    )

    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    y_train_temp = y.iloc[train_idx]
    y_test_temp = y.iloc[test_idx]

    if y_train_temp.nunique() == 2 and y_test_temp.nunique() == 2:
        selected_split = (train_idx, test_idx, seed)
        break

if selected_split is None:
    raise ValueError("Could not find a valid group split with both classes in train and test.")

train_idx, test_idx, selected_seed = selected_split

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()

groups_train = groups.iloc[train_idx].copy()
groups_test = groups.iloc[test_idx].copy()

print("\nSelected random seed for group split:", selected_seed)

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

train_substrates = set(groups_train.unique())
test_substrates = set(groups_test.unique())
overlap = train_substrates.intersection(test_substrates)

print("\nUnique substrates in train:", len(train_substrates))
print("Unique substrates in test:", len(test_substrates))
print("Substrate overlap between train and test:", len(overlap))

if len(overlap) > 0:
    raise ValueError("Group split failed: some substrates appear in both train and test.")
else:
    print("Group split check passed: no substrate overlap.")

# Save split information
split_info = {
    "selected_seed": selected_seed,
    "train_shape_before_preprocessing": list(X_train.shape),
    "test_shape_before_preprocessing": list(X_test.shape),
    "unique_train_substrates": len(train_substrates),
    "unique_test_substrates": len(test_substrates),
    "substrate_overlap": len(overlap),
    "train_target_distribution": {
        str(k): int(v)
        for k, v in y_train.value_counts().sort_index().items()
    },
    "test_target_distribution": {
        str(k): int(v)
        for k, v in y_test.value_counts().sort_index().items()
    }
}

with open(output_dir / "group_split_info.json", "w", encoding="utf-8") as f:
    json.dump(split_info, f, indent=4)

# ------------------------------------------------------------
# 6. Target encoding for Substrate
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

print("\nTarget encoding completed.")

# ------------------------------------------------------------
# 7. One-hot encode Reactor_Type
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

print("One-hot encoding completed.")

# ------------------------------------------------------------
# 8. Median imputation
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

print("Median imputation completed.")

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

numeric_cols = [col for col in numeric_cols if col in X_train.columns]

scaler = StandardScaler()

X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

print("Standard scaling completed.")

print("\nFinal X_train shape:", X_train.shape)
print("Final X_test shape:", X_test.shape)

print("\nMissing values in X_train:", int(X_train.isna().sum().sum()))
print("Missing values in X_test:", int(X_test.isna().sum().sum()))

if X_train.isna().sum().sum() > 0 or X_test.isna().sum().sum() > 0:
    raise ValueError("Missing values still present after preprocessing.")

# ------------------------------------------------------------
# 10. Class imbalance settings
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

scale_pos_weight = n_class_0 / n_class_1

print("\nClass weights:")
print(class_weight_dict)

print("\nscale_pos_weight for XGBoost:")
print(scale_pos_weight)

# ------------------------------------------------------------
# 11. Helper function for evaluation
# ------------------------------------------------------------

def evaluate_model(model_name, model, X_train_eval, y_train_eval, X_test_eval, y_test_eval):
    y_train_pred = model.predict(X_train_eval)
    y_test_pred = model.predict(X_test_eval)

    if hasattr(model, "predict_proba"):
        y_train_proba = model.predict_proba(X_train_eval)[:, 1]
        y_test_proba = model.predict_proba(X_test_eval)[:, 1]
    else:
        y_train_scores = model.decision_function(X_train_eval)
        y_test_scores = model.decision_function(X_test_eval)

        y_train_proba = (y_train_scores - y_train_scores.min()) / (y_train_scores.max() - y_train_scores.min())
        y_test_proba = (y_test_scores - y_test_scores.min()) / (y_test_scores.max() - y_test_scores.min())

    cm = confusion_matrix(y_test_eval, y_test_pred)

    tn, fp, fn, tp = cm.ravel()

    result = {
        "Model": model_name,

        "Train Accuracy": accuracy_score(y_train_eval, y_train_pred),
        "Test Accuracy": accuracy_score(y_test_eval, y_test_pred),

        "Train Precision Class 1": precision_score(y_train_eval, y_train_pred, pos_label=1, zero_division=0),
        "Test Precision Class 1": precision_score(y_test_eval, y_test_pred, pos_label=1, zero_division=0),

        "Train Recall Class 1": recall_score(y_train_eval, y_train_pred, pos_label=1, zero_division=0),
        "Test Recall Class 1": recall_score(y_test_eval, y_test_pred, pos_label=1, zero_division=0),

        "Train F1 Class 1": f1_score(y_train_eval, y_train_pred, pos_label=1, zero_division=0),
        "Test F1 Class 1": f1_score(y_test_eval, y_test_pred, pos_label=1, zero_division=0),

        "Train ROC-AUC": roc_auc_score(y_train_eval, y_train_proba),
        "Test ROC-AUC": roc_auc_score(y_test_eval, y_test_proba),

        "Train PR-AUC": average_precision_score(y_train_eval, y_train_proba),
        "Test PR-AUC": average_precision_score(y_test_eval, y_test_proba),

        "True Negatives": int(tn),
        "False Positives": int(fp),
        "False Negatives": int(fn),
        "True Positives": int(tp)
    }

    print("\n============================================================")
    print(f"{model_name} — GROUP SPLIT RESULTS")
    print("============================================================")

    for key, value in result.items():
        if key != "Model":
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

    print("\nTest confusion matrix:")
    print(cm)

    report = classification_report(
        y_test_eval,
        y_test_pred,
        target_names=["No inhibition", "Inhibition"],
        zero_division=0
    )

    print("\nTest classification report:")
    print(report)

    with open(output_dir / f"{model_name.lower().replace(' ', '_').replace('/', '_')}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    predictions = pd.DataFrame({
        "y_test_actual": y_test_eval.values,
        "y_test_predicted": y_test_pred,
        "probability_inhibition": y_test_proba
    })

    predictions.to_csv(
        output_dir / f"{model_name.lower().replace(' ', '_').replace('/', '_')}_test_predictions.csv",
        index=False
    )

    return result

# ------------------------------------------------------------
# 12. Train models
# ------------------------------------------------------------

results = []

# 12.1 Random Forest
rf = RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    class_weight=class_weight_dict,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

results.append(
    evaluate_model(
        "Random Forest",
        rf,
        X_train,
        y_train,
        X_test,
        y_test
    )
)

# 12.2 XGBoost
xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.0,
    reg_alpha=0.01,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)

xgb.fit(X_train, y_train)

results.append(
    evaluate_model(
        "XGBoost",
        xgb,
        X_train,
        y_train,
        X_test,
        y_test
    )
)

# 12.3 SVM
svm = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    class_weight=class_weight_dict,
    random_state=42
)

svm.fit(X_train, y_train)

results.append(
    evaluate_model(
        "SVM",
        svm,
        X_train,
        y_train,
        X_test,
        y_test
    )
)

# 12.4 Logistic Regression
logreg = LogisticRegression(
    C=1.0,
    solver="lbfgs",
    max_iter=5000,
    class_weight=class_weight_dict,
    random_state=42
)

logreg.fit(X_train, y_train)

results.append(
    evaluate_model(
        "Logistic Regression",
        logreg,
        X_train,
        y_train,
        X_test,
        y_test
    )
)

# 12.5 ANN / MLP with balanced upsampling
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

ann = MLPClassifier(
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

ann.fit(X_train_balanced, y_train_balanced)

results.append(
    evaluate_model(
        "ANN MLP",
        ann,
        X_train,
        y_train,
        X_test,
        y_test
    )
)

# ------------------------------------------------------------
# 13. Save final comparison
# ------------------------------------------------------------

comparison_df = pd.DataFrame(results)

comparison_df = comparison_df.sort_values(
    by=["Test F1 Class 1", "Test Recall Class 1", "Test Accuracy"],
    ascending=False
).reset_index(drop=True)

comparison_df.insert(0, "Rank", range(1, len(comparison_df) + 1))

comparison_df.to_csv(
    output_dir / "task2_group_split_model_comparison.csv",
    index=False
)

comparison_df.to_excel(
    output_dir / "task2_group_split_model_comparison.xlsx",
    index=False
)

print("\n============================================================")
print("FINAL GROUP SPLIT MODEL COMPARISON")
print("============================================================")
print(comparison_df)

# ------------------------------------------------------------
# 14. Plot comparison
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))
plt.barh(
    comparison_df["Model"][::-1],
    comparison_df["Test F1 Class 1"][::-1]
)
plt.xlabel("F1-score for inhibition class")
plt.title("Group Split Sanity Check — Inhibition F1-score")
plt.tight_layout()
plt.savefig(output_dir / "group_split_f1_class1_comparison.png", dpi=300)
plt.close()

plt.figure(figsize=(9, 5))
plt.barh(
    comparison_df["Model"][::-1],
    comparison_df["Test Recall Class 1"][::-1]
)
plt.xlabel("Recall for inhibition class")
plt.title("Group Split Sanity Check — Inhibition Recall")
plt.tight_layout()
plt.savefig(output_dir / "group_split_recall_class1_comparison.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 15. Final message
# ------------------------------------------------------------

print("\n============================================================")
print("GROUP SPLIT SANITY CHECK COMPLETED SUCCESSFULLY")
print("============================================================")

print("\nBest model under group split:")
print(comparison_df.iloc[0]["Model"])

print("\nSaved outputs in:")
print(output_dir.resolve())

print("\nFiles created:")
for file in sorted(output_dir.iterdir()):
    print("-", file.name)