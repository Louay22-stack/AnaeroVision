# ============================================================
# TASK 3C — RANDOM FOREST HYPERPARAMETER TUNING
# Interpretable Hybrid Inhibition-Type Classification
#
# Features:
# - Mechanistic operational variables
# - Substrate category
# - Reactor type
#
# Goal:
# Tune Random Forest using 5-fold stratified cross-validation.
# Main selection metric: macro F1-score.
# ============================================================

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    make_scorer
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

output_dir = script_dir / "task3c_rf_tuning_outputs"
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

with open(output_dir / "task3c_rf_tuning_label_mapping.json", "w", encoding="utf-8") as f:
    json.dump(label_mapping, f, indent=4, ensure_ascii=False)

joblib.dump(label_encoder, output_dir / "label_encoder_task3c_rf_tuned.pkl")

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
# 8. Baseline Random Forest pipeline
# ------------------------------------------------------------

baseline_rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

baseline_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", baseline_rf)
    ]
)

# ------------------------------------------------------------
# 9. Hyperparameter search space
# ------------------------------------------------------------

param_distributions = {
    "model__n_estimators": [200, 300, 500, 800],
    "model__max_depth": [None, 5, 8, 12, 16, 20],
    "model__min_samples_split": [2, 4, 6, 8, 10],
    "model__min_samples_leaf": [1, 2, 3, 4],
    "model__max_features": ["sqrt", "log2", 0.5, 0.7, None],
    "model__bootstrap": [True, False],
    "model__class_weight": ["balanced", "balanced_subsample"]
}

rf_for_tuning = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

tuning_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", rf_for_tuning)
    ]
)

# ------------------------------------------------------------
# 10. Cross-validation setup
# ------------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "macro_precision": "precision_macro",
    "macro_recall": "recall_macro",
    "macro_f1": "f1_macro",
    "weighted_f1": "f1_weighted"
}

# ------------------------------------------------------------
# 11. Baseline CV evaluation
# ------------------------------------------------------------

print("\n============================================================")
print("BASELINE RANDOM FOREST 5-FOLD CV")
print("============================================================")

baseline_cv_results = cross_validate(
    baseline_pipeline,
    X,
    y,
    cv=cv,
    scoring=scoring,
    n_jobs=-1,
    return_train_score=True
)

baseline_summary = {}

for metric in scoring.keys():
    test_scores = baseline_cv_results[f"test_{metric}"]
    train_scores = baseline_cv_results[f"train_{metric}"]

    baseline_summary[f"train_{metric}_mean"] = float(train_scores.mean())
    baseline_summary[f"train_{metric}_std"] = float(train_scores.std())
    baseline_summary[f"test_{metric}_mean"] = float(test_scores.mean())
    baseline_summary[f"test_{metric}_std"] = float(test_scores.std())

    print(
        f"{metric}: "
        f"train {train_scores.mean():.4f} ± {train_scores.std():.4f} | "
        f"test {test_scores.mean():.4f} ± {test_scores.std():.4f}"
    )

# ------------------------------------------------------------
# 12. RandomizedSearchCV
# ------------------------------------------------------------

print("\n============================================================")
print("STARTING RANDOM FOREST HYPERPARAMETER TUNING")
print("============================================================")

random_search = RandomizedSearchCV(
    estimator=tuning_pipeline,
    param_distributions=param_distributions,
    n_iter=60,
    scoring="f1_macro",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=2,
    return_train_score=True
)

random_search.fit(X, y)

print("\n============================================================")
print("RANDOM FOREST TUNING COMPLETED")
print("============================================================")

print("\nBest parameters:")
print(random_search.best_params_)

print("\nBest CV macro F1:")
print(f"{random_search.best_score_:.4f}")

best_pipeline = random_search.best_estimator_

# ------------------------------------------------------------
# 13. Save tuning results
# ------------------------------------------------------------

tuning_results_df = pd.DataFrame(random_search.cv_results_)
tuning_results_df.to_csv(
    output_dir / "rf_task3c_randomized_search_results.csv",
    index=False
)

best_params = random_search.best_params_

with open(output_dir / "rf_task3c_best_params.json", "w", encoding="utf-8") as f:
    json.dump(best_params, f, indent=4, ensure_ascii=False)

# ------------------------------------------------------------
# 14. Tuned model CV evaluation
# ------------------------------------------------------------

print("\n============================================================")
print("TUNED RANDOM FOREST 5-FOLD CV")
print("============================================================")

tuned_cv_results = cross_validate(
    best_pipeline,
    X,
    y,
    cv=cv,
    scoring=scoring,
    n_jobs=-1,
    return_train_score=True
)

tuned_summary = {}

for metric in scoring.keys():
    test_scores = tuned_cv_results[f"test_{metric}"]
    train_scores = tuned_cv_results[f"train_{metric}"]

    tuned_summary[f"train_{metric}_mean"] = float(train_scores.mean())
    tuned_summary[f"train_{metric}_std"] = float(train_scores.std())
    tuned_summary[f"test_{metric}_mean"] = float(test_scores.mean())
    tuned_summary[f"test_{metric}_std"] = float(test_scores.std())

    print(
        f"{metric}: "
        f"train {train_scores.mean():.4f} ± {train_scores.std():.4f} | "
        f"test {test_scores.mean():.4f} ± {test_scores.std():.4f}"
    )

# ------------------------------------------------------------
# 15. Compare baseline vs tuned
# ------------------------------------------------------------

comparison_rows = []

for metric in scoring.keys():
    comparison_rows.append({
        "Metric": metric,
        "Baseline_CV_Mean": baseline_summary[f"test_{metric}_mean"],
        "Baseline_CV_Std": baseline_summary[f"test_{metric}_std"],
        "Tuned_CV_Mean": tuned_summary[f"test_{metric}_mean"],
        "Tuned_CV_Std": tuned_summary[f"test_{metric}_std"],
        "Difference_Tuned_minus_Baseline": (
            tuned_summary[f"test_{metric}_mean"]
            - baseline_summary[f"test_{metric}_mean"]
        )
    })

comparison_df = pd.DataFrame(comparison_rows)

comparison_df.to_csv(
    output_dir / "rf_task3c_baseline_vs_tuned_comparison.csv",
    index=False
)

print("\n============================================================")
print("BASELINE VS TUNED RANDOM FOREST COMPARISON")
print("============================================================")
print(comparison_df)

# ------------------------------------------------------------
# 16. Out-of-fold predictions for tuned RF
# ------------------------------------------------------------

print("\nGenerating out-of-fold predictions for tuned Random Forest...")

y_pred_oof = cross_val_predict(
    best_pipeline,
    X,
    y,
    cv=cv,
    n_jobs=-1,
    method="predict"
)

y_proba_oof = cross_val_predict(
    best_pipeline,
    X,
    y,
    cv=cv,
    n_jobs=-1,
    method="predict_proba"
)

overall_metrics = {
    "overall_accuracy": accuracy_score(y, y_pred_oof),
    "overall_balanced_accuracy": balanced_accuracy_score(y, y_pred_oof),
    "overall_macro_precision": precision_score(y, y_pred_oof, average="macro", zero_division=0),
    "overall_macro_recall": recall_score(y, y_pred_oof, average="macro", zero_division=0),
    "overall_macro_f1": f1_score(y, y_pred_oof, average="macro", zero_division=0),
    "overall_weighted_f1": f1_score(y, y_pred_oof, average="weighted", zero_division=0)
}

try:
    overall_metrics["overall_roc_auc_ovr_macro"] = roc_auc_score(
        y,
        y_proba_oof,
        multi_class="ovr",
        average="macro"
    )
except ValueError:
    overall_metrics["overall_roc_auc_ovr_macro"] = np.nan

print("\nTuned Random Forest overall out-of-fold metrics:")
for key, value in overall_metrics.items():
    print(f"{key}: {value:.4f}")

overall_report = classification_report(
    y,
    y_pred_oof,
    target_names=target_names,
    zero_division=0
)

print("\nTuned Random Forest overall out-of-fold classification report:")
print(overall_report)

with open(output_dir / "rf_task3c_tuned_overall_classification_report.txt", "w", encoding="utf-8") as f:
    f.write(overall_report)

overall_cm = confusion_matrix(y, y_pred_oof)

print("\nTuned Random Forest overall out-of-fold confusion matrix:")
print(overall_cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=overall_cm,
    display_labels=target_names
)

fig, ax = plt.subplots(figsize=(9, 7))
disp.plot(ax=ax, values_format="d", xticks_rotation=45)
plt.title("Tuned Random Forest — Task 3C Overall CV Confusion Matrix")
plt.tight_layout()
plt.savefig(output_dir / "rf_task3c_tuned_overall_confusion_matrix.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 17. Train final selected model on all 220 known samples
# ------------------------------------------------------------

best_pipeline.fit(X, y)

joblib.dump(
    best_pipeline,
    output_dir / "rf_task3c_tuned_trained_on_all_known_samples.pkl"
)

print("\nTuned Random Forest retrained on all 220 known samples and saved.")

# ------------------------------------------------------------
# 18. Feature importance from final tuned model
# ------------------------------------------------------------

trained_preprocessor = best_pipeline.named_steps["preprocessor"]
trained_rf = best_pipeline.named_steps["model"]

numeric_feature_names = numeric_features

categorical_feature_names = (
    trained_preprocessor
    .named_transformers_["categorical"]
    .named_steps["onehot"]
    .get_feature_names_out(categorical_features)
    .tolist()
)

final_feature_names = numeric_feature_names + categorical_feature_names

feature_importance = pd.DataFrame({
    "Feature": final_feature_names,
    "Importance": trained_rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

feature_importance.to_csv(
    output_dir / "rf_task3c_tuned_feature_importance.csv",
    index=False
)

print("\nTop feature importances:")
print(feature_importance.head(20))

plt.figure(figsize=(10, 8))
plt.barh(
    feature_importance["Feature"].head(20)[::-1],
    feature_importance["Importance"].head(20)[::-1]
)
plt.xlabel("Feature importance")
plt.title("Tuned Random Forest — Task 3C Top Feature Importances")
plt.tight_layout()
plt.savefig(output_dir / "rf_task3c_tuned_feature_importance_top20.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 19. Save full summary
# ------------------------------------------------------------

final_decision = "tuned_random_forest"

baseline_macro_f1 = baseline_summary["test_macro_f1_mean"]
tuned_macro_f1 = tuned_summary["test_macro_f1_mean"]

if tuned_macro_f1 < baseline_macro_f1:
    final_decision = "baseline_random_forest_preferred"

summary = {
    "task": "Task 3C - Random Forest hyperparameter tuning",
    "rows_used": int(df_task3.shape[0]),
    "features_used": feature_cols,
    "category_column_used": category_col,
    "label_mapping": label_mapping,
    "baseline_summary": baseline_summary,
    "best_params": best_params,
    "best_cv_macro_f1_from_search": float(random_search.best_score_),
    "tuned_summary": tuned_summary,
    "baseline_vs_tuned_comparison": comparison_df.to_dict(orient="records"),
    "tuned_overall_out_of_fold_metrics": overall_metrics,
    "final_decision": final_decision,
    "note": (
        "If tuned RF improves or maintains macro F1 compared with baseline RF, "
        "select tuned RF. If it performs worse, retain baseline RF."
    )
}

with open(output_dir / "rf_task3c_tuning_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4, ensure_ascii=False)

# ------------------------------------------------------------
# 20. Final message
# ------------------------------------------------------------

print("\n============================================================")
print("TASK 3C RANDOM FOREST TUNING COMPLETED")
print("============================================================")

print("\nBaseline macro F1:")
print(f"{baseline_macro_f1:.4f}")

print("\nTuned macro F1:")
print(f"{tuned_macro_f1:.4f}")

print("\nFinal decision:")
print(final_decision)

print("\nSaved outputs in:")
print(output_dir.resolve())

print("\nFiles created:")
for file in sorted(output_dir.iterdir()):
    print("-", file.name)