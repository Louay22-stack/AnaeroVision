# ============================================================
# TASK 3C — TUNED RANDOM FOREST CONFIDENCE THRESHOLD ANALYSIS
#
# Goal:
# - Use out-of-fold predictions to evaluate prediction confidence
# - Decide when the model should output "Unknown / uncertain"
# - Test how the model behaves on the original "Unknown" inhibition rows
# ============================================================

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, f1_score, classification_report

# ------------------------------------------------------------
# 0. Paths
# ------------------------------------------------------------

script_dir = Path(__file__).resolve().parent

input_file = (
    script_dir
    / "task3_cleaned_inhibition_type_outputs"
    / "task3_inhibited_only_cleaned_dataset.csv"
)

tuning_dir = script_dir / "task3c_rf_tuning_outputs"
best_params_file = tuning_dir / "rf_task3c_best_params.json"

if not input_file.exists():
    raise FileNotFoundError(
        "Missing cleaned Task 3 dataset. Run task3_clean_inhibition_type_labels.py first."
    )

output_dir = script_dir / "task3c_rf_confidence_threshold_outputs"
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
# 3. Split known and unknown inhibition-type rows
# ------------------------------------------------------------

df_known = df[df["Inhibition_Type_Final"] != "Unknown"].copy()
df_unknown = df[df["Inhibition_Type_Final"] == "Unknown"].copy()

print("\nKnown inhibition-type rows:", df_known.shape[0])
print("Unknown inhibition-type rows:", df_unknown.shape[0])

print("\nKnown inhibition-type distribution:")
print(df_known["Inhibition_Type_Final"].value_counts())

# ------------------------------------------------------------
# 4. Define features
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

missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]

if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# ------------------------------------------------------------
# 5. Prepare X and y for known classes
# ------------------------------------------------------------

X_known = df_known[feature_cols].copy()
y_known_text = df_known[target_col].copy()

X_unknown = df_unknown[feature_cols].copy()

# Clean category
X_known[category_col] = X_known[category_col].fillna("Other/unknown").astype(str)
X_unknown[category_col] = X_unknown[category_col].fillna("Other/unknown").astype(str)

# Consolidate reactor type
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

X_known["Reactor_Type"] = X_known["Reactor_Type"].map(reactor_mapping)
X_known["Reactor_Type"] = X_known["Reactor_Type"].fillna("Other")

X_unknown["Reactor_Type"] = X_unknown["Reactor_Type"].map(reactor_mapping)
X_unknown["Reactor_Type"] = X_unknown["Reactor_Type"].fillna("Other")

# Encode labels
label_encoder = LabelEncoder()
y_known = label_encoder.fit_transform(y_known_text)

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

with open(output_dir / "task3c_confidence_label_mapping.json", "w", encoding="utf-8") as f:
    json.dump(label_mapping, f, indent=4, ensure_ascii=False)

joblib.dump(label_encoder, output_dir / "label_encoder_task3c_confidence.pkl")

# ------------------------------------------------------------
# 6. Build preprocessing pipeline
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
# 7. Load tuned RF parameters
# ------------------------------------------------------------

if best_params_file.exists():
    with open(best_params_file, "r", encoding="utf-8") as f:
        best_params_raw = json.load(f)

    rf_params = {
        key.replace("model__", ""): value
        for key, value in best_params_raw.items()
    }

else:
    print("\nBest params file not found. Using known tuned parameters from previous run.")

    rf_params = {
        "n_estimators": 200,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "max_depth": 16,
        "class_weight": "balanced",
        "bootstrap": False
    }

rf_params["random_state"] = 42
rf_params["n_jobs"] = -1

print("\nTuned RF parameters:")
print(rf_params)

rf_model = RandomForestClassifier(**rf_params)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", rf_model)
    ]
)

# ------------------------------------------------------------
# 8. Out-of-fold predictions on known classes
# ------------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\nGenerating out-of-fold predictions on known inhibition types...")

y_pred_oof = cross_val_predict(
    pipeline,
    X_known,
    y_known,
    cv=cv,
    method="predict",
    n_jobs=-1
)

y_proba_oof = cross_val_predict(
    pipeline,
    X_known,
    y_known,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)

confidence_oof = np.max(y_proba_oof, axis=1)

correct_oof = y_pred_oof == y_known

known_oof_predictions = pd.DataFrame({
    "Actual_Label_Encoded": y_known,
    "Predicted_Label_Encoded": y_pred_oof,
    "Actual_Inhibition_Type": y_known_text.values,
    "Predicted_Inhibition_Type": [
        label_mapping[int(label)]
        for label in y_pred_oof
    ],
    "Confidence": confidence_oof,
    "Correct": correct_oof
})

for class_index, class_name in label_mapping.items():
    known_oof_predictions[f"Probability_{class_name}"] = y_proba_oof[:, int(class_index)]

known_oof_predictions.to_csv(
    output_dir / "known_classes_out_of_fold_predictions_with_confidence.csv",
    index=False
)

print("\nKnown-class out-of-fold accuracy:")
print(f"{accuracy_score(y_known, y_pred_oof):.4f}")

print("\nKnown-class out-of-fold macro F1:")
print(f"{f1_score(y_known, y_pred_oof, average='macro'):.4f}")

print("\nOut-of-fold classification report:")
print(
    classification_report(
        y_known,
        y_pred_oof,
        target_names=target_names,
        zero_division=0
    )
)

# ------------------------------------------------------------
# 9. Fit on known classes and predict original Unknown rows
# ------------------------------------------------------------

print("\nTraining tuned RF on all known inhibition-type rows...")

pipeline.fit(X_known, y_known)

joblib.dump(
    pipeline,
    output_dir / "task3c_tuned_rf_final_pipeline_for_confidence.pkl"
)

print("\nPredicting original Unknown inhibition-type rows...")

if df_unknown.shape[0] > 0:
    y_unknown_pred = pipeline.predict(X_unknown)
    y_unknown_proba = pipeline.predict_proba(X_unknown)
    confidence_unknown = np.max(y_unknown_proba, axis=1)

    unknown_predictions = df_unknown.copy()

    unknown_predictions["Predicted_Known_Inhibition_Type"] = [
        label_mapping[int(label)]
        for label in y_unknown_pred
    ]

    unknown_predictions["Confidence"] = confidence_unknown

    for class_index, class_name in label_mapping.items():
        unknown_predictions[f"Probability_{class_name}"] = y_unknown_proba[:, int(class_index)]

    unknown_predictions.to_csv(
        output_dir / "original_unknown_rows_predictions_with_confidence.csv",
        index=False
    )

    print("\nUnknown rows predicted known-type distribution:")
    print(unknown_predictions["Predicted_Known_Inhibition_Type"].value_counts())

    print("\nUnknown rows confidence summary:")
    print(unknown_predictions["Confidence"].describe())

else:
    unknown_predictions = pd.DataFrame()
    confidence_unknown = np.array([])

# ------------------------------------------------------------
# 10. Threshold analysis
# ------------------------------------------------------------

thresholds = np.round(np.arange(0.40, 0.91, 0.05), 2)

threshold_rows = []

for threshold in thresholds:
    accepted_mask = confidence_oof >= threshold
    uncertain_mask = confidence_oof < threshold

    accepted_count = int(accepted_mask.sum())
    uncertain_count = int(uncertain_mask.sum())

    if accepted_count > 0:
        accepted_accuracy = accuracy_score(
            y_known[accepted_mask],
            y_pred_oof[accepted_mask]
        )

        accepted_macro_f1 = f1_score(
            y_known[accepted_mask],
            y_pred_oof[accepted_mask],
            average="macro",
            zero_division=0
        )

        confident_errors = int(
            ((y_pred_oof != y_known) & accepted_mask).sum()
        )

    else:
        accepted_accuracy = np.nan
        accepted_macro_f1 = np.nan
        confident_errors = 0

    if confidence_unknown.size > 0:
        unknown_uncertain_rate = float((confidence_unknown < threshold).mean())
        unknown_accepted_rate = float((confidence_unknown >= threshold).mean())
    else:
        unknown_uncertain_rate = np.nan
        unknown_accepted_rate = np.nan

    threshold_rows.append({
        "Threshold": float(threshold),
        "Known_Accepted_Count": accepted_count,
        "Known_Accepted_Rate": accepted_count / len(confidence_oof),
        "Known_Uncertain_Count": uncertain_count,
        "Known_Uncertain_Rate": uncertain_count / len(confidence_oof),
        "Accepted_Accuracy": accepted_accuracy,
        "Accepted_Macro_F1": accepted_macro_f1,
        "Confident_Errors": confident_errors,
        "Unknown_Uncertain_Rate": unknown_uncertain_rate,
        "Unknown_Accepted_Rate": unknown_accepted_rate
    })

threshold_analysis = pd.DataFrame(threshold_rows)

threshold_analysis.to_csv(
    output_dir / "confidence_threshold_analysis.csv",
    index=False
)

print("\n============================================================")
print("CONFIDENCE THRESHOLD ANALYSIS")
print("============================================================")

print(threshold_analysis)

# ------------------------------------------------------------
# 11. Suggest threshold
# ------------------------------------------------------------

# Rule:
# Prefer a threshold with accepted accuracy >= 0.95
# while keeping at least 70% of known samples accepted.
candidate_thresholds = threshold_analysis[
    (threshold_analysis["Accepted_Accuracy"] >= 0.95)
    & (threshold_analysis["Known_Accepted_Rate"] >= 0.70)
]

if not candidate_thresholds.empty:
    suggested_threshold = float(candidate_thresholds.iloc[0]["Threshold"])
else:
    # fallback: choose threshold with best accepted accuracy while accepted rate >= 0.60
    fallback_candidates = threshold_analysis[
        threshold_analysis["Known_Accepted_Rate"] >= 0.60
    ].copy()

    fallback_candidates = fallback_candidates.sort_values(
        by=["Accepted_Accuracy", "Accepted_Macro_F1"],
        ascending=False
    )

    suggested_threshold = float(fallback_candidates.iloc[0]["Threshold"])

print("\nSuggested confidence threshold:")
print(suggested_threshold)

# ------------------------------------------------------------
# 12. Plots
# ------------------------------------------------------------

# High-quality rendering settings (figures only — no data/logic changes)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["axes.linewidth"] = 1.1
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.15

plt.figure(figsize=(10, 6.5))
plt.hist(
    confidence_oof[correct_oof],
    bins=15,
    alpha=0.7,
    label="Correct known predictions",
    edgecolor="black",
    linewidth=0.6
)
plt.hist(
    confidence_oof[~correct_oof],
    bins=15,
    alpha=0.7,
    label="Incorrect known predictions",
    edgecolor="black",
    linewidth=0.6
)

if confidence_unknown.size > 0:
    plt.hist(
        confidence_unknown,
        bins=15,
        alpha=0.5,
        label="Original Unknown rows",
        edgecolor="black",
        linewidth=0.6
    )

plt.axvline(suggested_threshold, linestyle="--", linewidth=1.8, color="black",
            label=f"Suggested threshold = {suggested_threshold}")
plt.xlabel("Prediction confidence", labelpad=8)
plt.ylabel("Count", labelpad=8)
plt.title("Task 3C — Confidence Distribution", pad=14)
plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
plt.gca().set_axisbelow(True)
plt.legend(frameon=True, framealpha=0.9)
plt.setp(plt.gca().get_xticklabels(), fontweight="bold")
plt.setp(plt.gca().get_yticklabels(), fontweight="bold")
plt.tight_layout()
plt.savefig(output_dir / "confidence_distribution_known_vs_unknown.png", dpi=600)
plt.close()

plt.figure(figsize=(10, 6.5))
plt.plot(
    threshold_analysis["Threshold"],
    threshold_analysis["Accepted_Accuracy"],
    marker="o",
    markersize=6,
    linewidth=2,
    label="Accepted accuracy"
)
plt.plot(
    threshold_analysis["Threshold"],
    threshold_analysis["Known_Accepted_Rate"],
    marker="o",
    markersize=6,
    linewidth=2,
    label="Known accepted rate"
)

if confidence_unknown.size > 0:
    plt.plot(
        threshold_analysis["Threshold"],
        threshold_analysis["Unknown_Uncertain_Rate"],
        marker="o",
        markersize=6,
        linewidth=2,
        label="Unknown uncertain rate"
    )

plt.axvline(suggested_threshold, linestyle="--", linewidth=1.8, color="black",
            label=f"Suggested threshold = {suggested_threshold}")
plt.xlabel("Confidence threshold", labelpad=8)
plt.ylabel("Rate", labelpad=8)
plt.title("Task 3C — Threshold Trade-off", pad=14)
plt.grid(linestyle="--", linewidth=0.5, alpha=0.4)
plt.gca().set_axisbelow(True)
plt.legend(frameon=True, framealpha=0.9)
plt.setp(plt.gca().get_xticklabels(), fontweight="bold")
plt.setp(plt.gca().get_yticklabels(), fontweight="bold")
plt.tight_layout()
plt.savefig(output_dir / "confidence_threshold_tradeoff.png", dpi=600)
plt.close()

# ------------------------------------------------------------
# 13. Save summary
# ------------------------------------------------------------

summary = {
    "task": "Task 3C - RF confidence threshold analysis",
    "known_rows": int(df_known.shape[0]),
    "unknown_rows": int(df_unknown.shape[0]),
    "features_used": feature_cols,
    "label_mapping": label_mapping,
    "rf_params": rf_params,
    "known_oof_accuracy": float(accuracy_score(y_known, y_pred_oof)),
    "known_oof_macro_f1": float(f1_score(y_known, y_pred_oof, average="macro")),
    "suggested_confidence_threshold": suggested_threshold,
    "threshold_analysis": threshold_analysis.to_dict(orient="records")
}

with open(output_dir / "task3c_rf_confidence_threshold_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4, ensure_ascii=False)

# ------------------------------------------------------------
# 14. Final message
# ------------------------------------------------------------

print("\n============================================================")
print("TASK 3C CONFIDENCE THRESHOLD ANALYSIS COMPLETED")
print("============================================================")

print("\nSuggested threshold:")
print(suggested_threshold)

print("\nSaved outputs in:")
print(output_dir.resolve())

print("\nFiles created:")
for file in sorted(output_dir.iterdir()):
    print("-", file.name)