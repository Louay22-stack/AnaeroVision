# ============================================================
# TASK 2 — GROUP SPLIT THRESHOLD TUNING
# Inhibition Risk Classification
#
# Purpose:
# Tune the classification threshold using saved group-split
# prediction probabilities.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ------------------------------------------------------------
# 0. Paths
# ------------------------------------------------------------

script_dir = Path(__file__).resolve().parent

input_dir = script_dir / "task2_group_split_sanity_check_outputs"

output_dir = script_dir / "task2_group_split_threshold_tuning_outputs"
output_dir.mkdir(exist_ok=True)

if not input_dir.exists():
    raise FileNotFoundError("Missing folder: task2_group_split_sanity_check_outputs")

# ------------------------------------------------------------
# 1. Prediction files
# ------------------------------------------------------------

model_files = {
    "Random Forest": input_dir / "random_forest_test_predictions.csv",
    "XGBoost": input_dir / "xgboost_test_predictions.csv",
    "SVM": input_dir / "svm_test_predictions.csv",
    "Logistic Regression": input_dir / "logistic_regression_test_predictions.csv",
    "ANN MLP": input_dir / "ann_mlp_test_predictions.csv"
}

for model_name, file_path in model_files.items():
    if not file_path.exists():
        raise FileNotFoundError(f"Missing prediction file for {model_name}: {file_path}")

# ------------------------------------------------------------
# 2. Threshold tuning function
# ------------------------------------------------------------

def tune_thresholds(model_name, predictions_df):
    y_true = predictions_df["y_test_actual"].values
    y_proba = predictions_df["probability_inhibition"].values

    thresholds = np.arange(0.01, 1.00, 0.01)

    rows = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        rows.append({
            "Model": model_name,
            "Threshold": threshold,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision_Class_1": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
            "Recall_Class_1": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
            "F1_Class_1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
            "True_Negatives": int(tn),
            "False_Positives": int(fp),
            "False_Negatives": int(fn),
            "True_Positives": int(tp)
        })

    threshold_df = pd.DataFrame(rows)

    threshold_df.to_csv(
        output_dir / f"{model_name.lower().replace(' ', '_')}_threshold_results.csv",
        index=False
    )

    # Best threshold by F1-score
    best_f1_row = threshold_df.sort_values(
        by=["F1_Class_1", "Recall_Class_1", "Precision_Class_1"],
        ascending=False
    ).iloc[0]

    # Best recall-oriented threshold with precision >= 0.80
    precision_filtered = threshold_df[threshold_df["Precision_Class_1"] >= 0.80].copy()

    if len(precision_filtered) > 0:
        best_recall_row = precision_filtered.sort_values(
            by=["Recall_Class_1", "F1_Class_1", "Precision_Class_1"],
            ascending=False
        ).iloc[0]
    else:
        best_recall_row = threshold_df.sort_values(
            by=["Recall_Class_1", "F1_Class_1", "Precision_Class_1"],
            ascending=False
        ).iloc[0]

    # Plot F1, recall, precision vs threshold
    plt.figure(figsize=(8, 5))
    plt.plot(threshold_df["Threshold"], threshold_df["F1_Class_1"], label="F1 class 1")
    plt.plot(threshold_df["Threshold"], threshold_df["Recall_Class_1"], label="Recall class 1")
    plt.plot(threshold_df["Threshold"], threshold_df["Precision_Class_1"], label="Precision class 1")
    plt.axvline(best_f1_row["Threshold"], linestyle="--", label=f"Best F1 threshold = {best_f1_row['Threshold']:.2f}")
    plt.xlabel("Decision threshold")
    plt.ylabel("Score")
    plt.title(f"{model_name} — Threshold Tuning")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{model_name.lower().replace(' ', '_')}_threshold_tuning_curve.png",
        dpi=300
    )
    plt.close()

    return {
        "Model": model_name,

        "Best_F1_Threshold": best_f1_row["Threshold"],
        "Best_F1_Accuracy": best_f1_row["Accuracy"],
        "Best_F1_Precision_Class_1": best_f1_row["Precision_Class_1"],
        "Best_F1_Recall_Class_1": best_f1_row["Recall_Class_1"],
        "Best_F1_Class_1": best_f1_row["F1_Class_1"],
        "Best_F1_False_Positives": int(best_f1_row["False_Positives"]),
        "Best_F1_False_Negatives": int(best_f1_row["False_Negatives"]),
        "Best_F1_True_Positives": int(best_f1_row["True_Positives"]),
        "Best_F1_True_Negatives": int(best_f1_row["True_Negatives"]),

        "Recall_Optimized_Threshold": best_recall_row["Threshold"],
        "Recall_Optimized_Accuracy": best_recall_row["Accuracy"],
        "Recall_Optimized_Precision_Class_1": best_recall_row["Precision_Class_1"],
        "Recall_Optimized_Recall_Class_1": best_recall_row["Recall_Class_1"],
        "Recall_Optimized_F1_Class_1": best_recall_row["F1_Class_1"],
        "Recall_Optimized_False_Positives": int(best_recall_row["False_Positives"]),
        "Recall_Optimized_False_Negatives": int(best_recall_row["False_Negatives"]),
        "Recall_Optimized_True_Positives": int(best_recall_row["True_Positives"]),
        "Recall_Optimized_True_Negatives": int(best_recall_row["True_Negatives"])
    }

# ------------------------------------------------------------
# 3. Tune all models
# ------------------------------------------------------------

summary_rows = []

for model_name, file_path in model_files.items():
    print("\n============================================================")
    print(f"Tuning threshold for: {model_name}")
    print("============================================================")

    predictions_df = pd.read_csv(file_path)

    required_cols = [
        "y_test_actual",
        "probability_inhibition"
    ]

    missing_cols = [col for col in required_cols if col not in predictions_df.columns]

    if missing_cols:
        raise ValueError(f"Missing columns in {file_path}: {missing_cols}")

    result = tune_thresholds(model_name, predictions_df)
    summary_rows.append(result)

    print(f"Best F1 threshold: {result['Best_F1_Threshold']:.2f}")
    print(f"Best F1 class 1: {result['Best_F1_Class_1']:.4f}")
    print(f"Best recall class 1: {result['Best_F1_Recall_Class_1']:.4f}")
    print(f"False negatives at best F1: {result['Best_F1_False_Negatives']}")

# ------------------------------------------------------------
# 4. Save final threshold summary
# ------------------------------------------------------------

summary_df = pd.DataFrame(summary_rows)

summary_df = summary_df.sort_values(
    by=[
        "Best_F1_Class_1",
        "Best_F1_Recall_Class_1",
        "Best_F1_Precision_Class_1"
    ],
    ascending=False
).reset_index(drop=True)

summary_df.insert(0, "Rank", range(1, len(summary_df) + 1))

summary_df.to_csv(
    output_dir / "task2_group_split_threshold_tuning_summary.csv",
    index=False
)

summary_df.to_excel(
    output_dir / "task2_group_split_threshold_tuning_summary.xlsx",
    index=False
)

print("\n============================================================")
print("FINAL THRESHOLD TUNING SUMMARY")
print("============================================================")
print(summary_df)

# ------------------------------------------------------------
# 5. Plot final comparison after threshold tuning
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))
plt.barh(
    summary_df["Model"][::-1],
    summary_df["Best_F1_Class_1"][::-1]
)
plt.xlabel("Best F1-score for inhibition class")
plt.title("Task 2 Group Split — Best F1 After Threshold Tuning")
plt.tight_layout()
plt.savefig(output_dir / "threshold_tuned_f1_comparison.png", dpi=300)
plt.close()

plt.figure(figsize=(9, 5))
plt.barh(
    summary_df["Model"][::-1],
    summary_df["Best_F1_Recall_Class_1"][::-1]
)
plt.xlabel("Recall for inhibition class at best F1 threshold")
plt.title("Task 2 Group Split — Recall After Threshold Tuning")
plt.tight_layout()
plt.savefig(output_dir / "threshold_tuned_recall_comparison.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 6. Final message
# ------------------------------------------------------------

best_model = summary_df.iloc[0]

print("\n============================================================")
print("THRESHOLD TUNING COMPLETED SUCCESSFULLY")
print("============================================================")

print("\nBest model after threshold tuning:")
print(best_model["Model"])

print("\nBest threshold:")
print(f"{best_model['Best_F1_Threshold']:.2f}")

print("\nBest threshold-tuned metrics:")
print(f"Accuracy: {best_model['Best_F1_Accuracy']:.4f}")
print(f"Precision class 1: {best_model['Best_F1_Precision_Class_1']:.4f}")
print(f"Recall class 1: {best_model['Best_F1_Recall_Class_1']:.4f}")
print(f"F1 class 1: {best_model['Best_F1_Class_1']:.4f}")
print(f"False positives: {int(best_model['Best_F1_False_Positives'])}")
print(f"False negatives: {int(best_model['Best_F1_False_Negatives'])}")
print(f"True positives: {int(best_model['Best_F1_True_Positives'])}")
print(f"True negatives: {int(best_model['Best_F1_True_Negatives'])}")

print("\nSaved outputs in:")
print(output_dir.resolve())

print("\nFiles created:")
for file in sorted(output_dir.iterdir()):
    print("-", file.name)