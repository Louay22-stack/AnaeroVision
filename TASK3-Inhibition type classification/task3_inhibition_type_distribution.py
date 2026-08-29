# ============================================================
# TASK 3 — INHIBITION TYPE DISTRIBUTION INSPECTION
# Target: Inhibition_Type
#
# Purpose:
# Check whether inhibition-type classification is feasible,
# inspect class distribution, and identify rare classes.
# ============================================================

import json
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

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

output_dir = script_dir / "task3_inhibition_type_distribution_outputs"
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

# ------------------------------------------------------------
# 2. Check required columns
# ------------------------------------------------------------

required_cols = [
    "Inhibition_Label_Numeric",
    "Inhibition_Type"
]

missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# ------------------------------------------------------------
# 3. Inspect full Inhibition_Type distribution
# ------------------------------------------------------------

print("\nFull Inhibition_Type distribution including all rows:")
full_type_counts = df["Inhibition_Type"].value_counts(dropna=False)
print(full_type_counts)

full_type_counts.to_csv(
    output_dir / "full_inhibition_type_distribution.csv",
    header=["Count"]
)

# ------------------------------------------------------------
# 4. Keep only inhibited samples
# ------------------------------------------------------------

df_inhibited = df[df["Inhibition_Label_Numeric"] == 1].copy()

print("\nInhibited-only dataset shape:", df_inhibited.shape)

print("\nMissing Inhibition_Type values among inhibited samples:")
print(df_inhibited["Inhibition_Type"].isna().sum())

# Remove rows without inhibition type
df_inhibited = df_inhibited.dropna(subset=["Inhibition_Type"]).copy()

print("\nInhibited-only dataset shape after removing missing Inhibition_Type:")
print(df_inhibited.shape)

# ------------------------------------------------------------
# 5. Inhibition type distribution among inhibited samples
# ------------------------------------------------------------

type_counts = df_inhibited["Inhibition_Type"].value_counts()
type_percentages = (
    df_inhibited["Inhibition_Type"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

type_distribution = pd.DataFrame({
    "Inhibition_Type": type_counts.index,
    "Count": type_counts.values,
    "Percentage": type_percentages.values
})

print("\nInhibition type distribution among inhibited samples:")
print(type_distribution)

type_distribution.to_csv(
    output_dir / "inhibited_only_inhibition_type_distribution.csv",
    index=False
)

# ------------------------------------------------------------
# 6. Identify rare classes
# ------------------------------------------------------------

rare_thresholds = [5, 10, 20]

rare_class_summary = {}

for threshold in rare_thresholds:
    rare_classes = type_distribution[
        type_distribution["Count"] < threshold
    ]["Inhibition_Type"].tolist()

    rare_class_summary[f"classes_with_less_than_{threshold}_samples"] = rare_classes

print("\nRare class summary:")
print(json.dumps(rare_class_summary, indent=4))

with open(output_dir / "rare_class_summary.json", "w", encoding="utf-8") as f:
    json.dump(rare_class_summary, f, indent=4, ensure_ascii=False)

# ------------------------------------------------------------
# 7. Suggested merged class strategy
# ------------------------------------------------------------

# Suggested rule:
# Keep classes with at least 10 samples.
# Merge classes with fewer than 10 samples into "Other inhibition".

min_class_size = 10

classes_to_keep = type_distribution[
    type_distribution["Count"] >= min_class_size
]["Inhibition_Type"].tolist()

classes_to_merge = type_distribution[
    type_distribution["Count"] < min_class_size
]["Inhibition_Type"].tolist()

df_inhibited["Inhibition_Type_Merged"] = df_inhibited["Inhibition_Type"].apply(
    lambda x: x if x in classes_to_keep else "Other inhibition"
)

merged_counts = df_inhibited["Inhibition_Type_Merged"].value_counts()
merged_percentages = (
    df_inhibited["Inhibition_Type_Merged"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

merged_distribution = pd.DataFrame({
    "Inhibition_Type_Merged": merged_counts.index,
    "Count": merged_counts.values,
    "Percentage": merged_percentages.values
})

print("\nSuggested merged inhibition type distribution:")
print(merged_distribution)

merged_distribution.to_csv(
    output_dir / "suggested_merged_inhibition_type_distribution.csv",
    index=False
)

merge_strategy = {
    "min_class_size": min_class_size,
    "classes_kept": classes_to_keep,
    "classes_merged_into_other_inhibition": classes_to_merge
}

with open(output_dir / "suggested_merge_strategy.json", "w", encoding="utf-8") as f:
    json.dump(merge_strategy, f, indent=4, ensure_ascii=False)

# ------------------------------------------------------------
# 8. Save inhibited-only dataset with merged labels
# ------------------------------------------------------------

df_inhibited.to_csv(
    output_dir / "task3_inhibited_only_dataset_with_merged_labels.csv",
    index=False
)

# ------------------------------------------------------------
# 9. Plot original inhibition type distribution
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))
plt.barh(
    type_distribution["Inhibition_Type"][::-1],
    type_distribution["Count"][::-1]
)
plt.xlabel("Count")
plt.ylabel("Inhibition type")
plt.title("Task 3 — Original Inhibition Type Distribution")
plt.tight_layout()
plt.savefig(
    output_dir / "original_inhibition_type_distribution.png",
    dpi=300
)
plt.close()

# ------------------------------------------------------------
# 10. Plot merged inhibition type distribution
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))
plt.barh(
    merged_distribution["Inhibition_Type_Merged"][::-1],
    merged_distribution["Count"][::-1]
)
plt.xlabel("Count")
plt.ylabel("Merged inhibition type")
plt.title("Task 3 — Suggested Merged Inhibition Type Distribution")
plt.tight_layout()
plt.savefig(
    output_dir / "suggested_merged_inhibition_type_distribution.png",
    dpi=300
)
plt.close()

# ------------------------------------------------------------
# 11. Final summary
# ------------------------------------------------------------

summary = {
    "original_dataset_shape": list(df.shape),
    "inhibited_only_shape_before_dropping_missing_type": [
        int(df[df["Inhibition_Label_Numeric"] == 1].shape[0]),
        int(df[df["Inhibition_Label_Numeric"] == 1].shape[1])
    ],
    "inhibited_only_shape_after_dropping_missing_type": list(df_inhibited.shape),
    "number_of_original_inhibition_type_classes": int(type_distribution.shape[0]),
    "number_of_merged_inhibition_type_classes": int(merged_distribution.shape[0]),
    "min_class_size_for_merge": min_class_size,
    "classes_kept": classes_to_keep,
    "classes_merged": classes_to_merge
}

with open(output_dir / "task3_distribution_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4, ensure_ascii=False)

print("\n============================================================")
print("TASK 3 INHIBITION TYPE DISTRIBUTION INSPECTION COMPLETED")
print("============================================================")

print("\nNumber of original inhibition-type classes:")
print(type_distribution.shape[0])

print("\nNumber of suggested merged classes:")
print(merged_distribution.shape[0])

print("\nClasses kept:")
print(classes_to_keep)

print("\nClasses merged into 'Other inhibition':")
print(classes_to_merge)

print("\nSaved outputs in:")
print(output_dir.resolve())

print("\nFiles created:")
for file in sorted(output_dir.iterdir()):
    print("-", file.name)