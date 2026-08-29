# ============================================================
# TASK 3 — CLEAN AND MERGE INHIBITION TYPE LABELS
# Target: Inhibition_Type
#
# Purpose:
# Standardize duplicated inhibition-type labels and merge
# rare classes into "Other inhibition".
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

output_dir = script_dir / "task3_cleaned_inhibition_type_outputs"
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

required_cols = [
    "Inhibition_Label_Numeric",
    "Inhibition_Type"
]

missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# ------------------------------------------------------------
# 2. Keep only inhibited samples
# ------------------------------------------------------------

df_inhibited = df[df["Inhibition_Label_Numeric"] == 1].copy()

print("\nInhibited-only dataset shape:")
print(df_inhibited.shape)

print("\nMissing Inhibition_Type values:")
print(df_inhibited["Inhibition_Type"].isna().sum())

df_inhibited = df_inhibited.dropna(subset=["Inhibition_Type"]).copy()

# ------------------------------------------------------------
# 3. Original label distribution
# ------------------------------------------------------------

original_distribution = (
    df_inhibited["Inhibition_Type"]
    .value_counts()
    .reset_index()
)

original_distribution.columns = ["Original_Inhibition_Type", "Count"]

print("\nOriginal inhibition-type distribution:")
print(original_distribution)

original_distribution.to_csv(
    output_dir / "original_inhibition_type_distribution.csv",
    index=False
)

# ------------------------------------------------------------
# 4. Standardize duplicated labels
# ------------------------------------------------------------

label_mapping = {
    "Ammonia": "Ammonia",

    "Unknown": "Unknown",

    "LCFA": "LCFA",
    "LCFA_inhibition": "LCFA",

    "Salinity": "Salinity",
    "Salinity inhibition": "Salinity",

    "VFA_accumulation": "VFA accumulation",
    "VFA accumulation": "VFA accumulation",

    "OLR shock": "OLR shock",

    "Mixed": "Mixed",
    "Mixed inhibition": "Mixed",

    "Recovered": "Recovered"
}

df_inhibited["Inhibition_Type_Clean"] = df_inhibited["Inhibition_Type"].map(label_mapping)

if df_inhibited["Inhibition_Type_Clean"].isna().sum() > 0:
    print("\nWARNING: Some labels were not mapped:")
    print(df_inhibited[df_inhibited["Inhibition_Type_Clean"].isna()]["Inhibition_Type"].unique())
    raise ValueError("Some inhibition-type labels are not included in the mapping.")

clean_distribution = (
    df_inhibited["Inhibition_Type_Clean"]
    .value_counts()
    .reset_index()
)

clean_distribution.columns = ["Clean_Inhibition_Type", "Count"]

clean_distribution["Percentage"] = (
    clean_distribution["Count"] / clean_distribution["Count"].sum() * 100
).round(2)

print("\nCleaned inhibition-type distribution:")
print(clean_distribution)

clean_distribution.to_csv(
    output_dir / "cleaned_inhibition_type_distribution.csv",
    index=False
)

# ------------------------------------------------------------
# 5. Merge rare classes
# ------------------------------------------------------------

min_class_size = 10

class_counts = df_inhibited["Inhibition_Type_Clean"].value_counts()

classes_to_keep = class_counts[class_counts >= min_class_size].index.tolist()
classes_to_merge = class_counts[class_counts < min_class_size].index.tolist()

df_inhibited["Inhibition_Type_Final"] = df_inhibited["Inhibition_Type_Clean"].apply(
    lambda x: x if x in classes_to_keep else "Other inhibition"
)

final_distribution = (
    df_inhibited["Inhibition_Type_Final"]
    .value_counts()
    .reset_index()
)

final_distribution.columns = ["Final_Inhibition_Type", "Count"]

final_distribution["Percentage"] = (
    final_distribution["Count"] / final_distribution["Count"].sum() * 100
).round(2)

print("\nFinal merged inhibition-type distribution:")
print(final_distribution)

final_distribution.to_csv(
    output_dir / "final_merged_inhibition_type_distribution.csv",
    index=False
)

# ------------------------------------------------------------
# 6. Save cleaned inhibited-only dataset
# ------------------------------------------------------------

df_inhibited.to_csv(
    output_dir / "task3_inhibited_only_cleaned_dataset.csv",
    index=False
)

# ------------------------------------------------------------
# 7. Save mapping and summary
# ------------------------------------------------------------

merge_summary = {
    "original_dataset_shape": list(df.shape),
    "inhibited_only_shape": list(df_inhibited.shape),
    "original_number_of_classes": int(original_distribution.shape[0]),
    "cleaned_number_of_classes": int(clean_distribution.shape[0]),
    "final_number_of_classes_after_rare_merge": int(final_distribution.shape[0]),
    "min_class_size_for_rare_merge": min_class_size,
    "label_mapping": label_mapping,
    "classes_kept": classes_to_keep,
    "classes_merged_into_other_inhibition": classes_to_merge,
    "final_class_distribution": {
        row["Final_Inhibition_Type"]: int(row["Count"])
        for _, row in final_distribution.iterrows()
    }
}

with open(output_dir / "task3_cleaning_and_merging_summary.json", "w", encoding="utf-8") as f:
    json.dump(merge_summary, f, indent=4, ensure_ascii=False)

# ------------------------------------------------------------
# 8. Plot cleaned distribution
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))
plt.barh(
    clean_distribution["Clean_Inhibition_Type"][::-1],
    clean_distribution["Count"][::-1]
)
plt.xlabel("Count")
plt.ylabel("Cleaned inhibition type")
plt.title("Task 3 — Cleaned Inhibition Type Distribution")
plt.tight_layout()
plt.savefig(
    output_dir / "cleaned_inhibition_type_distribution.png",
    dpi=300
)
plt.close()

# ------------------------------------------------------------
# 9. Plot final merged distribution
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))
plt.barh(
    final_distribution["Final_Inhibition_Type"][::-1],
    final_distribution["Count"][::-1]
)
plt.xlabel("Count")
plt.ylabel("Final inhibition type")
plt.title("Task 3 — Final Merged Inhibition Type Distribution")
plt.tight_layout()
plt.savefig(
    output_dir / "final_merged_inhibition_type_distribution.png",
    dpi=300
)
plt.close()

# ------------------------------------------------------------
# 10. Final message
# ------------------------------------------------------------

print("\n============================================================")
print("TASK 3 LABEL CLEANING AND MERGING COMPLETED")
print("============================================================")

print("\nClasses kept:")
print(classes_to_keep)

print("\nClasses merged into 'Other inhibition':")
print(classes_to_merge)

print("\nFinal number of classes:")
print(final_distribution.shape[0])

print("\nSaved outputs in:")
print(output_dir.resolve())

print("\nFiles created:")
for file in sorted(output_dir.iterdir()):
    print("-", file.name)