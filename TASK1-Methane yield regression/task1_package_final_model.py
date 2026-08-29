# ============================================================
# TASK 1 — Package Final Methane Yield Regression Model
# Final selected model: Tuned ANN / MLPRegressor
# ============================================================

import shutil
from pathlib import Path

# ------------------------------------------------------------
# 0. Paths
# ------------------------------------------------------------

script_dir = Path(__file__).resolve().parent

preprocessing_dir = script_dir / "task1_preprocessing_outputs"
ann_tuning_dir = script_dir / "task1_ann_tuning_outputs"

final_model_dir = script_dir / "final_task1_methane_yield_model"
final_model_dir.mkdir(exist_ok=True)

# ------------------------------------------------------------
# 1. Check required folders
# ------------------------------------------------------------

if not preprocessing_dir.exists():
    raise FileNotFoundError("Missing folder: task1_preprocessing_outputs")

if not ann_tuning_dir.exists():
    raise FileNotFoundError("Missing folder: task1_ann_tuning_outputs")

# ------------------------------------------------------------
# 2. Files to copy
# ------------------------------------------------------------

files_to_copy = [
    # Final tuned ANN model
    (ann_tuning_dir / "ann_tuned_task1_model.pkl",
     final_model_dir / "ann_tuned_task1_model.pkl"),

    (ann_tuning_dir / "ann_tuned_metrics.json",
     final_model_dir / "ann_tuned_metrics.json"),

    (ann_tuning_dir / "ann_tuned_training_details.json",
     final_model_dir / "ann_tuned_training_details.json"),

    # Preprocessing objects
    (preprocessing_dir / "target_encoder_task1.pkl",
     final_model_dir / "target_encoder_task1.pkl"),

    (preprocessing_dir / "median_imputer_task1.pkl",
     final_model_dir / "median_imputer_task1.pkl"),

    (preprocessing_dir / "standard_scaler_task1.pkl",
     final_model_dir / "standard_scaler_task1.pkl"),

    # Metadata and mappings
    (preprocessing_dir / "task1_feature_names.json",
     final_model_dir / "task1_feature_names.json"),

    (preprocessing_dir / "task1_preprocessing_summary.json",
     final_model_dir / "task1_preprocessing_summary.json"),

    (preprocessing_dir / "reactor_mapping.json",
     final_model_dir / "reactor_mapping.json"),

    (preprocessing_dir / "substrate_category_map.json",
     final_model_dir / "substrate_category_map.json")
]

# ------------------------------------------------------------
# 3. Copy files
# ------------------------------------------------------------

for source, destination in files_to_copy:
    if not source.exists():
        raise FileNotFoundError(f"Missing required file: {source}")

    shutil.copy2(source, destination)

# ------------------------------------------------------------
# 4. Final message
# ------------------------------------------------------------

print("\n============================================================")
print("FINAL TASK 1 MODEL PACKAGE CREATED SUCCESSFULLY")
print("============================================================")

print("\nFinal model folder:")
print(final_model_dir.resolve())

print("\nFiles included:")
for file in sorted(final_model_dir.iterdir()):
    print("-", file.name)