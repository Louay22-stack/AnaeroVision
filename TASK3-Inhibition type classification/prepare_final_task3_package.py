# ============================================================
# TASK 3 FINAL PREDICTOR
# Final model: Tuned Random Forest
#
# Task 3:
# Predict inhibition type when Task 2 detects inhibition risk.
#
# Confidence logic:
# - confidence >= 0.60: high confidence
# - 0.45 <= confidence < 0.60: moderate confidence
# - 0.40 <= confidence < 0.45: low confidence
# - confidence < 0.40: uncertain / unknown
# ============================================================

import json
import joblib
import pandas as pd
from pathlib import Path


# ------------------------------------------------------------
# 1. Load model files
# ------------------------------------------------------------

model_dir = Path(__file__).resolve().parent

model_path = model_dir / "task3_final_tuned_rf_pipeline.pkl"
label_mapping_path = model_dir / "task3_label_mapping.json"

if not model_path.exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")

if not label_mapping_path.exists():
    raise FileNotFoundError(f"Label mapping file not found: {label_mapping_path}")

model = joblib.load(model_path)

with open(label_mapping_path, "r", encoding="utf-8") as f:
    label_mapping_raw = json.load(f)

label_mapping = {
    int(k): v
    for k, v in label_mapping_raw.items()
}


# ------------------------------------------------------------
# 2. Confidence thresholds
# ------------------------------------------------------------

HIGH_THRESHOLD = 0.60
MODERATE_THRESHOLD = 0.45
LOW_THRESHOLD = 0.40


# ------------------------------------------------------------
# 3. Required features
# ------------------------------------------------------------

feature_cols = [
    "Temperature_C",
    "pH",
    "HRT_days",
    "OLR_gVS_L_day",
    "ISR",
    "TAN_mg_L",
    "VFA_mg_L",
    "Substrate_Category",
    "Reactor_Type"
]


# ------------------------------------------------------------
# 4. Reactor type cleaning
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


# ------------------------------------------------------------
# 5. Prediction function
# ------------------------------------------------------------

def predict_inhibition_type(input_data):
    """
    Example input:

    {
        "Temperature_C": 37,
        "pH": 7.2,
        "HRT_days": 20,
        "OLR_gVS_L_day": 3.5,
        "ISR": 2.0,
        "TAN_mg_L": 3500,
        "VFA_mg_L": 2500,
        "Substrate_Category": "Protein-Rich",
        "Reactor_Type": "Batch"
    }
    """

    X = pd.DataFrame([input_data])

    missing_features = [
        col for col in feature_cols
        if col not in X.columns
    ]

    if missing_features:
        raise ValueError(f"Missing required input features: {missing_features}")

    X = X[feature_cols].copy()

    X["Substrate_Category"] = (
        X["Substrate_Category"]
        .fillna("Other/unknown")
        .astype(str)
    )

    X["Reactor_Type"] = X["Reactor_Type"].map(reactor_mapping)
    X["Reactor_Type"] = X["Reactor_Type"].fillna("Other")

    predicted_encoded = int(model.predict(X)[0])
    predicted_type = label_mapping[predicted_encoded]

    probabilities = model.predict_proba(X)[0]
    confidence = float(probabilities[predicted_encoded])

    class_probabilities = {
        label_mapping[int(i)]: float(probabilities[i])
        for i in range(len(probabilities))
    }

    if confidence >= HIGH_THRESHOLD:
        confidence_level = "high"
        final_message = f"Likely inhibition type: {predicted_type}."

    elif confidence >= MODERATE_THRESHOLD:
        confidence_level = "moderate"
        final_message = (
            f"Possible inhibition type: {predicted_type}. "
            "Recommendation: verify with process indicators."
        )

    elif confidence >= LOW_THRESHOLD:
        confidence_level = "low"
        final_message = (
            f"Tentative inhibition type: {predicted_type}. "
            "Confidence is low, so this prediction should be interpreted cautiously."
        )

    else:
        confidence_level = "uncertain"
        final_message = "Inhibition type uncertain / unknown."

    result = {
        "predicted_inhibition_type": predicted_type,
        "confidence": confidence,
        "confidence_level": confidence_level,
        "final_message": final_message,
        "class_probabilities": class_probabilities
    }

    return result


# ------------------------------------------------------------
# 6. Test example
# ------------------------------------------------------------

if __name__ == "__main__":

    example_input = {
        "Temperature_C": 37,
        "pH": 7.2,
        "HRT_days": 20,
        "OLR_gVS_L_day": 3.5,
        "ISR": 2.0,
        "TAN_mg_L": 3500,
        "VFA_mg_L": 2500,
        "Substrate_Category": "Protein-Rich",
        "Reactor_Type": "Batch"
    }

    prediction = predict_inhibition_type(example_input)

    print("\nTask 3 prediction result:")
    print(json.dumps(prediction, indent=4))