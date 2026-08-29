# ============================================================
# INTEGRATED AD BIOGAS PREDICTION INTERFACE
#
# Task 1: Methane yield prediction
# Task 2: Inhibition risk classification
# Task 3: Inhibition type classification
# ============================================================

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path


# ------------------------------------------------------------
# 0. Page configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="AD Biogas AI Predictor",
    page_icon="ðŸ§ª",
    layout="wide"
)


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

TASK1_DIR = BASE_DIR / "final_task1_methane_yield_model"
TASK2_DIR = BASE_DIR / "final_task2_inhibition_risk_model"
TASK3_DIR = BASE_DIR / "final_task3_inhibition_type_model"


# ------------------------------------------------------------
# 2. Utility functions
# ------------------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_feature_names(path):
    data = load_json(path)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["feature_names", "features", "final_features", "columns"]:
            if key in data:
                return data[key]

    raise ValueError(f"Could not read feature names from: {path}")


def apply_target_encoder(encoder, substrate_value):
    substrate_df = pd.DataFrame({"Substrate": [substrate_value]})
    encoded = encoder.transform(substrate_df)

    if isinstance(encoded, pd.DataFrame):
        return float(encoded.iloc[0, 0])

    return float(encoded[0][0])


def prepare_tabular_features(
    input_data,
    feature_names,
    target_encoder,
    reactor_mapping
):
    """
    Prepares the feature vector for Task 1 and Task 2.
    These models use:
    - Substrate target encoding
    - Numeric process variables
    - One-hot reactor type variables
    """

    substrate_encoded = apply_target_encoder(
        target_encoder,
        input_data["Substrate"]
    )

    raw_reactor = input_data["Reactor_Type"]
    cleaned_reactor = reactor_mapping.get(raw_reactor, raw_reactor)

    row = {
        "Substrate": substrate_encoded,
        "Temperature_C": input_data["Temperature_C"],
        "pH": input_data["pH"],
        "HRT_days": input_data["HRT_days"],
        "OLR_gVS_L_day": input_data["OLR_gVS_L_day"],
        "ISR": input_data["ISR"],
        "TAN_mg_L": input_data["TAN_mg_L"],
        "VFA_mg_L": input_data["VFA_mg_L"],
    }

    for feature in feature_names:
        if feature.startswith("Reactor_Type_"):
            row[feature] = 0

    reactor_feature = f"Reactor_Type_{cleaned_reactor}"

    if reactor_feature in feature_names:
        row[reactor_feature] = 1

    X = pd.DataFrame([row])

    for feature in feature_names:
        if feature not in X.columns:
            X[feature] = 0

    X = X[feature_names].copy()

    return X


# ------------------------------------------------------------
# 3. Load models
# ------------------------------------------------------------

@st.cache_resource
def load_all_models():

    # Task 1
    task1_model = joblib.load(TASK1_DIR / "ann_tuned_task1_model.pkl")
    task1_imputer = joblib.load(TASK1_DIR / "median_imputer_task1.pkl")
    task1_scaler = joblib.load(TASK1_DIR / "standard_scaler_task1.pkl")
    task1_encoder = joblib.load(TASK1_DIR / "target_encoder_task1.pkl")
    task1_reactor_mapping = load_json(TASK1_DIR / "reactor_mapping.json")
    task1_feature_names = load_feature_names(TASK1_DIR / "task1_feature_names.json")

    # Task 2
    task2_model = joblib.load(TASK2_DIR / "ann_mlp_task2_final_model.pkl")
    task2_imputer = joblib.load(TASK2_DIR / "median_imputer_task2_final.pkl")
    task2_scaler = joblib.load(TASK2_DIR / "standard_scaler_task2_final.pkl")
    task2_encoder = joblib.load(TASK2_DIR / "target_encoder_task2_final.pkl")
    task2_reactor_mapping = load_json(TASK2_DIR / "reactor_mapping_task2_final.json")
    task2_feature_names = load_feature_names(TASK2_DIR / "task2_final_feature_names.json")

    # Task 2 threshold
    task2_threshold = 0.14

    task2_summary_path = TASK2_DIR / "task2_final_model_summary.json"
    if task2_summary_path.exists():
        task2_summary = load_json(task2_summary_path)

        for key in ["threshold", "decision_threshold", "final_threshold", "selected_threshold"]:
            if key in task2_summary:
                task2_threshold = float(task2_summary[key])
                break

    # Task 3
    task3_model = joblib.load(TASK3_DIR / "task3_final_tuned_rf_pipeline.pkl")
    task3_label_mapping_raw = load_json(TASK3_DIR / "task3_label_mapping.json")
    task3_label_mapping = {
        int(k): v
        for k, v in task3_label_mapping_raw.items()
    }

    task3_thresholds = load_json(TASK3_DIR / "task3_confidence_thresholds.json")

    return {
        "task1_model": task1_model,
        "task1_imputer": task1_imputer,
        "task1_scaler": task1_scaler,
        "task1_encoder": task1_encoder,
        "task1_reactor_mapping": task1_reactor_mapping,
        "task1_feature_names": task1_feature_names,

        "task2_model": task2_model,
        "task2_imputer": task2_imputer,
        "task2_scaler": task2_scaler,
        "task2_encoder": task2_encoder,
        "task2_reactor_mapping": task2_reactor_mapping,
        "task2_feature_names": task2_feature_names,
        "task2_threshold": task2_threshold,

        "task3_model": task3_model,
        "task3_label_mapping": task3_label_mapping,
        "task3_thresholds": task3_thresholds,
    }


models = load_all_models()


# ------------------------------------------------------------
# 4. Prediction functions
# ------------------------------------------------------------

def predict_methane_yield(input_data):
    X = prepare_tabular_features(
        input_data=input_data,
        feature_names=models["task1_feature_names"],
        target_encoder=models["task1_encoder"],
        reactor_mapping=models["task1_reactor_mapping"]
    )

    X_scaled = models["task1_scaler"].transform(X.values)

    prediction = models["task1_model"].predict(X_scaled)

    methane_yield = float(np.ravel(prediction)[0])

    return methane_yield


def predict_inhibition_risk(input_data):
    X = prepare_tabular_features(
        input_data=input_data,
        feature_names=models["task2_feature_names"],
        target_encoder=models["task2_encoder"],
        reactor_mapping=models["task2_reactor_mapping"]
    )

    X_scaled = models["task2_scaler"].transform(X.values)

    if hasattr(models["task2_model"], "predict_proba"):
        probability = float(models["task2_model"].predict_proba(X_scaled)[0][1])
    else:
        raw_prediction = models["task2_model"].predict(X_scaled)
        probability = float(np.ravel(raw_prediction)[0])

    threshold = models["task2_threshold"]
    risk_detected = probability >= threshold

    return probability, threshold, risk_detected


def clean_task3_reactor_type(reactor_type):
    reactor_mapping = {
        "Batch": "Batch",
        "Batch (BMP)": "Batch",
        "Batch serum bottles": "Batch",

        "Thermophilic Batch": "Thermophilic_Batch",
        "Thermophilic AD": "Thermophilic_Batch",
        "Thermophilic_Batch": "Thermophilic_Batch",

        "CSTR": "CSTR",
        "CSTR (fed-batch)": "CSTR",
        "CSTR (mesophilic)": "CSTR",
        "Two-stage CSTR": "CSTR",
        "Serial CSTR": "CSTR",

        "Horizontal flow reactor": "Other",
        "Horizontal flow": "Other",
        "Other": "Other"
    }

    return reactor_mapping.get(reactor_type, "Other")


def predict_inhibition_type(input_data):
    task3_input = {
        "Temperature_C": input_data["Temperature_C"],
        "pH": input_data["pH"],
        "HRT_days": input_data["HRT_days"],
        "OLR_gVS_L_day": input_data["OLR_gVS_L_day"],
        "ISR": input_data["ISR"],
        "TAN_mg_L": input_data["TAN_mg_L"],
        "VFA_mg_L": input_data["VFA_mg_L"],
        "Substrate_Category": input_data["Substrate_Category"],
        "Reactor_Type": clean_task3_reactor_type(input_data["Reactor_Type"])
    }

    X = pd.DataFrame([task3_input])

    predicted_encoded = int(models["task3_model"].predict(X)[0])
    predicted_type = models["task3_label_mapping"][predicted_encoded]

    probabilities = models["task3_model"].predict_proba(X)[0]
    confidence = float(probabilities[predicted_encoded])

    class_probabilities = {
        models["task3_label_mapping"][int(i)]: float(probabilities[i])
        for i in range(len(probabilities))
    }

    high_threshold = float(models["task3_thresholds"].get("high_confidence_threshold", 0.60))
    moderate_threshold = float(models["task3_thresholds"].get("moderate_confidence_threshold", 0.45))
    low_threshold = float(models["task3_thresholds"].get("low_confidence_threshold", 0.40))

    if confidence >= high_threshold:
        confidence_level = "High"
        final_message = f"Likely inhibition type: {predicted_type}"

    elif confidence >= moderate_threshold:
        confidence_level = "Moderate"
        final_message = (
            f"Possible inhibition type: {predicted_type}. "
            "Verify with process indicators."
        )

    elif confidence >= low_threshold:
        confidence_level = "Low"
        final_message = (
            f"Tentative inhibition type: {predicted_type}. "
            "Interpret cautiously."
        )

    else:
        confidence_level = "Uncertain"
        final_message = "Inhibition type uncertain / unknown."

    return {
        "predicted_type": predicted_type,
        "confidence": confidence,
        "confidence_level": confidence_level,
        "final_message": final_message,
        "class_probabilities": class_probabilities
    }


# ------------------------------------------------------------
# 5. Interface
# ------------------------------------------------------------

st.title("ðŸ§ª AI-Based Anaerobic Digestion Prediction Interface")

st.markdown(
    """
This interface integrates three machine learning models for anaerobic digestion monitoring:

1. **Methane yield prediction**
2. **Inhibition risk classification**
3. **Inhibition type classification**
"""
)

with st.sidebar:
    st.header("Model Summary")

    st.markdown(
        """
**Task 1**  
Methane yield prediction  
Final model: Tuned ANN

**Task 2**  
Inhibition risk classification  
Final model: ANN  
Threshold: 0.14

**Task 3**  
Inhibition type classification  
Final model: Tuned Random Forest
"""
    )


st.subheader("Input Anaerobic Digestion Conditions")

col1, col2, col3 = st.columns(3)

with col1:
    substrate = st.text_input(
        "Substrate name",
        value="Protein-rich waste"
    )

    substrate_category = st.selectbox(
        "Substrate category",
        [
            "Protein-Rich",
            "Lipid-Rich",
            "Food Waste",
            "Manure/Sludge",
            "Agricultural residue",
            "Industrial organic waste",
            "Co-Digestion",
            "Other/unknown"
        ]
    )

    reactor_type = st.selectbox(
        "Reactor type",
        [
            "Batch",
            "CSTR",
            "Thermophilic_Batch",
            "Other"
        ]
    )

with col2:
    temperature = st.number_input(
        "Temperature (Â°C)",
        min_value=10.0,
        max_value=70.0,
        value=37.0,
        step=0.5
    )

    ph = st.number_input(
        "pH",
        min_value=3.0,
        max_value=10.0,
        value=7.2,
        step=0.1
    )

    hrt = st.number_input(
        "HRT (days)",
        min_value=0.1,
        max_value=100.0,
        value=20.0,
        step=0.5
    )

with col3:
    olr = st.number_input(
        "OLR (gVS/L/day)",
        min_value=0.0,
        max_value=30.0,
        value=3.5,
        step=0.1
    )

    isr = st.number_input(
        "ISR",
        min_value=0.0,
        max_value=20.0,
        value=2.0,
        step=0.1
    )

    tan = st.number_input(
        "TAN (mg/L)",
        min_value=0.0,
        max_value=20000.0,
        value=3500.0,
        step=50.0
    )

    vfa = st.number_input(
        "VFA (mg/L)",
        min_value=0.0,
        max_value=30000.0,
        value=2500.0,
        step=50.0
    )


input_data = {
    "Substrate": substrate,
    "Substrate_Category": substrate_category,
    "Reactor_Type": reactor_type,
    "Temperature_C": temperature,
    "pH": ph,
    "HRT_days": hrt,
    "OLR_gVS_L_day": olr,
    "ISR": isr,
    "TAN_mg_L": tan,
    "VFA_mg_L": vfa
}


st.divider()

if st.button("Run Prediction", type="primary"):

    with st.spinner("Running integrated prediction pipeline..."):

        methane_yield = predict_methane_yield(input_data)

        inhibition_probability, inhibition_threshold, risk_detected = predict_inhibition_risk(input_data)

        task3_result = None

        if risk_detected:
            task3_result = predict_inhibition_type(input_data)

    st.subheader("Prediction Results")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric(
            label="Predicted Methane Yield",
            value=f"{methane_yield:.2f}",
            help="Predicted methane yield in mL CHâ‚„/gVS"
        )
        st.caption("Unit: mL CHâ‚„/gVS")

    with result_col2:
        st.metric(
            label="Inhibition Probability",
            value=f"{inhibition_probability:.3f}",
            help=f"Decision threshold = {inhibition_threshold:.2f}"
        )

        if risk_detected:
            st.error("Inhibition risk detected")
        else:
            st.success("No inhibition risk detected")

    with result_col3:
        if risk_detected and task3_result is not None:
            st.metric(
                label="Inhibition Type Confidence",
                value=f"{task3_result['confidence']:.3f}"
            )
            st.caption(f"Confidence level: {task3_result['confidence_level']}")
        else:
            st.metric(
                label="Inhibition Type",
                value="Not required"
            )
            st.caption("Task 3 is only used when inhibition risk is detected.")

    st.divider()

    if risk_detected and task3_result is not None:
        st.subheader("Inhibition Type Interpretation")

        if task3_result["confidence_level"] == "High":
            st.success(task3_result["final_message"])
        elif task3_result["confidence_level"] == "Moderate":
            st.warning(task3_result["final_message"])
        elif task3_result["confidence_level"] == "Low":
            st.info(task3_result["final_message"])
        else:
            st.error(task3_result["final_message"])

        probability_df = pd.DataFrame(
            {
                "Inhibition Type": list(task3_result["class_probabilities"].keys()),
                "Probability": list(task3_result["class_probabilities"].values())
            }
        ).sort_values(by="Probability", ascending=False)

        st.markdown("#### Class probabilities")
        st.dataframe(probability_df, use_container_width=True)

    else:
        st.subheader("Inhibition Type Interpretation")
        st.success("Since no inhibition risk was detected, inhibition-type classification was not triggered.")

    with st.expander("Show input data used for prediction"):
        st.json(input_data)


