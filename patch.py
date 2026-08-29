with open('ad_biogas_interface.py', 'r', encoding='utf-8') as f:
    content = f.read()

old = 'def predict_inhibition_risk(input_data):\n    X = prepare_tabular_features(\n        input_data=input_data,\n        feature_names=models["task2_feature_names"],\n        target_encoder=models["task2_encoder"],\n        reactor_mapping=models["task2_reactor_mapping"]\n    )\n\n    X_scaled = models["task2_scaler"].transform(X.values)\n\n    if hasattr(models["task2_model"], "predict_proba"):\n        probability = float(models["task2_model"].predict_proba(X_scaled)[0][1])\n    else:\n        raw_prediction = models["task2_model"].predict(X_scaled)\n        probability = float(np.ravel(raw_prediction)[0])\n\n    threshold = models["task2_threshold"]\n    risk_detected = probability >= threshold\n\n    return probability, threshold, risk_detected'

new = '''def predict_inhibition_risk(input_data):
    substrate_encoded = apply_target_encoder(
        models["task2_encoder"],
        input_data["Substrate"]
    )
    impute_row = pd.DataFrame([{
        "HRT_days":      input_data["HRT_days"],
        "OLR_gVS_L_day": input_data["OLR_gVS_L_day"],
        "ISR":           input_data["ISR"],
    }])
    imputed = models["task2_imputer"].transform(impute_row)
    hrt_imp, olr_imp, isr_imp = imputed[0]
    scale_row = pd.DataFrame([{
        "Temperature_C":  input_data["Temperature_C"],
        "pH":             input_data["pH"],
        "HRT_days":       hrt_imp,
        "OLR_gVS_L_day":  olr_imp,
        "ISR":            isr_imp,
        "TAN_mg_L":       input_data["TAN_mg_L"],
        "VFA_mg_L":       input_data["VFA_mg_L"],
        "Substrate":      substrate_encoded,
    }])
    X_scaled = models["task2_scaler"].transform(scale_row)
    X_scaled_df = pd.DataFrame(X_scaled, columns=scale_row.columns)
    raw_reactor = input_data["Reactor_Type"]
    cleaned_reactor = models["task2_reactor_mapping"].get(raw_reactor, raw_reactor)
    for feature in models["task2_feature_names"]:
        if feature.startswith("Reactor_Type_"):
            X_scaled_df[feature] = 0
    reactor_feature = f"Reactor_Type_{cleaned_reactor}"
    if reactor_feature in models["task2_feature_names"]:
        X_scaled_df[reactor_feature] = 1
    X_final = X_scaled_df[models["task2_feature_names"]]
    if hasattr(models["task2_model"], "predict_proba"):
        probability = float(models["task2_model"].predict_proba(X_final)[0][1])
    else:
        raw_prediction = models["task2_model"].predict(X_final)
        probability = float(np.ravel(raw_prediction)[0])
    threshold = models["task2_threshold"]
    risk_detected = probability >= threshold
    return probability, threshold, risk_detected'''

if old in content:
    content = content.replace(old, new)
    with open('ad_biogas_interface.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('SUCCESS: predict_inhibition_risk patched')
else:
    print('ERROR: old code not found')
    if 'def predict_inhibition_risk' in content:
        print('Function name found')
    if 'X_scaled = models["task2_scaler"].transform(X.values)' in content:
        print('Old scaler line found')