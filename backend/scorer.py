import numpy as np
import joblib
import shap

MODEL_PATH = 'models_saved/bank_risk_model.pkl'
SCALER_PATH = 'models_saved/scaler.pkl'

feature_cols = ['npa_ratio', 'car', 'roa', 'liquidity_coverage',
                'debt_to_equity', 'cost_to_income']

rf = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
explainer = shap.TreeExplainer(rf)

def compute_finguard_score(ratios_dict):
    vals = np.array([[ratios_dict[f] for f in feature_cols]])
    vals_scaled = scaler.transform(vals)

    proba = rf.predict_proba(vals_scaled)[0]
    crisis_prob = proba[1]
    score = round((1 - crisis_prob) * 100, 1)

    if score >= 80:   risk_label = 'Safe'
    elif score >= 60: risk_label = 'Caution'
    elif score >= 40: risk_label = 'At-Risk'
    else:             risk_label = 'Critical'

    raw_sv = explainer.shap_values(vals_scaled)
    if isinstance(raw_sv, list):
        sv_single = np.array(raw_sv[1]).flatten()
    else:
        sv_single = np.array(raw_sv).flatten()

    shap_dict = {f: round(float(sv_single[i]), 4) for i, f in enumerate(feature_cols)}

    return {
        'score': float(score),
        'risk_label': risk_label,
        'crisis_probability': round(float(crisis_prob) * 100, 1),
        'shap_values': shap_dict
    }
