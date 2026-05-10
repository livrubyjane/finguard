import numpy as np
import joblib
import pandas as pd
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lstm_inference import LSTMModel

LSTM_PATH = 'models_saved/lstm_trend_model.keras'
SCALER_PATH = 'models_saved/scaler_lstm.pkl'

feature_cols = [
    'npa_ratio',
    'car',
    'roa',
    'liquidity_coverage',
    'debt_to_equity',
    'cost_to_income'
]

SEQUENCE_LENGTH = 12

# Load model and scaler
model = LSTMModel(weights_path='models_saved/lstm_weights.npz')
scaler_seq = joblib.load(SCALER_PATH)


def predict_trend(bank_name, df):
    bank_df = df[df['bank_name'] == bank_name].sort_values(
        ['year', 'quarter']
    ).reset_index(drop=True)

    if len(bank_df) < SEQUENCE_LENGTH:
        return {
            'error': f'Need at least {SEQUENCE_LENGTH} quarters'
        }

    seq = bank_df[feature_cols].values[-SEQUENCE_LENGTH:]

    seq_scaled = scaler_seq.transform(seq).reshape(
        1,
        SEQUENCE_LENGTH,
        len(feature_cols)
    )

    prob = float(model.predict(seq_scaled)[0])

    direction = (
        'Improving'
        if prob > 0.5
        else 'Deteriorating'
    )

    confidence = round(
        float(prob if prob > 0.5 else 1 - prob) * 100,
        1
    )

    return {
        'bank': bank_name,
        'trend_direction': direction,
        'trend_confidence': f'{confidence}%',
        'years_analysed':
            f"{bank_df['year'].iloc[-SEQUENCE_LENGTH]} "
            f"{bank_df['quarter'].iloc[-SEQUENCE_LENGTH]} — "
            f"{bank_df['year'].iloc[-1]} "
            f"{bank_df['quarter'].iloc[-1]}"
    }