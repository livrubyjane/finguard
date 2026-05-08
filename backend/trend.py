import numpy as np
import joblib
import tensorflow as tf
import pandas as pd

LSTM_PATH = 'models_saved/lstm_trend_model.h5'
SCALER_PATH = 'models_saved/scaler_lstm.pkl'

feature_cols = ['npa_ratio', 'car', 'roa', 'liquidity_coverage',
                'debt_to_equity', 'cost_to_income']
SEQUENCE_LENGTH = 12

model = tf.keras.models.load_model(LSTM_PATH)
scaler_seq = joblib.load(SCALER_PATH)

def predict_trend(bank_name, df):
    bank_df = df[df['bank_name'] == bank_name].sort_values(
        ['year', 'quarter']).reset_index(drop=True)

    if len(bank_df) < SEQUENCE_LENGTH:
        return {'error': f'Need at least {SEQUENCE_LENGTH} quarters'}

    seq = bank_df[feature_cols].values[-SEQUENCE_LENGTH:]
    seq_scaled = scaler_seq.transform(seq).reshape(1, SEQUENCE_LENGTH, len(feature_cols))

    prob = model.predict(seq_scaled, verbose=0)[0][0]
    direction = 'Improving' if prob > 0.5 else 'Deteriorating'
    confidence = round(float(prob if prob > 0.5 else 1 - prob) * 100, 1)

    return {
        'bank': bank_name,
        'trend_direction': direction,
        'trend_confidence': f'{confidence}%',
        'years_analysed': f"{bank_df['year'].iloc[-SEQUENCE_LENGTH]} Q{bank_df['quarter'].iloc[-SEQUENCE_LENGTH][-1]} — {bank_df['year'].iloc[-1]} {bank_df['quarter'].iloc[-1]}"
    }