import pandas as pd

FORECAST_PATH = 'models_saved/score_forecasts.csv'

forecasts_df = pd.read_csv(FORECAST_PATH)

def forecast_score(bank_name):
    row = forecasts_df[forecasts_df['bank'] == bank_name]
    if len(row) == 0:
        return {'error': f'No forecast found for {bank_name}'}
    row = row.iloc[0]
    return {
        'bank': bank_name,
        'current_score': float(row['current_score']),
        'current_label': row['current_label'] if 'current_label' in row else None,
        'q1_forecast': float(row['q1_forecast']),
        'q2_forecast': float(row['q2_forecast']),
        'direction': row['direction'],
        'will_cross_threshold': bool(row['will_cross_threshold']),
    }