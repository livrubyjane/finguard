from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from backend.scorer import compute_finguard_score, feature_cols
from backend.trend import predict_trend
from backend.forecast import forecast_score
from backend.news import get_news_and_sentiment

app = FastAPI(title="FinGuard API", version="1.0.0")

from map import router as map_router          # alongside your other router imports

app.include_router(map_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data once on startup
df = pd.read_csv('models_saved/finguard_dataset.csv')

# Precompute scores for all banks
def get_latest_score(bank_name):
    bank_df = df[df['bank_name'] == bank_name].sort_values(
        ['year', 'quarter']).iloc[-1]
    ratios = {f: float(bank_df[f]) for f in feature_cols}
    return compute_finguard_score(ratios)

# ── ENDPOINTS ──────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "FinGuard API is running"}

@app.get("/banks")
def get_all_banks():
    banks = []
    for bank in sorted(df['bank_name'].unique()):
        result = get_latest_score(bank)
        banks.append({
            'bank_name': bank,
            'score': result['score'],
            'risk_label': result['risk_label'],
            'crisis_probability': result['crisis_probability'],
        })
    return banks

@app.get("/bank/{bank_name}/score")
def get_score(bank_name: str):
    bank_name = bank_name.upper().replace('-', ' ')
    if bank_name not in df['bank_name'].unique():
        raise HTTPException(status_code=404, detail=f"Bank '{bank_name}' not found")
    bank_df = df[df['bank_name'] == bank_name].sort_values(['year', 'quarter']).iloc[-1]
    ratios = {f: float(bank_df[f]) for f in feature_cols}
    result = compute_finguard_score(ratios)
    result['ratios'] = ratios
    result['year'] = int(bank_df['year'])
    result['quarter'] = bank_df['quarter']
    return result

@app.get("/bank/{bank_name}/trend")
def get_trend(bank_name: str):
    bank_name = bank_name.upper().replace('-', ' ')
    bank_df = df[df['bank_name'] == bank_name].sort_values(['year', 'quarter'])
    if len(bank_df) == 0:
        raise HTTPException(status_code=404, detail="Bank not found")
    history = []
    for _, row in bank_df.iterrows():
        ratios = {f: float(row[f]) for f in feature_cols}
        score = compute_finguard_score(ratios)['score']
        history.append({
            'year': int(row['year']),
            'quarter': row['quarter'],
            'period': f"{int(row['year'])} {row['quarter']}",
            'score': score,
            'crisis_label': int(row['crisis_label'])
        })
    lstm_trend = predict_trend(bank_name, df)
    return {
        'bank': bank_name,
        'history': history,
        'lstm_trend': lstm_trend
    }

@app.get("/bank/{bank_name}/ratios")
def get_ratios(bank_name: str):
    bank_name = bank_name.upper().replace('-', ' ')
    bank_df = df[df['bank_name'] == bank_name].sort_values(['year', 'quarter'])
    if len(bank_df) == 0:
        raise HTTPException(status_code=404, detail="Bank not found")
    latest = bank_df.iloc[-1]
    industry_avg = df.groupby(['year', 'quarter'])[feature_cols].mean()
    latest_avg = industry_avg.loc[(latest['year'], latest['quarter'])]
    ratios = []
    for f in feature_cols:
        ratios.append({
            'ratio': f,
            'bank_value': round(float(latest[f]), 2),
            'industry_avg': round(float(latest_avg[f]), 2),
        })
    return {'bank': bank_name, 'ratios': ratios}

@app.get("/bank/{bank_name}/shap")
def get_shap(bank_name: str):
    bank_name = bank_name.upper().replace('-', ' ')
    bank_df = df[df['bank_name'] == bank_name].sort_values(['year', 'quarter'])
    if len(bank_df) == 0:
        raise HTTPException(status_code=404, detail="Bank not found")
    latest = bank_df.iloc[-1]
    ratios = {f: float(latest[f]) for f in feature_cols}
    result = compute_finguard_score(ratios)
    return {'bank': bank_name, 'shap_values': result['shap_values']}

@app.get("/bank/{bank_name}/forecast")
def get_forecast(bank_name: str):
    bank_name = bank_name.upper().replace('-', ' ')
    return forecast_score(bank_name)

@app.get("/compare")
def compare_banks(banks: str):
    bank_list = [b.strip().upper() for b in banks.split(',')]
    results = []
    for bank in bank_list:
        if bank not in df['bank_name'].unique():
            continue
        score_result = get_latest_score(bank)
        latest = df[df['bank_name'] == bank].sort_values(['year', 'quarter']).iloc[-1]
        results.append({
            'bank_name': bank,
            'score': score_result['score'],
            'risk_label': score_result['risk_label'],
            'ratios': {f: round(float(latest[f]), 2) for f in feature_cols},
            'shap_values': score_result['shap_values'],
        })
    return results

@app.get("/bank/{bank_name}/news")
def get_news(bank_name: str):
    bank_name = bank_name.upper().replace('-', ' ')
    return get_news_and_sentiment(bank_name)

class RatiosInput(BaseModel):
    npa_ratio: float
    car: float
    roa: float
    liquidity_coverage: float
    debt_to_equity: float
    cost_to_income: float

@app.post("/predict")
def predict_custom(ratios: RatiosInput):
    ratios_dict = ratios.dict()
    return compute_finguard_score(ratios_dict)