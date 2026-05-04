# FinGuard  
**AI-Powered Financial Health Screener for Indian Banks & NBFCs**

---

## 🚨 Problem

India has 80M+ retail investors actively investing in bank stocks and fixed deposits.  
However, investors typically rely on **stock prices or interest rates**, without understanding the **underlying financial health of institutions**.

Historical failures like:
- Yes Bank (2020)
- PMC Bank (2019)
- IL&FS (2018)
- DHFL (2019)

show that **warning signals existed in RBI data years in advance**, but:
- Data is fragmented across PDFs and Excel files  
- Requires financial expertise to interpret  
- No tool aggregates, compares, and explains it simply  

➡️ Retail investors are left exposed.

---

## 💡 Solution — FinGuard

FinGuard is a **free, AI-powered platform** that:

- Aggregates RBI financial data across banks and NBFCs  
- Computes a **normalized health score (0–100)**  
- Detects **early warning signals using ML**  
- Projects **future financial health trends**  
- Explains results in **plain language**  
- Connects insights to **user-specific investments**  

---

## 🎯 Core Features

### 1. FinGuard Health Score
- Composite score (0–100)
- Based on 6 financial ratios:
  - NPA Ratio
  - CAR (Capital Adequacy)
  - ROA (Profitability)
  - Liquidity Coverage
  - Debt-to-Equity
  - Cost-to-Income  

**Risk Levels:**
- 80–100 → Safe  
- 60–79 → Caution  
- 40–59 → At Risk  
- 0–39 → Critical  

📊 Powered by Random Forest ML model :contentReference[oaicite:0]{index=0}

---

### 2. 5-Year Trend Detection
- Uses LSTM to classify:
  - Improving 📈  
  - Deteriorating 📉  

➡️ Detects slow decline before collapse :contentReference[oaicite:1]{index=1}

---

### 3. Predictive Forecasting
- Projects next 2 quarters using ARIMA  
- Identifies **threshold crossings before risk escalation**

---

### 4. Explainability (SHAP)
- Breaks down score into:
  - Which ratio impacted risk  
  - By how much  

➡️ Converts black-box ML → actionable insights :contentReference[oaicite:2]{index=2}

---

### 5. News Sentiment (FinBERT)
- Analyzes latest financial news headlines  
- Detects governance risks (fraud, resignations, etc.)

---

### 6. Personal Investment Risk Connector
Users input:
- FD amount OR number of shares  

System outputs:
- Insurance coverage (DICGC)
- Key risk indicator to track  
- What a score drop means for *their money*  

---

### 7. Historical Crisis Validation
Demonstrates that FinGuard would have flagged:
- Yes Bank  
- PMC Bank  
- IL&FS  
- DHFL  

**12–18 months before collapse** :contentReference[oaicite:3]{index=3}

---

## 🏗️ System Architecture

### Backend
- FastAPI  
- PostgreSQL  
- REST APIs  
- ML inference layer  

### ML Models
- Random Forest → Risk scoring  
- LSTM → Trend detection  
- ARIMA → Forecasting  
- FinBERT → Sentiment analysis  

### Frontend
- Streamlit dashboard  
- Plotly visualizations  

### Deployment
- Docker + Docker Compose  
- AWS EC2 + S3 + RDS  
- CI/CD via GitHub Actions :contentReference[oaicite:4]{index=4}  

---

## 📊 Data Sources

- RBI Statistical Tables (2015–2024)  
- RBI Financial Stability Reports  
- Screener.in (financial ratios)  
- NewsAPI (real-time news)  

---

## ⚙️ Pipeline Overview

1. Data collection (RBI + Screener)
2. Data cleaning & merging
3. Feature engineering (6 ratios)
4. Model training (RF, LSTM, ARIMA)
5. Explainability (SHAP)
6. API deployment
7. Frontend visualization : yet to be added

---

