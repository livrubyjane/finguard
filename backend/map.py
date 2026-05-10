"""
map.py — FinGuard Health Map Backend
GET /map/regions         → all regions with avg + worst-case scores + hover tooltip
GET /map/region/{name}   → single region detail with full bank breakdown
"""

from fastapi import APIRouter, HTTPException
from typing import Optional
import pandas as pd
import numpy as np

router = APIRouter()

# ---------------------------------------------------------------------------
# Bank → Region mapping  (HQ-based, 70+ banks across 20 regions)
# ---------------------------------------------------------------------------
BANK_REGION_MAP = {

    # ── Maharashtra ─────────────────────────────────────────────────────────
    "Bank of Maharashtra":          "Maharashtra",
    "Central Bank of India":        "Maharashtra",
    "Union Bank of India":          "Maharashtra",
    "IDBI Bank":                    "Maharashtra",
    "Axis Bank":                    "Maharashtra",
    "Kotak Mahindra Bank":          "Maharashtra",
    "HDFC Bank":                    "Maharashtra",
    "Yes Bank":                     "Maharashtra",
    "IndusInd Bank":                "Maharashtra",
    "Bank of India":                "Maharashtra",
    "State Bank of India":          "Maharashtra",
    "RBL Bank":                     "Maharashtra",
    "ESAF Small Finance Bank":      "Maharashtra",

    # ── Delhi / NCR ─────────────────────────────────────────────────────────
    "Punjab National Bank":         "Delhi",
    "Oriental Bank of Commerce":    "Delhi",
    "Saraswat Bank":                "Delhi",
    "New India Co-operative Bank":  "Delhi",

    # ── West Bengal ─────────────────────────────────────────────────────────
    "UCO Bank":                     "West Bengal",
    "United Bank of India":         "West Bengal",
    "Allahabad Bank":               "West Bengal",
    "Bandhan Bank":                 "West Bengal",

    # ── Tamil Nadu ──────────────────────────────────────────────────────────
    "Indian Bank":                  "Tamil Nadu",
    "Indian Overseas Bank":         "Tamil Nadu",
    "City Union Bank":              "Tamil Nadu",
    "Lakshmi Vilas Bank":           "Tamil Nadu",
    "Tamilnad Mercantile Bank":     "Tamil Nadu",
    "Karur Vysya Bank":             "Tamil Nadu",
    "Equitas Small Finance Bank":   "Tamil Nadu",

    # ── Karnataka ───────────────────────────────────────────────────────────
    "Canara Bank":                  "Karnataka",
    "Syndicate Bank":               "Karnataka",
    "Vijaya Bank":                  "Karnataka",
    "Corporation Bank":             "Karnataka",
    "Karnataka Bank":               "Karnataka",
    "Shivalik Small Finance Bank":  "Karnataka",

    # ── Kerala ──────────────────────────────────────────────────────────────
    "South Indian Bank":            "Kerala",
    "Federal Bank":                 "Kerala",
    "Catholic Syrian Bank":         "Kerala",
    "Dhanlaxmi Bank":               "Kerala",
    "Kerala Gramin Bank":           "Kerala",

    # ── Gujarat ─────────────────────────────────────────────────────────────
    "Bank of Baroda":               "Gujarat",
    "Dena Bank":                    "Gujarat",
    "Saurashtra Gramin Bank":       "Gujarat",
    "Gujarat State Co-op Bank":     "Gujarat",

    # ── Andhra Pradesh ──────────────────────────────────────────────────────
    "Andhra Bank":                  "Andhra Pradesh",
    "Andhra Pradesh Grameena Vikas Bank": "Andhra Pradesh",
    "Saptagiri Grameena Bank":      "Andhra Pradesh",

    # ── Telangana ───────────────────────────────────────────────────────────
    "Telangana Grameena Bank":      "Telangana",
    "Deccan Grameena Bank":         "Telangana",

    # ── Punjab ──────────────────────────────────────────────────────────────
    "Punjab & Sind Bank":           "Punjab",
    "Punjab Gramin Bank":           "Punjab",
    "Sutlej Gramin Bank":           "Punjab",

    # ── Rajasthan ───────────────────────────────────────────────────────────
    "Bank of Rajasthan":            "Rajasthan",
    "Baroda Rajasthan Kshetriya Gramin Bank": "Rajasthan",
    "Rajasthan Marudhara Gramin Bank": "Rajasthan",

    # ── Uttar Pradesh ───────────────────────────────────────────────────────
    "Baroda UP Bank":               "Uttar Pradesh",
    "Prathama UP Gramin Bank":      "Uttar Pradesh",
    "Aryavart Bank":                "Uttar Pradesh",

    # ── Madhya Pradesh ──────────────────────────────────────────────────────
    "Central Madhya Pradesh Gramin Bank": "Madhya Pradesh",
    "Madhyanchal Gramin Bank":      "Madhya Pradesh",

    # ── Odisha ──────────────────────────────────────────────────────────────
    "Odisha Gramya Bank":           "Odisha",
    "Utkal Grameen Bank":           "Odisha",

    # ── Assam / North-East ──────────────────────────────────────────────────
    "Assam Gramin Vikash Bank":     "Assam",
    "Nagaland Rural Bank":          "Assam",
    "Meghalaya Rural Bank":         "Assam",

    # ── Jharkhand ───────────────────────────────────────────────────────────
    "Jharkhand Rajya Gramin Bank":  "Jharkhand",

    # ── Bihar ───────────────────────────────────────────────────────────────
    "Dakshin Bihar Gramin Bank":    "Bihar",
    "Uttar Bihar Gramin Bank":      "Bihar",

    # ── Himachal Pradesh ────────────────────────────────────────────────────
    "Himachal Pradesh Gramin Bank": "Himachal Pradesh",

    # ── Jammu & Kashmir ─────────────────────────────────────────────────────
    "J&K Bank":                     "Jammu & Kashmir",
    "J&K Grameen Bank":             "Jammu & Kashmir",

    # ── Haryana ─────────────────────────────────────────────────────────────
    "Sarva Haryana Gramin Bank":    "Haryana",

    # ── Chhattisgarh ────────────────────────────────────────────────────────
    "Chhattisgarh Rajya Gramin Bank": "Chhattisgarh",
}

# ---------------------------------------------------------------------------
# Hover tooltip blurbs — 2-3 sentences shown on region hover.
# Keyed by bank name; plain English, no jargon.
# ---------------------------------------------------------------------------
BANK_HOVER_BLURB: dict[str, str] = {

    # Maharashtra
    "Yes Bank": (
        "Yes Bank collapsed in March 2020 after years of concealing bad loans "
        "through accounting tricks. The RBI forced an emergency rescue involving "
        "SBI and other banks. It has stabilised since, but remains under close watch."
    ),
    "IDBI Bank": (
        "IDBI carried one of the highest NPA ratios among large Indian banks through "
        "the late 2010s. LIC took a majority stake in 2019 to avert a deeper crisis. "
        "Privatisation talks are ongoing as of 2024."
    ),
    "Central Bank of India": (
        "Central Bank spent five years under RBI's Prompt Corrective Action framework "
        "from 2017 to 2022, restricted from lending freely due to bad loans. "
        "It has exited PCA but stress indicators remain elevated."
    ),
    "Bank of India": (
        "Bank of India's gross NPA peaked above 16% during 2016–2019, "
        "making it one of the worst-hit public-sector lenders of that era. "
        "Government capital infusions helped stabilise it, but profitability recovery has been slow."
    ),
    "Bank of Maharashtra": (
        "Bank of Maharashtra is a mid-sized public-sector bank that narrowly avoided "
        "PCA restrictions through government support. It has shown steady improvement "
        "in asset quality since 2020 and returned to profit."
    ),
    "RBL Bank": (
        "RBL Bank, a fast-growing private lender, ran into trouble in 2022 when its "
        "CEO resigned abruptly and the RBI stepped in to appoint a new one. "
        "Asset quality stress in its microfinance and credit card book remains a concern."
    ),
    "Union Bank of India": (
        "Union Bank absorbed Andhra Bank and Corporation Bank in 2020, making it one of "
        "India's largest public-sector lenders by asset size. Post-merger integration "
        "pressures weighed on its metrics through 2022."
    ),

    # Delhi
    "Punjab National Bank": (
        "PNB was at the centre of India's biggest banking fraud in 2018, "
        "when the Nirav Modi scam exposed ₹14,000 crore in fraudulent guarantees. "
        "Subsequent mergers and recapitalisation have partially restored its health."
    ),
    "Oriental Bank of Commerce": (
        "OBC was merged into Punjab National Bank in April 2020 as part of India's "
        "bank consolidation programme. Its NPA ratios were elevated during 2017–2019, "
        "which weakened the combined entity's starting position."
    ),

    # West Bengal
    "UCO Bank": (
        "UCO Bank was under RBI's PCA framework from 2017 to 2019 due to high NPAs "
        "and negative return on assets. It exited PCA in 2019 but remains "
        "among the weaker public-sector banks in asset quality terms."
    ),
    "United Bank of India": (
        "United Bank struggled with NPAs above 15% and thin capital buffers for years "
        "before being merged into Punjab National Bank in 2020. "
        "Its weak state was a key driver behind the government's consolidation push."
    ),
    "Bandhan Bank": (
        "Bandhan Bank grew rapidly from a microfinance institution into a full-service bank "
        "but its heavy reliance on micro-loans creates vulnerability during downturns. "
        "Its NPA ratio spiked sharply after the COVID-19 disruption of 2020–21."
    ),
    "Allahabad Bank": (
        "Allahabad Bank, one of India's oldest banks, was merged into Indian Bank in 2020. "
        "It had been under PCA from 2018 with a high NPA ratio and dwindling capital. "
        "The merger was designed to create a stronger combined South-based entity."
    ),

    # Tamil Nadu
    "Lakshmi Vilas Bank": (
        "Lakshmi Vilas Bank was placed under a moratorium in November 2020 "
        "after years of rising NPAs, failed merger attempts, and governance failures. "
        "DBS Bank India rescued it through an RBI-orchestrated acquisition."
    ),
    "Indian Overseas Bank": (
        "Indian Overseas Bank spent six years under PCA — from 2015 to 2021 — "
        "the longest stint of any Indian bank in the corrective framework. "
        "Its NPA ratio peaked near 23% in 2018 before a gradual turnaround began."
    ),
    "Karur Vysya Bank": (
        "Karur Vysya Bank is a mid-sized private lender with a strong South India franchise. "
        "It faced NPA stress between 2018 and 2021 driven by MSME and jewellery loans, "
        "but has recovered steadily since with improving profitability."
    ),
    "Indian Bank": (
        "Indian Bank absorbed Allahabad Bank in 2020 to become a much larger lender. "
        "It has historically maintained relatively conservative underwriting standards "
        "and has shown consistent improvement in key health metrics since the merger."
    ),

    # Karnataka
    "Canara Bank": (
        "Canara Bank absorbed Syndicate Bank in the 2020 merger wave, becoming "
        "India's fourth-largest public-sector bank. NPA levels were high between "
        "2016 and 2019, but recapitalisation and merger synergies have improved its standing."
    ),
    "Vijaya Bank": (
        "Vijaya Bank was merged into Bank of Baroda in April 2019 in India's first "
        "three-way bank merger. It was broadly healthy at the time of the merger, "
        "absorbed mainly to strengthen the combined entity's reach in South India."
    ),
    "Corporation Bank": (
        "Corporation Bank was merged into Union Bank of India in April 2020. "
        "Like many public-sector peers, it had elevated NPAs during 2017–2019 "
        "and required capital support from the government before the merger."
    ),
    "Karnataka Bank": (
        "Karnataka Bank is an independent private-sector bank serving South India. "
        "It has maintained moderate profitability but faces pressure from competition "
        "by larger private banks encroaching on its home market."
    ),

    # Kerala
    "Dhanlaxmi Bank": (
        "Dhanlaxmi Bank has faced repeated governance crises and board-level disputes "
        "since 2010, making it one of India's most troubled small private banks. "
        "Its capital adequacy and profitability have remained persistently weak."
    ),
    "Federal Bank": (
        "Federal Bank is one of Kerala's strongest private lenders, "
        "with consistent profitability and improving asset quality since 2020. "
        "It is expanding aggressively in retail and NRI banking across India."
    ),
    "South Indian Bank": (
        "South Indian Bank went through significant NPA stress between 2019 and 2022, "
        "partly from a troubled corporate loan book. A turnaround plan is underway "
        "with focus on retail lending and digital banking."
    ),
    "Catholic Syrian Bank": (
        "Catholic Syrian Bank is a small Kerala-based private lender with a loyal "
        "NRI customer base. It has faced periodic capital adequacy concerns "
        "but remains community-oriented and locally influential."
    ),

    # Gujarat
    "Bank of Baroda": (
        "Bank of Baroda merged with Vijaya Bank and Dena Bank in 2019, "
        "creating India's third-largest public-sector bank by assets. "
        "Integration challenges initially weighed on metrics, but the recovery since has been solid."
    ),
    "Dena Bank": (
        "Dena Bank was placed under PCA in 2017 due to high NPAs and persistent losses, "
        "and was subsequently folded into Bank of Baroda in 2019. "
        "At the time of merger it was one of the weakest public-sector lenders."
    ),

    # Andhra Pradesh
    "Andhra Bank": (
        "Andhra Bank was merged into Union Bank of India in April 2020. "
        "It carried a stressed loan book with NPA ratios above 15% "
        "in the years leading up to the merger."
    ),

    # Punjab
    "Punjab & Sind Bank": (
        "Punjab & Sind Bank is a small government-owned lender primarily serving "
        "Punjab and the Sikh community. It has weak profitability metrics "
        "but benefits from implicit sovereign support."
    ),

    # Jammu & Kashmir
    "J&K Bank": (
        "J&K Bank is the dominant lender in Jammu & Kashmir, controlled by the "
        "state government. Political uncertainty and regional economic disruptions "
        "post-2019 pushed its NPA levels significantly higher."
    ),

    # Rajasthan
    "Bank of Rajasthan": (
        "Bank of Rajasthan was a private lender that was absorbed by ICICI Bank "
        "in 2010 after RBI raised concerns about governance and related-party lending. "
        "It was one of the earlier examples of a private bank rescue in India."
    ),

    # Odisha
    "Odisha Gramya Bank": (
        "Odisha Gramya Bank is a regional rural bank serving one of India's "
        "less financially included states. It focuses on agricultural lending "
        "and faces seasonal NPA volatility tied to monsoon cycles."
    ),

    # Bihar
    "Dakshin Bihar Gramin Bank": (
        "Dakshin Bihar Gramin Bank serves the southern districts of Bihar, "
        "one of India's most underbanked regions. Its loan book is dominated "
        "by small agricultural and priority-sector borrowers."
    ),

    # Assam
    "Assam Gramin Vikash Bank": (
        "Assam Gramin Vikash Bank is the primary rural lender in Assam, "
        "covering a geographically vast and flood-prone region. "
        "Agricultural loan defaults spike sharply in years with poor monsoons or floods."
    ),

    # Fallback
    "__default__": (
        "This bank contributes to credit flow in its region. "
        "Check the Bank Detail page for its latest FinGuard score, SHAP risk drivers, "
        "and ARIMA forecast for the next two quarters."
    ),
}


# ---------------------------------------------------------------------------
# Risk helpers
# ---------------------------------------------------------------------------
RISK_ORDER = {"Critical": 4, "At-Risk": 3, "Caution": 2, "Safe": 1}
RISK_COLOR  = {
    "Critical": "#ef4444",
    "At-Risk":  "#f97316",
    "Caution":  "#eab308",
    "Safe":     "#22c55e",
}

FEATURE_COLS = [
    "npa_ratio", "car", "roa",
    "liquidity_coverage", "debt_to_equity", "cost_to_income",
]


# ---------------------------------------------------------------------------
# Score loaders
# ---------------------------------------------------------------------------
def _get_all_bank_scores() -> pd.DataFrame:
    try:
        from scorer import compute_finguard_score
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="scorer module not available — ensure scorer.py is on PYTHONPATH",
        )
    try:
        df = pd.read_csv("finguard_quarterly_interpolated.csv")
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="finguard_quarterly_interpolated.csv not found",
        )

    latest = (
        df.sort_values(["bank_name", "year", "quarter"])
          .groupby("bank_name")
          .last()
          .reset_index()
    )

    records = []
    for _, row in latest.iterrows():
        bank = row["bank_name"]
        ratios = {col: row[col] for col in FEATURE_COLS if col in row}
        try:
            result = compute_finguard_score(ratios)
            records.append({
                "bank_name":          bank,
                "score":              round(float(result["score"]), 1),
                "risk_label":         result["risk_label"],
                "crisis_probability": round(float(result["crisis_probability"]), 4),
                "region":             BANK_REGION_MAP.get(bank, "Other"),
            })
        except Exception:
            continue

    return pd.DataFrame(records)


def _score_at_quarter(year: int, quarter: int) -> pd.DataFrame:
    try:
        from scorer import compute_finguard_score
    except ImportError:
        raise HTTPException(status_code=503, detail="scorer module unavailable")
    try:
        raw = pd.read_csv("finguard_quarterly_interpolated.csv")
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="finguard_quarterly_interpolated.csv not found",
        )

    raw = raw[(raw["year"] == year) & (raw["quarter"] == quarter)]
    if raw.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for {year}-Q{quarter}",
        )

    records = []
    for _, row in raw.iterrows():
        bank = row["bank_name"]
        ratios = {col: row[col] for col in FEATURE_COLS if col in row}
        try:
            result = compute_finguard_score(ratios)
            records.append({
                "bank_name":          bank,
                "score":              round(float(result["score"]), 1),
                "risk_label":         result["risk_label"],
                "crisis_probability": round(float(result["crisis_probability"]), 4),
                "region":             BANK_REGION_MAP.get(bank, "Other"),
            })
        except Exception:
            continue

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Hover tooltip builder
# ---------------------------------------------------------------------------
def _build_hover_tooltip(
    worst_bank: str,
    worst_label: str,
    worst_score: float,
    avg_score: float,
    bank_count: int,
) -> dict:
    blurb = BANK_HOVER_BLURB.get(worst_bank, BANK_HOVER_BLURB["__default__"])
    summary = (
        f"{bank_count} bank{'s' if bank_count != 1 else ''} tracked in this region. "
        f"Regional average FinGuard score: {avg_score}. "
        f"Highest-risk bank is {worst_bank} ({worst_label}, score {worst_score})."
    )
    return {"blurb": blurb, "summary": summary}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _aggregate_by_region(scores_df: pd.DataFrame) -> list[dict]:
    results = []

    for region, group in scores_df.groupby("region"):
        avg_score        = round(group["score"].mean(), 1)
        avg_crisis_prob  = round(group["crisis_probability"].mean(), 4)
        bank_count       = len(group)

        group = group.copy()
        group["_risk_order"] = group["risk_label"].map(RISK_ORDER).fillna(0)
        worst_row = group.sort_values(
            ["_risk_order", "score"], ascending=[False, True]
        ).iloc[0]

        worst_bank        = worst_row["bank_name"]
        worst_score       = round(float(worst_row["score"]), 1)
        worst_label       = worst_row["risk_label"]
        worst_crisis_prob = round(float(worst_row["crisis_probability"]), 4)

        region_label = worst_label
        region_color = RISK_COLOR.get(region_label, "#6b7280")

        label_counts = group["risk_label"].value_counts().to_dict()

        hover = _build_hover_tooltip(
            worst_bank, worst_label, worst_score, avg_score, bank_count
        )

        results.append({
            "region":             region,
            "bank_count":         bank_count,
            "avg_score":          avg_score,
            "avg_crisis_prob":    avg_crisis_prob,
            "worst_bank":         worst_bank,
            "worst_score":        worst_score,
            "worst_risk_label":   worst_label,
            "worst_crisis_prob":  worst_crisis_prob,
            "region_risk_label":  region_label,
            "region_color":       region_color,
            "label_counts": {
                "Safe":     label_counts.get("Safe", 0),
                "Caution":  label_counts.get("Caution", 0),
                "At-Risk":  label_counts.get("At-Risk", 0),
                "Critical": label_counts.get("Critical", 0),
            },
            "hover": hover,
        })

    results.sort(
        key=lambda r: (
            -RISK_ORDER.get(r["region_risk_label"], 0),
            r["avg_score"],
        )
    )
    return results


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/map/regions", summary="All regions — avg + worst-case scores + hover")
def get_all_regions(as_of_quarter: Optional[str] = None):
    """
    Returns health data for every tracked region.

    **Query param** `as_of_quarter` (optional, format `YYYY-QN`):
    Scores each bank using data from that specific quarter.
    Useful for the crisis timeline scrubber on the frontend.

    **Sample response item:**
    ```json
    {
      "region": "Tamil Nadu",
      "bank_count": 7,
      "avg_score": 42.3,
      "avg_crisis_prob": 0.31,
      "worst_bank": "Lakshmi Vilas Bank",
      "worst_score": 18.7,
      "worst_risk_label": "Critical",
      "worst_crisis_prob": 0.81,
      "region_risk_label": "Critical",
      "region_color": "#ef4444",
      "label_counts": {"Safe": 1, "Caution": 3, "At-Risk": 2, "Critical": 1},
      "hover": {
        "blurb": "Lakshmi Vilas Bank was placed under a moratorium in November 2020...",
        "summary": "7 banks tracked. Avg score: 42.3. Highest-risk: Lakshmi Vilas Bank (Critical, 18.7)."
      }
    }
    ```
    """
    if as_of_quarter:
        try:
            year_str, q_str = as_of_quarter.split("-Q")
            scores_df = _score_at_quarter(int(year_str), int(q_str))
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail="as_of_quarter must be in format YYYY-QN, e.g. 2024-Q3",
            )
    else:
        scores_df = _get_all_bank_scores()

    regions = _aggregate_by_region(scores_df)
    return {"as_of": as_of_quarter or "latest", "regions": regions}


@router.get(
    "/map/region/{region_name}",
    summary="Single region — full bank breakdown + hover blurbs",
)
def get_region_detail(region_name: str):
    """
    Returns aggregate + full per-bank list for one region, sorted worst → best.

    Each bank row includes its own plain-English `blurb` (used for a deeper
    tooltip or a sidebar card on the Bank Detail page).

    **Sample response:**
    ```json
    {
      "region": "Maharashtra",
      "bank_count": 13,
      "avg_score": 55.2,
      "worst_bank": "Yes Bank",
      "worst_risk_label": "At-Risk",
      "region_color": "#f97316",
      "hover": {
        "blurb": "Yes Bank collapsed in March 2020...",
        "summary": "13 banks tracked. Avg score: 55.2. Highest-risk: Yes Bank (At-Risk, 21.4)."
      },
      "banks": [
        {
          "bank_name": "Yes Bank",
          "score": 21.4,
          "risk_label": "At-Risk",
          "crisis_probability": 0.62,
          "is_worst": true,
          "blurb": "Yes Bank collapsed in March 2020 after years of concealing bad loans..."
        }
      ]
    }
    ```
    """
    scores_df = _get_all_bank_scores()

    region_df = scores_df[
        scores_df["region"].str.lower() == region_name.strip().lower()
    ]

    if region_df.empty:
        valid = sorted(scores_df["region"].unique().tolist())
        raise HTTPException(
            status_code=404,
            detail=f"Region '{region_name}' not found. Valid regions: {valid}",
        )

    aggregated = _aggregate_by_region(region_df)
    agg = aggregated[0]

    region_df = region_df.copy()
    region_df["_risk_order"] = region_df["risk_label"].map(RISK_ORDER).fillna(0)
    region_df = region_df.sort_values(
        ["_risk_order", "score"], ascending=[False, True]
    )

    worst_bank = agg["worst_bank"]
    banks = [
        {
            "bank_name":          row["bank_name"],
            "score":              row["score"],
            "risk_label":         row["risk_label"],
            "crisis_probability": row["crisis_probability"],
            "is_worst":           row["bank_name"] == worst_bank,
            "blurb":              BANK_HOVER_BLURB.get(
                                      row["bank_name"],
                                      BANK_HOVER_BLURB["__default__"]
                                  ),
        }
        for _, row in region_df.iterrows()
    ]

    return {
        "region":            agg["region"],
        "bank_count":        agg["bank_count"],
        "avg_score":         agg["avg_score"],
        "avg_crisis_prob":   agg["avg_crisis_prob"],
        "worst_bank":        agg["worst_bank"],
        "worst_score":       agg["worst_score"],
        "worst_risk_label":  agg["worst_risk_label"],
        "worst_crisis_prob": agg["worst_crisis_prob"],
        "region_risk_label": agg["region_risk_label"],
        "region_color":      agg["region_color"],
        "label_counts":      agg["label_counts"],
        "hover":             agg["hover"],
        "banks":             banks,
    }
# ---------------------------------------------------------------------------
# Dropdown menu endpoints
# ---------------------------------------------------------------------------

@router.get("/map/dropdowns", summary="All regions and their banks — for dropdown menus")
def get_dropdown_options():
    """
    Returns every region and the banks inside it.
    Use this to populate both dropdown menus on the Health Map page.
    The frontend should call this once on mount and cache the result.

    **Sample response:**
```json
    {
      "regions": ["Assam", "Bihar", "Delhi", ...],
      "banks_by_region": {
        "Maharashtra": ["Axis Bank", "Bank of India", ...],
        "Tamil Nadu":  ["City Union Bank", "Indian Bank", ...]
      },
      "all_banks": ["Allahabad Bank", "Andhra Bank", ...]
    }
```
    """
    banks_by_region: dict[str, list[str]] = {}
    for bank, region in BANK_REGION_MAP.items():
        banks_by_region.setdefault(region, []).append(bank)

    # sort everything so the dropdowns are alphabetical
    for region in banks_by_region:
        banks_by_region[region].sort()

    return {
        "regions":         sorted(banks_by_region.keys()),
        "banks_by_region": banks_by_region,
        "all_banks":       sorted(BANK_REGION_MAP.keys()),
    }


@router.get("/map/bank/{bank_name}", summary="Single bank — score + blurb + region")
def get_bank_detail(bank_name: str):
    """
    Returns the FinGuard score, risk label, region, and hover blurb for one bank.
    Called when the user selects a bank from the second dropdown.

    Path param is case-insensitive and trims whitespace.

    **Sample response:**
```json
    {
      "bank_name": "Yes Bank",
      "region": "Maharashtra",
      "score": 21.4,
      "risk_label": "At-Risk",
      "crisis_probability": 0.62,
      "risk_color": "#f97316",
      "blurb": "Yes Bank collapsed in March 2020 after years of concealing bad loans..."
    }
```
    """
    # case-insensitive lookup against BANK_REGION_MAP keys
    name_lower = bank_name.strip().lower()
    matched = next(
        (b for b in BANK_REGION_MAP if b.lower() == name_lower), None
    )
    if matched is None:
        raise HTTPException(
            status_code=404,
            detail=f"Bank '{bank_name}' not found. Use GET /map/dropdowns to see valid names.",
        )

    scores_df = _get_all_bank_scores()
    row = scores_df[scores_df["bank_name"] == matched]

    if row.empty:
        # bank is in the map but not in the scored data (no CSV rows)
        raise HTTPException(
            status_code=404,
            detail=f"No score data found for '{matched}' in the dataset.",
        )

    row = row.iloc[0]
    risk_label = row["risk_label"]

    return {
        "bank_name":          matched,
        "region":             BANK_REGION_MAP[matched],
        "score":              row["score"],
        "risk_label":         risk_label,
        "crisis_probability": row["crisis_probability"],
        "risk_color":         RISK_COLOR.get(risk_label, "#6b7280"),
        "blurb":              BANK_HOVER_BLURB.get(matched, BANK_HOVER_BLURB["__default__"]),
    }