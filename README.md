# PAN Capstone — Corporate Purpose & Financial Performance

> Quantifying the relationship between corporate purpose and long-term financial performance across NYSE-listed companies.
> BA298A Capstone Research Project · UC Irvine · 2026

---

## Overview

This repository contains the full data pipeline and analysis code for the PAN Group capstone research project. The project builds a quantitative framework to score corporate purpose and financial performance for NYSE-listed companies, then tests whether purpose-driven companies deliver superior long-term financial outcomes.

**Core hypothesis:** Companies that genuinely operationalise their stated purpose — evidenced by consistent language, structural embedding, and strategic integration — deliver superior long-term financial performance compared to peers that treat purpose as a communications exercise.

---

## Key Results

| Statistic | Value |
|---|---|
| Companies scored | 904 |
| Measurement window | 2015–2025 |
| Pearson r (purpose vs performance) | 0.1441 (p < 0.001) |
| OLS coefficient (sector-controlled) | 0.265 (p < 0.001) |
| Sectors with positive correlation | 18 of 20 |
| Strongest sector | Software & Services (r = 0.509) |

The aggregate correlation is statistically significant and strengthens after sector controls — sector membership was suppressing, not inflating, the relationship.

---

## Two-Framework Structure

### Purpose Score (0.0 – 1.0)
Measured at time T (2015–2020) across five dimensions:

| Section | Description | Max Points |
|---|---|---|
| S1 | Mission Clarity | 3.0 |
| S2 | Longitudinal Consistency | 3.0 |
| S3 | NLP Signals (Loughran-McDonald + PAN lexicon) | 3.5 |
| S4 | Structural Embedding | 2.5 |
| S5 | Stakeholder Integration | 2.0 |
| Penalties | Up to −4.5 pts for vagueness, greenwashing, rebranding | — |
| Bonuses | Up to +2.5 pts for accountability, adversity resilience | — |

**Sources:** SEC EDGAR (10-K, DEF 14A, 20-F for foreign filers), earnings call transcripts, Wayback Machine mission snapshots  
**Scoring method:** ANTHROPIC OPUS 4.7 LLM pipeline + Python NLP + manual validation

### Performance Score (0.0 – 1.0)
Measured at T+3 to T+5 (2020–2025). All metrics benchmarked relative to GICS sector peers.

| Section | Metric | Weight |
|---|---|---|
| S1 | Total Shareholder Return (10yr) | 35% |
| S2 | Return on Assets (10yr avg) | 25% |
| S3 | EBITDA Margin (10yr avg) | 20% |
| S4 | Revenue CAGR (10yr) | 15% |
| Penalties | Sector tailwinds, growth-performance mismatch, profitability distortion | Up to −1.5 pts |

**Sources:** S&P Capital IQ, yfinance  
**Scoring method:** Manual Excel scoring — peer-relative benchmarking across GICS sectors

### Measurement Lag Design
Purpose is scored at T. Performance is scored at T+3 to T+5. This separation makes the causal direction more defensible — purpose is measured before performance outcomes are observed.

---

## Quadrant Classification

Every company is assigned to one of four quadrants:

| Quadrant | Description |
|---|---|
| **Purpose Leaders** | High purpose, high performance — the thesis confirmed |
| **Value Extractors** | Low purpose, high performance — financial returns without purpose operationalisation |
| **Purpose Promise** | High purpose, low performance — purpose not yet translating to outcomes |
| **Dual Weakness** | Low purpose, low performance |

---

## Score Interpretation Bands

Both frameworks use the same 5-band scale:

| Band | Interpretation |
|---|---|
| 0.85–1.0 | Structural / Exceptional |
| 0.65–0.84 | Strong |
| 0.45–0.64 | Moderate / Inconsistent |
| 0.25–0.44 | Weak / Performative |
| 0.00–0.24 | Absent / Value Destruction |

---

## Repository Structure

```
pan_capstone/
│
├── scripts/
│   ├── data_collection/
│   │   ├── get_tickers.py           # Pull NYSE ticker universe
│   │   ├── downloader.py            # SEC EDGAR filing downloader
│   │   ├── download_and_extract.py  # Download + text extraction pipeline
│   │   └── prefilter.py             # Filter companies with insufficient filings
│   │
│   ├── purpose_scoring/
│   │   ├── nlp_scorer.py            # NLP scoring using Loughran-McDonald + PAN lexicon
│   │   ├── llm_scorer_v2.py         # LLM scoring pipeline (Anthropic Opus 4.7)
│   │   ├── penalty_scorer.py        # Penalty detection and scoring
│   │   ├── merge_penalties.py       # Merge penalty scores into master dataset
│   │   └── pan_purpose_lexicon_v3.py # PAN extended purpose lexicon
│   │
│   └── analysis/
│       ├── pan_analysis.py          # Correlation, OLS regression, sector analysis
│       ├── quadrant_final.py        # Quadrant classification and distribution
│       └── quadrant_chart.py        # Quadrant visualisation
│
├── data/
│   ├── reference/
│   │   └── Loughran-McDonald_MasterDictionary_1993-2025.csv  # LM financial sentiment lexicon
│   ├── tickers/
│   │   └── nyse_tickers_filtered.csv   # Final filtered NYSE universe
│   ├── scores/
│   │   └── purpose_scores.csv          # Raw purpose scores (1,446 companies pre-filter)
│   └── output/
│       ├── sector_averages.csv         # Sector-level aggregated scores
│       └── graphs/
│           ├── chart_quadrant_map.png
│           ├── chart_scatter_trend.png
│           ├── chart_sector_bottom.png
│           ├── chart_sector_correlations.png
│           ├── chart_sector_top.png
│           └── chart_size_metrics.png
│
└── datasets/
    └── PAN_Client_Dataset.xlsx     # Client-facing dataset (three tabs: Overall, Purpose, Performance)
```

---

## Data Sources

| Source | What We Pull | Used For |
|---|---|---|
| SEC EDGAR | 10-K, DEF 14A (domestic); 20-F (foreign filers) | Purpose scoring S1–S5 |
| S&P Capital IQ | Earnings call transcripts, financial metrics | NLP signals, performance scoring |
| Wayback Machine | Annual mission statement snapshots 2010–2024 | S2 longitudinal consistency |
| Yahoo Finance / yfinance | TSR, ROA, EBITDA margin, Revenue | Performance scoring |
| Mergent Intellect | NYSE company universe, master ticker list | Master company list |
| Nexis Uni | Rebranding / purpose-shift news events | Penalty validation |

**Note on foreign filers:** Companies listed on NYSE that file 20-F forms (rather than 10-K) are handled separately in the data collection pipeline. These include cross-listed foreign private issuers and are processed through the same purpose scoring framework with filing-type-appropriate extraction logic.

---

## Pipeline Order

```
get_tickers.py
    → prefilter.py
        → downloader.py / download_and_extract.py
            → nlp_scorer.py
                → llm_scorer_v2.py
                    → penalty_scorer.py
                        → merge_penalties.py
                            → pan_analysis.py / quadrant_final.py
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/manaalf/pan_capstone.git
cd pan_capstone

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Tech Stack

- **Python** — core pipeline
- **Anthropic Opus 4.7** — LLM scoring
- **sec-edgar-downloader** — SEC filing retrieval
- **pdfplumber** — PDF text extraction
- **yfinance** — financial data
- **pandas, scipy, statsmodels** — analysis
- **matplotlib** — visualisation
- **Tableau** — interactive dashboard (PAN_Dashboard.twb)

---

## Client Dataset

`datasets/PAN_Client_Dataset.xlsx` contains three tabs:

- **Overall** — final purpose score, performance score, band, and quadrant for all 904 companies
- **Purpose** — full purpose sub-score breakdown (S1–S5, all penalties and bonuses)
- **Performance** — full performance sub-score breakdown (S1–S4, raw metrics, penalty flags)

---

*PAN Group · paneffect.co · BA298A Capstone Research Project · UC Irvine · 2026*
