# PAN Group — Full Cohort Pipeline

BA298A Capstone | UC Irvine | 2026

## What This Is

Automated pipeline to score 2,300 NYSE companies on Purpose and Performance
frameworks, producing `master_dataset.csv`.

---

## Setup

```bash
# 1. Create virtual environment (do this once)
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Run Order — Phase 1: Data Download (No API Key Needed)

### Step 1 — Get NYSE tickers

```bash
python get_tickers.py
```

Output: `data/tickers/nyse_tickers.csv`

### Step 2 — Test downloader on pilot tickers first

```bash
python downloader.py --pilot
```

Output: `data/filings/NKE/`, `data/filings/IBM/`, `data/filings/GHC/`

Check it looks right before running full:

```
data/filings/
  NKE/
    10-K/
      0000320187-20-000004/   ← accession number folders
      0000320187-19-000011/
      ...
    DEF 14A/
      ...
  IBM/
    ...
```

### Step 3 — Full download (run overnight)

```bash
python downloader.py --full
```

Estimated time: 6–10 hours for 2,300 companies.
Safe to stop and restart — resumes from where it left off.

### Step 4 — Audit coverage

```bash
python audit_downloads.py
```

Output: `data/audit/coverage_report.csv` + `data/audit/coverage_summary.txt`

Key question this answers: **how many companies have 10+ years of filings
covering the 2015–2020 purpose window?**

---

## Run Order — Phase 2: Scoring (Needs API Key)

*(Build these modules next — not part of Phase 1)*

```bash
python pipeline.py --ticker NKE    # score single company
python run_batch.py                # score all eligible companies
```

---

## File Structure

```
pan_pipeline/
├── config.py              # all settings — edit this first
├── get_tickers.py         # Step 1: NYSE company list
├── downloader.py          # Step 2-3: EDGAR filings
├── audit_downloads.py     # Step 4: coverage analysis
├── requirements.txt
├── data/
│   ├── tickers/
│   │   ├── nyse_tickers.csv          # all NYSE companies
│   │   └── nyse_tickers_filtered.csv # eligible after audit
│   ├── filings/                      # downloaded 10-Ks + DEF 14As
│   └── audit/                        # coverage reports
└── logs/
    └── download_log.txt
```

---

## Key Config Settings (config.py)


| Setting               | Default         | Notes                                            |
| --------------------- | --------------- | ------------------------------------------------ |
| `SEC_USER_AGENT`      | PAN Group email | **Must be real** — SEC blocks generic strings    |
| `FILINGS_LIMIT`       | 10              | 10-Ks per company — covers 2014–2024 comfortably |
| `MIN_YEARS_REQUIRED`  | 10              | Minimum 10-K filings to be eligible              |
| `EDGAR_DELAY_SECONDS` | 0.15            | ~6 req/sec — safe under SEC's 10 req/sec limit   |


---

## Troubleshooting

**403 from SEC**: Your User-Agent string in `config.py` is invalid.
SEC requires: `"Real Name OrganisationDomain email@domain.com"`.
Generic strings like "Mozilla/5.0" get blocked.

**Download stalls**: Kill it (`Ctrl+C`) and restart — it resumes from checkpoint.

**Accession folders look wrong**: Run `audit_downloads.py --pilot` and check
the year extraction is working on your three pilot tickers.