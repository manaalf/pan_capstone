"""
PAN Group — NYSE Ticker Fetcher
BA298A Capstone | UC Irvine | 2026

Pulls the full NYSE company list from SEC EDGAR's public company_tickers_exchange.json.
No API key required — this is a public SEC endpoint.

Output: data/tickers/nyse_tickers.csv
Columns: cik, ticker, name, exchange

Run: python get_tickers.py
"""

import requests
import pandas as pd
import time
import sys
from pathlib import Path
from config import (
    SEC_USER_AGENT,
    TICKERS_CSV,
    LOGS_DIR,
)

EXCHANGE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"


def fetch_nyse_tickers() -> pd.DataFrame:
    """
    Fetches all NYSE-listed companies from SEC EDGAR.
    Returns a DataFrame with columns: cik, ticker, name, exchange.

    Why this source instead of Mergent:
    - Mergent requires a university library export which isn't automatable.
    - The SEC EDGAR exchange file is updated daily, free, and is the definitive
      source for CIK numbers (which we need for EDGAR downloads anyway).
    - The tradeoff: no IPO date or founding date, so we can't pre-filter by age
      here — that filter happens in audit_downloads.py after checking filing history.
    """
    print(f"Fetching company list from SEC EDGAR...")
    print(f"URL: {EXCHANGE_URL}\n")

    headers = {"User-Agent": SEC_USER_AGENT}

    try:
        response = requests.get(EXCHANGE_URL, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"ERROR: HTTP {response.status_code} — {e}")
        print("If you get a 403, check your User-Agent string in config.py.")
        print("SEC requires a real name, domain, and email — not a generic string.")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Network request failed — {e}")
        sys.exit(1)

    data = response.json()

    # The JSON structure is: {"fields": [...], "data": [[...], [...]]}
    df = pd.DataFrame(data["data"], columns=data["fields"])
    print(f"Total companies across all exchanges: {len(df):,}")
    print(f"\nExchange breakdown:")
    print(df["exchange"].value_counts().to_string())

    # Filter to NYSE only
    # Note: 'NYSE' is the main board. 'NYSEArca' is ETFs/funds — exclude those.
    # 'NASDAQ' is separate and not in scope for this project.
    nyse = df[df["exchange"] == "NYSE"].copy()
    nyse = nyse.reset_index(drop=True)

    print(f"\nNYSE (main board only): {len(nyse):,} companies")

    # Pad CIK to 10 digits — EDGAR uses zero-padded CIKs in URLs
    nyse["cik_str"] = nyse["cik"].astype(str).str.zfill(10)

    return nyse


def save_tickers(df: pd.DataFrame) -> None:
    df.to_csv(TICKERS_CSV, index=False)
    print(f"\nSaved to: {TICKERS_CSV}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head().to_string())


def main():
    df = fetch_nyse_tickers()
    save_tickers(df)

    print(f"\n{'='*60}")
    print(f"NEXT STEP: Run downloader.py")
    print(f"  python downloader.py --pilot   # test on Nike, IBM, GHC first")
    print(f"  python downloader.py --full    # all {len(df):,} companies (overnight)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
