"""
PAN Group — EDGAR Pre-Filter
BA298A Capstone | UC Irvine | 2026

Checks each NYSE company's filing history via EDGAR submissions API
WITHOUT downloading any actual filings.

For each company:
  - Hits https://data.sec.gov/submissions/CIK{cik}.json
  - Counts 10-K and DEF 14A filings in the 2015-2025 window
  - Flags as eligible if it meets minimum criteria

Takes ~2 seconds per company = ~2 hours for 3,276 companies.
Compare to download_and_extract.py which takes ~60 seconds per company.

Output:
  data/tickers/nyse_tickers_filtered.csv  — eligible companies only
  data/tickers/prefilter_report.csv       — full results for all companies

Eligibility criteria:
  1. Has at least 6 10-K filings in the 2015-2025 window
     (6 not 10 — some companies have fiscal years ending mid-year,
      so 10 calendar years may only show 6-9 in EDGAR)
  2. Has at least one DEF 14A in the 2015-2025 window
  3. Files 10-K (not 20-F — foreign private issuers are excluded)

Usage:
  python prefilter.py           # all tickers in nyse_tickers.csv
  python prefilter.py --pilot   # NKE, IBM, GHC only
"""

import argparse
import time
import sys
import logging
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd

from config import (
    SEC_USER_AGENT,
    TICKERS_CSV,
    TICKERS_DIR,
    LOGS_DIR,
    EDGAR_DELAY_SECONDS,
    EDGAR_RETRY_ATTEMPTS,
    EDGAR_RETRY_DELAY,
    WINDOW_START,
    WINDOW_END,
    PILOT_TICKERS,
)

# ── Output paths ──────────────────────────────────────────────────────────────
PREFILTER_REPORT   = TICKERS_DIR / "prefilter_report.csv"
FILTERED_TICKERS   = TICKERS_DIR / "nyse_tickers_filtered.csv"
PREFILTER_LOG      = LOGS_DIR / "prefilter_log.txt"

# Minimum 10-K filings in window to be eligible
MIN_10K_IN_WINDOW  = 10

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PREFILTER_LOG, mode="a"),
    ],
)
log = logging.getLogger(__name__)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"


def get_submissions(cik_str: str, session: requests.Session) -> dict | None:
    """
    Fetches the EDGAR submissions JSON for a company.
    This is a lightweight metadata call — no filing content is downloaded.

    The submissions JSON contains a list of all filings with their types
    and dates. We only need the filingDate and form fields.
    """
    url = SUBMISSIONS_URL.format(cik=cik_str)

    for attempt in range(1, EDGAR_RETRY_ATTEMPTS + 1):
        try:
            r = session.get(url, timeout=15)
            if r.status_code == 404:
                return None  # Company not found in EDGAR — skip
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            if attempt == EDGAR_RETRY_ATTEMPTS:
                log.warning(f"  Failed after {EDGAR_RETRY_ATTEMPTS} attempts: {e}")
                return None
            time.sleep(EDGAR_RETRY_DELAY)

    return None


def count_filings_in_window(submissions: dict) -> dict:
    """
    Counts 10-K and DEF 14A filings within the measurement window.

    The submissions JSON has a 'filings' key with 'recent' sub-key
    containing parallel arrays: forms[], filingDates[], etc.
    Older filings (beyond ~40 most recent) are in separate paginated files
    but for our window (2015-2025) recent filings are sufficient for
    most active companies.
    """
    result = {
        "tenk_in_window": 0,
        "def14a_in_window": 0,
        "tenk_years": [],
        "earliest_tenk": None,
        "latest_tenk": None,
        "files_20f": False,  # foreign private issuer flag
    }

    try:
        recent = submissions.get("filings", {}).get("recent", {})
        forms  = recent.get("form", [])
        dates  = recent.get("filingDate", [])
    except (KeyError, AttributeError):
        return result

    for form, date in zip(forms, dates):
        try:
            year = int(date[:4])
        except (ValueError, TypeError):
            continue

        if form == "20-F":
            result["files_20f"] = True

        if not (WINDOW_START <= year <= WINDOW_END):
            continue

        if form == "10-K":
            result["tenk_in_window"] += 1
            result["tenk_years"].append(year)
        elif form in ("DEF 14A", "DEFA14A"):
            result["def14a_in_window"] += 1

    if result["tenk_years"]:
        result["earliest_tenk"] = min(result["tenk_years"])
        result["latest_tenk"]   = max(result["tenk_years"])

    return result


def check_ticker(ticker: str, cik_str: str, session: requests.Session) -> dict:
    """
    Checks a single ticker's eligibility without downloading filings.
    """
    row = {
        "ticker":           ticker,
        "cik":              cik_str,
        "tenk_in_window":   0,
        "def14a_in_window": 0,
        "earliest_tenk":    None,
        "latest_tenk":      None,
        "files_20f":        False,
        "eligible":         False,
        "reason":           "",
    }

    submissions = get_submissions(cik_str, session)

    if submissions is None:
        row["reason"] = "edgar_not_found"
        return row

    counts = count_filings_in_window(submissions)
    row.update({
        "tenk_in_window":   counts["tenk_in_window"],
        "def14a_in_window": counts["def14a_in_window"],
        "earliest_tenk":    counts["earliest_tenk"],
        "latest_tenk":      counts["latest_tenk"],
        "files_20f":        counts["files_20f"],
    })

    # Eligibility checks
    if counts["files_20f"]:
        row["reason"] = "foreign_issuer_20F"
    elif counts["tenk_in_window"] < MIN_10K_IN_WINDOW:
        row["reason"] = f"only_{counts['tenk_in_window']}_10Ks_in_window"
    elif counts["def14a_in_window"] == 0:
        row["reason"] = "no_DEF14A_in_window"
    else:
        row["eligible"] = True
        row["reason"]   = "eligible"

    return row


def load_tickers_with_cik() -> list[tuple[str, str]]:
    """
    Loads tickers and their CIK numbers from nyse_tickers.csv.
    CIK is needed to construct the EDGAR submissions URL.
    """
    if not TICKERS_CSV.exists():
        print(f"ERROR: {TICKERS_CSV} not found. Run get_tickers.py first.")
        sys.exit(1)

    df = pd.read_csv(TICKERS_CSV)
    df["cik_str"] = df["cik"].astype(str).str.zfill(10)
    return list(zip(df["ticker"].str.upper(), df["cik_str"]))


def get_completed_tickers() -> set:
    """Resume support — skip tickers already in the report."""
    if not PREFILTER_REPORT.exists():
        return set()
    try:
        df = pd.read_csv(PREFILTER_REPORT, usecols=["ticker"])
        return set(df["ticker"].unique())
    except Exception:
        return set()


def save_row(row: dict) -> None:
    df = pd.DataFrame([row])
    write_header = not PREFILTER_REPORT.exists()
    df.to_csv(PREFILTER_REPORT, mode="a", header=write_header, index=False)


def print_summary(df: pd.DataFrame) -> None:
    total    = len(df)
    eligible = df["eligible"].sum()
    foreign  = (df["reason"] == "foreign_issuer_20F").sum()
    too_few  = df["reason"].str.startswith("only_").sum()
    no_proxy = (df["reason"] == "no_DEF14A_in_window").sum()
    not_found = (df["reason"] == "edgar_not_found").sum()

    print(f"\n{'='*60}")
    print(f"PAN PRE-FILTER RESULTS")
    print(f"{'='*60}")
    print(f"Total tickers checked:     {total:>6,}")
    print(f"ELIGIBLE for download:     {eligible:>6,}  ({eligible/total*100:.1f}%)")
    print(f"Ineligible — foreign 20-F: {foreign:>6,}")
    print(f"Ineligible — <{MIN_10K_IN_WINDOW} 10-Ks:      {too_few:>6,}")
    print(f"Ineligible — no DEF 14A:   {no_proxy:>6,}")
    print(f"Ineligible — not in EDGAR: {not_found:>6,}")
    print(f"{'='*60}")
    print(f"Eligible tickers saved to: {FILTERED_TICKERS}")
    print(f"\nEstimated download time: ~{eligible} minutes ({eligible/60:.1f} hours)")
    print(f"{'='*60}\n")


def run(ticker_cik_pairs: list[tuple[str, str]]) -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": SEC_USER_AGENT})

    completed  = get_completed_tickers()
    remaining  = [(t, c) for t, c in ticker_cik_pairs if t not in completed]

    log.info(f"{'='*60}")
    log.info(f"PAN Pre-Filter — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info(f"Total tickers:   {len(ticker_cik_pairs):>6,}")
    log.info(f"Already checked: {len(completed):>6,}")
    log.info(f"To check:        {len(remaining):>6,}")
    log.info(f"Window:          {WINDOW_START}–{WINDOW_END}")
    log.info(f"Min 10-Ks:       {MIN_10K_IN_WINDOW}")
    log.info(f"{'='*60}\n")

    est_minutes = len(remaining) * EDGAR_DELAY_SECONDS / 60
    log.info(f"Estimated time: ~{est_minutes:.0f} minutes")

    for i, (ticker, cik_str) in enumerate(remaining, 1):
        if i % 100 == 0 or i <= 5:
            log.info(f"[{i:>4}/{len(remaining)}]  {ticker}")

        row = check_ticker(ticker, cik_str, session)
        save_row(row)
        time.sleep(EDGAR_DELAY_SECONDS)

    # Load full report and save eligible subset
    df = pd.read_csv(PREFILTER_REPORT)
    eligible_df = df[df["eligible"]][["ticker", "cik", "tenk_in_window", "def14a_in_window", "earliest_tenk", "latest_tenk"]]
    eligible_df.to_csv(FILTERED_TICKERS, index=False)

    print_summary(df)


def parse_args():
    parser = argparse.ArgumentParser(description="PAN EDGAR Pre-Filter")
    parser.add_argument("--pilot", action="store_true", help="Check pilot tickers only")
    return parser.parse_args()


def main():
    args = parse_args()

    all_pairs = load_tickers_with_cik()

    if args.pilot:
        pilot_set = set(PILOT_TICKERS)
        pairs = [(t, c) for t, c in all_pairs if t in pilot_set]
        log.info(f"Mode: PILOT — {PILOT_TICKERS}")
    else:
        pairs = all_pairs
        log.info(f"Mode: FULL — {len(pairs):,} tickers")

    run(pairs)


if __name__ == "__main__":
    main()
