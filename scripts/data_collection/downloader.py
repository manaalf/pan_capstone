"""
PAN Group — EDGAR Filing Downloader
BA298A Capstone | UC Irvine | 2026

Downloads 10-K and DEF 14A filings for all tickers in nyse_tickers.csv.
Designed to run unattended overnight for the full 2,300-company cohort.

Key design decisions:
  - Checkpoint to disk after every company: crash-safe, resumable
  - Skip tickers already downloaded: never re-download
  - Rate limit at ~6 req/sec: well inside SEC's 10 req/sec limit
  - Retry on transient failures: 3 attempts with backoff
  - Separate log file: full audit trail without cluttering stdout

Usage:
  python downloader.py --pilot        # NKE, IBM, GHC only
  python downloader.py --full         # all tickers in nyse_tickers.csv
  python downloader.py --ticker AAPL  # single ticker

Output structure:
  data/filings/{TICKER}/10-K/{accession_number}/...
  data/filings/{TICKER}/DEF 14A/{accession_number}/...
"""

import argparse
import time
import sys
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
from sec_edgar_downloader import Downloader

from config import (
    SEC_USER_AGENT,
    TICKERS_CSV,
    FILINGS_DIR,
    DOWNLOAD_LOG,
    FILING_TYPES,
    FILINGS_LIMIT,
    EDGAR_DELAY_SECONDS,
    EDGAR_RETRY_ATTEMPTS,
    EDGAR_RETRY_DELAY,
    PILOT_TICKERS,
)

# ── Logging setup ─────────────────────────────────────────────────────────────
# Logs to both stdout and file so you can watch progress AND review later.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(DOWNLOAD_LOG, mode="a"),
    ],
)
log = logging.getLogger(__name__)


def get_completed_tickers() -> set:
    """
    Returns the set of tickers that already have a download folder.
    Used to skip re-downloading on resume after crash or interruption.

    Why folder existence rather than a separate status file:
    - Simpler: no separate state to get out of sync with reality.
    - Folder is created by sec-edgar-downloader only after a successful attempt.
    - Edge case: if a download was interrupted mid-filing, the folder exists
      but is incomplete. audit_downloads.py catches these — they show as
      having fewer filings than expected.
    """
    if not FILINGS_DIR.exists():
        return set()
    return {d.name for d in FILINGS_DIR.iterdir() if d.is_dir()}


def download_ticker(dl: Downloader, ticker: str) -> dict:
    """
    Downloads all configured filing types for a single ticker.
    Returns a status dict for logging.

    Why sec-edgar-downloader vs raw requests:
    - Handles EDGAR full-text search API pagination automatically.
    - Manages accession number formatting.
    - Saves files in a clean folder structure.
    - We still need our own rate limiting on top (the library doesn't enforce it).
    """
    result = {
        "ticker": ticker,
        "status": "ok",
        "filings_downloaded": {},
        "error": None,
    }

    for filing_type in FILING_TYPES:
        for attempt in range(1, EDGAR_RETRY_ATTEMPTS + 1):
            try:
                dl.get(
                    filing_type,
                    ticker,
                    limit=FILINGS_LIMIT,
                    download_details=True,   # gets full filing, not just index
                )
                result["filings_downloaded"][filing_type] = "ok"
                break  # success — exit retry loop

            except Exception as e:
                if attempt == EDGAR_RETRY_ATTEMPTS:
                    result["filings_downloaded"][filing_type] = f"FAILED: {e}"
                    result["status"] = "partial"
                    log.warning(f"  {ticker} / {filing_type}: failed after {EDGAR_RETRY_ATTEMPTS} attempts — {e}")
                else:
                    log.warning(f"  {ticker} / {filing_type}: attempt {attempt} failed, retrying in {EDGAR_RETRY_DELAY}s — {e}")
                    time.sleep(EDGAR_RETRY_DELAY)

        # Rate limit between filing type requests for the same company
        time.sleep(EDGAR_DELAY_SECONDS)

    return result


def run_download(tickers: list[str]) -> None:
    """
    Main download loop. Initialises downloader, skips completed tickers,
    logs progress.
    """
    dl = Downloader("PAN Group Research", "research@paneffect.co", FILINGS_DIR)

    completed = get_completed_tickers()
    remaining = [t for t in tickers if t not in completed]

    log.info(f"{'='*60}")
    log.info(f"PAN EDGAR Downloader — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info(f"Total tickers:     {len(tickers):>6,}")
    log.info(f"Already done:      {len(completed):>6,}")
    log.info(f"To download:       {len(remaining):>6,}")
    log.info(f"Filing types:      {FILING_TYPES}")
    log.info(f"Filings per type:  {FILINGS_LIMIT}")
    log.info(f"Output dir:        {FILINGS_DIR}")
    log.info(f"{'='*60}")

    if not remaining:
        log.info("All tickers already downloaded. Run audit_downloads.py to check coverage.")
        return

    # Rough time estimate: ~2 filing types × EDGAR_DELAY_SECONDS per ticker
    # Plus network latency — usually 2-5 sec per ticker in practice
    est_minutes = len(remaining) * (FILING_TYPES.__len__() * 3) / 60
    log.info(f"Estimated time: ~{est_minutes:.0f} minutes ({est_minutes/60:.1f} hours)")
    log.info("")

    errors = []
    for i, ticker in enumerate(remaining, 1):
        log.info(f"[{i:>4}/{len(remaining)}]  {ticker}")

        result = download_ticker(dl, ticker)

        if result["status"] != "ok":
            errors.append(result)
            log.warning(f"  → partial/failed: {result['filings_downloaded']}")
        else:
            log.info(f"  → ok: {result['filings_downloaded']}")

        # Rate limit between companies
        time.sleep(EDGAR_DELAY_SECONDS)

    # Summary
    log.info(f"\n{'='*60}")
    log.info(f"Download complete.")
    log.info(f"Successful: {len(remaining) - len(errors)}")
    log.info(f"Errors:     {len(errors)}")
    if errors:
        log.info("Failed tickers:")
        for e in errors:
            log.info(f"  {e['ticker']}: {e['filings_downloaded']}")
    log.info(f"{'='*60}")
    log.info(f"NEXT STEP: python audit_downloads.py")


def load_tickers() -> list[str]:
    if not TICKERS_CSV.exists():
        print(f"ERROR: {TICKERS_CSV} not found.")
        print("Run get_tickers.py first.")
        sys.exit(1)
    df = pd.read_csv(TICKERS_CSV)
    return df["ticker"].dropna().str.upper().tolist()


def parse_args():
    parser = argparse.ArgumentParser(description="PAN EDGAR Downloader")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot",  action="store_true", help="Download pilot tickers only (NKE, IBM, GHC)")
    group.add_argument("--full",   action="store_true", help="Download all tickers in nyse_tickers.csv")
    group.add_argument("--ticker", type=str,            help="Download a single ticker (e.g. --ticker AAPL)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.pilot:
        tickers = PILOT_TICKERS
        log.info(f"Mode: PILOT — {tickers}")
    elif args.ticker:
        tickers = [args.ticker.upper()]
        log.info(f"Mode: SINGLE — {tickers}")
    else:
        tickers = load_tickers()
        log.info(f"Mode: FULL — {len(tickers):,} tickers from {TICKERS_CSV}")

    run_download(tickers)


if __name__ == "__main__":
    main()
