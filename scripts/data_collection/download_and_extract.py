"""
PAN Group — Download & Extract Pipeline
BA298A Capstone | UC Irvine | 2026

Replaces the old two-step downloader.py + extractor.py approach.
For each company:
  1. Download 10-K and DEF 14A filings from EDGAR
  2. Immediately extract purpose-relevant text passages
  3. Save passages to master CSV
  4. Delete raw filing to reclaim disk space

Storage: ~500MB-1GB total for 3,000 companies vs 1TB+ for raw filings.

Output: data/extracted/passages.csv
Columns:
  ticker, filing_type, filing_year, section, passage_text, word_count

Usage:
  python download_and_extract.py --pilot       # NKE, IBM, GHC
  python download_and_extract.py --full        # all tickers
  python download_and_extract.py --ticker NKE  # single company
"""

import argparse
import re
import time
import logging
import sys
import shutil
from pathlib import Path
from datetime import datetime

import pandas as pd
from sec_edgar_downloader import Downloader

from config import (
    TICKERS_DIR,
    TICKERS_CSV,
    FILINGS_DIR,
    LOGS_DIR,
    EDGAR_DELAY_SECONDS,
    EDGAR_RETRY_ATTEMPTS,
    EDGAR_RETRY_DELAY,
    FILINGS_LIMIT,
    FILING_TYPES,
    WINDOW_START,
    WINDOW_END,
    PILOT_TICKERS,
)

# ── Output setup ─────────────────────────────────────────────────────────────
EXTRACTED_DIR  = FILINGS_DIR.parent.parent / "extracted"
PASSAGES_CSV   = EXTRACTED_DIR / "passages.csv"
EXTRACT_LOG    = LOGS_DIR / "extract_log.txt"
EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(EXTRACT_LOG, mode="a"),
    ],
)
log = logging.getLogger(__name__)


# ── Section anchors ───────────────────────────────────────────────────────────
# These are the keyword patterns we use to find purpose-relevant sections.
# For 10-Ks: Item 1 business description, mission/purpose/values sections, 
#             MD&A opening, risk factors (for adversity context)
# For DEF 14As: CEO letter, compensation discussion, board committees

SECTION_PATTERNS = {
    "10-K": [
        # Item 1 — Business (always first, always has mission language)
        r"item\s+1[\.\s]+business",
        r"our\s+mission",
        r"our\s+purpose",
        r"our\s+values",
        r"who\s+we\s+are",
        r"about\s+us",
        # MD&A opening — capital allocation and purpose in operations
        r"item\s+7[\.\s]+management",
        r"management.s\s+discussion",
        # Risk factors — adversity context for Cat D
        r"item\s+1a[\.\s]+risk",
    ],
    "20-F": [
        # Item 1 — Identity and business overview (equivalent to 10-K Item 1)
        r"item\s+1[\.\s]+identity",
        r"item\s+1[\.\s]+information\s+on\s+the\s+company",
        r"item\s+4[\.\s]+information\s+on\s+the\s+company",
        r"our\s+mission",
        r"our\s+purpose",
        r"our\s+values",
        r"who\s+we\s+are",
        r"about\s+us",
        r"our\s+business",
        # Item 5 — Operating results (equivalent to MD&A)
        r"item\s+5[\.\s]+operating",
        r"management.s\s+discussion",
        # Item 6 — Directors and compensation (equivalent to DEF 14A)
        r"item\s+6[\.\s]+directors",
        r"compensation\s+discussion",
        r"executive\s+compensation",
        r"corporate\s+governance",
        # Sustainability/ESG — for P8 penalty
        r"sustainability",
        r"environmental",
        r"social\s+responsibility",
        r"stakeholder",
        # Risk factors — adversity context
        r"item\s+3[\.\s]+risk",
        r"key\s+information.*risk",
    ],
    "DEF 14A": [
        # CEO/Chairman letter
        r"dear\s+(fellow\s+)?(shareholder|stockholder)",
        r"letter\s+from\s+(the\s+)?(chair|ceo|president)",
        r"message\s+from\s+(the\s+)?(chair|ceo|president)",
        # Compensation discussion — purpose linked to exec pay (S4)
        r"compensation\s+discussion",
        r"cd\&a",
        r"executive\s+compensation",
        # Board governance — structural embedding (S4)
        r"board\s+committee",
        r"governance\s+committee",
        r"corporate\s+governance",
        # ESG/sustainability section — for P8 penalty comparison
        r"sustainability",
        r"environmental",
        r"social\s+responsibility",
        r"stakeholder",
    ],
}

# How many characters to extract after finding a section header
# 3000 chars ≈ 500 words ≈ enough for 1-2 paragraphs of purpose language
PASSAGE_LENGTH = 3000

# Minimum passage length to save — filters out false positives
MIN_PASSAGE_LENGTH = 100


def extract_text_from_file(filepath: Path) -> str:
    """
    Extracts raw text from a filing file.
    EDGAR filings come as HTML, HTM, or TXT.
    We strip HTML tags and clean whitespace.
    """
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

    # Strip HTML tags
    content = re.sub(r"<[^>]+>", " ", content)
    # Collapse whitespace
    content = re.sub(r"\s+", " ", content)
    return content.strip()


def find_main_filing_file(accession_dir: Path) -> Path | None:
    """
    Finds the main document in an accession folder.
    EDGAR accession folders contain multiple files. The main filing is 
    usually the largest .htm or .txt file that isn't an exhibit.

    Why not just read all files: exhibits (EX-21, EX-31, EX-32) contain
    subsidiary lists, certifications, and consent letters — not purpose text.
    We want the main 10-K or DEF 14A document only.
    """
    if not accession_dir.exists():
        return None

    candidates = []
    for f in accession_dir.iterdir():
        if f.is_file() and f.suffix.lower() in [".htm", ".html", ".txt"]:
            # Skip known exhibit files
            name_lower = f.name.lower()
            if any(skip in name_lower for skip in ["ex-", "ex_", "exhibit", "xbrl", "r1.", "r2.", "r3."]):
                continue
            candidates.append(f)

    if not candidates:
        return None

    # Return the largest file — main filing is always the biggest
    return max(candidates, key=lambda f: f.stat().st_size)


def extract_passages(text: str, filing_type: str) -> list[dict]:
    """
    Finds and extracts purpose-relevant passages from filing text.
    Returns a list of dicts with section name and passage text.

    Why regex anchors instead of reading structure:
    - EDGAR HTML structure varies wildly across companies and years.
    - Keyword anchors are more robust across 3,000 companies.
    - Tradeoff: we may miss some sections and get some false positives,
      but at scale this is acceptable — NLP scoring handles noise.
    """
    passages = []
    text_lower = text.lower()
    patterns = SECTION_PATTERNS.get(filing_type, [])

    seen_positions = []  # avoid extracting overlapping passages

    for pattern in patterns:
        for match in re.finditer(pattern, text_lower):
            start = match.start()

            # Skip if this position overlaps with an already-extracted passage
            too_close = any(abs(start - pos) < PASSAGE_LENGTH // 2 for pos in seen_positions)
            if too_close:
                continue

            # Extract passage from match position
            passage = text[start:start + PASSAGE_LENGTH].strip()

            if len(passage) < MIN_PASSAGE_LENGTH:
                continue

            section_name = pattern.replace(r"\s+", " ").replace(r"[\.\s]+", " ").strip()

            passages.append({
                "section": section_name,
                "passage_text": passage,
                "word_count": len(passage.split()),
            })
            seen_positions.append(start)

    return passages


def extract_filing_year_from_path(accession_dir: Path) -> int | None:
    """Extract year from accession number folder name."""
    match = re.match(r"^\d{10}-(\d{2})-\d{6}$", accession_dir.name)
    if not match:
        return None
    yy = int(match.group(1))
    return 2000 + yy if yy < 96 else 1900 + yy


def process_ticker(dl: Downloader, ticker: str) -> list[dict]:
    """
    Downloads filings for one ticker, extracts passages, deletes raw files.
    Returns list of passage rows ready for the CSV.
    """
    rows = []
    ticker_dir = FILINGS_DIR / ticker

    for filing_type in FILING_TYPES:
        # Download
        for attempt in range(1, EDGAR_RETRY_ATTEMPTS + 1):
            try:
                dl.get(filing_type, ticker, limit=FILINGS_LIMIT, download_details=True)
                break
            except Exception as e:
                if attempt == EDGAR_RETRY_ATTEMPTS:
                    log.warning(f"  {ticker}/{filing_type}: failed after {EDGAR_RETRY_ATTEMPTS} attempts — {e}")
                else:
                    log.warning(f"  {ticker}/{filing_type}: attempt {attempt} failed, retrying — {e}")
                    time.sleep(EDGAR_RETRY_DELAY)

        time.sleep(EDGAR_DELAY_SECONDS)

        # Extract from downloaded files
        filing_dir = ticker_dir / filing_type
        if not filing_dir.exists():
            continue

        for accession_dir in filing_dir.iterdir():
            if not accession_dir.is_dir():
                continue

            year = extract_filing_year_from_path(accession_dir)
            if not year or not (WINDOW_START <= year <= WINDOW_END):
                # Delete files outside measurement window immediately
                shutil.rmtree(accession_dir, ignore_errors=True)
                continue

            main_file = find_main_filing_file(accession_dir)
            if not main_file:
                shutil.rmtree(accession_dir, ignore_errors=True)
                continue

            text = extract_text_from_file(main_file)
            if not text:
                shutil.rmtree(accession_dir, ignore_errors=True)
                continue

            passages = extract_passages(text, filing_type)

            for p in passages:
                rows.append({
                    "ticker":       ticker,
                    "filing_type":  filing_type,
                    "filing_year":  year,
                    "section":      p["section"],
                    "passage_text": p["passage_text"],
                    "word_count":   p["word_count"],
                })

            # Delete raw filing immediately after extraction
            shutil.rmtree(accession_dir, ignore_errors=True)
            log.info(f"    {filing_type} {year}: {len(passages)} passages extracted, raw filing deleted")

    # Clean up empty ticker directory
    if ticker_dir.exists():
        shutil.rmtree(ticker_dir, ignore_errors=True)

    return rows


FOREIGN_FILERS_CSV = TICKERS_DIR / "foreign_filers_for_teammate.csv"
FOREIGN_FILING_TYPES = ["20-F"]  # foreign filers don't file DEF 14A


def load_foreign_tickers() -> list[str]:
    """Loads foreign filers that have 10+ 20-F filings."""
    if not FOREIGN_FILERS_CSV.exists():
        print(f"ERROR: {FOREIGN_FILERS_CSV} not found. Run the foreign filer scan first.")
        sys.exit(1)
    df = pd.read_csv(FOREIGN_FILERS_CSV)
    tickers = df["ticker"].dropna().str.upper().tolist()
    log.info(f"Loaded {len(tickers)} foreign filers from {FOREIGN_FILERS_CSV}")
    return tickers


def process_foreign_ticker(dl: Downloader, ticker: str) -> list[dict]:
    """
    Downloads 20-F filings for one foreign ticker, extracts passages.
    Same logic as process_ticker but uses 20-F only — no DEF 14A.
    """
    rows = []
    ticker_dir = FILINGS_DIR / ticker

    for filing_type in FOREIGN_FILING_TYPES:
        for attempt in range(1, EDGAR_RETRY_ATTEMPTS + 1):
            try:
                dl.get(filing_type, ticker, limit=FILINGS_LIMIT, download_details=True)
                break
            except Exception as e:
                if attempt == EDGAR_RETRY_ATTEMPTS:
                    log.warning(f"  {ticker}/{filing_type}: failed after {EDGAR_RETRY_ATTEMPTS} attempts — {e}")
                else:
                    log.warning(f"  {ticker}/{filing_type}: attempt {attempt} failed — {e}")
                    time.sleep(EDGAR_RETRY_DELAY)

        time.sleep(EDGAR_DELAY_SECONDS)

        filing_dir = ticker_dir / filing_type
        if not filing_dir.exists():
            continue

        for accession_dir in filing_dir.iterdir():
            if not accession_dir.is_dir():
                continue

            year = extract_filing_year_from_path(accession_dir)
            if not year or not (WINDOW_START <= year <= WINDOW_END):
                shutil.rmtree(accession_dir, ignore_errors=True)
                continue

            main_file = find_main_filing_file(accession_dir)
            if not main_file:
                shutil.rmtree(accession_dir, ignore_errors=True)
                continue

            text = extract_text_from_file(main_file)
            if not text:
                shutil.rmtree(accession_dir, ignore_errors=True)
                continue

            passages = extract_passages(text, filing_type)

            for p in passages:
                rows.append({
                    "ticker":       ticker,
                    "filing_type":  filing_type,
                    "filing_year":  year,
                    "section":      p["section"],
                    "passage_text": p["passage_text"],
                    "word_count":   p["word_count"],
                })

            shutil.rmtree(accession_dir, ignore_errors=True)
            log.info(f"    {filing_type} {year}: {len(passages)} passages extracted")

    if ticker_dir.exists():
        shutil.rmtree(ticker_dir, ignore_errors=True)

    return rows


def get_completed_tickers() -> set:
    """Returns tickers already in passages.csv to enable resuming."""
    if not PASSAGES_CSV.exists():
        return set()
    try:
        df = pd.read_csv(PASSAGES_CSV, usecols=["ticker"])
        return set(df["ticker"].unique())
    except Exception:
        return set()


def save_rows(rows: list[dict]) -> None:
    """Appends rows to passages.csv, creating it if needed."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    write_header = not PASSAGES_CSV.exists()
    df.to_csv(PASSAGES_CSV, mode="a", header=write_header, index=False)


def load_tickers() -> list[str]:
    if not TICKERS_CSV.exists():
        print(f"ERROR: {TICKERS_CSV} not found. Run get_tickers.py first.")
        sys.exit(1)
    df = pd.read_csv(TICKERS_CSV)
    return df["ticker"].dropna().str.upper().tolist()


def run(tickers: list[str]) -> None:
    dl = Downloader("PAN Group Research", "research@paneffect.co", FILINGS_DIR)
    completed = get_completed_tickers()
    remaining = [t for t in tickers if t not in completed]

    log.info(f"{'='*60}")
    log.info(f"PAN Extract Pipeline — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info(f"Total tickers:   {len(tickers):>6,}")
    log.info(f"Already done:    {len(completed):>6,}")
    log.info(f"To process:      {len(remaining):>6,}")
    log.info(f"Window:          {WINDOW_START}–{WINDOW_END}")
    log.info(f"Output:          {PASSAGES_CSV}")
    log.info(f"{'='*60}\n")

    for i, ticker in enumerate(remaining, 1):
        log.info(f"[{i:>4}/{len(remaining)}]  {ticker}")
        try:
            rows = process_ticker(dl, ticker)
            save_rows(rows)
            log.info(f"  → {len(rows)} passages saved")
        except Exception as e:
            log.error(f"  → ERROR: {e}")

        time.sleep(EDGAR_DELAY_SECONDS)

    log.info(f"\n{'='*60}")
    log.info(f"Done. Output: {PASSAGES_CSV}")
    if PASSAGES_CSV.exists():
        df = pd.read_csv(PASSAGES_CSV)
        log.info(f"Total rows: {len(df):,} passages from {df['ticker'].nunique():,} companies")
        log.info(f"File size:  {PASSAGES_CSV.stat().st_size / 1e6:.1f} MB")
    log.info(f"{'='*60}")
    log.info(f"NEXT STEP: python audit_downloads.py")


def parse_args():
    parser = argparse.ArgumentParser(description="PAN Download & Extract")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot",           action="store_true", help="Download pilot tickers (NKE, IBM, GHC)")
    group.add_argument("--full",            action="store_true", help="Download all domestic tickers")
    group.add_argument("--ticker",          type=str,            help="Download a single ticker")
    group.add_argument("--foreign",         action="store_true", help="Download 20-F for all foreign filers list")
    group.add_argument("--tickers",         type=str,            help="CSV file with ticker column — download 10-K for these")
    group.add_argument("--foreign-tickers", type=str,            help="CSV file with ticker column — download 20-F for these")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.pilot:
        tickers     = PILOT_TICKERS
        use_foreign = False
    elif args.ticker:
        tickers     = [args.ticker.upper()]
        use_foreign = False
    elif args.foreign:
        tickers     = load_foreign_tickers()
        use_foreign = True
    elif args.tickers:
        tickers     = pd.read_csv(args.tickers)["ticker"].str.upper().tolist()
        use_foreign = False
        log.info(f"Loaded {len(tickers)} domestic tickers from {args.tickers}")
    elif getattr(args, "foreign_tickers", None):
        tickers     = pd.read_csv(args.foreign_tickers)["ticker"].str.upper().tolist()
        use_foreign = True
        log.info(f"Loaded {len(tickers)} foreign tickers from {args.foreign_tickers}")
    else:
        tickers     = load_tickers()
        use_foreign = False

    dl        = Downloader("PAN Group Research", "research@paneffect.co", FILINGS_DIR.parent)
    completed = get_completed_tickers()
    remaining = [t for t in tickers if t not in completed]

    log.info(f"{'='*60}")
    log.info(f"PAN Extract Pipeline — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info(f"Mode:            {'Foreign 20-F' if use_foreign else 'Domestic 10-K/DEF14A'}")
    log.info(f"Total tickers:   {len(tickers):>6,}")
    log.info(f"Already done:    {len(completed):>6,}")
    log.info(f"To process:      {len(remaining):>6,}")
    log.info(f"Window:          {WINDOW_START}–{WINDOW_END}")
    log.info(f"Output:          {PASSAGES_CSV}")
    log.info(f"{'='*60}\n")

    for i, ticker in enumerate(remaining, 1):
        log.info(f"[{i:>4}/{len(remaining)}]  {ticker}")
        try:
            if use_foreign:
                rows = process_foreign_ticker(dl, ticker)
            else:
                rows = process_ticker(dl, ticker)
            save_rows(rows)
            log.info(f"  → {len(rows)} passages saved")
        except Exception as e:
            log.error(f"  → ERROR: {e}")
        time.sleep(EDGAR_DELAY_SECONDS)

    log.info(f"\n{'='*60}")
    log.info(f"Done. Output: {PASSAGES_CSV}")
    if PASSAGES_CSV.exists():
        df = pd.read_csv(PASSAGES_CSV)
        log.info(f"Total rows: {len(df):,} passages from {df['ticker'].nunique():,} companies")
        log.info(f"File size:  {PASSAGES_CSV.stat().st_size / 1e6:.1f} MB")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
