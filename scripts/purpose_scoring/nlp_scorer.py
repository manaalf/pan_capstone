"""
PAN Group — NLP Scorer
BA298A Capstone | UC Irvine | 2026

Runs two lexicon layers on passages.csv:
  Layer 1 — Loughran-McDonald (2011) master dictionary
  Layer 2 — PAN Purpose Extension (pan_purpose_lexicon_v3.py)

Produces S3 sub-scores and all NLP intermediate scores.
Zero API cost — runs entirely locally.

Output: data/scores/purpose_scores.csv
  One row per company with S3a, S3b, S3c, S3d, S3_total
  plus all NLP intermediates for audit trail.

Usage:
  python nlp_scorer.py           # score all companies
  python nlp_scorer.py --pilot   # NKE, IBM, GHC, JPM only
"""

import argparse
import logging
import sys
import re
import numpy as np
from pathlib import Path
from collections import defaultdict

import pandas as pd
from scipy import stats

from pan_purpose_lexicon_v3 import score_all_extension
from config import LOGS_DIR, PILOT_TICKERS

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.resolve()
PASSAGES_CSV = BASE_DIR / "data" / "extracted" / "passages.csv"
SCORES_DIR   = BASE_DIR / "data" / "scores"
OUTPUT_CSV   = SCORES_DIR / "purpose_scores.csv"
NLP_LOG      = LOGS_DIR / "nlp_scorer.log"
LM_DICT_PATH = BASE_DIR / "Loughran-McDonald_MasterDictionary_1993-2025.csv"

SCORES_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(NLP_LOG, mode="w"),
    ],
)
log = logging.getLogger(__name__)


# ── Load Loughran-McDonald Dictionary ────────────────────────────────────────
def load_lm_dictionary(path: Path) -> dict:
    """
    Loads the LM master dictionary CSV.
    Returns a dict: word -> {category: value}

    The LM CSV has one row per word. Relevant columns:
      Word, Negative, Positive, Uncertainty, Litigious,
      Strong_Modal, Weak_Modal, Constraining

    Non-zero value in a column means word belongs to that category.
    We only need the categories relevant to our S3 sub-scores.
    """
    log.info(f"Loading LM dictionary from {path.name}...")

    if not path.exists():
        log.error(f"LM dictionary not found at {path}")
        log.error("Download from: https://sraf.nd.edu/loughranmcdonald-master-dictionary/")
        sys.exit(1)

    df = pd.read_csv(path, low_memory=False)

    # Normalize column names — different versions use different casing
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Find the word column
    word_col = next((c for c in df.columns if c in ["word", "words"]), None)
    if not word_col:
        log.error(f"Cannot find word column. Columns: {list(df.columns)}")
        sys.exit(1)

    # Categories we need for S3
    needed = {
        "negative":    "negative",
        "positive":    "positive",
        "uncertainty": "uncertainty",
        "litigious":   "litigious",
        "strong_modal": "strong_modal",
        "weak_modal":   "weak_modal",
    }

    # Map available columns
    available = {}
    for col, label in needed.items():
        if col in df.columns:
            available[label] = col
        else:
            log.warning(f"Column '{col}' not found in LM dict — will default to 0")

    # Build lookup dict
    lm = {}
    for _, row in df.iterrows():
        word = str(row[word_col]).lower().strip()
        lm[word] = {
            label: int(row[col]) if col in df.columns else 0
            for label, col in available.items()
        }

    log.info(f"LM dictionary loaded: {len(lm):,} words")
    return lm


def score_passage_lm(tokens: list[str], lm: dict) -> dict:
    """
    Scores a tokenized passage against the LM dictionary.
    Returns raw counts and ratios.
    """
    total = len(tokens)
    if total == 0:
        return {k: 0 for k in [
            "n_negative", "n_positive", "n_uncertainty",
            "n_litigious", "n_strong_modal", "n_weak_modal",
            "lm_negative_ratio", "lm_positive_ratio",
            "uncertainty_density", "litigious_density",
            "commitment_ratio",
        ]}

    counts = defaultdict(int)
    for t in tokens:
        if t in lm:
            for cat, val in lm[t].items():
                if val > 0:
                    counts[cat] += 1

    n_strong = counts["strong_modal"]
    n_weak   = counts["weak_modal"]
    modal_total = n_strong + n_weak

    return {
        "n_negative":       counts["negative"],
        "n_positive":       counts["positive"],
        "n_uncertainty":    counts["uncertainty"],
        "n_litigious":      counts["litigious"],
        "n_strong_modal":   n_strong,
        "n_weak_modal":     n_weak,
        "lm_negative_ratio":   counts["negative"] / total,
        "lm_positive_ratio":   counts["positive"] / total,
        "uncertainty_density": counts["uncertainty"] / total,
        "litigious_density":   counts["litigious"] / total,
        # S3a: commitment ratio = strong_modal / (strong_modal + weak_modal)
        "commitment_ratio": n_strong / modal_total if modal_total > 0 else 0.5,
    }


# ── S3 Sub-score Functions ────────────────────────────────────────────────────

def compute_s3a(commitment_ratio: float) -> float:
    """
    S3a — Commitment ratio (strong modal / total modal)
    Max: 1.0 pt
    Thresholds from framework:
      >0.65 = assertive (1.0)
      0.50-0.65 = mixed (0.70)
      0.35-0.50 = hedged (0.35)
      <0.35 = non-committal (0.10)
    """
    if commitment_ratio > 0.65:
        return 1.0
    elif commitment_ratio >= 0.50:
        return 0.70
    elif commitment_ratio >= 0.35:
        return 0.35
    else:
        return 0.10


def compute_s3b(uncertainty_density: float) -> float:
    """
    S3b — Uncertainty density (uncertainty words / total words)
    Max: 1.0 pt
    Thresholds from framework:
      <2% = specific (1.0)
      2-4% = acceptable (0.55)
      >4% = flag (0.15)
    """
    if uncertainty_density < 0.02:
        return 1.0
    elif uncertainty_density <= 0.04:
        return 0.55
    else:
        return 0.15


def compute_s3c(yearly_sentiments: dict) -> float:
    """
    S3c — Temporal decay (linear regression on net sentiment per year)
    Max: 1.0 pt
    Requires minimum 5 years of data.

    Net sentiment = (positive - negative) / total_words per year.
    Slope of regression line is the signal:
      Positive slope = strengthening (0.9)
      Flat (|slope| < threshold) = stable (0.50)
      Negative slope = erosion (0.10)

    yearly_sentiments: {year: net_sentiment_score}
    """
    if len(yearly_sentiments) < 5:
        return 0.50  # insufficient data — neutral score

    years  = sorted(yearly_sentiments.keys())
    values = [yearly_sentiments[y] for y in years]

    slope, _, _, p_value, _ = stats.linregress(years, values)

    # Only treat slope as meaningful if statistically significant
    # p < 0.10 is sufficient for this application
    if p_value > 0.10:
        return 0.50  # flat / no significant trend

    if slope > 0:
        return 0.90  # strengthening
    else:
        return 0.10  # erosion — also triggers P1 review flag


def compute_s3d(litigious_density: float) -> float:
    """
    S3d — Litigious density (litigious words / total words)
    Max: 0.5 pt
    Thresholds from framework:
      <1% = authentic (0.50)
      1-2% = borderline (0.25)
      >2% = compliance-driven (0.05)
    """
    if litigious_density < 0.01:
        return 0.50
    elif litigious_density <= 0.02:
        return 0.25
    else:
        return 0.05


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_company(ticker: str, passages: pd.DataFrame, lm: dict) -> dict:
    """
    Aggregates all passages for one company into final NLP scores.

    Strategy:
    - LM scores: aggregate across all passages (weighted by word count)
    - PAN extension: aggregate across all passages
    - S3c: compute per year first, then run regression
    - S3a/S3b/S3d: apply thresholds to aggregated ratios
    """
    row = {"ticker": ticker}

    if passages.empty:
        log.warning(f"  {ticker}: no passages found")
        for col in ["S3a", "S3b", "S3c", "S3d", "S3_total",
                    "lm_positive_ratio", "lm_negative_ratio",
                    "commitment_ratio", "uncertainty_density",
                    "litigious_density", "stakeholder_specificity",
                    "accountability_ratio", "temporal_commitment",
                    "adversity_resilience", "s3c_slope", "s3c_years",
                    "p1_flag"]:
            row[col] = 0.0
        return row

    # Tokenize all passages
    all_lm_scores    = []
    all_pan_scores   = []
    yearly_sentiment = defaultdict(list)
    total_words      = 0

    for _, p in passages.iterrows():
        text = str(p["passage_text"])
        year = int(p["filing_year"]) if pd.notna(p["filing_year"]) else 0
        wc   = int(p["word_count"]) if pd.notna(p["word_count"]) else 0

        tokens = re.findall(r"[a-z]+(?:-[a-z]+)*", text.lower())
        if not tokens:
            continue

        total_words += len(tokens)

        # LM scoring
        lm_result = score_passage_lm(tokens, lm)
        all_lm_scores.append((len(tokens), lm_result))

        # PAN extension scoring
        pan_result = score_all_extension(text)
        all_pan_scores.append(pan_result)

        # For S3c: track net sentiment per year
        if year >= 2015:
            net = lm_result["lm_positive_ratio"] - lm_result["lm_negative_ratio"]
            yearly_sentiment[year].append(net)

    if not all_lm_scores:
        log.warning(f"  {ticker}: no scoreable tokens found")
        for col in ["S3a", "S3b", "S3c", "S3d", "S3_total",
                    "lm_positive_ratio", "lm_negative_ratio",
                    "commitment_ratio", "uncertainty_density",
                    "litigious_density", "stakeholder_specificity",
                    "accountability_ratio", "temporal_commitment",
                    "adversity_resilience", "s3c_slope", "s3c_years",
                    "p1_flag"]:
            row[col] = 0.0
        return row

    # Weighted average of LM ratios (weight by token count)
    total_tok = sum(w for w, _ in all_lm_scores)
    def wavg(key):
        return sum(w * s[key] for w, s in all_lm_scores) / total_tok if total_tok else 0

    commitment_ratio   = wavg("commitment_ratio")
    uncertainty_density = wavg("uncertainty_density")
    litigious_density  = wavg("litigious_density")
    lm_positive_ratio  = wavg("lm_positive_ratio")
    lm_negative_ratio  = wavg("lm_negative_ratio")

    # Average PAN extension scores across passages
    def pan_avg(key):
        vals = [s[key] for s in all_pan_scores if key in s]
        return float(np.mean(vals)) if vals else 0.0

    stakeholder_specificity = pan_avg("stakeholder_specificity")
    accountability_ratio    = pan_avg("accountability_ratio")
    temporal_commitment     = pan_avg("temporal_commitment")
    adversity_resilience    = pan_avg("adversity_resilience")

    # S3c: average net sentiment per year, then regress
    yearly_avg = {yr: float(np.mean(vals)) for yr, vals in yearly_sentiment.items()}
    s3c_score  = compute_s3c(yearly_avg)
    s3c_slope  = 0.0
    if len(yearly_avg) >= 5:
        years  = sorted(yearly_avg.keys())
        values = [yearly_avg[y] for y in years]
        s3c_slope, *_ = stats.linregress(years, values)

    # S3 sub-scores
    S3a = compute_s3a(commitment_ratio)
    S3b = compute_s3b(uncertainty_density)
    S3c = s3c_score
    S3d = compute_s3d(litigious_density)
    S3_total = round(S3a + S3b + S3c + S3d, 4)

    # P1 flag: negative S3c slope (mission erosion signal)
    p1_flag = 1 if s3c_score == 0.10 else 0

    row.update({
        # S3 sub-scores
        "S3a":  round(S3a, 4),
        "S3b":  round(S3b, 4),
        "S3c":  round(S3c, 4),
        "S3d":  round(S3d, 4),
        "S3_total": S3_total,

        # LM intermediates
        "lm_positive_ratio":    round(lm_positive_ratio, 4),
        "lm_negative_ratio":    round(lm_negative_ratio, 4),
        "commitment_ratio":     round(commitment_ratio, 4),
        "uncertainty_density":  round(uncertainty_density, 4),
        "litigious_density":    round(litigious_density, 4),

        # PAN extension intermediates
        "stakeholder_specificity": round(stakeholder_specificity, 4),
        "accountability_ratio":    round(accountability_ratio, 4),
        "temporal_commitment":     round(temporal_commitment, 4),
        "adversity_resilience":    round(adversity_resilience, 4),

        # S3c diagnostics
        "s3c_slope": round(float(s3c_slope), 6),
        "s3c_years": len(yearly_avg),

        # Penalty flags
        "p1_flag": p1_flag,
    })

    return row


# ── Main ──────────────────────────────────────────────────────────────────────

def run(tickers: list[str] | None = None):
    log.info("=" * 60)
    log.info("PAN NLP Scorer")
    log.info("=" * 60)

    # Load LM dictionary
    lm = load_lm_dictionary(LM_DICT_PATH)

    # Load passages
    log.info(f"Loading passages from {PASSAGES_CSV}...")
    if not PASSAGES_CSV.exists():
        log.error(f"passages.csv not found. Run download_and_extract.py first.")
        sys.exit(1)

    passages_df = pd.read_csv(PASSAGES_CSV, low_memory=False)
    log.info(f"Loaded {len(passages_df):,} passages from {passages_df['ticker'].nunique():,} companies")

    # Filter to requested tickers
    all_tickers = sorted(passages_df["ticker"].unique())
    if tickers:
        all_tickers = [t for t in tickers if t in passages_df["ticker"].values]
        log.info(f"Pilot mode: scoring {len(all_tickers)} tickers")

    log.info(f"Scoring {len(all_tickers):,} companies...\n")

    rows = []
    for i, ticker in enumerate(all_tickers, 1):
        if i % 100 == 0 or i <= 3:
            log.info(f"[{i:>4}/{len(all_tickers)}]  {ticker}")

        company_passages = passages_df[passages_df["ticker"] == ticker]
        row = aggregate_company(ticker, company_passages, lm)
        rows.append(row)

    # Merge with existing scores — preserve previously scored companies
    new_df = pd.DataFrame(rows)

    if OUTPUT_CSV.exists():
        existing = pd.read_csv(OUTPUT_CSV)
        # Drop rows for tickers we just rescored
        existing = existing[~existing["ticker"].isin(new_df["ticker"])]
        df = pd.concat([existing, new_df], ignore_index=True).sort_values("ticker")
        log.info(f"Merged with existing {len(existing):,} rows — total: {len(df):,}")
    else:
        df = new_df

    # Recalculate S3_total = S3a + S3c only (S3b and S3d excluded as non-discriminatory)
    df["S3_total"] = df["S3a"].fillna(0) + df["S3c"].fillna(0)

    df.to_csv(OUTPUT_CSV, index=False)

    log.info(f"\n{'='*60}")
    log.info(f"NLP scoring complete.")
    log.info(f"Newly scored: {len(new_df):,}  |  Total in file: {len(df):,}")
    log.info(f"Output: {OUTPUT_CSV}")
    log.info(f"\nS3 score summary (new companies only):")
    log.info(f"  S3a (commitment):   mean={new_df['S3a'].mean():.3f}  std={new_df['S3a'].std():.3f}")
    log.info(f"  S3c (temporal):     mean={new_df['S3c'].mean():.3f}  std={new_df['S3c'].std():.3f}")
    log.info(f"  S3_total:           mean={new_df['S3_total'].mean():.3f}  std={new_df['S3_total'].std():.3f}")
    log.info(f"\nP1 flags (mission erosion): {new_df['p1_flag'].sum():,} companies")
    log.info(f"{'='*60}")
    log.info(f"NEXT STEP: python llm_scorer_v2.py --submit")


def parse_args():
    parser = argparse.ArgumentParser(description="PAN NLP Scorer")
    parser.add_argument("--pilot",   action="store_true", help="Score pilot tickers only (NKE, IBM, GHC)")
    parser.add_argument("--tickers", type=str, help="Path to CSV file with ticker column — score only these companies")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.pilot:
        tickers = PILOT_TICKERS + ["JPM"]
    elif args.tickers:
        t_df = pd.read_csv(args.tickers)
        tickers = t_df["ticker"].str.upper().tolist()
        log.info(f"Loaded {len(tickers)} tickers from {args.tickers}")
    else:
        tickers = None
    run(tickers)


if __name__ == "__main__":
    main()
