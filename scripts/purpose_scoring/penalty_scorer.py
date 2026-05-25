"""
PAN Group — Penalty & Bonus Scorer
BA298A Capstone | UC Irvine | 2026

Submits a lightweight batch to score ONLY penalties and bonuses.
Assumes S1/S2/S4/S5 scores already exist in purpose_scores.csv.
Costs roughly 75% less than a full rescore.

Penalties detected by LLM:
  P1  (-1.00): Mission materially rewritten across years
  P2  (-1.25): Mission rewritten during financial distress
  P4  (-1.00): Purpose contradicted by documented actions
  P8  (-0.50): Sustainability language clustered in ESG sections only

Penalties applied automatically from NLP:
  P1_auto     : s3c_slope < -0.05 corroboration

Bonuses detected by LLM:
  B1  (+0.50): Purpose predates leadership
  B2  (+0.75): Purpose operationalised under adversity
  B3  (+0.75): Formal binding stakeholder accountability mechanism
  B4  (+0.50): Purpose cited in adversity

Usage:
  python penalty_scorer.py --submit    # submit penalty/bonus batch
  python penalty_scorer.py --check     # check status / download + parse
  python penalty_scorer.py --parse     # re-parse downloaded results
  python penalty_scorer.py --retry     # resubmit errored tickers
"""

import argparse
import json
import os
import re
import sys
import time
import logging
from pathlib import Path

import pandas as pd
import requests

from config import LOGS_DIR, PILOT_TICKERS

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.resolve()
PASSAGES_CSV  = BASE_DIR / "data" / "extracted" / "passages.csv"
SCORES_CSV    = BASE_DIR / "data" / "scores" / "purpose_scores.csv"
PERF_FILTERED = BASE_DIR / "data" / "scores" / "performance_filtered.xlsx"
LLM_DIR       = BASE_DIR / "data" / "llm"
BATCH_ID_FILE = LLM_DIR / "penalty_batch_id.txt"
RESULTS_FILE  = LLM_DIR / "penalty_batch_results.jsonl"
ERROR_FILE    = LLM_DIR / "penalty_batch_errors.txt"
LLM_LOG       = LOGS_DIR / "penalty_scorer.log"

LLM_DIR.mkdir(parents=True, exist_ok=True)

# ── Settings ──────────────────────────────────────────────────────────────────
MODEL                    = "claude-haiku-4-5-20251001"
MAX_TOKENS               = 1500   # smaller — only penalty/bonus JSON output
MAX_PASSAGES_PER_SECTION = 8      # fewer passages — only need adversity evidence
ACTUAL_MAX               = 14.0

# Automated penalty thresholds
P1_SLOPE_THRESHOLD = -0.05

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LLM_LOG, mode="a"),
    ],
)
log = logging.getLogger(__name__)


# ── API helpers ───────────────────────────────────────────────────────────────
def get_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        log.error("ANTHROPIC_API_KEY not set. Run: export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)
    return key


def api_headers(key: str) -> dict:
    return {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "message-batches-2024-09-24",
        "content-type": "application/json",
    }


# ── Passage selection ─────────────────────────────────────────────────────────
def select_passages(
    passages: pd.DataFrame,
    filing_type: str,
    section_keywords: list,
    content_keywords: list = None,
    years_labeled: bool = False,
) -> list:
    mask     = passages["filing_type"] == filing_type
    filtered = passages[mask].copy()
    if filtered.empty:
        return []

    def score_row(row):
        section = str(row["section"]).lower()
        text    = str(row["passage_text"]).lower()
        s_score = sum(1 for kw in section_keywords if kw in section) * 2
        c_score = sum(1 for kw in (content_keywords or []) if kw in text)
        return s_score + c_score

    filtered["_score"] = filtered.apply(score_row, axis=1)
    filtered = filtered.sort_values("_score", ascending=False)

    results = []
    for _, row in filtered.head(MAX_PASSAGES_PER_SECTION).iterrows():
        text = str(row["passage_text"])[:1500]
        if years_labeled and "filing_year" in row and pd.notna(row["filing_year"]):
            text = f"[{int(row['filing_year'])}] {text}"
        results.append(text)
    return results


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_penalty_prompt(ticker: str, passages: pd.DataFrame, nlp: dict) -> str:

    # Adversity passages — for P1, P2, P4, B2, B4
    adversity_10k = select_passages(
        passages, "10-K",
        section_keywords=["business", "management", "values", "mission", "risk"],
        content_keywords=["despite", "challenging", "difficult", "crisis", "downturn",
                         "headwinds", "pressure", "maintained", "reaffirm", "committed",
                         "mission", "purpose", "values", "restructur", "layoff", "workforce"],
        years_labeled=True,
    )

    # Mission year-over-year — for P1, P2
    mission_passages = select_passages(
        passages, "10-K",
        section_keywords=["mission", "purpose", "values", "business"],
        content_keywords=["mission", "purpose", "values", "believe", "serve", "committed"],
        years_labeled=True,
    )

    # ESG/sustainability — for P8
    esg_passages = select_passages(
        passages, "10-K",
        section_keywords=["sustainability", "environmental", "social", "esg", "csr"],
        content_keywords=["purpose", "mission", "values", "community", "environment",
                         "stakeholder", "committed"],
    )

    # Stakeholder accountability — for B3
    accountability_passages = select_passages(
        passages, "DEF 14A",
        section_keywords=["compensation", "governance", "committee", "stakeholder"],
        content_keywords=["binding", "profit-sharing", "representation", "board",
                         "employee", "community", "veto", "formula", "mechanism",
                         "accountable", "measured", "target"],
    )

    # Year coverage for P1 context
    all_years = sorted(passages["filing_year"].dropna().unique().astype(int).tolist())
    mission_years = sorted(passages[
        (passages["filing_type"] == "10-K") &
        (passages["section"].str.contains("mission|purpose|values", case=False, na=False))
    ]["filing_year"].dropna().unique().astype(int).tolist())

    def fmt(plist):
        if not plist:
            return "[No relevant passages found]"
        return "".join(f"\n[Passage {i}]\n{p}\n" for i, p in enumerate(plist, 1))

    prompt = f"""You are assessing penalties and bonuses for {ticker} under the PAN Purpose Framework.
Evaluate ONLY based on the passages provided from SEC EDGAR filings. Do not use external knowledge.

## CONTEXT
NLP temporal decay slope: {nlp.get('s3c_slope', 0.0):.4f}  [negative = purpose language eroding over time]
Filing years in dataset: {all_years}
Years WITH mission/purpose sections: {mission_years}

---

## PENALTIES — trigger ONLY with clear evidence from passages

P1 — Mission materially rewritten (-1.0):
Trigger if year-labeled passages show the mission/purpose statement was SUBSTANTIALLY REPLACED across years — not minor wording adjustment but a complete change in core framing. A slope of {nlp.get('s3c_slope', 0.0):.4f} corroborates this if negative.
Do NOT trigger for: gradual evolution, minor rewording, or natural refinement.

P2 — Mission rewritten during financial distress (-1.25):
Trigger ONLY if P1 is ALSO triggered AND the rewrite coincides with visible financial pressure in the same year (cost cuts, restructuring, layoffs mentioned in same period's filings).
Cannot trigger without P1.

P4 — Purpose contradicted by documented actions (-1.0):
Trigger if passages explicitly state a stakeholder group as a purpose beneficiary BUT also show material adverse action against that same group (mass layoffs without acknowledgment, documented environmental violations, community harm) within the same filing window.
Do NOT trigger for: general business decisions, routine restructuring, or competitive actions.

P8 — Sustainability clustering (-0.5):
Trigger if purpose/mission/values keywords appear ONLY in dedicated ESG/sustainability sections and are largely ABSENT from the main 10-K business and strategy sections.
Do NOT trigger if purpose language appears throughout the 10-K, not just in ESG sections.

Year-labeled Mission Passages (for P1, P2):
{fmt(mission_passages)}

Year-labeled Adversity Passages (for P2, P4, B2, B4):
{fmt(adversity_10k)}

ESG/Sustainability Passages (for P8):
{fmt(esg_passages)}

---

## BONUSES — trigger ONLY with clear documented evidence

B1 — Purpose predates leadership (+0.5):
Trigger if mission language appears UNCHANGED across what appear to be multiple leadership periods in the filing window — purpose reads as institutional not personal to one leader.
Do NOT trigger based on assumption — requires visible evidence of continuity across leadership change.

B2 — Purpose operationalised under adversity (+0.75):
Trigger if passages show the company MAINTAINED or STRENGTHENED purpose commitments during a documented period of financial or operational stress — requires a COSTLY DOCUMENTED DECISION, not just language. Example: maintained employee programmes or community investment through revenue decline.
Do NOT trigger for: language only, vague commitments, or normal business operations.

B3 — Formal binding stakeholder accountability mechanism (+0.75):
Trigger if passages show a NON-ADVISORY, formally constituted mechanism through which non-shareholder stakeholders can influence decisions — binding profit-sharing formula, employee board representation, community veto rights.
Must be LEGALLY BINDING or FORMALLY CONSTITUTED, not discretionary or aspirational.

B4 — Purpose cited in adversity (+0.5):
Trigger if adversity passages show the company EXPLICITLY invokes purpose/mission/values language during periods of financial difficulty or operational challenge — purpose used as anchor during stress.
Do NOT trigger if purpose language appears only in good times.

Accountability/Governance Passages (for B3):
{fmt(accountability_passages)}

---

## OUTPUT — RETURN ONLY THIS JSON. No preamble, no markdown, no text outside the JSON.

{{
  "P1": {{"triggered": <true/false>, "deduction": <0 or -1.0>, "reason": "<one sentence citing specific evidence>"}},
  "P2": {{"triggered": <true/false>, "deduction": <0 or -1.25>, "reason": "<one sentence — only if P1 triggered>"}},
  "P4": {{"triggered": <true/false>, "deduction": <0 or -1.0>, "reason": "<one sentence citing specific contradiction>"}},
  "P8": {{"triggered": <true/false>, "deduction": <0 or -0.5>, "reason": "<one sentence citing distribution evidence>"}},
  "B1": {{"triggered": <true/false>, "addition": <0 or 0.5>, "reason": "<one sentence citing leadership continuity evidence>"}},
  "B2": {{"triggered": <true/false>, "addition": <0 or 0.75>, "reason": "<one sentence citing specific costly decision>"}},
  "B3": {{"triggered": <true/false>, "addition": <0 or 0.75>, "reason": "<one sentence citing the specific binding mechanism>"}},
  "B4": {{"triggered": <true/false>, "addition": <0 or 0.5>, "reason": "<one sentence citing adversity + purpose language>"}},
  "overall_confidence": "<low/medium/high — how confident are you in these assessments given available evidence>"
}}"""

    return prompt


# ── Batch submission ──────────────────────────────────────────────────────────
def build_batch_requests(tickers: list) -> list:
    log.info("Loading passages...")
    passages_df = pd.read_csv(PASSAGES_CSV, low_memory=False)

    log.info("Loading NLP scores...")
    nlp_df  = pd.read_csv(SCORES_CSV)
    nlp_map = nlp_df.set_index("ticker").to_dict("index")

    requests_list = []
    skipped = 0

    for ticker in tickers:
        company_passages = passages_df[passages_df["ticker"] == ticker]
        if company_passages.empty:
            log.warning(f"  {ticker}: no passages — skipping")
            skipped += 1
            continue

        nlp    = nlp_map.get(ticker, {})
        prompt = build_penalty_prompt(ticker, company_passages, nlp)

        requests_list.append({
            "custom_id": ticker,
            "params": {
                "model":      MODEL,
                "max_tokens": MAX_TOKENS,
                "messages":   [{"role": "user", "content": prompt}],
            }
        })

    log.info(f"Built {len(requests_list)} requests ({skipped} skipped)")
    return requests_list


def submit_batch(tickers: list) -> str:
    key           = get_api_key()
    requests_list = build_batch_requests(tickers)

    if not requests_list:
        log.error("No requests to submit")
        sys.exit(1)

    log.info(f"Submitting batch of {len(requests_list)} requests...")
    log.info(f"Model: {MODEL} | Max tokens: {MAX_TOKENS}")

    # Cost estimate: ~2k input tokens + 300 output tokens per company
    est_input  = len(requests_list) * 2000 / 1_000_000 * 2.50
    est_output = len(requests_list) * 300  / 1_000_000 * 12.50
    log.info(f"Estimated cost: ~${est_input + est_output:.2f} (Batch API pricing)")

    response = None
    for attempt in range(3):
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages/batches",
                headers=api_headers(key),
                json={"requests": requests_list},
                timeout=300,
                stream=True,
            )
            content = b""
            for chunk in response.iter_content(chunk_size=8192):
                content += chunk
            response._content = content
            break
        except Exception as e:
            log.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                log.info("Retrying in 10 seconds...")
                time.sleep(10)
            else:
                log.error("All 3 attempts failed")
                sys.exit(1)

    if response.status_code != 200:
        log.error(f"Submission failed: {response.status_code} — {response.text[:500]}")
        sys.exit(1)

    data     = response.json()
    batch_id = data["id"]

    BATCH_ID_FILE.write_text(batch_id)
    log.info(f"Batch submitted! ID: {batch_id}")
    log.info(f"Now run: python penalty_scorer.py --check")

    return batch_id


# ── Batch status check ────────────────────────────────────────────────────────
def check_batch() -> None:
    key = get_api_key()

    if not BATCH_ID_FILE.exists():
        log.error("No batch ID found. Run --submit first.")
        sys.exit(1)

    batch_id = BATCH_ID_FILE.read_text().strip()
    log.info(f"Checking batch: {batch_id}")

    response = requests.get(
        f"https://api.anthropic.com/v1/messages/batches/{batch_id}",
        headers=api_headers(key),
        timeout=30,
    )

    if response.status_code != 200:
        log.error(f"Status check failed: {response.status_code} — {response.text}")
        sys.exit(1)

    data   = response.json()
    status = data.get("processing_status", "unknown")
    counts = data.get("request_counts", {})

    log.info(f"Status: {status}")
    log.info(f"Requests: {counts}")

    if status != "ended":
        log.info("Not complete yet. Run: python penalty_scorer.py --check")
        return

    log.info("Complete — downloading results...")
    results_url = data.get("results_url")
    results_response = requests.get(results_url, headers=api_headers(key), timeout=120)
    RESULTS_FILE.write_text(results_response.text)
    log.info(f"Results saved to: {RESULTS_FILE}")
    parse_results()


# ── Results parsing ───────────────────────────────────────────────────────────
def parse_results() -> None:
    if not RESULTS_FILE.exists():
        log.error(f"Results file not found. Run --check first.")
        sys.exit(1)

    log.info(f"Parsing results...")

    rows   = []
    errors = []

    for line in RESULTS_FILE.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            continue

        ticker      = result.get("custom_id", "")
        result_type = result.get("result", {}).get("type", "")

        if result_type == "errored":
            errors.append(ticker)
            log.warning(f"  {ticker}: API error")
            continue

        content = result.get("result", {}).get("message", {}).get("content", [])
        text    = ""
        for block in content:
            if block.get("type") == "text":
                text = block["text"].strip()
                break

        try:
            clean  = re.sub(r"```json|```", "", text).strip()
            scores = json.loads(clean)

            def get_flag(key, amount_field):
                flag = scores.get(key, {})
                if isinstance(flag, dict):
                    triggered = flag.get("triggered", False)
                    amount    = float(flag.get(amount_field, 0))
                    reason    = str(flag.get("reason", ""))
                    return amount if triggered else 0.0, reason
                return 0.0, ""

            p1_val, p1_r = get_flag("P1", "deduction")
            p2_val, p2_r = get_flag("P2", "deduction")
            p4_val, p4_r = get_flag("P4", "deduction")
            p8_val, p8_r = get_flag("P8", "deduction")
            b1_val, b1_r = get_flag("B1", "addition")
            b2_val, b2_r = get_flag("B2", "addition")
            b3_val, b3_r = get_flag("B3", "addition")
            b4_val, b4_r = get_flag("B4", "addition")

            rows.append({
                "ticker": ticker,
                "llm_P1": p1_val, "llm_P1_reason": p1_r,
                "llm_P2": p2_val, "llm_P2_reason": p2_r,
                "llm_P4": p4_val, "llm_P4_reason": p4_r,
                "llm_P8": p8_val, "llm_P8_reason": p8_r,
                "llm_B1": b1_val, "llm_B1_reason": b1_r,
                "llm_B2": b2_val, "llm_B2_reason": b2_r,
                "llm_B3": b3_val, "llm_B3_reason": b3_r,
                "llm_B4": b4_val, "llm_B4_reason": b4_r,
                "confidence": str(scores.get("overall_confidence", "medium")),
            })

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            log.warning(f"  {ticker}: parse error — {e} | raw: {text[:100]}")
            errors.append(ticker)

    log.info(f"Parsed: {len(rows)} successful, {len(errors)} errors")

    if errors:
        ERROR_FILE.write_text("\n".join(errors))
        log.info(f"Errored tickers saved to: {ERROR_FILE}")
        log.info(f"To retry: python penalty_scorer.py --retry")

    if not rows:
        log.error("No results parsed.")
        return

    penalty_df = pd.DataFrame(rows)

    # Load existing purpose scores
    purpose_df = pd.read_csv(SCORES_CSV)

    # Drop old penalty/bonus columns if present
    drop_cols = [
        "llm_P1", "llm_P1_reason", "llm_P2", "llm_P2_reason",
        "llm_P4", "llm_P4_reason", "llm_P8", "llm_P8_reason",
        "llm_B1", "llm_B1_reason", "llm_B2", "llm_B2_reason",
        "llm_B3", "llm_B3_reason", "llm_B4", "llm_B4_reason",
        "P1_auto", "confidence",
        "llm_penalties_total", "llm_bonuses_total",
        "adjusted_score", "purpose_score",
    ]
    purpose_df = purpose_df.drop(columns=[c for c in drop_cols if c in purpose_df.columns])

    merged = purpose_df.merge(penalty_df, on="ticker", how="left")

    # ── Score calculation ─────────────────────────────────────────────────────

    # Base score (S1 + S2 + S3_total + S4 + S5)
    merged["base_score"] = (
        merged["S1"].fillna(0) +
        merged["S2"].fillna(0) +
        merged["S3_total"].fillna(0) +
        merged["S4"].fillna(0) +
        merged["S5"].fillna(0)
    )

    # P1_auto: NLP slope corroboration for companies LLM didn't flag
    merged["P1_auto"] = merged.apply(
        lambda r: -1.0 if (
            pd.notna(r.get("s3c_slope")) and
            r["s3c_slope"] < P1_SLOPE_THRESHOLD and
            r.get("llm_P1", 0) == 0
        ) else 0.0,
        axis=1,
    )

    # Totals
    merged["llm_penalties_total"] = (
        merged["llm_P1"].fillna(0) +
        merged["llm_P2"].fillna(0) +
        merged["llm_P4"].fillna(0) +
        merged["llm_P8"].fillna(0)
    )

    merged["llm_bonuses_total"] = (
        merged["llm_B1"].fillna(0) +
        merged["llm_B2"].fillna(0) +
        merged["llm_B3"].fillna(0) +
        merged["llm_B4"].fillna(0)
    )

    merged["adjusted_score"] = (
        merged["base_score"] +
        merged["P1_auto"] +
        merged["llm_penalties_total"] +
        merged["llm_bonuses_total"]
    ).clip(lower=0)

    merged["purpose_score"] = (merged["adjusted_score"] / ACTUAL_MAX).clip(0, 1).round(4)

    merged.to_csv(SCORES_CSV, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    scored = merged[merged["llm_P1"].notna()]
    log.info(f"\nPurpose scores updated: {SCORES_CSV}")
    log.info(f"Companies with penalty/bonus scores: {len(scored)}")
    log.info(f"\nPenalty/bonus summary:")
    log.info(f"  P1 triggered: {(scored['llm_P1'] != 0).sum()}")
    log.info(f"  P2 triggered: {(scored['llm_P2'] != 0).sum()}")
    log.info(f"  P4 triggered: {(scored['llm_P4'] != 0).sum()}")
    log.info(f"  P8 triggered: {(scored['llm_P8'] != 0).sum()}")
    log.info(f"  B1 triggered: {(scored['llm_B1'] != 0).sum()}")
    log.info(f"  B2 triggered: {(scored['llm_B2'] != 0).sum()}")
    log.info(f"  B3 triggered: {(scored['llm_B3'] != 0).sum()}")
    log.info(f"  B4 triggered: {(scored['llm_B4'] != 0).sum()}")
    log.info(f"\nFinal purpose score summary:")
    log.info(f"  Mean:  {merged['purpose_score'].mean():.3f}")
    log.info(f"  Std:   {merged['purpose_score'].std():.3f}")
    log.info(f"  Min:   {merged['purpose_score'].min():.3f}")
    log.info(f"  Max:   {merged['purpose_score'].max():.3f}")

    # Calibration
    pilots = merged[merged["ticker"].isin(["NKE", "IBM", "GHC"])]
    if len(pilots) > 0:
        log.info(f"\nPilot calibration:")
        for _, row in pilots.iterrows():
            log.info(
                f"  {row['ticker']}: base={row.get('base_score','?'):.2f} "
                f"penalties={row.get('llm_penalties_total',0)++row.get('P1_auto',0):.2f} "
                f"bonuses={row.get('llm_bonuses_total',0):.2f} "
                f"→ purpose={row.get('purpose_score','?')}"
            )
        log.info("  Expected: NKE~0.40, IBM~0.88, GHC~0.45")


# ── Main ──────────────────────────────────────────────────────────────────────
def load_scored_tickers() -> list:
    """Load tickers that have S1/S2/S4/S5 scores but NOT yet penalty scores.
    Filtered to client master list only."""
    # Load master list
    master = pd.read_excel(
        BASE_DIR / "5-10_Purpose_Dataset.xlsx"
    )
    master["ticker"] = master["clean_ticker"].str.upper().str.strip()
    master_tickers = set(master["ticker"])

    df = pd.read_csv(SCORES_CSV)
    df["ticker"] = df["ticker"].str.upper()

    has_base    = df["S1"].notna()
    has_penalty = df["llm_P1"].notna() if "llm_P1" in df.columns else pd.Series([False] * len(df))
    in_master   = df["ticker"].isin(master_tickers)

    remaining    = df[has_base & ~has_penalty & in_master]["ticker"].tolist()
    already_done = df[has_base & has_penalty & in_master]["ticker"].tolist()

    log.info(f"Already penalty-scored (master list): {len(already_done)} companies (skipping)")
    log.info(f"Remaining to score (master list):     {len(remaining)} companies")
    return sorted(remaining)


def load_retry_tickers() -> list:
    if not ERROR_FILE.exists():
        log.error(f"No error file found at {ERROR_FILE}")
        sys.exit(1)
    tickers = [t.strip() for t in ERROR_FILE.read_text().strip().split("\n") if t.strip()]
    log.info(f"Retrying {len(tickers)} errored tickers")
    return tickers
load_retry_tickers

def main():
    parser = argparse.ArgumentParser(description="PAN Penalty & Bonus Scorer")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--submit", action="store_true", help="Submit penalty/bonus batch for all scored companies")
    group.add_argument("--check",  action="store_true", help="Check status / download + parse")
    group.add_argument("--parse",  action="store_true", help="Re-parse downloaded results")
    group.add_argument("--retry",  action="store_true", help="Resubmit errored tickers")
    group.add_argument("--pilot",  action="store_true", help="Submit pilot tickers only")
    args = parser.parse_args()

    if args.pilot:
        tickers = PILOT_TICKERS
        log.info(f"Mode: PILOT — {tickers}")
        submit_batch(tickers)

    elif args.submit:
        tickers = load_scored_tickers()
        log.info(f"Mode: FULL — {len(tickers)} scored companies")
        submit_batch(tickers)

    elif args.retry:
        tickers = load_retry_tickers()
        submit_batch(tickers)

    elif args.check:
        check_batch()

    elif args.parse:
        parse_results()


if __name__ == "__main__":
    main()
