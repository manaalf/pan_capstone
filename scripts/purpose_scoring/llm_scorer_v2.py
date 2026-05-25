"""
PAN Group — LLM Scorer
BA298A Capstone | UC Irvine | 2026

Scores S1, S2, S4, S5 using Claude via the Anthropic Batch API.
Batch API is 50% cheaper than standard and runs overnight — ideal for 1,044 companies.

Pipeline:
  1. submit_batch()  — builds prompts, submits batch, saves batch_id to disk
  2. check_batch()   — polls status, downloads results when done
  3. parse_results() — extracts scores, merges into purpose_scores.csv

Usage:
  python llm_scorer.py --submit          # submit batch (run once)
  python llm_scorer.py --check           # check status + download if done
  python llm_scorer.py --pilot           # submit pilot (NKE, IBM, GHC, JPM) only
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
NLP_SCORES    = BASE_DIR / "data" / "scores" / "purpose_scores.csv"
PERF_FILTERED = BASE_DIR / "data" / "scores" / "performance_filtered.xlsx"
LLM_DIR       = BASE_DIR / "data" / "llm"
BATCH_ID_FILE = LLM_DIR / "batch_id.txt"
RESULTS_FILE  = LLM_DIR / "batch_results.jsonl"
OUTPUT_CSV    = BASE_DIR / "data" / "scores" / "purpose_scores.csv"
LLM_LOG       = LOGS_DIR / "llm_scorer.log"

LLM_DIR.mkdir(parents=True, exist_ok=True)

# ── Settings ──────────────────────────────────────────────────────────────────
MODEL      = "claude-sonnet-4-6"
MAX_TOKENS = 2000
MAX_PASSAGES_PER_SECTION = 8  # keep prompts small — controls cost

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


# ── Prompt builder ────────────────────────────────────────────────────────────

def select_passages_smart(
    passages: pd.DataFrame,
    filing_types: list,
    section_keywords: list,
    content_keywords: list = None,
    years_labeled: bool = False,
) -> list:
    """
    Accepts a list of filing types so foreign firms (20-F only) are handled
    alongside domestic firms (10-K + DEF 14A).
    """
    mask = passages["filing_type"].isin(filing_types)
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
        text = str(row["passage_text"])[:2000]
        if years_labeled and "filing_year" in row and pd.notna(row["filing_year"]):
            text = f"[{int(row['filing_year'])}] {text}"
        results.append(text)
    return results


def build_prompt(ticker: str, passages: pd.DataFrame, nlp: dict) -> str:

    # Detect if this is a foreign filer (has 20-F but no 10-K)
    filing_types = passages["filing_type"].unique().tolist()
    is_foreign = "20-F" in filing_types and "10-K" not in filing_types

    # For foreign firms: 20-F covers everything (business + compensation in one doc)
    # For domestic firms: 10-K for S1/S2, DEF 14A for S4/S5
    annual_types  = ["20-F"] if is_foreign else ["10-K"]
    proxy_types   = ["20-F"] if is_foreign else ["DEF 14A"]
    all_types     = ["20-F"] if is_foreign else ["10-K", "DEF 14A"]

    foreign_note = "\nNOTE: This is a foreign private issuer filing 20-F (not 10-K/DEF 14A). Compensation and governance are in the 20-F body (Item 6). Score accordingly.\n" if is_foreign else ""

    s1_10k = select_passages_smart(
        passages, annual_types,
        section_keywords=["business", "mission", "purpose", "values", "about", "identity", "information on the company"],
        content_keywords=["mission", "purpose", "values", "believe", "serve", "committed", "vision"],
    )
    s1_proxy = select_passages_smart(
        passages, proxy_types,
        section_keywords=["dear", "letter", "message", "stakeholder", "overview"],
        content_keywords=["mission", "purpose", "values", "believe", "serve"],
    )
    s2_passages = select_passages_smart(
        passages, annual_types,
        section_keywords=["business", "mission", "purpose", "values", "identity"],
        content_keywords=["mission", "purpose", "values", "believe", "committed"],
        years_labeled=True,
    )

    # Year coverage — check annual filing types only
    all_years = sorted(passages["filing_year"].dropna().unique().astype(int).tolist())
    mission_years = sorted(passages[
        (passages["filing_type"].isin(annual_types)) &
        (passages["section"].str.contains("mission|purpose|values", case=False, na=False))
    ]["filing_year"].dropna().unique().astype(int).tolist())
    missing_years = [y for y in all_years if y not in mission_years]
    year_coverage = f"Filing years in dataset: {all_years}\nYears WITH mission/purpose/values sections: {mission_years}\nYears WITHOUT any mission/purpose/values sections: {missing_years}"

    s4_comp = select_passages_smart(
        passages, proxy_types,
        section_keywords=["compensation", "cd&a", "executive compensation", "directors", "item 6"],
        content_keywords=["metric", "target", "bonus", "incentive", "non-financial",
                         "esg", "sustainability", "employee", "diversity", "safety",
                         "community", "purpose", "values", "stakeholder"],
    )
    s4_gov = select_passages_smart(
        passages, proxy_types,
        section_keywords=["governance", "committee", "board"],
        content_keywords=["mandate", "oversight", "responsible", "stakeholder",
                         "sustainability", "purpose", "mission", "values"],
    )
    s4_10k = select_passages_smart(
        passages, annual_types,
        section_keywords=["business", "strategy", "management"],
        content_keywords=["acquisition", "invest", "capital", "purpose", "mission", "strategy"],
    )
    s5_passages = select_passages_smart(
        passages, proxy_types,
        section_keywords=["stakeholder", "employee", "community", "sustainability", "social"],
        content_keywords=["employees", "workers", "communities", "customers", "suppliers",
                         "patients", "environment", "commitment", "target", "goal",
                         "invested", "donated", "training", "safety"],
    )
    s5_10k = select_passages_smart(
        passages, annual_types,
        section_keywords=["business", "stakeholder", "sustainability", "values"],
        content_keywords=["employees", "communities", "customers", "commitment",
                         "responsible", "invest", "support"],
    )

    def fmt(plist):
        if not plist:
            return "[No relevant passages found]"
        result = ""
        for i, p in enumerate(plist, 1):
            result += f"\n[Passage {i}]\n{p}\n"
        return result

    prompt = f"""You are an expert research analyst scoring {ticker} on the PAN Purpose Framework.
Score each section based ONLY on the passages provided from SEC EDGAR filings. Do not use external knowledge.
{foreign_note}
## QUANTITATIVE NLP PRE-SCORES
- Commitment ratio: {nlp.get('commitment_ratio', 0.5):.3f}  [0=non-committal, 1=assertive]
- Temporal commitment: {nlp.get('temporal_commitment', 0.5):.3f}  [0=short-term, 1=long-term framing]
- Stakeholder specificity: {nlp.get('stakeholder_specificity', 0.0):.3f}  [0=generic, 1=specific groups]
- Accountability ratio: {nlp.get('accountability_ratio', 0.0):.3f}  [0=aspirational, 1=measurable]
- Temporal decay slope: {nlp.get('s3c_slope', 0.0):.4f}  [negative=eroding, positive=strengthening]

---

## S1 — MISSION CLARITY (max 3.0 pts)
Criteria 1.1 — Mission quality:
- 3.0: Specific, names who is served and how, clearly differentiated, genuine strategic commitment
- 1.5-2.5: Reasonably specific but generic in one element
- 0-1.0: No mission, or purely financial framing

Criteria 1.2 — Presence across filing types:
- 1.0: Present in both annual report body AND governance/proxy filing in operational language
- 0.5: Present in one filing type only
- 0: Absent or only in marketing cover letters

Annual Report Passages:
{fmt(s1_10k)}

Governance/Proxy Passages:
{fmt(s1_proxy)}

---

## S2 — LONGITUDINAL CONSISTENCY (max 3.0 pts)

CRITICAL FOR S2: Check whether purpose language appears in EARLY filings (2016-2018).
- If purpose sections are ABSENT from early filings and only appear from 2021 onwards — score S2 maximum 1.5
- If purpose sections are present AND consistent from the earliest filings — can score 2.5-3.0

NLP anchors: Temporal commitment={nlp.get('temporal_commitment', 0.5):.3f}, Decay slope={nlp.get('s3c_slope', 0.0):.4f}
A NEGATIVE slope means purpose language weakened over time — lower your S2 score.

- 3.0: Language identical or strengthening across all years
- 2.0: Mostly stable with minor wording shifts
- 1.0: Noticeable drift — key terms dropped
- 0: Major rewrite or abrupt shift

Year Coverage Analysis:
{year_coverage}

Year-labeled Annual Report Passages:
{fmt(s2_passages)}

---

## S4 — STRUCTURAL EMBEDDING (max 2.5 pts)
CRITICAL: Score ONLY on explicit binding mechanisms. Most companies score 0-0.5.

DOES NOT COUNT:
- General statements about company values or culture
- Aspirational language about purpose or ESG
- Financial performance metrics in compensation (TSR, EPS, revenue, stock price)

DOES COUNT:
- Annual report or governance filing explicitly names a non-financial metric (employee satisfaction %, safety rates, diversity %) in the bonus/incentive formula
- Board committee charter with binding stakeholder mandate
- Capital allocation explicitly justified by purpose metrics
- Business model where revenue lines are natural extensions of stated purpose

- 2.5: Multiple mechanisms with explicit evidence
- 1.5-2.0: At least one mechanism with EXPLICIT documented evidence
- 0.5-1.25: Purpose mentioned but no binding mechanism
- 0: No structural embedding

Compensation/Governance Passages:
{fmt(s4_comp)}

Board/Committee Passages:
{fmt(s4_gov)}

Strategy/Capital Allocation Passages:
{fmt(s4_10k)}

---

## S5 — STAKEHOLDER INTEGRATION (max 2.0 pts)
- 2.0: Multiple SPECIFIC groups named with MEASURABLE commitments, consistent across filings
- 1.25-1.75: Specific groups named but commitments are aspirational
- 0.5-1.0: Generic language — "We care about our people and communities"
- 0: Shareholder primacy only

Governance/Proxy Stakeholder Passages:
{fmt(s5_passages)}

Annual Report Stakeholder Passages:
{fmt(s5_10k)}

---

Return ONLY valid JSON. No preamble, no markdown.

{{"S1": <0.0-3.0>, "S1_rationale": "<one sentence citing specific evidence>", "S2": <0.0-3.0>, "S2_rationale": "<one sentence citing year-over-year evidence>", "S4": <0.0-2.5>, "S4_rationale": "<if above 0.5, cite the exact mechanism found>", "S5": <0.0-2.0>, "S5_rationale": "<cite specific stakeholder groups and commitments found>"}}"""

    return prompt


# ── Batch submission ──────────────────────────────────────────────────────────

def build_batch_requests(tickers: list[str]) -> list[dict]:
    """
    Builds the list of batch request objects for the Anthropic Batch API.
    """
    log.info(f"Loading passages...")
    passages_df = pd.read_csv(PASSAGES_CSV, low_memory=False)

    log.info(f"Loading NLP scores...")
    nlp_df = pd.read_csv(NLP_SCORES)
    nlp_map = nlp_df.set_index("ticker").to_dict("index")

    requests_list = []
    skipped = 0

    for ticker in tickers:
        company_passages = passages_df[passages_df["ticker"] == ticker]
        if company_passages.empty:
            log.warning(f"  {ticker}: no passages — skipping")
            skipped += 1
            continue

        nlp = nlp_map.get(ticker, {})
        prompt = build_prompt(ticker, company_passages, nlp)

        requests_list.append({
            "custom_id": ticker,
            "params": {
                "model": MODEL,
                "max_tokens": MAX_TOKENS,
                "messages": [{"role": "user", "content": prompt}],
            }
        })

    log.info(f"Built {len(requests_list)} requests ({skipped} skipped — no passages)")
    return requests_list


def submit_batch(tickers: list[str]) -> str:
    """
    Submits the batch to the Anthropic API.
    Returns the batch_id.
    """
    key = get_api_key()
    requests_list = build_batch_requests(tickers)

    if not requests_list:
        log.error("No requests to submit")
        sys.exit(1)

    log.info(f"Submitting batch of {len(requests_list)} requests to Anthropic...")
    log.info(f"Model: {MODEL} | Max tokens: {MAX_TOKENS}")

    # Rough cost estimate: Opus 4.6 batch = $2.50/MTok input, $12.50/MTok output
    # ~3000 input tokens + 200 output tokens per company
    est_input_cost  = len(requests_list) * 3000 / 1_000_000 * 2.50
    est_output_cost = len(requests_list) * 200  / 1_000_000 * 12.50
    log.info(f"Estimated cost: ~${est_input_cost + est_output_cost:.2f} (Batch API pricing)")

    import time
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
        log.error(f"Batch submission failed: {response.status_code} — {response.text}")
        sys.exit(1)

    data = response.json()
    batch_id = data["id"]

    BATCH_ID_FILE.write_text(batch_id)
    log.info(f"Batch submitted successfully!")
    log.info(f"Batch ID: {batch_id}")
    log.info(f"Saved to: {BATCH_ID_FILE}")
    log.info(f"\nNow run: python llm_scorer_v2.py --check")
    log.info(f"Batches typically complete within 1-24 hours.")

    return batch_id


# ── Batch status check ────────────────────────────────────────────────────────

def check_batch() -> None:
    """
    Checks batch status. Downloads and parses results if complete.
    """
    key = get_api_key()

    if not BATCH_ID_FILE.exists():
        log.error(f"No batch ID found. Run --submit first.")
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

    data = response.json()
    status = data.get("processing_status", "unknown")
    counts = data.get("request_counts", {})

    log.info(f"Status: {status}")
    log.info(f"Requests: {counts}")

    if status != "ended":
        log.info(f"Batch not complete yet. Check again later.")
        log.info(f"Run: python llm_scorer.py --check")
        return

    # Download results
    log.info("Batch complete — downloading results...")
    results_url = data.get("results_url")
    if not results_url:
        log.error("No results_url in response")
        sys.exit(1)

    results_response = requests.get(results_url, headers=api_headers(key), timeout=60)
    RESULTS_FILE.write_text(results_response.text)
    log.info(f"Results saved to: {RESULTS_FILE}")

    # Parse immediately
    parse_results()


# ── Results parsing ───────────────────────────────────────────────────────────

def parse_results() -> None:
    """
    Parses batch results and merges S1, S2, S4, S5 into purpose_scores.csv.
    """
    if not RESULTS_FILE.exists():
        log.error(f"Results file not found: {RESULTS_FILE}")
        log.error("Run --check first to download results.")
        sys.exit(1)

    log.info(f"Parsing results from {RESULTS_FILE}...")

    rows = []
    errors = []

    for line in RESULTS_FILE.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            continue

        ticker = result.get("custom_id", "")
        result_type = result.get("result", {}).get("type", "")

        if result_type == "error":
            errors.append(ticker)
            log.warning(f"  {ticker}: API error — {result['result'].get('error', {})}")
            continue

        # Extract text content
        content = result.get("result", {}).get("message", {}).get("content", [])
        text = ""
        for block in content:
            if block.get("type") == "text":
                text = block["text"].strip()
                break

        # Parse JSON from response
        try:
            # Strip markdown code fences if present
            clean = re.sub(r"```json|```", "", text).strip()
            scores = json.loads(clean)

            rows.append({
                "ticker":        ticker,
                "S1":            float(scores.get("S1", 0)),
                "S1_rationale":  str(scores.get("S1_rationale", "")),
                "S2":            float(scores.get("S2", 0)),
                "S2_rationale":  str(scores.get("S2_rationale", "")),
                "S4":            float(scores.get("S4", 0)),
                "S4_rationale":  str(scores.get("S4_rationale", "")),
                "S5":            float(scores.get("S5", 0)),
                "S5_rationale":  str(scores.get("S5_rationale", "")),
            })
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            log.warning(f"  {ticker}: parse error — {e} | raw: {text[:100]}")
            errors.append(ticker)

    log.info(f"Parsed: {len(rows)} successful, {len(errors)} errors")

    if not rows:
        log.error("No scores parsed. Check results file.")
        return

    llm_df = pd.DataFrame(rows)

    # Merge into purpose_scores.csv — ONLY update companies in this batch
    # Preserve existing S1/S2/S4/S5 for all other companies
    purpose_df = pd.read_csv(OUTPUT_CSV)

    # For companies in this batch: drop their existing S1/S2/S4/S5 and replace
    batch_tickers = set(llm_df["ticker"])
    keep_mask = ~purpose_df["ticker"].isin(batch_tickers)
    preserved = purpose_df[keep_mask]
    to_update  = purpose_df[~keep_mask].drop(
        columns=[c for c in ["S1","S1_rationale","S2","S2_rationale",
                              "S4","S4_rationale","S5","S5_rationale"] if c in purpose_df.columns]
    )
    to_update = to_update.merge(llm_df, on="ticker", how="left")

    merged = pd.concat([preserved, to_update], ignore_index=True).sort_values("ticker")

    # Calculate base score and normalised purpose score
    # S3_total is already S3a + S3c (max 2.0 after dropping S3b/S3d)
    # We renormalise S3 to fit the original 3.5 weight proportionally
    # Framework max: S1(3) + S2(3) + S3(3.5→2.0 actual) + S4(2.5) + S5(2.0) = 12.5 actual max
    # Use actual max for normalisation to preserve relative scoring
    ACTUAL_MAX = 14

    merged["base_score"] = (
        merged["S1"].fillna(0) +
        merged["S2"].fillna(0) +
        merged["S3_total"].fillna(0) +
        merged["S4"].fillna(0) +
        merged["S5"].fillna(0)
    )

    # Apply P5 penalty (stakeholder specificity below threshold)
    P5_THRESHOLD = 0.15
    merged["P5"] = merged["stakeholder_specificity"].apply(
        lambda x: -0.50 if pd.notna(x) and x < P5_THRESHOLD else 0.0
    )

    merged["adjusted_score"] = (merged["base_score"] + merged["P5"]).clip(lower=0)
    merged["purpose_score"]  = (merged["adjusted_score"] / ACTUAL_MAX).clip(0, 1).round(4)

    merged.to_csv(OUTPUT_CSV, index=False)

    log.info(f"\nPurpose scores updated: {OUTPUT_CSV}")
    log.info(f"Companies scored: {merged['S1'].notna().sum()}")
    log.info(f"\nScore summary:")
    log.info(f"  S1 mean: {merged['S1'].mean():.3f}")
    log.info(f"  S2 mean: {merged['S2'].mean():.3f}")
    log.info(f"  S4 mean: {merged['S4'].mean():.3f}")
    log.info(f"  S5 mean: {merged['S5'].mean():.3f}")
    log.info(f"  Purpose score mean: {merged['purpose_score'].mean():.3f}")
    log.info(f"  Purpose score std:  {merged['purpose_score'].std():.3f}")

    # Calibration check against pilot companies
    pilots = merged[merged["ticker"].isin(["NKE", "IBM", "GHC"])]
    if len(pilots) > 0:
        log.info(f"\nPilot calibration:")
        for _, row in pilots.iterrows():
            log.info(f"  {row['ticker']}: S1={row.get('S1','?')} S2={row.get('S2','?')} S4={row.get('S4','?')} S5={row.get('S5','?')} → purpose={row.get('purpose_score','?')}")
        log.info(f"  Expected: NKE~0.40, IBM~0.88, GHC~0.45")


# ── Main ──────────────────────────────────────────────────────────────────────

def load_tickers_to_score() -> list:
    """
    Loads only companies that:
    1. Are in the client master list
    2. Have passages downloaded
    3. Have NLP scores (S3a not null)
    4. Do NOT already have LLM scores (S1 is null)
    """
    master = pd.read_excel(
        BASE_DIR / "5-10_Purpose_Dataset.xlsx"
    )
    master["ticker"] = master["clean_ticker"].str.upper().str.strip()
    master_tickers = set(master["ticker"])

    passages_df = pd.read_csv(PASSAGES_CSV, usecols=["ticker"])
    has_passages = set(passages_df["ticker"].str.upper().unique())

    purpose_df = pd.read_csv(OUTPUT_CSV)
    has_nlp    = set(purpose_df[purpose_df["S3a"].notna()]["ticker"].str.upper())
    has_llm    = set(purpose_df[purpose_df["S1"].notna()]["ticker"].str.upper())

    to_score = sorted(master_tickers & has_passages & has_nlp - has_llm)
    log.info(f"Auto-detected {len(to_score)} companies needing LLM scoring")
    log.info(f"  (in master list + have passages + NLP, missing S1/S2/S4/S5)")
    return to_score

def parse_args():
    parser = argparse.ArgumentParser(description="PAN LLM Scorer v2")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--submit",  action="store_true", help="Submit batch for all unscored companies")
    group.add_argument("--check",   action="store_true", help="Check status / download results")
    group.add_argument("--parse",   action="store_true", help="Parse already-downloaded results")
    group.add_argument("--pilot",   action="store_true", help="Submit pilot only (NKE, IBM, GHC, JPM)")
    group.add_argument("--tickers", type=str,            help="Path to CSV with ticker column — score only these")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.pilot:
        tickers = PILOT_TICKERS + ["JPM"]
        log.info(f"Mode: PILOT — {tickers}")
        submit_batch(tickers)

    elif args.tickers:
        t_df = pd.read_csv(args.tickers)
        tickers = t_df["ticker"].str.upper().tolist()
        log.info(f"Mode: TICKERS FILE — {len(tickers)} companies from {args.tickers}")
        submit_batch(tickers)

    elif args.submit:
        tickers = load_tickers_to_score()
        log.info(f"Mode: AUTO — {len(tickers):,} unscored companies")
        submit_batch(tickers)

    elif args.check:
        check_batch()

    elif args.parse:
        parse_results()


if __name__ == "__main__":
    main()
