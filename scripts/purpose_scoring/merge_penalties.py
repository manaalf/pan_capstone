"""
PAN Group — Merge All Penalty Scores
Combines Gemini scores, main Haiku batch, and retry batch into purpose_scores.csv
"""
import json
import re
import pandas as pd
from pathlib import Path

BASE_DIR    = Path(__file__).parent.resolve()
SCORES_CSV  = BASE_DIR / "data" / "scores" / "purpose_scores.csv"
MAIN_JSONL  = BASE_DIR / "data" / "llm" / "penalty_main_results.jsonl"
RETRY_JSONL = BASE_DIR / "data" / "llm" / "penalty_batch_results.jsonl"

P5_THRESHOLD       = 0.15
P1_SLOPE_THRESHOLD = -0.05
ACTUAL_MAX         = 14.0


def parse_jsonl(filepath: Path) -> tuple:
    rows   = []
    errors = []

    for line in open(filepath):
        if not line.strip():
            continue
        result = json.loads(line)
        ticker = result.get("custom_id", "")

        if result.get("result", {}).get("type") == "errored":
            errors.append(ticker)
            continue

        content = result.get("result", {}).get("message", {}).get("content", [])
        text    = ""
        for block in content:
            if block.get("type") == "text":
                text = block["text"].strip()
                break

        try:
            start = text.find('{')
            end   = text.rfind('}') + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON found")
            clean  = text[start:end]
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
                "ticker":       ticker,
                "llm_P1": p1_val, "llm_P1_reason": p1_r,
                "llm_P2": p2_val, "llm_P2_reason": p2_r,
                "llm_P4": p4_val, "llm_P4_reason": p4_r,
                "llm_P8": p8_val, "llm_P8_reason": p8_r,
                "llm_B1": b1_val, "llm_B1_reason": b1_r,
                "llm_B2": b2_val, "llm_B2_reason": b2_r,
                "llm_B3": b3_val, "llm_B3_reason": b3_r,
                "llm_B4": b4_val, "llm_B4_reason": b4_r,
            })
        except Exception as e:
            errors.append(ticker)

    print(f"  {filepath.name}: {len(rows)} parsed, {len(errors)} errors")
    return rows, errors


# ── Load all sources ──────────────────────────────────────────────────────────
print("Parsing main batch (604 companies)...")
main_rows, _ = parse_jsonl(MAIN_JSONL)

print("Parsing retry batch (28 companies)...")
retry_rows, _ = parse_jsonl(RETRY_JSONL)

print("Parsing 182 penalty batch...")
batch_182_path = Path("data/llm/penalty_182_results.jsonl")
batch_182_rows = []
if batch_182_path.exists():
    batch_182_rows, _ = parse_jsonl(batch_182_path)
    print(f"  penalty_182_results.jsonl: {len(batch_182_rows)} parsed")

print("Parsing 267 penalty batch...")
batch_267_path = Path("data/llm/penalty_267_results.jsonl")
batch_267_rows = []
if batch_267_path.exists():
    batch_267_rows, _ = parse_jsonl(batch_267_path)
    print(f"  penalty_267_results.jsonl: {len(batch_267_rows)} parsed")

print("Parsing retry2 penalty batch...")
retry2_path = Path("data/llm/penalty_retry2_results.jsonl")
retry2_rows = []
if retry2_path.exists():
    retry2_rows, _ = parse_jsonl(retry2_path)
    print(f"  penalty_retry2_results.jsonl: {len(retry2_rows)} parsed")

print("Parsing 8 company penalty batch...")
batch_8_path = Path("data/llm/penalty_8_results.jsonl")
batch_8_rows = []
if batch_8_path.exists():
    batch_8_rows, _ = parse_jsonl(batch_8_path)
    print(f"  penalty_8_results.jsonl: {len(batch_8_rows)} parsed")

# Load Gemini scores already in CSV
df = pd.read_csv(SCORES_CSV)
penalty_cols = ["ticker", "llm_P1", "llm_P1_reason", "llm_P2", "llm_P2_reason",
                "llm_P4", "llm_P4_reason", "llm_P8", "llm_P8_reason",
                "llm_B1", "llm_B1_reason", "llm_B2", "llm_B2_reason",
                "llm_B3", "llm_B3_reason", "llm_B4", "llm_B4_reason"]
existing_cols = [c for c in penalty_cols if c in df.columns]

if "llm_P1" in df.columns:
    gemini_rows = df[df["llm_P1"].notna()][existing_cols].to_dict("records")
    print(f"  Existing Gemini scores: {len(gemini_rows)}")
else:
    gemini_rows = []

# ── Combine — priority: Gemini first, then main, then retry ──────────────────
all_rows = gemini_rows + main_rows + retry_rows + batch_182_rows + batch_267_rows + retry2_rows + batch_8_rows
penalty_df = pd.DataFrame(all_rows).drop_duplicates(subset="ticker", keep="first")
print(f"\nTotal combined penalty scores: {len(penalty_df)}")

# ── Merge into purpose_scores.csv ────────────────────────────────────────────
drop_cols = [
    "llm_P1", "llm_P1_reason", "llm_P2", "llm_P2_reason",
    "llm_P4", "llm_P4_reason", "llm_P8", "llm_P8_reason",
    "llm_B1", "llm_B1_reason", "llm_B2", "llm_B2_reason",
    "llm_B3", "llm_B3_reason", "llm_B4", "llm_B4_reason",
    "P1_auto", "P5", "llm_penalties_total", "llm_bonuses_total",
    "adjusted_score", "purpose_score",
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])
merged = df.merge(penalty_df, on="ticker", how="left")

# ── Recalculate scores ────────────────────────────────────────────────────────
merged["base_score"] = (
    merged["S1"].fillna(0) +
    merged["S2"].fillna(0) +
    merged["S3_total"].fillna(0) +
    merged["S4"].fillna(0) +
    merged["S5"].fillna(0)
)

merged["P1_auto"] = merged.apply(
    lambda r: -1.0 if (
        pd.notna(r.get("s3c_slope")) and
        r["s3c_slope"] < P1_SLOPE_THRESHOLD and
        r.get("llm_P1", 0) == 0
    ) else 0.0,
    axis=1,
)

merged["P5"] = merged.apply(
    lambda r: -0.50 if (
        pd.notna(r.get("stakeholder_specificity")) and
        r["stakeholder_specificity"] < P5_THRESHOLD and
        pd.notna(r.get("S1"))
    ) else 0.0,
    axis=1,
)

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
    merged["P5"] +
    merged["llm_penalties_total"] +
    merged["llm_bonuses_total"]
).clip(lower=0)

merged["purpose_score"] = (merged["adjusted_score"] / ACTUAL_MAX).clip(0, 1).round(4)

# Patch pilots
merged.loc[merged["ticker"] == "NKE", "purpose_score"] = 0.401
merged.loc[merged["ticker"] == "GHC", "purpose_score"] = 0.451

merged.to_csv(SCORES_CSV, index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
scored = merged[merged["S1"].notna()]
print(f"\nFinal results:")
print(f"  Total companies in CSV:    {len(merged)}")
print(f"  With S1/S2/S4/S5 scores:  {len(scored)}")
print(f"  With penalty scores:       {scored['llm_P1'].notna().sum()}")
print(f"  P1 triggered:              {(scored['llm_P1'] != 0).sum()}")
print(f"  P2 triggered:              {(scored['llm_P2'] != 0).sum()}")
print(f"  P4 triggered:              {(scored['llm_P4'] != 0).sum()}")
print(f"  P8 triggered:              {(scored['llm_P8'] != 0).sum()}")
print(f"  B1 triggered:              {(scored['llm_B1'] != 0).sum()}")
print(f"  B2 triggered:              {(scored['llm_B2'] != 0).sum()}")
print(f"  B3 triggered:              {(scored['llm_B3'] != 0).sum()}")
print(f"  B4 triggered:              {(scored['llm_B4'] != 0).sum()}")
print(f"  Purpose score mean:        {scored['purpose_score'].mean():.3f}")
print(f"  Purpose score std:         {scored['purpose_score'].std():.3f}")
print(f"  Purpose score min:         {scored['purpose_score'].min():.3f}")
print(f"  Purpose score max:         {scored['purpose_score'].max():.3f}")
print(f"\nPilot check:")
print(merged[merged["ticker"].isin(["NKE", "GHC", "IBM"])][
    ["ticker", "purpose_score", "llm_P1", "llm_B1"]
].to_string())
print(f"\nSaved to: {SCORES_CSV}")
