"""
PAN Group — Purpose Lexicon v3
BA298A Capstone | UC Irvine | 2026

Two-layer architecture:
  Layer 1 — Loughran-McDonald (2011) — loaded from CSV at runtime
  Layer 2 — PAN Purpose Extension — four theory-grounded categories

Where extension scores feed into the framework:
  stakeholder_specificity  → S5 (Stakeholder Integration) pre-score input
  accountability_ratio     → S4 (Structural Embedding) pre-score input
  temporal_commitment      → S2 (Longitudinal Consistency) quantitative
                             input to Gemini prompt — measures whether
                             purpose language is framed durably across years
  adversity_resilience     → Penalty/bonus layer — triggers B4 (purpose
                             cited in adversity) and P4 (purpose contradicted
                             by actions) review

Citation map:
  [LM11]  Loughran & McDonald (2011). Journal of Finance, 66(1), 35-65.
  [F84]   Freeman (1984). Strategic Management: A Stakeholder Approach.
  [EIS14] Eccles, Ioannou & Serafeim (2014). Management Science, 60(11), 2835-2857.
  [HVS15] Henderson & Van den Steen (2015). American Economic Review, 105(5), 326-330.
"""

import re


# ── Shared anchor sets ───────────────────────────────────────────────────────
# Used for co-occurrence filtering in both stakeholder and accountability
# scoring. An anchor word must appear within WINDOW tokens of a scored term
# for that term to count. Prevents financial/legal boilerplate from inflating
# scores in sections of the 10-K that are not about purpose.

PURPOSE_ANCHORS = {
    "mission", "purpose", "values", "commitment", "dedicated", "dedicate",
    "serve", "serving", "impact", "benefit", "empower", "enabling",
    "believe", "vision", "principle", "principles",
}

WINDOW = 50   # tokens either side — tunable at calibration stage


def _has_nearby_anchor(idx, tokens, window=WINDOW):
    start = max(0, idx - window)
    end   = min(len(tokens), idx + window)
    return any(t in PURPOSE_ANCHORS for t in tokens[start:end])


# ── CATEGORY A: Stakeholder Specificity ─────────────────────────────────────
# Rationale: Freeman (1984) — genuine stakeholder orientation names specific
# groups with specific obligations, not generic placeholders.
# Co-occurrence filter applied: stakeholder term only counted if a purpose
# anchor appears within WINDOW tokens. [F84]

STAKEHOLDER_SPECIFIC = {
    # Employee groups
    "employees", "workers", "workforce", "associates", "colleagues",
    # Customer/patient groups — domain-specific only
    "customers", "patients", "consumers", "athletes", "subscribers", "students",
    # Community/societal groups
    "communities", "neighborhoods", "residents",
    "suppliers", "farmers", "artisans", "vendors",
    # Environmental subjects
    "planet", "ecosystem", "biodiversity", "oceans", "forests",
}

STAKEHOLDER_GENERIC = {
    "stakeholders", "constituents", "parties", "society", "everyone",
}


def score_stakeholder_specificity(tokens):
    """
    Ratio of specific to (specific + generic) stakeholder terms.
    Specific terms filtered by purpose anchor co-occurrence. [F84]
    """
    n_specific = sum(
        1 for i, t in enumerate(tokens)
        if t in STAKEHOLDER_SPECIFIC and _has_nearby_anchor(i, tokens)
    )
    n_generic = sum(1 for t in tokens if t in STAKEHOLDER_GENERIC)
    total = n_specific + n_generic
    if total == 0:
        return 0.0
    return round(n_specific / total, 3)


# ── CATEGORY B: Accountability Language ─────────────────────────────────────
# Rationale: Eccles et al. (2014) — high-sustainability companies embed
# measurable accountability mechanisms tied to purpose outcomes, not just
# financial/legal reporting requirements. [EIS14]
#
# Fix applied: split ACCOUNTABILITY into two sub-lists.
#   ACCOUNTABILITY_PURPOSE: terms that only count near a purpose anchor —
#     "report", "disclose", "transparency" appear constantly in SEC filings
#     as legal obligations, not purpose operationalisation. Co-occurrence
#     filter required to distinguish purpose-linked reporting from compliance.
#   ACCOUNTABILITY_STRONG: terms unambiguously tied to purpose measurement —
#     "measure our impact", "track progress", "accountable to communities"
#     These are specific enough that false positives are rare even without filter.

ACCOUNTABILITY_STRONG = {
    # Unambiguously purpose-linked — no co-occurrence filter needed
    "accountable", "accountability",
    "measure", "measured", "measuring",
    "track", "tracked", "tracking",
    "monitor", "monitored", "monitoring",
    "audit", "audits", "audited", "auditing",
    "verify", "verifies", "verified", "verifying",
    "target", "targets", "targeted",
    "milestone", "milestones",
    "achieve", "achieves", "achieved", "achieving",
    "attain", "attains", "attained",
}

ACCOUNTABILITY_FILTERED = {
    # Common in SEC boilerplate — only count near purpose anchor
    "report", "reports", "reported", "reporting",
    "disclose", "disclosed", "disclosing", "disclosure",
    "transparent", "transparency",
    "goal", "goals",
    "deliver", "delivers", "delivered", "delivering",
}

ASPIRATION = {
    # Commitment stated but unmeasurable [EIS14, LM11]
    "aspire", "aspires", "aspiring", "aspiration",
    "strive", "strives", "striving",
    "endeavor", "endeavors", "endeavoring",
    "hope", "hopes", "hoping",
    "intend", "intends", "intention",
    "seek", "seeks", "seeking",
    "journey",
}


def score_accountability(tokens):
    """
    Ratio of accountability to (accountability + aspiration) terms.
    ACCOUNTABILITY_STRONG counts unconditionally.
    ACCOUNTABILITY_FILTERED counts only near a purpose anchor.
    Returns 0-1. Higher = more measurable commitment. [EIS14]
    """
    n_strong = sum(1 for t in tokens if t in ACCOUNTABILITY_STRONG)

    n_filtered = sum(
        1 for i, t in enumerate(tokens)
        if t in ACCOUNTABILITY_FILTERED and _has_nearby_anchor(i, tokens)
    )

    n_acc = n_strong + n_filtered
    n_asp = sum(1 for t in tokens if t in ASPIRATION)
    total = n_acc + n_asp
    if total == 0:
        return 0.0
    return round(n_acc / total, 3)


# ── CATEGORY C: Temporal Commitment ─────────────────────────────────────────
# Rationale: Henderson & Van den Steen (2015) — genuine purpose functions as
# a long-term identity carrier. Short-term framing in purpose language signals
# financial communication rather than strategic anchor.
#
# Framework mapping: feeds into S2 (Longitudinal Consistency) as quantitative
# input to Gemini prompt. Measures whether purpose language is framed durably
# across the 2015-2020 window. Not mapped to P5 (that is single-stakeholder
# framing, a different construct). [HVS15]
#
# Note: "sustainability", "sustainable" deliberately excluded — ESG/compliance
# boilerplate that inflates scores for companies meeting disclosure requirements.
# Hyphenated terms ("long-term", "near-term", "year-over-year") confirmed to
# tokenize correctly under regex r"[a-z]+(?:-[a-z]+)*". [HVS15]

LONG_TERM = {
    "enduring", "endure", "endures",
    "lasting",
    "permanent", "permanence",
    "generational", "generation", "generations",
    "legacy", "legacies",
    "decade", "decades",
    "perpetual", "perpetually",
    "foundational", "foundation",
    "durable", "durability",
    "embedded",
    "long-term",
    "ongoing",
}

SHORT_TERM = {
    "quarterly", "quarter",
    "near-term",
    "short-term",
    "annually",
    "immediate", "immediately",
    "year-over-year",
}


def score_temporal_commitment(tokens):
    """
    Ratio of long-term to (long-term + short-term) framing terms.
    Returns 0-1. Neutral 0.5 when no temporal signal present. [HVS15]
    """
    n_long  = sum(1 for t in tokens if t in LONG_TERM)
    n_short = sum(1 for t in tokens if t in SHORT_TERM)
    total = n_long + n_short
    if total == 0:
        return 0.5
    return round(n_long / total, 3)


# ── CATEGORY D: Adversity Resilience ────────────────────────────────────────
# Rationale: Eccles et al. (2014) — genuine purpose is maintained under
# financial stress. Performative purpose softens under pressure. [EIS14]
#
# Design: hard gate — both ADVERSITY_CONTEXT and MAINTAINED_COMMITMENT must
# be present or score is zero. A filing that only mentions challenges without
# maintained commitment does not score positively.
#
# Fix applied: removed "continued" and "continuing" from MAINTAINED_COMMITMENT.
# These appear constantly in 10-K boilerplate ("continued operations",
# "continued growth", "continued investment") and would fire the gate
# prematurely with any adversity language in the same passage.
# "Reaffirm", "upheld", "unwavering", "doubled", "deepened" are retained —
# these are specific enough to carry the signal without "continued". [EIS14]

ADVERSITY_CONTEXT = {
    "despite", "notwithstanding",
    "challenging", "challenges", "challenge",
    "difficult", "difficulty",
    "volatile", "volatility",
    "crisis", "downturn", "disruption", "disruptions",
    "headwinds", "pressure", "pressures",
}

MAINTAINED_COMMITMENT = {
    # Removed: "continued", "continuing" — too common in boilerplate [EIS14]
    "nonetheless", "nevertheless",
    "maintained", "maintain",
    "upheld", "uphold",
    "remained", "remain",
    "unwavering", "undiminished",
    "reaffirm", "reaffirmed", "reaffirms",
    "accelerated", "accelerate",
    "doubled", "deepened",
}

HEDGE_UNDER_PRESSURE = {
    "pausing", "pause", "temporarily",
    "contingent", "dependent",
    "reassessing", "reassess",
    "reevaluating", "revisiting",
    "adjusting",
    "prioritizing",
}


def score_adversity_resilience(tokens):
    """
    Gate: both adversity context AND maintained commitment must be present.
    Score: ratio of maintained_commitment / (adversity + maintained),
    penalised by hedge density. No magic number scaling. [EIS14]
    """
    n_adversity  = sum(1 for t in tokens if t in ADVERSITY_CONTEXT)
    n_maintained = sum(1 for t in tokens if t in MAINTAINED_COMMITMENT)
    n_hedge      = sum(1 for t in tokens if t in HEDGE_UNDER_PRESSURE)
    total        = len(tokens)

    if n_adversity == 0 or n_maintained == 0:
        return 0.0

    resilience_ratio = n_maintained / (n_adversity + n_maintained)
    hedge_density    = n_hedge / total if total > 0 else 0
    score = resilience_ratio - hedge_density

    return round(max(0.0, min(score, 1.0)), 3)


# ── Master Scoring Function ──────────────────────────────────────────────────

def score_all_extension(passage_text):
    """
    Call after score_passage_lm() for the same passage.
    Returns all four extension scores plus raw counts for audit trail.
    """
    tokens = re.findall(r"[a-z]+(?:-[a-z]+)*", passage_text.lower())

    return {
        "stakeholder_specificity": score_stakeholder_specificity(tokens),
        "accountability_ratio":    score_accountability(tokens),
        "temporal_commitment":     score_temporal_commitment(tokens),
        "adversity_resilience":    score_adversity_resilience(tokens),

        "n_stakeholder_specific":  sum(1 for t in tokens if t in STAKEHOLDER_SPECIFIC),
        "n_stakeholder_generic":   sum(1 for t in tokens if t in STAKEHOLDER_GENERIC),
        "n_accountability_strong": sum(1 for t in tokens if t in ACCOUNTABILITY_STRONG),
        "n_accountability_filter": sum(
            1 for i, t in enumerate(tokens)
            if t in ACCOUNTABILITY_FILTERED and _has_nearby_anchor(i, tokens)
        ),
        "n_aspiration":            sum(1 for t in tokens if t in ASPIRATION),
        "n_long_term":             sum(1 for t in tokens if t in LONG_TERM),
        "n_short_term":            sum(1 for t in tokens if t in SHORT_TERM),
        "n_adversity_context":     sum(1 for t in tokens if t in ADVERSITY_CONTEXT),
        "n_maintained_commitment": sum(1 for t in tokens if t in MAINTAINED_COMMITMENT),
        "n_hedge_under_pressure":  sum(1 for t in tokens if t in HEDGE_UNDER_PRESSURE),

        "stakeholder_specific_found":  [t for t in tokens if t in STAKEHOLDER_SPECIFIC],
        "accountability_strong_found": [t for t in tokens if t in ACCOUNTABILITY_STRONG],
        "accountability_filter_found": [
            t for i, t in enumerate(tokens)
            if t in ACCOUNTABILITY_FILTERED and _has_nearby_anchor(i, tokens)
        ],
        "long_term_found":             [t for t in tokens if t in LONG_TERM],
        "maintained_commitment_found": [t for t in tokens if t in MAINTAINED_COMMITMENT],
        "hedge_found":                 [t for t in tokens if t in HEDGE_UNDER_PRESSURE],
    }


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":

    ibm_passage = """
    At IBM, we are dedicated to our clients' success and to the world's progress.
    We measure our impact by the outcomes we deliver for communities and employees.
    We track progress toward our sustainability goals and report transparently on
    our commitment to workforce investment. Despite challenging market conditions,
    we maintained our dedication to equitable access to technology. Our foundational
    purpose remains embedded in every long-term decision we make.
    """

    nike_passage = """
    We aim to bring inspiration and innovation to every athlete in the world.
    We believe in the power of sport to move the world forward. We strive to
    create products that make athletes better. We hope to continue growing our
    business while serving our stakeholders. This quarterly performance reflects
    our near-term strategic priorities. Revenue was reported to shareholders.
    We disclosed our financial results in compliance with SEC requirements.
    """

    adversity_passage = """
    Despite significant headwinds and a challenging economic environment,
    we reaffirmed our mission-driven commitment to our employees and communities.
    We maintained our dedication to workforce development and upheld our values
    even as market conditions remained difficult. Our purpose remained unwavering.
    """

    boilerplate_passage = """
    We reported our financial results to shareholders in compliance with SEC
    requirements. We disclosed material information and continued operations
    across all business segments. Revenue growth continued year-over-year.
    The company reported earnings consistent with quarterly guidance.
    """

    for label, passage in [
        ("IBM", ibm_passage),
        ("Nike", nike_passage),
        ("Adversity (should score high on resilience)", adversity_passage),
        ("Boilerplate (should score low overall)", boilerplate_passage),
    ]:
        scores = score_all_extension(passage)
        print(f"\n=== {label} ===")
        for k in ["stakeholder_specificity", "accountability_ratio",
                  "temporal_commitment", "adversity_resilience"]:
            print(f"  {k}: {scores[k]}")
        print(f"  accountability_strong_found: {scores['accountability_strong_found']}")
        print(f"  accountability_filter_found: {scores['accountability_filter_found']}")
        print(f"  maintained_commitment_found: {scores['maintained_commitment_found']}")


CITATIONS = {
    "LM11":  "Loughran & McDonald (2011). Journal of Finance, 66(1), 35-65.",
    "F84":   "Freeman (1984). Strategic Management: A Stakeholder Approach. Pitman.",
    "EIS14": "Eccles, Ioannou & Serafeim (2014). Management Science, 60(11), 2835-2857.",
    "HVS15": "Henderson & Van den Steen (2015). American Economic Review, 105(5), 326-330.",
}
