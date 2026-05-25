# pan_analysis.py
# PAN Group — Full Correlation Analysis Suite
# Run from the same directory as master_dataset.csv

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for Cursor terminal
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import kruskal, chi2_contingency, pearsonr, spearmanr
import warnings
warnings.filterwarnings("ignore")

# ── Output directory ──────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("master_dataset.csv")

# Normalise column names defensively — strip whitespace
df.columns = df.columns.str.strip()

print(f"Dataset loaded: {len(df)} companies, {df['GICS Sector'].nunique()} sectors")
print(f"Columns: {list(df.columns)}\n")


# =============================================================================
# ANALYSIS 1 — Sector-stratified correlations
# =============================================================================
# Why: The pooled r=0.12 averages across all GICS sectors.
# Structural differences between sectors (Energy vs Healthcare) make that
# number meaningless as a standalone. Stratifying reveals WHERE the thesis
# holds and where it breaks — which is the actual finding for the professor.
# We run both Pearson (parametric, assumes linearity) and Spearman
# (rank-based, no linearity assumption) because with n per sector often <100,
# distribution assumptions are fragile.

print("=" * 60)
print("ANALYSIS 1: Sector-stratified correlations")
print("=" * 60)

sector_results = []
min_n = 10  # exclude sectors with too few companies to be meaningful

for sector, group in df.groupby("GICS Sector"):
    if len(group) < min_n:
        continue
    n = len(group)
    # Pearson — tests linear relationship
    pearson_r, pearson_p = pearsonr(group["purpose_score"], group["performance_score"])
    # Spearman — tests monotonic relationship (more robust to outliers)
    spearman_r, spearman_p = spearmanr(group["purpose_score"], group["performance_score"])

    sector_results.append({
        "sector": sector,
        "n": n,
        "pearson_r": round(pearson_r, 4),
        "pearson_p": round(pearson_p, 4),
        "pearson_sig": "***" if pearson_p < 0.001 else "**" if pearson_p < 0.01 else "*" if pearson_p < 0.05 else "",
        "spearman_r": round(spearman_r, 4),
        "spearman_p": round(spearman_p, 4),
        "spearman_sig": "***" if spearman_p < 0.001 else "**" if spearman_p < 0.01 else "*" if spearman_p < 0.05 else "",
    })

sector_df = pd.DataFrame(sector_results).sort_values("pearson_r", ascending=False)
sector_df.to_csv("results/01_sector_correlations.csv", index=False)
print(sector_df[["sector", "n", "pearson_r", "pearson_sig", "spearman_r", "spearman_sig"]].to_string(index=False))

# Plot: sector correlation heatmap (sorted by pearson_r)
fig, ax = plt.subplots(figsize=(10, max(5, len(sector_df) * 0.45)))
colors = ["#c0392b" if r < 0 else "#27ae60" if r > 0.2 else "#f39c12" for r in sector_df["pearson_r"]]
bars = ax.barh(sector_df["sector"], sector_df["pearson_r"], color=colors, alpha=0.8, height=0.6)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.axvline(0.12, color="gray", linewidth=0.8, linestyle=":", label="Pooled r=0.12")
ax.set_xlabel("Pearson r (Purpose vs Performance)", fontsize=11)
ax.set_title("Within-Sector Correlations: Purpose vs Performance\nPAN Group — Sector-Stratified Analysis", fontsize=12)
for i, (r, sig, n) in enumerate(zip(sector_df["pearson_r"], sector_df["pearson_sig"], sector_df["n"])):
    ax.text(r + (0.005 if r >= 0 else -0.005), i, f"{sig} (n={n})",
            va="center", ha="left" if r >= 0 else "right", fontsize=8)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("results/01_sector_correlations.png", dpi=150)
plt.close()
print("\nSaved: results/01_sector_correlations.csv + .png\n")


# =============================================================================
# ANALYSIS 2 — Quadrant distribution + chi-square independence test
# =============================================================================
# Why: The investment thesis predicts the bottom-right quadrant (high purpose,
# low performance) should be sparse. A chi-square test tells us whether the
# quadrant distribution is significantly different from what you'd expect if
# purpose and performance were completely independent.
# If chi-square is significant → the quadrant distribution is non-random →
# purpose and performance are associated in a way that shows up in the 2x2 table.

print("=" * 60)
print("ANALYSIS 2: Quadrant distribution + chi-square test")
print("=" * 60)

quadrant_counts = df["quadrant"].value_counts()
print("Quadrant distribution:")
print(quadrant_counts)
print(f"\nTotal: {quadrant_counts.sum()}")

# Build 2x2 contingency table
# High purpose = purpose_score >= 0.5, High performance = performance_score >= 0.5
df["high_purpose"] = df["purpose_score"] >= 0.5
df["high_performance"] = df["performance_score"] >= 0.5

contingency = pd.crosstab(df["high_purpose"], df["high_performance"],
                           rownames=["High Purpose"], colnames=["High Performance"])
print("\n2x2 Contingency Table:")
print(contingency)

chi2, p_chi2, dof, expected = chi2_contingency(contingency)
print(f"\nChi-square statistic: {chi2:.4f}")
print(f"p-value: {p_chi2:.6f}")
print(f"Degrees of freedom: {dof}")
print(f"\nExpected counts under independence:")
print(pd.DataFrame(expected, index=contingency.index, columns=contingency.columns).round(1))

# What % of companies are in each quadrant?
total = len(df)
for quad, count in quadrant_counts.items():
    print(f"  {quad}: {count} ({count/total*100:.1f}%)")

quadrant_counts.to_csv("results/02_quadrant_distribution.csv")

# Plot: quadrant pie or bar
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: actual distribution
colors_q = {"Purpose Leaders": "#27ae60", "Value Extraction": "#c0392b",
             "Purpose Promise": "#2980b9", "Dual Weakness": "#95a5a6"}
q_labels = list(quadrant_counts.index)
q_vals = list(quadrant_counts.values)
q_colors = [colors_q.get(q, "#bdc3c7") for q in q_labels]
axes[0].bar(q_labels, q_vals, color=q_colors, alpha=0.85, edgecolor="white")
axes[0].set_title("Actual Quadrant Distribution", fontsize=11)
axes[0].set_ylabel("Number of Companies")
for i, v in enumerate(q_vals):
    axes[0].text(i, v + 2, f"{v}\n({v/total*100:.0f}%)", ha="center", fontsize=9)
axes[0].tick_params(axis="x", labelrotation=20)

# Right: actual vs expected from chi-square
expected_flat = expected.flatten()
actual_flat = contingency.values.flatten()
labels_2x2 = ["Low P, Low Perf", "Low P, High Perf", "High P, Low Perf", "High P, High Perf"]
x = np.arange(len(labels_2x2))
w = 0.35
axes[1].bar(x - w/2, actual_flat, w, label="Actual", color="#2980b9", alpha=0.8)
axes[1].bar(x + w/2, expected_flat, w, label="Expected (independence)", color="#e67e22", alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels_2x2, rotation=15, fontsize=8)
axes[1].set_title(f"Actual vs Expected: χ²={chi2:.2f}, p={p_chi2:.4f}", fontsize=11)
axes[1].set_ylabel("Number of Companies")
axes[1].legend()

plt.suptitle("PAN Group — Quadrant Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("results/02_quadrant_analysis.png", dpi=150)
plt.close()
print("\nSaved: results/02_quadrant_distribution.csv + 02_quadrant_analysis.png\n")


# =============================================================================
# ANALYSIS 3 — Band-level analysis (Kruskal-Wallis + pairwise)
# =============================================================================
# Why: Pearson r treats the scores as continuous and linear. But your framework
# is fundamentally band-based (5 bands). Kruskal-Wallis asks: does mean
# performance score differ significantly across purpose bands?
# It's the non-parametric ANOVA equivalent — no normality assumption needed.
# If significant, pairwise Mann-Whitney U tests show WHICH bands differ.

print("=" * 60)
print("ANALYSIS 3: Band-level Kruskal-Wallis + pairwise comparisons")
print("=" * 60)


actual_bands = df["purpose_band"].unique()
print(f"Purpose bands in data: {sorted(actual_bands)}")

# Sort bands by their numeric prefix so they display low → high
band_order = sorted(actual_bands, key=lambda x: float(x.split("-")[0].strip()))

band_groups = {band: df[df["purpose_band"] == band]["performance_score"].dropna()
               for band in band_order}
band_groups = {k: v for k, v in band_groups.items() if len(v) >= 5}

# Band summary stats
band_summary = []
for band, scores in band_groups.items():
    band_summary.append({
        "purpose_band": band,
        "n": len(scores),
        "mean_performance": round(scores.mean(), 4),
        "median_performance": round(scores.median(), 4),
        "std": round(scores.std(), 4),
    })
band_summary_df = pd.DataFrame(band_summary)
print("\nBand summary:")
print(band_summary_df.to_string(index=False))

# Kruskal-Wallis test
if len(band_groups) >= 2:
    kw_stat, kw_p = kruskal(*band_groups.values())
    print(f"\nKruskal-Wallis H={kw_stat:.4f}, p={kw_p:.6f}")
    print("Interpretation:", "Significant — performance differs across purpose bands" if kw_p < 0.05
          else "Not significant — no evidence bands differ in performance")

    # Pairwise Mann-Whitney U (post-hoc) — Bonferroni corrected
    band_keys = list(band_groups.keys())
    pairwise_results = []
    n_comparisons = len(band_keys) * (len(band_keys) - 1) // 2
    for i in range(len(band_keys)):
        for j in range(i + 1, len(band_keys)):
            a, b = band_keys[i], band_keys[j]
            stat, p = stats.mannwhitneyu(band_groups[a], band_groups[b], alternative="two-sided")
            p_bonf = min(p * n_comparisons, 1.0)  # Bonferroni correction
            pairwise_results.append({
                "band_a": a, "band_b": b,
                "U_stat": round(stat, 2),
                "p_raw": round(p, 5),
                "p_bonferroni": round(p_bonf, 5),
                "significant": p_bonf < 0.05,
            })
    pairwise_df = pd.DataFrame(pairwise_results)
    print("\nPairwise Mann-Whitney U (Bonferroni corrected):")
    print(pairwise_df.to_string(index=False))
    pairwise_df.to_csv("results/03_pairwise_comparisons.csv", index=False)

band_summary_df.to_csv("results/03_band_summary.csv", index=False)

# Plot: boxplot of performance by purpose band
existing_bands = [b for b in band_order if b in band_groups]
fig, ax = plt.subplots(figsize=(10, 5))
data_to_plot = [band_groups[b].values for b in existing_bands]
bp = ax.boxplot(data_to_plot, labels=existing_bands, patch_artist=True, notch=False,
                medianprops={"color": "white", "linewidth": 2})
palette = ["#c0392b", "#e67e22", "#f1c40f", "#27ae60", "#1abc9c"]
for patch, color in zip(bp["boxes"], palette[:len(existing_bands)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Overlay mean dots
for i, band in enumerate(existing_bands):
    mean_val = band_groups[band].mean()
    ax.scatter(i + 1, mean_val, color="white", zorder=5, s=50, marker="D")

ax.set_xlabel("Purpose Score Band", fontsize=11)
ax.set_ylabel("Performance Score", fontsize=11)
ax.set_title(f"Performance Score Distribution by Purpose Band\nKruskal-Wallis H={kw_stat:.2f}, p={kw_p:.4f}",
             fontsize=12)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig("results/03_band_boxplot.png", dpi=150)
plt.close()
print("\nSaved: results/03_band_summary.csv + pairwise + 03_band_boxplot.png\n")


# =============================================================================
# ANALYSIS 4 — Sub-score correlations (S1–S5 vs performance)
# =============================================================================
# Why: The composite purpose score hides WHICH dimension of purpose predicts
# performance. If S4 (structural embedding) has r=0.25 but S1 (mission clarity)
# has r=0.01, that tells you FAR more than the aggregate r=0.12.
# This is the finding the professor will be most interested in — it connects
# specific theoretical constructs to outcomes.

print("=" * 60)
print("ANALYSIS 4: Sub-score correlations (S1–S5 vs performance)")
print("=" * 60)

sub_scores = ["S1", "S2", "S3_total", "S4", "S5"]
sub_score_labels = {
    "S1": "Mission Clarity",
    "S2": "Longitudinal Consistency",
    "S3_total": "NLP Signals",
    "S4": "Structural Embedding",
    "S5": "Stakeholder Integration",
}

sub_results = []
for col in sub_scores:
    if col not in df.columns:
        print(f"  WARNING: {col} not found in dataset, skipping")
        continue
    clean = df[["performance_score", col]].dropna()
    n = len(clean)
    pr, pp = pearsonr(clean["performance_score"], clean[col])
    sr, sp = spearmanr(clean["performance_score"], clean[col])
    sub_results.append({
        "sub_score": col,
        "label": sub_score_labels.get(col, col),
        "n": n,
        "pearson_r": round(pr, 4),
        "pearson_p": round(pp, 5),
        "pearson_sig": "***" if pp < 0.001 else "**" if pp < 0.01 else "*" if pp < 0.05 else "ns",
        "spearman_r": round(sr, 4),
        "spearman_p": round(sp, 5),
    })

sub_df = pd.DataFrame(sub_results).sort_values("pearson_r", ascending=False)
sub_df.to_csv("results/04_subscore_correlations.csv", index=False)
print(sub_df[["label", "n", "pearson_r", "pearson_sig", "spearman_r"]].to_string(index=False))

# Plot: horizontal bar chart of sub-score correlations
fig, ax = plt.subplots(figsize=(9, 4))
colors_sub = ["#27ae60" if r > 0 else "#c0392b" for r in sub_df["pearson_r"]]
bars = ax.barh(sub_df["label"], sub_df["pearson_r"], color=colors_sub, alpha=0.8, height=0.5)
ax.axvline(0, color="black", linewidth=0.8)
ax.axvline(0.12, color="gray", linewidth=0.8, linestyle=":", label="Composite r=0.12")
for i, (r, sig) in enumerate(zip(sub_df["pearson_r"], sub_df["pearson_sig"])):
    ax.text(r + (0.003 if r >= 0 else -0.003), i, sig,
            va="center", ha="left" if r >= 0 else "right", fontsize=10, fontweight="bold")
ax.set_xlabel("Pearson r vs Performance Score", fontsize=11)
ax.set_title("Which Dimension of Purpose Predicts Performance?\nSub-Score Correlations — PAN Group", fontsize=12)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("results/04_subscore_correlations.png", dpi=150)
plt.close()
print("\nSaved: results/04_subscore_correlations.csv + .png\n")


# =============================================================================
# ANALYSIS 5 — Value extraction flag analysis
# =============================================================================
# Why: The value_extraction_flag is your key investment signal — companies with
# high performance but zero purpose operationalisation. Understanding which
# sectors they cluster in, and what their sub-score profiles look like,
# gives the client a concrete screening tool.

print("=" * 60)
print("ANALYSIS 5: Value extraction flag analysis")
print("=" * 60)

flagged = df[df["value_extraction_flag"] == 1].copy()
unflagged = df[df["value_extraction_flag"] == 0].copy()

print(f"Total value extraction flags: {len(flagged)} ({len(flagged)/len(df)*100:.1f}% of dataset)")
print(f"\nFlagged company stats:")
print(f"  Mean purpose score:     {flagged['purpose_score'].mean():.4f}")
print(f"  Mean performance score: {flagged['performance_score'].mean():.4f}")
print(f"\nUnflagged company stats:")
print(f"  Mean purpose score:     {unflagged['purpose_score'].mean():.4f}")
print(f"  Mean performance score: {unflagged['performance_score'].mean():.4f}")

# Sector breakdown of flags
sector_flag_rate = (df.groupby("GICS Sector")["value_extraction_flag"]
                    .agg(["sum", "count"])
                    .rename(columns={"sum": "flagged", "count": "total"}))
sector_flag_rate["flag_rate"] = (sector_flag_rate["flagged"] / sector_flag_rate["total"]).round(4)
sector_flag_rate = sector_flag_rate.sort_values("flag_rate", ascending=False)
print("\nValue extraction flag rate by sector:")
print(sector_flag_rate.to_string())
sector_flag_rate.to_csv("results/05_value_extraction_by_sector.csv")

# Sub-score profile of flagged vs unflagged
print("\nSub-score profile — flagged vs unflagged:")
for col in sub_scores:
    if col in df.columns:
        f_mean = flagged[col].mean()
        u_mean = unflagged[col].mean()
        print(f"  {col}: flagged={f_mean:.4f}, unflagged={u_mean:.4f}, diff={f_mean-u_mean:+.4f}")

# Export flagged companies
flagged_export = flagged[["ticker", "Company Name", "GICS Sector",
                           "purpose_score", "performance_score"] + sub_scores].copy()
flagged_export = flagged_export.sort_values("performance_score", ascending=False)
flagged_export.to_csv("results/05_flagged_companies.csv", index=False)

# Plot: sector flag rate + sub-score profile side by side
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: sector flag rate
top_sectors = sector_flag_rate[sector_flag_rate["flagged"] > 0].head(12)
axes[0].barh(top_sectors.index, top_sectors["flag_rate"] * 100,
             color="#c0392b", alpha=0.75, height=0.6)
axes[0].set_xlabel("Value Extraction Flag Rate (%)", fontsize=10)
axes[0].set_title("Sectors with Highest Value Extraction Flag Rate", fontsize=11)
for i, (rate, n) in enumerate(zip(top_sectors["flag_rate"], top_sectors["flagged"])):
    axes[0].text(rate * 100 + 0.3, i, f"n={n}", va="center", fontsize=8)

# Right: sub-score comparison
if len(flagged) > 0:
    sub_labels = [sub_score_labels.get(s, s) for s in sub_scores if s in df.columns]
    flagged_means = [flagged[s].mean() for s in sub_scores if s in df.columns]
    unflagged_means = [unflagged[s].mean() for s in sub_scores if s in df.columns]
    x = np.arange(len(sub_labels))
    w = 0.35
    axes[1].bar(x - w/2, flagged_means, w, label="Value Extraction (flagged)", color="#c0392b", alpha=0.8)
    axes[1].bar(x + w/2, unflagged_means, w, label="Rest of dataset", color="#2980b9", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(sub_labels, rotation=20, ha="right", fontsize=8)
    axes[1].set_ylabel("Mean Sub-Score")
    axes[1].set_title("Purpose Sub-Score Profile\nValue Extraction vs Rest", fontsize=11)
    axes[1].legend(fontsize=8)

plt.suptitle("PAN Group — Value Extraction Flag Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("results/05_value_extraction.png", dpi=150)
plt.close()
print("\nSaved: results/05_value_extraction_by_sector.csv + 05_flagged_companies.csv + .png\n")


# =============================================================================
# SUMMARY TABLE — all key numbers in one place
# =============================================================================
print("=" * 60)
print("SUMMARY: Key findings")
print("=" * 60)
print(f"  Dataset: n={len(df)}, sectors={df['GICS Sector'].nunique()}")
print(f"  Pooled Pearson r: 0.1224 (p=0.0002)")
print(f"  Pooled Spearman r: 0.1295")
print(f"  Chi-square (2x2): H={chi2:.4f}, p={p_chi2:.6f}")
print(f"  Kruskal-Wallis across bands: H={kw_stat:.4f}, p={kw_p:.6f}")
print(f"  Value extraction flags: {len(flagged)} ({len(flagged)/len(df)*100:.1f}%)")
print(f"\nAll outputs saved to /results/")
print("  01_sector_correlations.csv + .png")
print("  02_quadrant_distribution.csv + 02_quadrant_analysis.png")
print("  03_band_summary.csv + 03_pairwise_comparisons.csv + 03_band_boxplot.png")
print("  04_subscore_correlations.csv + .png")
print("  05_value_extraction_by_sector.csv + 05_flagged_companies.csv + .png")


# =============================================================================
# ANALYSIS 6 — Sector-stratified sub-score correlations
# =============================================================================
# Why: We know S2 is the only significant sub-score at the aggregate level.
# This checks whether S2 is also driving the strong sector results (e.g.
# Food/Beverage r=0.55), or whether a different dimension explains it there.
# If S2 dominates across strong sectors too, the theoretical story is tight.
# If a different sub-score drives strong sectors, that's a separate finding.

print("=" * 60)
print("ANALYSIS 6: Sector-stratified sub-score correlations")
print("=" * 60)

sub_scores = ["S1", "S2", "S3_total", "S4", "S5"]
sub_labels = {
    "S1": "Mission Clarity",
    "S2": "Longitudinal Consistency",
    "S3_total": "NLP Signals",
    "S4": "Structural Embedding",
    "S5": "Stakeholder Integration",
}

min_n_sector = 15  # only sectors with enough companies to be meaningful
sector_subscore_results = []

for sector, group in df.groupby("GICS Sector"):
    if len(group) < min_n_sector:
        continue
    row = {"sector": sector, "n": len(group)}
    for col in sub_scores:
        if col in group.columns:
            clean = group[["performance_score", col]].dropna()
            if len(clean) < 10:
                row[col + "_r"] = np.nan
                row[col + "_sig"] = ""
                continue
            r, p = pearsonr(clean["performance_score"], clean[col])
            row[col + "_r"] = round(r, 4)
            row[col + "_sig"] = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    sector_subscore_results.append(row)

ss_df = pd.DataFrame(sector_subscore_results)

# Sort by S2 correlation descending — that's our hypothesis leader
ss_df = ss_df.sort_values("S2_r", ascending=False)
ss_df.to_csv("results/06_sector_subscore_correlations.csv", index=False)

# Print readable table
print_cols = ["sector", "n"] + [c + "_r" for c in sub_scores]
print(ss_df[print_cols].to_string(index=False))

# Plot: heatmap — sectors as rows, sub-scores as columns, color = pearson r
r_cols = [c + "_r" for c in sub_scores]
heat_data = ss_df.set_index("sector")[r_cols].copy()
heat_data.columns = [sub_labels[c.replace("_r", "")] for c in r_cols]

fig, ax = plt.subplots(figsize=(11, max(5, len(heat_data) * 0.45)))
sns.heatmap(
    heat_data,
    annot=True, fmt=".2f",
    cmap="RdYlGn", center=0, vmin=-0.5, vmax=0.5,
    linewidths=0.5, linecolor="white",
    ax=ax, cbar_kws={"label": "Pearson r"}
)
ax.set_title("Sector × Purpose Dimension Correlation Heatmap\nPAN Group — Which dimension predicts performance, and where?",
             fontsize=12, pad=12)
ax.set_xlabel("")
ax.set_ylabel("")
ax.tick_params(axis="x", rotation=25, labelsize=9)
ax.tick_params(axis="y", rotation=0, labelsize=9)
plt.tight_layout()
plt.savefig("results/06_sector_subscore_heatmap.png", dpi=150)
plt.close()
print("\nSaved: results/06_sector_subscore_correlations.csv + 06_sector_subscore_heatmap.png\n")


# =============================================================================
# ANALYSIS 7 — OLS regression with sector controls
# =============================================================================
# Why: The pooled r=0.12 is uncontrolled. A sector like Food/Beverage might
# have both high purpose scores AND high performance for structural reasons
# unrelated to purpose (strong brands, pricing power). OLS with sector dummies
# isolates the purpose effect AFTER accounting for sector membership.
# The coefficient on purpose_score in this model is what you actually want —
# it answers: "holding sector constant, does higher purpose predict higher
# performance?" This is the number the professor will ask for.

print("=" * 60)
print("ANALYSIS 7: OLS regression with sector controls")
print("=" * 60)

from statsmodels.formula.api import ols
import statsmodels.api as sm

# Clean column name for formula — statsmodels can't handle spaces
df["GICS_Sector"] = df["GICS Sector"].str.replace(r"[^a-zA-Z0-9]", "_", regex=True)

# Model 1: purpose_score only (baseline — should reproduce r=0.12 story)
model1 = ols("performance_score ~ purpose_score", data=df).fit()

# Model 2: purpose_score + sector dummies (main model)
# C(GICS_Sector) tells statsmodels to treat it as categorical
model2 = ols("performance_score ~ purpose_score + C(GICS_Sector)", data=df).fit()

# Model 3: all sub-scores + sector dummies (which sub-score drives it?)
sub_formula = " + ".join(sub_scores)
model3 = ols(f"performance_score ~ {sub_formula} + C(GICS_Sector)", data=df).fit()

# Print summaries
print("\n--- Model 1: Baseline (no sector controls) ---")
print(f"  purpose_score coef: {model1.params['purpose_score']:.4f}")
print(f"  purpose_score p:    {model1.pvalues['purpose_score']:.5f}")
print(f"  R²: {model1.rsquared:.4f}")

print("\n--- Model 2: Purpose + Sector Controls ---")
print(f"  purpose_score coef: {model2.params['purpose_score']:.4f}")
print(f"  purpose_score p:    {model2.pvalues['purpose_score']:.5f}")
print(f"  R²: {model2.rsquared:.4f}")
print(f"  Adjusted R²: {model2.rsquared_adj:.4f}")
print(f"  N: {int(model2.nobs)}")

print("\n--- Model 3: Sub-scores + Sector Controls ---")
for col in sub_scores:
    coef = model3.params.get(col, np.nan)
    pval = model3.pvalues.get(col, np.nan)
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
    print(f"  {sub_labels.get(col, col)}: coef={coef:.4f}, p={pval:.5f} {sig}")
print(f"  R²: {model3.rsquared:.4f}, Adjusted R²: {model3.rsquared_adj:.4f}")

# Save full model 2 summary
with open("results/07_regression_model2_full.txt", "w") as f:
    f.write(model2.summary().as_text())
with open("results/07_regression_model3_full.txt", "w") as f:
    f.write(model3.summary().as_text())

# Build clean regression comparison table
reg_summary = pd.DataFrame({
    "Model": ["Baseline", "With Sector Controls", "Sub-scores + Sectors"],
    "purpose_score_coef": [
        round(model1.params.get("purpose_score", np.nan), 4),
        round(model2.params.get("purpose_score", np.nan), 4),
        np.nan,
    ],
    "purpose_score_p": [
        round(model1.pvalues.get("purpose_score", np.nan), 5),
        round(model2.pvalues.get("purpose_score", np.nan), 5),
        np.nan,
    ],
    "R_squared": [round(model1.rsquared, 4), round(model2.rsquared, 4), round(model3.rsquared, 4)],
    "Adj_R_squared": [round(model1.rsquared_adj, 4), round(model2.rsquared_adj, 4), round(model3.rsquared_adj, 4)],
    "N": [int(model1.nobs), int(model2.nobs), int(model3.nobs)],
})
reg_summary.to_csv("results/07_regression_summary.csv", index=False)

# Plot: coefficient comparison across models + sub-score forest plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: purpose_score coefficient in model 1 vs model 2 with confidence intervals
coefs = [model1.params["purpose_score"], model2.params["purpose_score"]]
cis = [
    model1.conf_int().loc["purpose_score"].values,
    model2.conf_int().loc["purpose_score"].values,
]
labels_reg = ["No sector\ncontrols", "With sector\ncontrols"]
colors_reg = ["#e67e22", "#2980b9"]
for i, (coef, ci, label, color) in enumerate(zip(coefs, cis, labels_reg, colors_reg)):
    axes[0].barh(i, coef, color=color, alpha=0.8, height=0.4)
    axes[0].errorbar(coef, i, xerr=[[coef - ci[0]], [ci[1] - coef]],
                     fmt="none", color="black", capsize=5, linewidth=1.5)
axes[0].axvline(0, color="black", linewidth=0.8, linestyle="--")
axes[0].set_yticks(range(len(labels_reg)))
axes[0].set_yticklabels(labels_reg, fontsize=10)
axes[0].set_xlabel("Purpose Score Coefficient (OLS)", fontsize=10)
axes[0].set_title("Does sector membership explain\nthe purpose-performance relationship?", fontsize=11)

# Right: sub-score coefficients from model 3 (forest plot)
sub_coefs = [model3.params.get(c, np.nan) for c in sub_scores]
sub_cis = [model3.conf_int().loc[c].values if c in model3.conf_int().index else [np.nan, np.nan]
           for c in sub_scores]
sub_pvals = [model3.pvalues.get(c, np.nan) for c in sub_scores]
sub_label_list = [sub_labels[c] for c in sub_scores]

# Sort by coefficient
sorted_idx = np.argsort(sub_coefs)[::-1]
for rank, i in enumerate(sorted_idx):
    color = "#27ae60" if sub_pvals[i] < 0.05 else "#95a5a6"
    axes[1].barh(rank, sub_coefs[i], color=color, alpha=0.8, height=0.5)
    if not any(np.isnan(sub_cis[i])):
        axes[1].errorbar(sub_coefs[i], rank,
                         xerr=[[sub_coefs[i] - sub_cis[i][0]], [sub_cis[i][1] - sub_coefs[i]]],
                         fmt="none", color="black", capsize=4, linewidth=1.2)
axes[1].axvline(0, color="black", linewidth=0.8, linestyle="--")
axes[1].set_yticks(range(len(sub_scores)))
axes[1].set_yticklabels([sub_label_list[i] for i in sorted_idx], fontsize=9)
axes[1].set_xlabel("Coefficient (controlling for sector)", fontsize=10)
axes[1].set_title("Which purpose dimension predicts performance\n(sector-controlled)?", fontsize=11)
green_patch = mpatches.Patch(color="#27ae60", alpha=0.8, label="p < 0.05")
gray_patch = mpatches.Patch(color="#95a5a6", alpha=0.8, label="not significant")
axes[1].legend(handles=[green_patch, gray_patch], fontsize=8)

plt.suptitle("PAN Group — Regression Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("results/07_regression.png", dpi=150)
plt.close()
print("\nSaved: results/07_regression_summary.csv + full model txts + 07_regression.png\n")


# =============================================================================
# ANALYSIS 8 — Score distribution diagnostics
# =============================================================================
# Why: The professor or client will ask about score calibration.
# No companies score above 0.85 on purpose — that needs to be visible and
# addressed, not discovered during Q&A. Distribution plots also show whether
# scores are roughly normal (required assumption for OLS), heavily skewed
# (which would explain why Spearman and Pearson r differ), or bimodal
# (which would suggest the framework is sorting companies into two groups
# rather than producing a continuous scale).

print("=" * 60)
print("ANALYSIS 8: Score distribution diagnostics")
print("=" * 60)

from scipy.stats import shapiro, skew, kurtosis

for score_col, label in [("purpose_score", "Purpose Score"), ("performance_score", "Performance Score")]:
    vals = df[score_col].dropna()
    sk = skew(vals)
    kurt = kurtosis(vals)
    # Shapiro-Wilk only reliable up to n~5000, fine here
    # Use a sample if n > 5000
    sample = vals.sample(min(len(vals), 500), random_state=42)
    stat, p_sw = shapiro(sample)
    print(f"\n{label}:")
    print(f"  Mean:     {vals.mean():.4f}")
    print(f"  Median:   {vals.median():.4f}")
    print(f"  Std:      {vals.std():.4f}")
    print(f"  Min:      {vals.min():.4f}")
    print(f"  Max:      {vals.max():.4f}")
    print(f"  Skewness: {sk:.4f}  ({'right-skewed' if sk > 0.5 else 'left-skewed' if sk < -0.5 else 'roughly symmetric'})")
    print(f"  Kurtosis: {kurt:.4f}")
    print(f"  Shapiro-Wilk p: {p_sw:.5f}  ({'not normal' if p_sw < 0.05 else 'normal'})")
    print(f"  % scoring 0:    {(vals == 0).sum() / len(vals) * 100:.1f}%")
    print(f"  % scoring >0.65: {(vals > 0.65).sum() / len(vals) * 100:.1f}%")
    print(f"  % scoring >0.85: {(vals > 0.85).sum() / len(vals) * 100:.1f}%")

# Plot: distribution grid — histograms, KDE, Q-Q plots, sub-score distributions
fig, axes = plt.subplots(3, 3, figsize=(14, 11))

# Row 1: Purpose and Performance histograms + scatter
for idx, (col, label, color) in enumerate([
    ("purpose_score", "Purpose Score", "#2980b9"),
    ("performance_score", "Performance Score", "#27ae60"),
]):
    ax = axes[0][idx]
    vals = df[col].dropna()
    ax.hist(vals, bins=40, color=color, alpha=0.75, edgecolor="white", linewidth=0.5)
    ax.axvline(vals.mean(), color="red", linewidth=1.5, linestyle="--", label=f"Mean={vals.mean():.2f}")
    ax.axvline(vals.median(), color="orange", linewidth=1.5, linestyle=":", label=f"Median={vals.median():.2f}")
    ax.set_title(f"{label} Distribution", fontsize=11)
    ax.set_xlabel(label)
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    # Add band boundaries
    for boundary in [0.25, 0.45, 0.65, 0.85]:
        ax.axvline(boundary, color="gray", linewidth=0.8, alpha=0.5)

# Scatter with marginal density hint
ax = axes[0][2]
ax.scatter(df["purpose_score"], df["performance_score"],
           alpha=0.3, s=8, color="#7f8c8d")
ax.set_xlabel("Purpose Score")
ax.set_ylabel("Performance Score")
ax.set_title("Joint Distribution", fontsize=11)
# Add quadrant lines
ax.axvline(0.5, color="red", linewidth=0.8, linestyle="--", alpha=0.6)
ax.axhline(0.5, color="red", linewidth=0.8, linestyle="--", alpha=0.6)

# Row 2: Sub-score distributions
for idx, col in enumerate(sub_scores[:3]):
    ax = axes[1][idx]
    vals = df[col].dropna()
    ax.hist(vals, bins=30, color="#8e44ad", alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.set_title(f"{sub_labels[col]}", fontsize=10)
    ax.set_xlabel("Raw Score")
    ax.axvline(vals.mean(), color="red", linewidth=1.2, linestyle="--")

# Row 3: Remaining sub-scores + purpose score by sector boxplot
for idx, col in enumerate(sub_scores[3:]):
    ax = axes[2][idx]
    vals = df[col].dropna()
    ax.hist(vals, bins=30, color="#8e44ad", alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.set_title(f"{sub_labels[col]}", fontsize=10)
    ax.set_xlabel("Raw Score")
    ax.axvline(vals.mean(), color="red", linewidth=1.2, linestyle="--")

# Last panel: purpose score by sector (shows calibration consistency)
ax = axes[2][2]
sector_purpose_means = df.groupby("GICS Sector")["purpose_score"].mean().sort_values()
top10 = sector_purpose_means.tail(10)
ax.barh(range(len(top10)), top10.values, color="#2980b9", alpha=0.75)
ax.set_yticks(range(len(top10)))
ax.set_yticklabels(top10.index, fontsize=7)
ax.set_xlabel("Mean Purpose Score")
ax.set_title("Mean Purpose Score\nby Sector (top 10)", fontsize=10)

plt.suptitle("PAN Group — Score Distribution Diagnostics", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("results/08_distribution_diagnostics.png", dpi=150)
plt.close()

# Save summary stats
dist_summary = []
for col, label in [("purpose_score", "Purpose Score"), ("performance_score", "Performance Score")] + \
                   [(c, sub_labels[c]) for c in sub_scores]:
    vals = df[col].dropna()
    dist_summary.append({
        "score": label,
        "mean": round(vals.mean(), 4),
        "median": round(vals.median(), 4),
        "std": round(vals.std(), 4),
        "min": round(vals.min(), 4),
        "max": round(vals.max(), 4),
        "skewness": round(skew(vals), 4),
        "pct_zero": round((vals == 0).sum() / len(vals) * 100, 2),
        "pct_above_065": round((vals > 0.65).sum() / len(vals) * 100, 2),
    })
pd.DataFrame(dist_summary).to_csv("results/08_distribution_summary.csv", index=False)
print("\nSaved: results/08_distribution_diagnostics.png + 08_distribution_summary.csv\n")

print("=" * 60)
print("ALL ANALYSES COMPLETE")
print("New outputs:")
print("  06_sector_subscore_correlations.csv + 06_sector_subscore_heatmap.png")
print("  07_regression_summary.csv + 07_regression.png + model txt files")
print("  08_distribution_diagnostics.png + 08_distribution_summary.csv")
print("=" * 60)