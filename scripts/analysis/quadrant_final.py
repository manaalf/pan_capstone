import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── PALETTE ───────────────────────────────────────────────────────────────────
CREAM       = "#F2EDE4"
CRIMSON     = "#8B1A1A"
BLACK       = "#1A1A1A"
GRAY        = "#6B6B6B"
MGRAY       = "#AAAAAA"
LGRAY       = "#D9D3C8"

# Quadrant fills — more vivid
Q_PURPOSE_LEADERS  = "#E8C4C4"   # top-right  — rose
Q_VALUE_EXTRACTION = "#E8E0D4"   # top-left   — warm tan
Q_DUAL_WEAKNESS    = "#E2DDD6"   # bottom-left — muted cream
Q_PURPOSE_PROMISE  = "#EDD8D0"   # bottom-right — dusty rose

# Dot colors
DOT_PURPOSE_LEADERS  = "#8B1A1A"   # crimson
DOT_VALUE_EXTRACTION = "#7A6655"   # warm brown
DOT_DUAL_WEAKNESS    = "#999999"   # gray
DOT_PURPOSE_PROMISE  = "#B89080"   # muted rose

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
df = pd.read_csv("master_dataset.csv")
df = df.dropna(subset=["purpose_score", "performance_score"])

# ── QUADRANT ASSIGNMENT ───────────────────────────────────────────────────────
t = 0.5
df["q"] = "dual_weakness"
df.loc[(df["purpose_score"] >= t) & (df["performance_score"] >= t), "q"] = "purpose_leaders"
df.loc[(df["purpose_score"] <  t) & (df["performance_score"] >= t), "q"] = "value_extraction"
df.loc[(df["purpose_score"] >= t) & (df["performance_score"] <  t), "q"] = "purpose_promise"

dot_map = {
    "purpose_leaders":  DOT_PURPOSE_LEADERS,
    "value_extraction": DOT_VALUE_EXTRACTION,
    "dual_weakness":    DOT_DUAL_WEAKNESS,
    "purpose_promise":  DOT_PURPOSE_PROMISE,
}
df["dot_color"] = df["q"].map(dot_map)

# ── PILOT VALUES (hardcoded) ───────────────────────────────────────────────────
pilots = {
    "IBM": {"x": 0.875, "y": 0.700, "lx":  0.06,  "ly":  0.04},
    "GHC": {"x": 0.451, "y": 0.667, "lx": -0.14,  "ly":  0.05},
    "NKE": {"x": 0.401, "y": 0.492, "lx": -0.14,  "ly": -0.06},
}

# ── FIGURE ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8.2), facecolor=CREAM)
ax.set_facecolor(CREAM)

# Quadrant fills
ax.fill_between([0, t], [t, t], [1, 1],   color=Q_VALUE_EXTRACTION, zorder=0)
ax.fill_between([t, 1], [t, t], [1, 1],   color=Q_PURPOSE_LEADERS,  zorder=0)
ax.fill_between([0, t], [0, 0], [t, t],   color=Q_DUAL_WEAKNESS,    zorder=0)
ax.fill_between([t, 1], [0, 0], [t, t],   color=Q_PURPOSE_PROMISE,  zorder=0)

# Divider lines
ax.axvline(t, color=LGRAY, linewidth=1.0, zorder=1, linestyle="--", alpha=0.9)
ax.axhline(t, color=LGRAY, linewidth=1.0, zorder=1, linestyle="--", alpha=0.9)

# All dots — larger and clearer
ax.scatter(
    df["purpose_score"],
    df["performance_score"],
    c=df["dot_color"],
    alpha=0.65,
    s=28,
    zorder=2,
    linewidths=0,
)

# ── PILOT ANCHORS ─────────────────────────────────────────────────────────────
for name, p in pilots.items():
    # Large highlighted dot
    ax.scatter(p["x"], p["y"],
               color=CRIMSON, s=120, zorder=5,
               edgecolors=BLACK, linewidths=1.5)

    # Label box
    lx = p["x"] + p["lx"]
    ly = p["y"] + p["ly"]

    ax.annotate(
        name,
        xy=(p["x"], p["y"]),
        xytext=(lx, ly),
        fontsize=10.5,
        fontweight="bold",
        color=BLACK,
        fontfamily="sans-serif",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor=CREAM,
            edgecolor=CRIMSON,
            linewidth=1.4,
        ),
        arrowprops=dict(
            arrowstyle="-",
            color=CRIMSON,
            linewidth=1.0,
        ),
        zorder=6,
    )

# ── QUADRANT LABELS ───────────────────────────────────────────────────────────
ax.text(0.02, 0.98, "VALUE EXTRACTION",
        transform=ax.transAxes, va="top",
        fontsize=10, fontweight="bold",
        color="#6B5B4A", fontfamily="sans-serif", alpha=0.85)

ax.text(0.52, 0.98, "PURPOSE LEADERS",
        transform=ax.transAxes, va="top",
        fontsize=10, fontweight="bold",
        color=CRIMSON, fontfamily="sans-serif", alpha=0.85)

ax.text(0.02, 0.46, "DUAL WEAKNESS",
        transform=ax.transAxes, va="top",
        fontsize=10, fontweight="bold",
        color="#777777", fontfamily="sans-serif", alpha=0.85)

ax.text(0.52, 0.46, "PURPOSE PROMISE",
        transform=ax.transAxes, va="top",
        fontsize=10, fontweight="bold",
        color="#9B7060", fontfamily="sans-serif", alpha=0.85)

# ── AXES ──────────────────────────────────────────────────────────────────────
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.0)
ax.set_xlabel("PURPOSE SCORE  →",
              fontsize=10, color=GRAY,
              fontfamily="sans-serif", labelpad=10, fontweight="bold")
ax.set_ylabel("PERFORMANCE SCORE  →",
              fontsize=10, color=GRAY,
              fontfamily="sans-serif", labelpad=10, fontweight="bold")
ax.tick_params(colors=GRAY, labelsize=9)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color(LGRAY)
ax.spines["bottom"].set_color(LGRAY)

plt.tight_layout()
plt.savefig("quadrant_final.png", dpi=180,
            bbox_inches="tight", facecolor=CREAM)
print("✅  Saved: quadrant_final.png")
plt.show()
