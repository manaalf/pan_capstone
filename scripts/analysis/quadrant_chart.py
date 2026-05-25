import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

df = pd.read_csv('data/output/master_dataset.csv')
mid = 0.5

# ── Top 3 per quadrant ─────────────────────────────────────────────────────
hh = df[(df['purpose_score']>=mid)&(df['performance_score']>=mid)].nlargest(3,'purpose_score')
lh = df[(df['purpose_score']<mid) &(df['performance_score']>=mid)].nlargest(3,'performance_score')
hl = df[(df['purpose_score']>=mid)&(df['performance_score']<mid)].nlargest(3,'purpose_score')
ll = df[(df['purpose_score']<mid) &(df['performance_score']<mid)].nsmallest(3,'performance_score')

anchors = ['NKE','GHC','IBM']
highlight = pd.concat([hh,lh,hl,ll])

# ── Figure ────────────────────────────────,[mid,mid],[1,1],  alpha=0.08, color='#4ADE80')
ax.fill_between([0,mid],[mid,mid],[1,1],  alpha=0.08, color='#F87171')
ax.fill_between([mid,1],[0,0],  [mid,mid],alpha=0.08, color='#60A5FA')
ax.fill_between([0,mid],[0,0],  [mid,mid],alpha=0.06, color='#94A3B8')

ax.axhline(mid, color='#334155', linewidth=1.2, linestyle='--', alpha=0.7)
ax.axvline(mid, color='#334155', linewidth=1.2, linestyle='--', alpha=0.7)

# ── All dots by quadrant colour ────────────────────────────────────────────
qcolor = {
    'High Purpose / High Performance': '#4ADE80',
    'Low Purpose / High Performance':  '#F87171',
    'High Purpose / Low Performance':  '#60A5FA',
    'Low Purpose / Low Performance':   '#94A3B8',
}
for q, color in qcolor.items():
    sub = df[df['quadrant']==q]
    ax.scatter(sub['purpose_score'], sub['performance_score'],
               c=color, alpha=0.30, s=22, edgecolors='none', zorder=2)

# ── Highlighted top-3 TMO':  (-0.14,  0.02), 'KEYS': ( 0.02, -0.04),
    'MLI':  (-0.12,  0.02), 'ANET': ( 0.02,  0.02), 'APH':  ( 0.02, -0.04),
    'MDT':  (-0.10,  0.02), 'BBY':  ( 0.02,  0.02), 'AMN':  ( 0.02, -0.04),
    'AAP':  ( 0.02,  0.02), 'AER':  ( 0.02, -0.04), 'AFL':  (-0.10,  0.02),
}
for _, row in highlight.iterrows():
    q = row['quadrant']
    color = qcolor[q]
    ax.scatter(row['purpose_score'], row['performance_score'],
               c=color, s=100, edgecolors='white', linewidths=1.5, zorder=5)
    dx, dy = offsets.get(row['ticker'], (0.02, 0.02))
    short_name = row['Company Name'].split('(')[0].strip()
    short_name = short_name if len(short_name)<=22 else short_name[:20]+'…'
    ax.annotate(f"{row['ticker']}\n{short_name}",
                (row['purpose_score'], row['performance_score']),
                xytext=(row['purpose_score']+dx, row['performance_score']+dy),
                fontsize=7.5, color=color, fontweight='bold',
                arrowprops=dict(arrowstyle='-', color=color, alpha=0.5, lw=8),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#0D0D1A',
                          edgecolor=color, alpha=0.9, linewidth=0.8), zorder=6)

# ── Pilot anchors ──────────────────────────────────────────────────────────
anchor_data = df[df['ticker'].isin(anchors)]
anchor_offsets = {'NKE':( 0.02,-0.04), 'GHC':( 0.02, 0.02), 'IBM':(-0.12, 0.02)}
for _, row in anchor_data.iterrows():
    ax.scatter(row['purpose_score'], row['performance_score'],
               c='#FBBF24', s=140, edgecolors='white', linewidths=2,
               marker='*', zorder=7)
    dx, dy = anchor_offsets.get(row['ticker'], (0.02, 0.02))
    ax.annotate(f"{row['ticker']} ★",
                (row['purpose_score'], row['performance_score']),
                xytext=(row['purpose_score']+dx, row['performance_score']+dy),
                fontsize=9, color='#FBBF24', fontweight='bold',
                bbox=di), zorder=8)

# ── Quadrant labels ────────────────────────────────────────────────────────
kw = dict(transform=ax.transAxes, fontweight='bold', fontsize=11, va='center', ha='center')
ax.text(0.75, 0.95, 'Purpose Leaders',      color='#4ADE80', **kw)
ax.text(0.75, 0.92, f"{len(df[(df['purpose_score']>=mid)&(df['performance_score']>=mid)])} companies",
        transform=ax.transAxes, fontsize=9, color='#4ADE80', ha='center', va='center')
ax.text(0.25, 0.95, 'Value Extraction',     color='#F87171', **kw)
ax.text(0.25, 0.92, f"{len(df[(df['purpose_score']<mid)&(df['performance_score']>=mid)])} companies",
        transform=ax.transAxes, fontsize=9, color='#F87171', ha='center', va='center')
ax.text(0.75, 0.08, 'Purpose Promise',      color='#60A5FA', **kw)
ax.text(0.75, 0.05, f"{len(df[(df['purpose_score']>=mid)&(df['performance_score']<mid)])} companies",
        transform=ax.transAxes, fontsize.text(0.25, 0.05, f"{len(df[(df['purpose_score']<mid)&(df['performance_score']<mid)])} companies",
        transform=ax.transAxes, fontsize=9, color='#94A3B8', ha='center', va='center')

# ── Legend ─────────────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(color='#FBBF24', label='Pilot anchors (NKE, GHC, IBM)'),
    plt.scatter([],[], c='white', s=80, edgecolors='white', label='Top 3 per quadrant'),
]
ax.legend(handles=[
    mpatches.Patch(color='#FBBF24', label='★ Pilot anchors (NKE, GHC, IBM)'),
    mpatches.Patch(color='white',   label='● Top 3 per quadrant (labelled)'),
], loc='lower right', facecolor='#1E293B', edgecolor='#334155',
   labelcolor='#CBD5E1', fontsize=9, framealpha=0.9)

ax.set_xlim(0.02, 1.0)
ax.set_ylim(-0.02, 1.0)
ax.set_xlabel('Purpose Score →', color='#CBD5E1', fontsize=13, labelpad=12)
ax.set_ylabel\n'
    'Quadrant Analysis  |  904 NYSE Companies  |  Pearson r=0.122  p<0.001',
    color='white', fontsize=13, pad=18)
ax.tick_params(colors='#64748B')
for spine in ax.spines.values():
    spine.set_edgecolor('#1E293B')

plt.tight_layout()
plt.savefig('data/output/quadrant_analysis_904.png', dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
print('Saved: data/output/quadrant_analysis_904.png')
