# -*- coding: utf-8 -*-
"""Regenerate Figure 1: MERCI architecture diagram (matplotlib, deterministic)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "DejaVu Sans"

FIG = r"E:\claude\deepmeans_repo\research\figures\fig1_merci_architecture.png"

fig, ax = plt.subplots(figsize=(13.5, 9.2), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.axis("off")

# ---------------------------------------------------------------- colors
C_INPUT  = "#e8eef7"; C_FEAT = "#dbe8f4"
C_MOD    = "#fdeee0"; C_HEAD = "#e2f0e2"
C_OUT    = "#eaf3fb"; C_EVAL = "#f6ecf6"
C_EDGE   = "#2f5d8a"; C_TXT = "#1a1a1a"

def box(x, y, w, h, text, fc, fs=10.5, weight="normal", ec=C_EDGE, lw=1.4, align="center"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=0.8",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=C_TXT, fontweight=weight, zorder=3)
    return (x, y, w, h)

def arrow(x1, y1, x2, y2, color=C_EDGE, lw=1.8, style="-|>"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                        linewidth=lw, color=color, zorder=1)
    ax.add_patch(a)

# ---------------------------------------------------------------- title
ax.text(50, 96.5, "MERCI: multi-task residual learning for sgRNA efficiency "
        "prediction across Cas9 variants", ha="center", va="center",
        fontsize=15, fontweight="bold", color="#123a5e")
ax.text(50, 92.6, "Multi-task ensemble with residual correction and "
        "uncertainty-aware interpretation", ha="center", va="center",
        fontsize=11.5, color="#4a6b8a", style="italic")

# ---------------------------------------------------------------- input
box(38, 84.5, 24, 6.0, "Input: sgRNA 21-nt sequence\n(20-nt guide + NGG PAM)",
    C_INPUT, fs=10.5, weight="bold")

# ---------------------------------------------------------------- features
feats = [
    (2,  "Sequence features\nk-mer / positional / poly-T /\nGC / base composition (178)"),
    (27, "Thermodynamic\nfeatures\nTm, DNA binding \u0394G (4)"),
    (52, "DNA-shape features\nMGW / HelT / Roll / ProT / EP\n(35)"),
    (77, "Interaction features\ndimer counts +\nsequence\u2013structure terms (8)"),
]
for x, t in feats:
    box(x, 71.5, 21, 10.5, t, C_FEAT, fs=9.5)

arrow(50, 84.2, 50, 82.4)
for x, t in feats:
    arrow(x + 10.5, 71.4, x + 10.5, 68.6)

box(31, 60.0, 38, 6.2, "Concatenated feature vector  (225 dimensions)",
    "#cfe0ee", fs=10.5, weight="bold")
arrow(50, 67.9, 50, 66.6)

# ---------------------------------------------------------------- modules
box(2.5, 40.5, 21, 15.0, "Module A\nUncertainty-aware\nsample weighting\n\n5 shallow LightGBM\nOOF prediction variance \u2192 w\u1d62",
    C_MOD, fs=9.5, weight="bold")
box(28, 40.5, 21, 15.0, "Module B\nFeature-subspace\nbagging\n\n20 random 70%\nfeature subsets",
    C_MOD, fs=9.5, weight="bold")
box(54, 40.5, 21, 15.0, "Module C\nMulti-task residual\ncorrection\n\nWT head + two\nresidual heads",
    C_MOD, fs=9.5, weight="bold")

arrow(50, 59.4, 50, 56.2)
arrow(13, 59.6, 13, 55.8, color="#8a6d3b")
arrow(38.5, 59.6, 38.5, 55.8, color="#8a6d3b")
arrow(64.5, 59.6, 64.5, 55.8, color="#8a6d3b")

# ---------------------------------------------------------------- heads
box(2.5, 28.5, 21, 8.5, "Base learner 1..20\n(shared residual targets)\n\u0177_WT, \u0394\u0177_eSp, \u0394\u0177_HF1",
    C_HEAD, fs=9.5)
box(54, 28.5, 21, 8.5, "Weighted multi-task loss\nw\u1d62 \u00b7 [MSE_WT + MSE_\u0394eSp + MSE_\u0394HF1]",
    C_HEAD, fs=9.5)

box(28, 28.5, 21, 8.5, "Stacking meta-learner\nCatBoost on 60 OOF\nmeta-features (20\u00d73)",
    "#f5ead3", fs=9.5, weight="bold")

arrow(13, 40.2, 13, 37.4)
arrow(64.5, 40.2, 64.5, 37.4)
arrow(38.5, 40.2, 38.5, 37.4)
arrow(13, 28.2, 27.6, 28.2)
arrow(59.5, 28.2, 45.0, 28.2, style="-|>")  # into stacking from loss
arrow(24.5, 32.8, 34.8, 32.8, color="#8a6d3b")  # heads into stacking

# ---------------------------------------------------------------- output
box(31, 16.5, 38, 7.5, "Final predictions:  \u0177_WT,  \u0177_eSp,  \u0177_HF1\n+ per-sgRNA uncertainty proxy  \u03c3\u00b2",
    C_OUT, fs=10.5, weight="bold")
arrow(38.5, 28.2, 38.5, 24.4)

box(2.5, 5.0, 30, 7.0, "Evaluation\ninternal hold-out + 10 independent datasets\n(Spearman / Pearson / MSE, overlap-filtered)",
    C_EVAL, fs=9.0)
box(67.5, 5.0, 30, 7.0, "Interpretation\nSHAP attribution + residual analysis\n(enzyme-specific determinants)",
    C_EVAL, fs=9.0)
arrow(38.5, 16.2, 38.5, 12.6)
arrow(33.0, 12.4, 33.0, 12.4, color="none")
arrow(40.5, 12.6, 40.5, 12.6, color="none")

fig.savefig(FIG, bbox_inches="tight", facecolor="white")
print("saved", FIG)
