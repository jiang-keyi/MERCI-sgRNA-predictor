# -*- coding: utf-8 -*-
"""
Figure 1：MERCI 架构图（审稿建议 22-24，替代原 predicted-vs-true 散点）
用 matplotlib 绘制，无外部依赖，白色背景（期刊要求）。
输出：figures/fig1_merci_architecture.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG, exist_ok=True)

C_SEQ = "#4C72B0"; C_THERMO = "#55A868"; C_SHAPE = "#C44E52"
C_INTER = "#8172B2"; C_BAG = "#CCB974"; C_HEAD = "#64B5CD"
C_META = "#E08A1E"; C_OUT = "#333333"


def box(ax, x, y, w, h, text, fc, ec="#333333", fs=8.5, bold=False, rounded=True):
    style = "round,pad=0.02,rounding_size=0.02" if rounded else "square,pad=0.02"
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=style,
                       linewidth=1.0, edgecolor=ec, facecolor=fc, alpha=0.92)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#111111")
    return (x, y)


def arrow(ax, xy1, xy2, color="#555555", lw=1.4, style="-|>", ls="-"):
    a = FancyArrowPatch(xy1, xy2, arrowstyle=style, mutation_scale=12,
                        linewidth=lw, color=color, linestyle=ls)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(9.5, 11.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 13.2)
    ax.axis("off")

    # 标题
    ax.text(5, 12.85, "MERCI: Multi-task Ensemble with Residual Correction and\n"
                       "uncertainty-aware Interpretation",
            ha="center", va="center", fontsize=12.5, fontweight="bold", color="#111111")

    # 1. 输入
    box(ax, 5, 12.0, 3.2, 0.55, "Input: sgRNA 21-nt sequence\n(20-nt guide + N of NGG PAM)", "#F5F5F5", fs=8)

    # 2. 特征族
    fam = [
        ("Sequence features\nk-mer / positional / poly-T /\nGC / base composition (178)", C_SEQ, 1.75),
        ("Thermodynamic features\nTm, ΔG binding (4)", C_THERMO, 4.15),
        ("DNA-shape features\nMGW / HelT / Roll / ProT / EP\n5 shapes × 7 stats (35)", C_SHAPE, 6.55),
        ("Interaction features\ndimer counts + shape–motif\ncross terms (8)", C_INTER, 8.95),
    ]
    for text, c, xc in fam:
        box(ax, xc, 10.95, 2.15, 1.15, text, c, fs=7.6)
        arrow(ax, (5, 11.7), (xc, 11.55))

    # 3. 拼接
    box(ax, 5, 9.95, 3.6, 0.5, "Concatenated feature vector (225 dims)", "#DDDDDD", fs=8, bold=True)
    for xc in [1.75, 4.15, 6.55, 8.95]:
        arrow(ax, (xc, 10.3), (5, 10.2))

    # 4. 不确定性加权（模块 A）
    box(ax, 5, 9.0, 4.6, 0.78,
        "Module A — Uncertainty-aware sample weighting\n"
        "OOF variance of shallow models → w_i = 1/(1+τσ²ᵢ)",
        "#EAF2F8", fs=7.6, bold=False)
    ax.text(1.15, 9.3, "Module A", fontsize=8, fontweight="bold", color="#1F4E79")
    arrow(ax, (5, 9.7), (5, 9.42))

    # 5. 特征子空间 bagging（模块 B）
    box(ax, 5, 7.7, 7.4, 1.15,
        "Module B — Feature-subspace bagging\n"
        "20 random 70% feature subsets × 5 bootstrap folds\n"
        "(100 LightGBM base learners, shared residual target)",
        "#FDF6E3", fs=8)
    ax.text(1.0, 7.75, "Module B", fontsize=8, fontweight="bold", color="#7D6608")
    arrow(ax, (5, 8.6), (5, 8.32))

    # 6. 多任务残差头
    hd_y = 6.4
    box(ax, 2.0, hd_y, 2.5, 0.75, "WT main head\nŷ_WT", C_HEAD, fs=8, bold=True)
    box(ax, 5.0, hd_y, 2.5, 0.75, "Residual head 1\nΔeSp = y_eSp − y_WT\n→ ŷ_eSp = ŷ_WT + Δ̂eSp", C_HEAD, fs=7.2)
    box(ax, 8.0, hd_y, 2.5, 0.75, "Residual head 2\nΔHF1 = y_HF1 − y_WT\n→ ŷ_HF1 = ŷ_WT + Δ̂HF1", C_HEAD, fs=7.2)
    for xc in [2.0, 5.0, 8.0]:
        arrow(ax, (5, 7.05), (xc, hd_y + 0.42))
    ax.text(9.6, 6.9, "Residual\nlearning", fontsize=7.5, color="#2C6E8F", fontweight="bold")

    # 7. stacking（模块 C）
    box(ax, 5, 5.0, 5.6, 0.78,
        "Module C — Stacking meta-learner\n"
        "CatBoost trained on OOF base predictions\n"
        "(sample-weight-aware, uncertainty-weighted losses)",
        "#FBE5D6", fs=7.6)
    ax.text(1.3, 5.05, "Module C", fontsize=8, fontweight="bold", color="#9C4A00")
    for xc in [2.0, 5.0, 8.0]:
        arrow(ax, (xc, hd_y - 0.42), (5, 5.42))

    # 8. 输出
    box(ax, 5, 3.9, 4.2, 0.55,
        "Ensembled predictions: ŷ_WT, ŷ_eSp, ŷ_HF1\n(+ per-sgRNA uncertainty proxy σ²)",
        "#D9EAD3", fs=7.6, bold=True)
    arrow(ax, (5, 4.6), (5, 4.2))

    # 9. 评估/解释
    box(ax, 5, 2.9, 6.2, 0.7,
        "Evaluation: internal held-out + 10 independent datasets\n"
        "(Spearman, Pearson, MSE; overlap-filtered test sets)",
        "#F5F5F5", fs=7.6)
    arrow(ax, (5, 3.6), (5, 3.27))
    box(ax, 5, 1.9, 6.2, 0.7,
        "Interpretation: SHAP attribution + residual analysis\n"
        "(enzyme-specific efficiency determinants)",
        "#F5F5F5", fs=7.6)
    arrow(ax, (5, 2.55), (5, 2.27))

    # 模块说明
    ax.text(5, 0.9,
            "Module A: down-weights low-confidence sgRNAs (ensemble-disagreement proxy).\n"
            "Module B: decorrelates learners via random feature subsets.\n"
            "Module C: combines base predictions with an uncertainty-weighted meta-learner.",
            ha="center", va="center", fontsize=7.2, color="#444444")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig1_merci_architecture.png"), dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("saved figures/fig1_merci_architecture.png", flush=True)


if __name__ == "__main__":
    main()
