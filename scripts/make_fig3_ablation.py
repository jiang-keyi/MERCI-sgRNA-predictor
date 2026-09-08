# -*- coding: utf-8 -*-
"""
图：多随机种子消融结果（mean ± SD 柱状图，按目标分面板）
输入：results/revision/ablation_multiseed.json（summary 为 per config+target 行）
输出：figures/fig3_ablation_multiseed.png
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(BASE, "results", "revision")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

ORDER = ["M0_plain", "M1_weight", "M2_mtl_direct", "M3_residual",
         "M4_bagging_avg", "M5_bagging_stack", "M6_merci"]
LABELS = ["Plain\nsingle-task", "+Uncertainty\nweighting", "Multi-task\ndirect",
          "+Residual\ncorrection", "+Feature-subspace\nbagging (avg)",
          "+Stacking", "MERCI\n(full)"]
COLORS = ["#bdbdbd", "#9ecae1", "#9ecae1", "#a1d99b", "#a1d99b", "#a1d99b", "#fd8d3c"]


def main():
    with open(os.path.join(RES, "ablation_multiseed.json")) as f:
        data = json.load(f)
    rows = data["summary"]  # [{config,target,spearman_mean,spearman_sd,...}]
    by = {(r["config"], r["target"]): r for r in rows}

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, tgt, title in zip(axes, ["wt", "esp", "hf1"],
                              ["SpCas9-WT", "eSpCas9(1.1)", "SpCas9-HF1"]):
        means = [by[(c, tgt)]["spearman_mean"] for c in ORDER]
        sds = [by[(c, tgt)]["spearman_sd"] for c in ORDER]
        x = np.arange(len(ORDER))
        ax.bar(x, means, yerr=sds, capsize=3, color=COLORS, edgecolor="k",
               linewidth=0.5, error_kw=dict(lw=0.8))
        for xi, (m, s) in enumerate(zip(means, sds)):
            ax.text(xi, m + s + 0.0015, f"{m:.4f}\u00b1{s:.4f}",
                    ha="center", fontsize=6.6, rotation=0)
        ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=7.2)
        ax.set_ylim(0.77, max(means) + 0.02)
        ax.set_ylabel("Validation Spearman" if tgt == "wt" else "")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(by[("M6_merci", tgt)]["spearman_mean"], color="gray",
                   ls="--", lw=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig3_ablation_multiseed.png"))
    plt.close()
    print("saved figures/fig3_ablation_multiseed.png", flush=True)


if __name__ == "__main__":
    main()
