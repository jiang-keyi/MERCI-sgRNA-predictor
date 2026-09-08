# -*- coding: utf-8 -*-
"""
域偏移分析（审稿建议 4 / Figure 7 建议）

目标：证明外部集性能下降与特征分布偏移一致
  - PCA 降维后对比 DeepHF 训练/验证与 10 个外部集
  - 每数据集计算相对训练中心的标准化欧氏距离（shift score）
  - 关联 shift score 与外部 Spearman → 若负相关，则
    "0.368 不是模型差，而是分布偏移的可量化后果"

输出：results/revision/domain_shift.json + figures/fig_domain_shift.png
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from eval_test_sets import load_test_set, build_test_features, TEST_ROOT
from models import predict_pipeline

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PROC = os.path.join(BASE, "data", "processed")
RES = os.path.join(BASE, "results")
OUT = os.path.join(RES, "revision")
FIG = os.path.join(BASE, "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

EXTERNAL_RES = os.path.join(RES, "tables", "independent_test_results.json")


def main():
    with open(os.path.join(RES, "models", "trained_full.pkl"), "rb") as f:
        import pickle
        obj = pickle.load(f)
    trained, feat_names = obj["trained"], obj["feat_names"]

    merged = pd.read_pickle(os.path.join(PROC, "merged_all.pkl"))
    features = pd.read_pickle(os.path.join(PROC, "features_all.pkl"))
    X = features[feat_names].values.astype(np.float64)
    train_mask = (merged["split"] == "train").values
    val_mask = (merged["split"] == "val").values
    train_seqs = set(merged.loc[train_mask, "seq21"].tolist())

    # 采样：训练集最多 8000，验证集全部（用于投影）
    rng = np.random.RandomState(0)
    n_tr = min(8000, int(train_mask.sum()))
    tr_idx = rng.choice(np.where(train_mask)[0], n_tr, replace=False)
    va_idx = np.where(val_mask)[0]

    # 标准scaler 基于训练样本
    scaler = StandardScaler().fit(X[tr_idx])
    Xs_tr = scaler.transform(X[tr_idx])
    Xs_va = scaler.transform(X[va_idx])
    pca2 = PCA(n_components=2, random_state=0).fit(Xs_tr)
    pca20 = PCA(n_components=20, random_state=0).fit(Xs_tr)

    proj = {"train": pca2.transform(Xs_tr), "val": pca2.transform(Xs_va)}
    Ztr = pca20.transform(Xs_tr)

    def rbf_mmd(A, B, sigma=4.0):
        """无偏 RBF-MMD² 估计。"""
        n = A.shape[0]; m = B.shape[0]
        A2 = (A ** 2).sum(1, keepdims=True)
        B2 = (B ** 2).sum(1, keepdims=True)
        Kaa = np.exp(-(A2 + A2.T - 2 * A @ A.T) / (2 * sigma ** 2))
        Kbb = np.exp(-(B2 + B2.T - 2 * B @ B.T) / (2 * sigma ** 2))
        Kab = np.exp(-(A2 + B2.T - 2 * A @ B.T) / (2 * sigma ** 2))
        np.fill_diagonal(Kaa, 0); np.fill_diagonal(Kbb, 0)
        return float(Kaa.sum() / (n * (n - 1)) + Kbb.sum() / (m * (m - 1)) - 2 * Kab.mean())

    # 验证集 MMD 锚点（同分布应接近 0；随机采样 2000）
    rng_v = np.random.RandomState(1)
    va_samp = rng_v.choice(len(Xs_va), min(2000, len(Xs_va)), replace=False)
    val_mmd = rbf_mmd(Ztr[:2000], pca20.transform(Xs_va)[va_samp])
    print(f"VAL MMD anchor = {val_mmd:.6f}", flush=True)

    # 外部集
    ext_res = json.load(open(EXTERNAL_RES))
    rows = []
    proj_ext = {}
    for d in sorted(os.listdir(TEST_ROOT)):
        dirpath = os.path.join(TEST_ROOT, d)
        if not os.path.isdir(dirpath):
            continue
        seqs, labels, shape_dict = load_test_set(dirpath)
        keep = [i for i, s in enumerate(seqs) if s not in train_seqs]
        seqs_k = [seqs[i] for i in keep]
        labels_k = labels[keep]
        shape_k = {sh: v[keep] for sh, v in shape_dict.items()}
        df_feat = build_test_features(seqs_k, shape_k)
        for c in feat_names:
            if c not in df_feat.columns:
                df_feat[c] = 0.0
        Xt = df_feat[feat_names].values.astype(np.float64)
        Xt_s = scaler.transform(Xt)
        proj_ext[d] = pca2.transform(Xt_s)
        # shift score：RBF-MMD（PCA-20 空间，随机采样 2000）
        rng_e = np.random.RandomState(2)
        samp = rng_e.choice(len(Xt_s), min(2000, len(Xt_s)), replace=False)
        Zt = pca20.transform(Xt_s)[samp]
        shift = rbf_mmd(Ztr[:2000], Zt)
        sp = ext_res.get(d, {}).get("WT", {}).get("spearman", np.nan)
        rows.append({"dataset": d, "n": int(len(labels_k)), "mmd": shift,
                     "spearman": sp, "val_mmd_anchor": val_mmd})
        print(f"{d:20s} MMD={shift:.5f}  spearman={sp:.3f}", flush=True)

    # 关联
    rho, pval = spearmanr([r["mmd"] for r in rows],
                          [r["spearman"] for r in rows])
    print(f"Spearman(MMD, external spearman) = {rho:.4f} (p={pval:.4f})", flush=True)

    with open(os.path.join(OUT, "domain_shift.json"), "w") as f:
        json.dump({"rows": rows, "mmd_spearman": float(rho),
                   "p_value": float(pval), "val_mmd_anchor": val_mmd}, f, indent=2)

    # ===== 图 =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.scatter(proj["train"][:, 0], proj["train"][:, 1], s=3, alpha=0.25,
               color="lightgray", label="DeepHF train (sample)")
    ax.scatter(proj["val"][:, 0], proj["val"][:, 1], s=3, alpha=0.35,
               color="darkgray", label="DeepHF val")
    colors = plt.cm.tab10(np.linspace(0, 1, len(proj_ext)))
    for (d, P), c in zip(proj_ext.items(), colors):
        ax.scatter(P[:, 0], P[:, 1], s=4, alpha=0.45, color=c, label=d)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("(A) PCA: training vs external distributions")
    ax.legend(fontsize=6.5, markerscale=3, loc="upper left", framealpha=0.5)
    ax.grid(alpha=0.3)

    ax = axes[1]
    r = rows
    ax.scatter([x["mmd"] * 1000 for x in r], [x["spearman"] for x in r],
               s=45, color="steelblue", edgecolor="k", alpha=0.8)
    for x in r:
        ax.annotate(x["dataset"].replace("_", "-"), (x["mmd"] * 1000, x["spearman"]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.axvline(val_mmd * 1000, color="red", ls="--", lw=1)
    ax.text(val_mmd * 1000, 0.52, f"val anchor\n({val_mmd*1000:.2f})",
            fontsize=7, color="red", ha="left")
    ax.set_xlabel("RBF-MMD × 1000 (feature-space shift vs training)")
    ax.set_ylabel("External Spearman")
    ax.set_title(f"(B) All external sets shift ≫ validation\n"
                 f"Spearman(MMD, perf) = {rho:.3f} (p={pval:.3g})")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_domain_shift.png"))
    plt.close()
    print("saved figures/fig_domain_shift.png", flush=True)
    print("DONE domain_shift", flush=True)


if __name__ == "__main__":
    main()
