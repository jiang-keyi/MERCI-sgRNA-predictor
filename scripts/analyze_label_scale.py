# -*- coding: utf-8 -*-
"""
外部标签尺度核查（审稿意见 13/14：external MSE 出现 3.3-3.6 的根源）

1. 统计 DeepHF 训练标签范围（应为 [0,1]）与 10 个外部数据集标签范围。
2. 用训练好的 MERCI 对每个外部集预测（与论文相同的去重叠流程）。
3. 输出：
   - 原始 MSE（尺度不可比的，明确标注）
   - 每数据集 min-max 归一化后的 MSE（尺度无关，可跨集比较）
   - Spearman / Pearson（本就不受线性尺度影响）
4. 画图：标签范围对比 + 尺度错配数据集的 pred-vs-true 散点。
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from models import predict_pipeline, compute_metrics
from eval_test_sets import load_test_set, build_test_features, TEST_ROOT, SHAPES

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PROC = os.path.join(BASE, "data", "processed")
RES = os.path.join(BASE, "results")
OUT = os.path.join(RES, "revision")
FIG = os.path.join(BASE, "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})


def main():
    # 1. 训练模型与特征
    import pickle
    with open(os.path.join(RES, "models", "trained_full.pkl"), "rb") as f:
        obj = pickle.load(f)
    trained, feat_names = obj["trained"], obj["feat_names"]

    merged = pd.read_pickle(os.path.join(PROC, "merged_all.pkl"))
    features = pd.read_pickle(os.path.join(PROC, "features_all.pkl"))
    X = features.values.astype(np.float64)
    y_wt = merged["y_wt"].values.astype(np.float64)
    train_mask = (merged["split"] == "train").values
    train_seqs = set(merged.loc[merged["split"] == "train", "seq21"].tolist())

    # 训练标签范围（[0,1] 归一化 indel 效率）
    train_range = {"min": float(np.nanmin(y_wt[train_mask])),
                   "max": float(np.nanmax(y_wt[train_mask]))}

    # 2. 逐数据集评估
    results = {}
    rows = []
    fig_axes = None
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
        pred_wt, _, _ = predict_pipeline(Xt, trained)

        # 标签尺度
        lab = labels_k
        lmin, lmax = float(lab.min()), float(lab.max())
        # 原始 MSE（尺度混比）
        m_raw = compute_metrics(lab, pred_wt)
        # min-max 归一化后 MSE（两个序列都按本数据集标签范围映射到 [0,1]）
        span = max(lmax - lmin, 1e-9)
        lab_n = (lab - lmin) / span
        pred_n = (pred_wt - lmin) / span
        m_norm = compute_metrics(lab_n, pred_n)
        # 预测值的范围（检查预测是否被压在某区间）
        prange = (float(pred_wt.min()), float(pred_wt.max()))
        results[d] = {
            "n": int(len(lab)), "removed": int(len(seqs) - len(keep)),
            "label_min": lmin, "label_max": lmax, "label_mean": float(lab.mean()),
            "label_sd": float(lab.std()),
            "in_0_1": bool(lmin >= 0 and lmax <= 1),
            "raw_mse": m_raw["mse"], "spearman": m_raw["spearman"],
            "pearson": m_raw["pearson"],
            "norm_mse": m_norm["mse"],
            "pred_min": prange[0], "pred_max": prange[1],
        }
        rows.append({"dataset": d, **results[d]})
        print(f"{d:20s} n={len(lab):5d} label=[{lmin:.3f},{lmax:.3f}] "
              f"in[0,1]={lmin>=0 and lmax<=1}  rawMSE={m_raw['mse']:.3f} "
              f"normMSE={m_norm['mse']:.4f} ρ={m_raw['spearman']:.3f} "
              f"pred=[{prange[0]:.3f},{prange[1]:.3f}]", flush=True)

    with open(os.path.join(OUT, "external_label_scale.json"), "w") as f:
        json.dump({"train_range": train_range, "datasets": results}, f, indent=2)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "external_label_scale.csv"), index=False)

    # 3. 画图：标签范围对比
    ds_names = [r["dataset"] for r in rows]
    mins = [r["label_min"] for r in rows]
    maxs = [r["label_max"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ypos = np.arange(len(ds_names))
    for yi, (mn, mx) in enumerate(zip(mins, maxs)):
        ax.plot([mn, mx], [yi, yi], "o-", color="steelblue", lw=2.5, ms=5)
        ax.text(mx + 0.05, yi, f"{mn:.2f}~{mx:.2f}", fontsize=8, va="center")
    ax.axvspan(0, 1, color="orange", alpha=0.15, label="DeepHF training [0,1]")
    ax.set_yticks(ypos); ax.set_yticklabels(ds_names, fontsize=8)
    ax.set_xlabel("Efficiency label range")
    ax.set_title("(A) Label scales across the 10 external datasets")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 散点：尺度错配最严重的 Doench_A375（标签含负值，疑似 log2 尺度）
    ax = axes[1]
    d_sel = "Doench_A375_2016"
    dirpath = os.path.join(TEST_ROOT, d_sel)
    seqs, labels, shape_dict = load_test_set(dirpath)
    keep = [i for i, s in enumerate(seqs) if s not in train_seqs]
    seqs_k = [seqs[i] for i in keep]
    shape_k = {sh: v[keep] for sh, v in shape_dict.items()}
    df_feat = build_test_features(seqs_k, shape_k)
    for c in feat_names:
        if c not in df_feat.columns:
            df_feat[c] = 0.0
    pred_wt, _, _ = predict_pipeline(df_feat[feat_names].values.astype(np.float64), trained)
    ax.scatter(labels[keep], pred_wt, s=6, alpha=0.4, color="steelblue")
    ax.set_xlabel("True efficiency (Doench-A375, log-ish scale)")
    ax.set_ylabel("MERCI prediction (DeepHF [0,1] scale)")
    ax.set_title("(B) Scale mismatch: Doench-A375 labels vs [0,1] predictions")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_ext_label_scale.png"))
    plt.close()
    print("saved figures/fig_ext_label_scale.png", flush=True)
    print("DONE analyze_label_scale", flush=True)


if __name__ == "__main__":
    main()
