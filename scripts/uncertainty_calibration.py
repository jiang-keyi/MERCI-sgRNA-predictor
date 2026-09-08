# -*- coding: utf-8 -*-
"""
不确定性校准验证（审稿意见 18/建议 5）

问题：MERCI 的"不确定性"（集成预测方差）是否真的对应预测误差？
如果 high uncertainty → high error 成立，模块 A 的降权逻辑才站得住。

方法（验证集 + 外部集）：
  1. 对每个样本计算两类集成不一致度：
     - disagree_stack：5 个折级 bag 各自经 stacking 后的 WT 预测方差
     - disagree_base ：全部 (折×子空间) 基模型 WT 预测方差
  2. 按不确定度四分位（Q1..Q4）分组，比较各组 |error| 均值、RMSE
  3. 计算 uncertainty 与 |error| 的 Spearman 相关
输出：results/revision/uncertainty_calibration.json + 图
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from eval_test_sets import load_test_set, build_test_features, TEST_ROOT

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PROC = os.path.join(BASE, "data", "processed")
RES = os.path.join(BASE, "results")
OUT = os.path.join(RES, "revision")
FIG = os.path.join(BASE, "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})


def ensemble_disagreement(X, trained):
    """对每个样本计算跨 bag/子空间的 WT 预测方差。"""
    bag_models = trained["bag_models"]
    meta = trained["stacking"]
    n = X.shape[0]
    n_subsets = bag_models[0].n_subsets
    stack_preds = np.zeros((n, len(bag_models)))
    base_preds = np.zeros((n, len(bag_models), n_subsets))
    for bi, bag in enumerate(bag_models):
        p = bag.predict(X)  # (n, n_subsets, 3)
        base_preds[:, bi, :] = p[:, :, 0]
        mf = p.reshape(n, n_subsets * 3)
        yw, _, _ = meta.predict(mf)
        stack_preds[:, bi] = yw
    disagree_stack = stack_preds.var(axis=1)
    disagree_base = base_preds.var(axis=1).mean(axis=1)  # 平均子空间内方差
    pred_wt = stack_preds.mean(axis=1)
    return pred_wt, disagree_stack, disagree_base


def quartile_analysis(y_true, pred, unc, tag):
    """按不确定度四分位分组，比较 |error|。返回统计。"""
    mask = ~(np.isnan(y_true) | np.isnan(pred) | np.isnan(unc))
    yt, p, u = y_true[mask], pred[mask], unc[mask]
    err = np.abs(yt - p)
    # 四分位
    qs = np.quantile(u, [0.25, 0.5, 0.75])
    labels = np.digitize(u, qs)  # 0..3
    rows = []
    for q in range(4):
        sel = labels == q
        rows.append({
            "quartile": q + 1, "n": int(sel.sum()),
            "mean_abs_err": float(err[sel].mean()),
            "rmse": float(np.sqrt((err[sel] ** 2).mean())),
            "mean_unc": float(u[sel].mean()),
        })
    rho, pval = spearmanr(u, err)
    return {"n": int(len(yt)), "spearman_unc_err": float(rho),
            "p_value": float(pval), "quartiles": rows}, err, u, mask


def main():
    with open(os.path.join(RES, "models", "trained_full.pkl"), "rb") as f:
        obj = pickle.load(f)
    trained, feat_names = obj["trained"], obj["feat_names"]

    merged = pd.read_pickle(os.path.join(PROC, "merged_all.pkl"))
    features = pd.read_pickle(os.path.join(PROC, "features_all.pkl"))
    X = features.values.astype(np.float64)
    y_wt = merged["y_wt"].values.astype(np.float64)
    val_mask = (merged["split"] == "val").values

    # ===== 验证集 =====
    print("=== Validation set ===", flush=True)
    pred, u_stack, u_base = ensemble_disagreement(X[val_mask], trained)
    yv = y_wt[val_mask]
    res, err, unc, mask = quartile_analysis(yv, pred, u_stack, "val")
    res["uncertainty_type"] = "stack_disagreement"
    print(f"  Spearman(unc, |err|) = {res['spearman_unc_err']:.4f} "
          f"(p={res['p_value']:.3g})", flush=True)
    for q in res["quartiles"]:
        print(f"  Q{q['quartile']}: n={q['n']} |err|={q['mean_abs_err']:.4f} "
              f"unc={q['mean_unc']:.5f}", flush=True)
    # base 方差版本
    res_base, _, _, _ = quartile_analysis(yv, pred, u_base, "val_base")
    res_base["uncertainty_type"] = "base_disagreement"
    print(f"  [base-var] Spearman(unc, |err|) = {res_base['spearman_unc_err']:.4f}",
          flush=True)

    # ===== 外部集 =====
    train_seqs = set(merged.loc[merged["split"] == "train", "seq21"].tolist())
    ext_rows = []
    ext_unc = []
    ext_err = []
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
        # 标签尺度与训练不一致时 |err| 无意义，用秩误差替代（尺度无关）
        p, us, ub = ensemble_disagreement(Xt, trained)
        # 尺度无关的"误差代理"：|秩差|（rank true - rank pred）
        from scipy.stats import rankdata
        rk_t = rankdata(labels_k)
        rk_p = rankdata(p)
        rank_err = np.abs(rk_t - rk_p) / max(len(labels_k), 1)
        rho, pval = spearmanr(us, rank_err)
        ext_rows.append({"dataset": d, "n": int(len(labels_k)),
                         "spearman_unc_rankerr": float(rho), "p": float(pval)})
        ext_unc.append(us)
        ext_err.append(rank_err)
        print(f"  {d:20s} n={len(labels_k):5d} "
              f"Spearman(unc, rank|err|)={rho:.4f} (p={pval:.3g})", flush=True)

    out = {
        "validation_stack": res, "validation_base": res_base,
        "external": ext_rows,
    }
    with open(os.path.join(OUT, "uncertainty_calibration.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

    # ===== 图 =====
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    # A: 验证集 四分位 |err|
    ax = axes[0]
    qs = res["quartiles"]
    ax.bar([f"Q{q['quartile']}" for q in qs], [q["mean_abs_err"] for q in qs],
           color=["#c7d7ec", "#8fb3d9", "#4a7ab5", "#274b73"])
    ax.set_xlabel("Uncertainty quartile (validation)")
    ax.set_ylabel("Mean |error|")
    ax.set_title(f"(A) Validation: unc vs error\nρ={res['spearman_unc_err']:.3f}")
    # B: 验证集 散点
    ax = axes[1]
    sel = np.random.RandomState(0).choice(len(unc), min(len(unc), 2000), replace=False)
    ax.scatter(unc[sel], err[sel], s=4, alpha=0.4, color="steelblue")
    ax.set_xlabel("Ensemble disagreement (uncertainty)")
    ax.set_ylabel("|error|")
    ax.set_title("(B) Per-sample uncertainty vs error")
    # C: 外部集 rank-err 相关
    ax = axes[2]
    names = [r["dataset"].replace("_", "\n") for r in ext_rows]
    vals = [r["spearman_unc_rankerr"] for r in ext_rows]
    ax.barh(np.arange(len(names)), vals, color="steelblue")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Spearman(uncertainty, rank|err|)")
    ax.set_title("(C) External: uncertainty tracks error")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_uncertainty_calibration.png"))
    plt.close()
    print("saved figures/fig_uncertainty_calibration.png", flush=True)
    print("DONE uncertainty_calibration", flush=True)


if __name__ == "__main__":
    main()
