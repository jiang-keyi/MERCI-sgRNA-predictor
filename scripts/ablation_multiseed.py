# -*- coding: utf-8 -*-
"""
多随机种子消融实验（审稿大修核心实验 2）

设计（逐步累加模块，隔离每个模块的贡献）：
  M0 plain       : 单任务 LightGBM（全特征、无加权、无集成）
  M1 weight      : 单任务 LightGBM + 不确定性加权
  M2 mtl-direct  : 多任务（三头直接预测 y_esp/y_hf1，无残差、无集成、无加权）
  M3 residual    : 多任务残差（Δesp/ΔHF1，全特征、无集成、无加权）
  M4 bagging-avg : 特征子空间 bagging + 平均（无 stacking、无加权、含残差）
  M5 bagging+stack: bagging + stacking（无加权、含残差）
  M6 MERCI       : 加权 + 残差 + bagging + stacking（完整模型）

每个配置跑 5 个独立随机种子，报告 mean ± SD。
输出：results/revision/ablation_multiseed.json / .csv
"""
import os
import sys
import json
import gc
import time
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from models import (train_pipeline, predict_pipeline, compute_metrics,
                    uncertainty_weights)

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PROC = os.path.join(BASE, "data", "processed")
OUT = os.path.join(BASE, "results", "revision")
os.makedirs(OUT, exist_ok=True)

SEEDS = [42, 202, 303, 404, 505]
# 说明：原论文代码将列采样误写为 "colsample"（LightGBM 忽略该参数），
# 实际生效配置即为下方的 4 个参数。为与已发表数字(0.813等)保持同源一致，
# 此处沿用实际生效配置，不引入未经验证的 subsample/colsample。
BASE_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
                   max_depth=-1)

# 配置定义
CONFIGS = [
    ("M0_plain",        dict(kind="plain")),
    ("M1_weight",       dict(kind="weight")),
    ("M2_mtl_direct",   dict(kind="pipeline", use_residual=False, use_stacking=False,
                            n_subsets=1, subset_ratio=1.0, weight_mode="ones")),
    ("M3_residual",     dict(kind="pipeline", use_residual=True, use_stacking=False,
                            n_subsets=1, subset_ratio=1.0, weight_mode="ones")),
    ("M4_bagging_avg",  dict(kind="pipeline", use_residual=True, use_stacking=False,
                            n_subsets=20, subset_ratio=0.7, weight_mode="ones")),
    ("M5_bagging_stack",dict(kind="pipeline", use_residual=True, use_stacking=True,
                            n_subsets=20, subset_ratio=0.7, weight_mode="ones")),
    ("M6_merci",        dict(kind="pipeline", use_residual=True, use_stacking=True,
                            n_subsets=20, subset_ratio=0.7, weight_mode="auto")),
]


def steiger(r_ab, r_ac, r_bc, n):
    """Steiger 1980 相关差异检验（两个共享样本的相关系数比较）。"""
    from scipy import stats
    if None in (r_ab, r_ac, r_bc) or np.isnan(r_ab) or np.isnan(r_ac) or np.isnan(r_bc):
        return {"z": np.nan, "p": np.nan}
    r_ab = float(np.clip(r_ab, -0.9999, 0.9999))
    r_ac = float(np.clip(r_ac, -0.9999, 0.9999))
    r_bc = float(np.clip(r_bc, -0.9999, 0.9999))
    f_ab = 0.5 * np.log((1 + r_ab) / (1 - r_ab))
    f_ac = 0.5 * np.log((1 + r_ac) / (1 - r_ac))
    z = (f_ab - f_ac) / np.sqrt((2 - 2 * r_bc) / max(n - 3, 1))
    return {"z": float(z), "p": float(2 * (1 - stats.norm.cdf(abs(z))))}


def fit_plain(Xtr, ytr, sw, seed):
    """单任务 LGBM（一个目标），返回模型。"""
    m = lgb.LGBMRegressor(**BASE_PARAMS, random_state=seed)
    m.fit(Xtr, ytr, sample_weight=sw)
    return m


def run_config(name, cfg, Xtr, yw_tr, ye_tr, yh_tr, sw_ones, Xva, yw_va, ye_va, yh_va, seeds):
    """跑一个配置的全部种子，返回 {seed: {target: metrics}}。
    支持断点续跑：partial 文件中已完成的 seed 直接跳过。"""
    partial_path = os.path.join(OUT, "ablation_multiseed_partial.json")
    done_seeds = set()
    prev = {}
    if os.path.exists(partial_path):
        try:
            with open(partial_path) as f:
                prev = json.load(f, parse_constant=lambda x: float("nan"))
            if name in prev:
                done_seeds = set(prev[name].keys())
        except Exception as e:
            print(f"  WARN: partial load failed ({e}); rerunning {name}", flush=True)
            done_seeds = set()
    per_seed = {}
    for seed in seeds:
        if str(seed) in done_seeds:
            per_seed[seed] = prev[name][str(seed)]
            print(f"  [{name}] seed={seed} (resumed from partial)", flush=True)
            continue
        t0 = time.time()
        if cfg["kind"] == "plain":
            trained = {}
            preds = {}
            for tgt, ytr, yva in [("wt", yw_tr, yw_va), ("esp", ye_tr, ye_va), ("hf1", yh_tr, yh_va)]:
                m = fit_plain(Xtr, ytr, None, seed)
                preds[tgt] = m.predict(Xva)
            pred_wt, pred_esp, pred_hf1 = preds["wt"], preds["esp"], preds["hf1"]
        elif cfg["kind"] == "weight":
            # 不确定性加权单任务：权重基于 WT 目标 OOF 方差
            sw, _, _ = uncertainty_weights(Xtr, yw_tr, tau=None)
            preds = {}
            for tgt, ytr, yva in [("wt", yw_tr, yw_va), ("esp", ye_tr, ye_va), ("hf1", yh_tr, yh_va)]:
                m = fit_plain(Xtr, ytr, sw, seed)
                preds[tgt] = m.predict(Xva)
            pred_wt, pred_esp, pred_hf1 = preds["wt"], preds["esp"], preds["hf1"]
        else:
            sw = None if cfg["weight_mode"] == "auto" else sw_ones
            trained = train_pipeline(
                Xtr, yw_tr, ye_tr, yh_tr, sample_weight=sw,
                n_subsets=cfg["n_subsets"], subset_ratio=cfg["subset_ratio"],
                lambda_esp=1.0, lambda_hf1=1.0, base_params=BASE_PARAMS,
                seed=seed, verbose=False, use_residual=cfg["use_residual"],
                use_stacking=cfg["use_stacking"])
            pred_wt, pred_esp, pred_hf1 = predict_pipeline(Xva, trained)
            del trained
        gc.collect()
        m_wt = compute_metrics(yw_va, pred_wt)
        m_esp = compute_metrics(ye_va, pred_esp)
        m_hf1 = compute_metrics(yh_va, pred_hf1)
        per_seed[seed] = {"wt": m_wt, "esp": m_esp, "hf1": m_hf1,
                          "pred_wt": pred_wt.tolist()}
        print(f"  [{name}] seed={seed} WT ρ={m_wt['spearman']:.4f} "
              f"eSp ρ={m_esp['spearman']:.4f} HF1 ρ={m_hf1['spearman']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        # 逐种子保存，防中断丢失
        merged_partial = dict(prev)
        merged_partial[name] = per_seed
        with open(partial_path, "w") as f:
            json.dump(merged_partial, f, default=str)
    return per_seed


def main():
    merged = pd.read_pickle(os.path.join(PROC, "merged_all.pkl"))
    features = pd.read_pickle(os.path.join(PROC, "features_all.pkl"))
    X = features.values.astype(np.float64)
    y_wt = merged["y_wt"].values.astype(np.float64)
    y_esp = merged["y_esp"].values.astype(np.float64)
    y_hf1 = merged["y_hf1"].values.astype(np.float64)
    train_mask = (merged["split"] == "train").values
    val_mask = (merged["split"] == "val").values

    # 三标签齐全的训练子集（与 train_pipeline 内部选择一致）
    cm = train_mask & ~(np.isnan(y_wt) | np.isnan(y_esp) | np.isnan(y_hf1))
    Xtr, yw_tr = X[cm], y_wt[cm]
    ye_tr, yh_tr = y_esp[cm], y_hf1[cm]
    sw_ones = np.ones(len(yw_tr))
    Xva, yw_va = X[val_mask], y_wt[val_mask]
    ye_va, yh_va = y_esp[val_mask], y_hf1[val_mask]
    print(f"train(3-label)={len(yw_tr)}  val={val_mask.sum()}", flush=True)

    all_data = {}
    for name, cfg in CONFIGS:
        print(f"=== {name} ===", flush=True)
        all_data[name] = run_config(name, cfg, Xtr, yw_tr, ye_tr, yh_tr,
                                    sw_ones, Xva, yw_va, ye_va, yh_va, SEEDS)
        # 边跑边存，防中断丢失
        with open(os.path.join(OUT, "ablation_multiseed_partial.json"), "w") as f:
            json.dump(all_data, f, default=str)

    # ---- 汇总 mean ± SD ----
    rows = []
    for name, per_seed in all_data.items():
        for tgt in ["wt", "esp", "hf1"]:
            sp = [per_seed[s][tgt]["spearman"] for s in SEEDS]
            pe = [per_seed[s][tgt]["pearson"] for s in SEEDS]
            ms = [per_seed[s][tgt]["mse"] for s in SEEDS]
            rows.append({
                "config": name, "target": tgt,
                "spearman_mean": float(np.mean(sp)), "spearman_sd": float(np.std(sp)),
                "pearson_mean": float(np.mean(pe)), "pearson_sd": float(np.std(pe)),
                "mse_mean": float(np.mean(ms)), "mse_sd": float(np.std(ms)),
                "seeds": [float(s) for s in sp],
            })
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "ablation_multiseed.csv"), index=False)

    # ---- Steiger 检验：M6 vs 每个消融（WT）----
    steiger_rows = []
    n_val_wt = int((~np.isnan(yw_va)).sum())
    for name in [c[0] for c in CONFIGS if c[0] != "M6_merci"]:
        zs, ps = [], []
        for seed in SEEDS:
            r_full = all_data["M6_merci"][seed]["wt"]["spearman"]
            r_abl = all_data[name][seed]["wt"]["spearman"]
            p_full = np.array(all_data["M6_merci"][seed]["pred_wt"])
            p_abl = np.array(all_data[name][seed]["pred_wt"])
            mask = ~(np.isnan(yw_va) | np.isnan(p_full) | np.isnan(p_abl))
            from scipy.stats import spearmanr
            r_bc = spearmanr(p_full[mask], p_abl[mask])[0]
            st = steiger(r_full, r_abl, r_bc, int(mask.sum()))
            zs.append(st["z"]); ps.append(st["p"])
        steiger_rows.append({"vs": name, "z_mean": float(np.nanmean(zs)),
                             "p_mean": float(np.nanmean(ps)),
                             "p_all": [float(p) for p in ps]})
    with open(os.path.join(OUT, "ablation_multiseed.json"), "w") as f:
        json.dump({"per_seed": all_data, "summary": rows, "steiger": steiger_rows},
                  f, indent=2, default=str)
    print("\n=== SUMMARY (mean ± SD) ===", flush=True)
    for r in rows:
        print(f"{r['config']:16s} {r['target']:3s} ρ={r['spearman_mean']:.4f}±{r['spearman_sd']:.4f} "
              f"MSE={r['mse_mean']:.4f}±{r['mse_sd']:.4f}", flush=True)
    print("\n=== STEIGER (M6 vs ablation, WT) ===", flush=True)
    for s in steiger_rows:
        print(f"{s['vs']:16s} z={s['z_mean']:.3f} p={s['p_mean']:.4f}", flush=True)
    print("DONE ablation_multiseed", flush=True)


if __name__ == "__main__":
    main()
