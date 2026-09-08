# -*- coding: utf-8 -*-
"""
特征族消融（审稿建议 6）
序列(178) → +热力学(4) → +DNA形状(35) → +交互(8)，逐族累加，
用完整 MERCI 流水线（seed=42），看每个特征族的边际贡献。
输出：results/revision/feature_family_ablation.json
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from models import train_pipeline, predict_pipeline, compute_metrics

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PROC = os.path.join(BASE, "data", "processed")
OUT = os.path.join(BASE, "results", "revision")
os.makedirs(OUT, exist_ok=True)

BASE_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=-1)


def main():
    merged = pd.read_pickle(os.path.join(PROC, "merged_all.pkl"))
    features = pd.read_pickle(os.path.join(PROC, "features_all.pkl"))
    cols = list(features.columns)
    seq_cols = [c for c in cols if c.startswith(("kmer", "pos", "polyT", "gc", "base"))
                or c in ("at_content", "at_ratio")]
    thermo_cols = [c for c in cols if c.startswith(("tm_", "dG_"))]
    inter_cols = [c for c in cols if c.startswith("dimer_") or "_x_" in c]
    inter_set = set(inter_cols)
    shape_cols = [c for c in cols if c.split("_")[0] in
                  ("MGW", "HelT", "Roll", "ProT", "EP") and c not in inter_set]
    assert len(seq_cols) == 178 and len(thermo_cols) == 4 and \
        len(shape_cols) == 35 and len(inter_cols) == 8, \
        f"counts: seq={len(seq_cols)} thermo={len(thermo_cols)} " \
        f"shape={len(shape_cols)} inter={len(inter_cols)}"
    print(f"seq={len(seq_cols)} thermo={len(thermo_cols)} "
          f"shape={len(shape_cols)} inter={len(inter_cols)}", flush=True)

    families = [seq_cols, seq_cols + thermo_cols,
                seq_cols + thermo_cols + shape_cols,
                seq_cols + thermo_cols + shape_cols + inter_cols]
    names = ["sequence", "+thermodynamic", "+DNA-shape", "+interaction (full)"]

    X = features.values.astype(np.float64)
    y_wt = merged["y_wt"].values.astype(np.float64)
    y_esp = merged["y_esp"].values.astype(np.float64)
    y_hf1 = merged["y_hf1"].values.astype(np.float64)
    train_mask = (merged["split"] == "train").values
    val_mask = (merged["split"] == "val").values
    cm = train_mask & ~(np.isnan(y_wt) | np.isnan(y_esp) | np.isnan(y_hf1))
    idx_tr = np.where(cm)[0]
    idx_va = np.where(val_mask)[0]

    rows = []
    for name, fam in zip(names, families):
        keep_idx = [features.columns.get_loc(c) for c in fam]
        Xtr = X[np.ix_(idx_tr, keep_idx)]
        Xva = X[np.ix_(idx_va, keep_idx)]
        trained = train_pipeline(Xtr, y_wt[idx_tr], y_esp[idx_tr], y_hf1[idx_tr],
                                 sample_weight=None, n_subsets=20, subset_ratio=0.7,
                                 base_params=BASE_PARAMS, seed=42, verbose=False)
        pw, pe, ph = predict_pipeline(Xva, trained)
        m_wt = compute_metrics(y_wt[idx_va], pw)
        m_esp = compute_metrics(y_esp[idx_va], pe)
        m_hf1 = compute_metrics(y_hf1[idx_va], ph)
        row = {"family": name, "n_features": len(fam),
               "WT_rho": m_wt["spearman"], "eSp_rho": m_esp["spearman"],
               "HF1_rho": m_hf1["spearman"], "WT_mse": m_wt["mse"]}
        rows.append(row)
        print(f"{name:24s} n={len(fam):3d}  WT ρ={m_wt['spearman']:.4f} "
              f"eSp ρ={m_esp['spearman']:.4f} HF1 ρ={m_hf1['spearman']:.4f}",
              flush=True)

    with open(os.path.join(OUT, "feature_family_ablation.json"), "w") as f:
        json.dump(rows, f, indent=2)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "feature_family_ablation.csv"), index=False)
    print("DONE feature_family_ablation", flush=True)


if __name__ == "__main__":
    main()
