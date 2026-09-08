# -*- coding: utf-8 -*-
"""
同测试集基准 2/2：DeepMEns（同去重叠测试集）
================================================================
用 DeepMEns 预训练的 5 个 Wt-SpCas9 子模型，
在 MERCI 论文使用的同一批"去重叠"外部测试集上评估。
与 DeepHF / MERCI 构成真正同数据集、同分割、同指标的对比。

输出：results/revision/benchmark_deepmeans.json
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import scipy.stats as stats

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(BASE, "..", "data")          # DeepMEns 模型/测试集（仓库根）
PROC = os.path.join(BASE, "data", "processed")   # 训练 pickle（research/data）
RES = os.path.join(BASE, "results")
OUT = os.path.join(RES, "revision")
TEST_ROOT = os.path.join(DATA, "DeepMEns", "independent_test_datasets")
MODEL_DIR = os.path.join(DATA, "DeepMEns")
os.makedirs(OUT, exist_ok=True)

DATASETS = ['Chari_293T_2015', 'Doench_A375_2016', 'Doench_Mm_2014',
            'Doench_NB4_2014', 'Hart_Hct116_2016', 'Hart_hela_2016',
            'Moreno_Zb_2015', 'Ren_Ff_2015', 'Varshney_Zb_2015',
            'Wang_HL60_2014']

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")
from predict_deepmeans import (build_features_for_independent,  # noqa: E402
                               load_ensemble, predict_ensemble)


def main():
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.config.set_visible_devices([], 'GPU')

    merged = pd.read_pickle(os.path.join(PROC, "merged_all.pkl"))
    train_seqs = set(merged.loc[merged["split"] == "train", "seq21"].tolist())

    wt_models = load_ensemble(MODEL_DIR, 'Wt')
    print(f"loaded {len(wt_models)} Wt DeepMEns models", flush=True)

    rows = []
    for ds in DATASETS:
        feats = build_features_for_independent(DATA, ds)
        y = feats['y'].reshape(-1)
        seqs = feats['seqs']
        keep = [i for i, s in enumerate(seqs) if s not in train_seqs]
        inputs = [v[keep] for v in feats['inputs']]
        y_k = y[keep]
        pred = predict_ensemble(wt_models, inputs).reshape(-1)
        sp = stats.spearmanr(y_k, pred)[0]
        pe = stats.pearsonr(y_k, pred)[0]
        mse = float(np.mean((y_k - pred) ** 2))
        rows.append({"dataset": ds, "n": int(len(y_k)),
                     "spearman": float(sp), "pearson": float(pe),
                     "mse": mse})
        print(f"  {ds:20s} n={len(y_k):5d} ρ={sp:.4f} Pearson={pe:.4f} MSE={mse:.4f}",
              flush=True)

    mean_sp = float(np.mean([r["spearman"] for r in rows]))
    print(f"MEAN Spearman = {mean_sp:.4f}", flush=True)
    with open(os.path.join(OUT, "benchmark_deepmeans.json"), "w") as f:
        json.dump({"model": "DeepMEns (5-Wt ensemble, pretrained)",
                   "note": "same overlap-filtered test sets as MERCI",
                   "mean_spearman": mean_sp, "datasets": rows}, f, indent=2)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "benchmark_deepmeans.csv"), index=False)
    print("DONE benchmark_deepmeans", flush=True)


if __name__ == "__main__":
    main()
