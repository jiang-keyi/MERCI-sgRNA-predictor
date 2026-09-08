# -*- coding: utf-8 -*-
"""
同测试集基准 1/2：DeepHF（同去重叠测试集）
================================================================
用 DeepHF fork 的预训练 DeepWt.hd5（RNN, SpCas9-WT），
在 MERCI 论文使用的同一批"去重叠"外部测试集上评估，
实现真正同数据集、同分割、同指标的对比。

DeepHF 模型输入：
  - X_seq    : (N, 22) 整数编码（START + 21mer 字符映射）
  - X_biofeat: (N, 11) [stem, dG, dG_binding_20, dg_binding_7to20,
                        gc_above_10, gc_below_10, gc_count,
                        Tm global, 5mer_end, 8mer_middle, 4mer_start]
  11 维生物特征从 DeepMEns 独立数据集文件夹中已预计算的
  bio_feature/*.csv 精确组装（结构特征由 RNAfold 预计算，无需本机安装）。

输出：results/revision/benchmark_deephf.json
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import scipy.stats as stats

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(BASE, "..", "data")          # DeepHF 权重（仓库根）
PROC = os.path.join(BASE, "data", "processed")   # 训练 pickle（research/data）
RES = os.path.join(BASE, "results")
OUT = os.path.join(RES, "revision")
TEST_ROOT = os.path.join(DATA, "DeepMEns", "independent_test_datasets")
MODEL_PATH = os.path.join(DATA, "DeepHF_models", "DeepWt.hd5")
os.makedirs(OUT, exist_ok=True)

DATASETS = ['Chari_293T_2015', 'Doench_A375_2016', 'Doench_Mm_2014',
            'Doench_NB4_2014', 'Hart_Hct116_2016', 'Hart_hela_2016',
            'Moreno_Zb_2015', 'Ren_Ff_2015', 'Varshney_Zb_2015',
            'Wang_HL60_2014']

CHAR_MAP = {'A': 2, 'T': 3, 'C': 4, 'G': 5}  # DeepHF: PAD=0, START=1


def encode_seq(seqs, length=21):
    """复刻 DeepHF make_data：START(1) + 字符映射(A=2,T=3,C=4,G=5) + 前填充到 22。
    未知字符（如 N）按 keras Tokenizer 行为丢弃，短序列 pre-pad 0。"""
    out = np.zeros((len(seqs), length + 1), dtype=np.int32)
    for i, s in enumerate(seqs):
        toks = [CHAR_MAP[c] for c in str(s).upper() if c in CHAR_MAP][:length]
        body = [1] + toks                       # START + 21 字符
        out[i, length + 1 - len(body):] = body  # pre-pad 0 到 22
    return out


def read_feature_matrix(path, expected_cols=None):
    """读取特征矩阵：自动去掉误读的整数索引列（'Unnamed: 0' 或 0..n-1）。"""
    df = pd.read_csv(path)
    arr = df.values.astype(np.float32)
    if arr.shape[1] == expected_cols:
        return arr
    if arr.shape[1] == expected_cols + 1:
        first = arr[:, 0]
        if np.all(first == np.arange(len(first))):
            return arr[:, 1:]
    return arr[:, :expected_cols]


def assemble_bio11_full(folder):
    """返回对齐到最小行数的 (n, 11) 生物特征（DeepHF 顺序）。"""
    bf = os.path.join(folder, 'bio_feature')
    ds = os.path.basename(folder)
    sp = None
    for cand in [f'{ds}_structure_feature.csv', f'{ds}_struct_feature.csv',
                 f'{ds}_feature.csv']:
        p = os.path.join(bf, cand)
        if os.path.exists(p):
            sp = read_feature_matrix(p, expected_cols=4)
            break
    assert sp is not None, f'{ds}: no structure feature file'
    struct = sp
    g_above = pd.read_csv(os.path.join(bf, f'{ds}_gc_above_10.csv'),
                          header=None).values.astype(np.float32).reshape(-1, 1)
    g_below = pd.read_csv(os.path.join(bf, f'{ds}_gc_below_10.csv'),
                          header=None).values.astype(np.float32).reshape(-1, 1)
    g_count = pd.read_csv(os.path.join(bf, f'{ds}_gc_count.csv'),
                          header=None).values.astype(np.float32).reshape(-1, 1)
    tm = read_feature_matrix(os.path.join(bf, f'{ds}_Tm_feature.csv'),
                             expected_cols=4)
    n = min(len(struct), len(g_above), len(g_below), len(g_count), len(tm))
    if n < len(tm):
        print(f"  WARN {ds}: truncating bio rows {len(tm)} -> {n}", flush=True)
    return np.hstack([struct[:n], g_above[:n], g_below[:n], g_count[:n], tm[:n]])


def main():
    import tensorflow as tf
    from tensorflow import keras
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.config.set_visible_devices([], 'GPU')  # 强制 CPU

    print("Loading DeepHF DeepWt model ...", flush=True)
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded.", flush=True)

    # 训练集 seq（去重叠用）
    merged = pd.read_pickle(os.path.join(PROC, "merged_all.pkl"))
    train_seqs = set(merged.loc[merged["split"] == "train", "seq21"].tolist())

    rows = []
    for ds in DATASETS:
        folder = os.path.join(TEST_ROOT, ds)
        df = pd.read_csv(os.path.join(folder, f'{ds}.csv'))
        if 'seq' in df.columns:
            seqs = df['seq'].astype(str).tolist()
        elif '21mer' in df.columns:
            seqs = df['21mer'].astype(str).tolist()
        else:
            seqs = df[df.columns[0]].astype(str).tolist()
        y = df['Efficiency'].astype(np.float32).values
        X_bio_full = assemble_bio11_full(folder)
        n = X_bio_full.shape[0]
        if n < len(seqs):
            print(f"  WARN {ds}: aligning rows {len(seqs)} -> {n}", flush=True)
            seqs = seqs[:n]
            y = y[:n]
        keep = [i for i, s in enumerate(seqs) if s not in train_seqs]
        seqs_k = [seqs[i] for i in keep]
        y_k = y[keep]
        X_seq = encode_seq(seqs_k)
        X_bio = X_bio_full[keep]
        assert len(seqs_k) == X_bio.shape[0]
        pred = model.predict([X_seq, X_bio], batch_size=4096, verbose=0).reshape(-1)
        pred = np.clip(pred, 0, 1)
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
    with open(os.path.join(OUT, "benchmark_deephf.json"), "w") as f:
        json.dump({"model": "DeepHF (DeepWt, pretrained)",
                   "note": "same overlap-filtered test sets as MERCI",
                   "mean_spearman": mean_sp, "datasets": rows}, f, indent=2)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "benchmark_deephf.csv"), index=False)
    print("DONE benchmark_deephf", flush=True)


if __name__ == "__main__":
    main()
