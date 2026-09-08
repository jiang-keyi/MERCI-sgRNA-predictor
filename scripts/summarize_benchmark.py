# -*- coding: utf-8 -*-
"""
三方同测试集基准汇总（审稿建议 1 的核心表格）
MERCI vs DeepMEns vs DeepHF，同一批去重叠外部测试集。
读取：
  - results/tables/independent_test_results.json   (MERCI, WT)
  - results/revision/benchmark_deepmeans.json
  - results/revision/benchmark_deephf.json
输出：results/revision/benchmark_threeway.csv / .json + 表格文本
"""
import os
import json
import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(BASE, "results")
OUT = os.path.join(RES, "revision")
os.makedirs(OUT, exist_ok=True)

DATASETS = ['Chari_293T_2015', 'Doench_A375_2016', 'Doench_Mm_2014',
            'Doench_NB4_2014', 'Hart_Hct116_2016', 'Hart_hela_2016',
            'Moreno_Zb_2015', 'Ren_Ff_2015', 'Varshney_Zb_2015',
            'Wang_HL60_2014']


def main():
    merci = json.load(open(os.path.join(RES, "tables", "independent_test_results.json")))
    dm = {r["dataset"]: r for r in json.load(open(os.path.join(OUT, "benchmark_deepmeans.json")))["datasets"]}
    dh = {r["dataset"]: r for r in json.load(open(os.path.join(OUT, "benchmark_deephf.json")))["datasets"]}

    rows = []
    for ds in DATASETS:
        m = merci.get(ds, {}).get("WT", {})
        rows.append({
            "dataset": ds,
            "merci_n": m.get("n", merci.get(ds, {}).get("n", "")),
            "merci_spearman": m.get("spearman", ""),
            "deepmeans_spearman": dm[ds]["spearman"],
            "deephf_spearman": dh[ds]["spearman"],
            "deepmeans_pearson": dm[ds]["pearson"],
            "deephf_pearson": dh[ds]["pearson"],
        })
    df = pd.DataFrame(rows)
    # 均值行
    mean_row = {c: np.nanmean([float(r[c]) for r in rows if str(r[c]) != ""])
                for c in ["merci_spearman", "deepmeans_spearman", "deephf_spearman"]}
    df.loc[len(df)] = ["Mean", "", *[round(v, 4) for v in mean_row.values()],
                       "", ""]

    df.to_csv(os.path.join(OUT, "benchmark_threeway.csv"), index=False)
    print(df.to_string(index=False))
    print("\nDONE benchmark_threeway", flush=True)


if __name__ == "__main__":
    main()
