# -*- coding: utf-8 -*-
"""消融/特征族消融结果回填到英文版与中文版 tex。
读取 results/revision/ablation_multiseed.json + feature_family_ablation.json，
替换 \FILLMAIN / \FILLABL / \FILLFAM 占位，并更新第 3.2 节叙述与摘要数字。
用法: python fill_paper_numbers.py
"""
import os
import json
import re

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(BASE, "results", "revision")
EN = os.path.join(BASE, "paper", "MERCI_paper.tex")
ZH = os.path.join(BASE, "paper", "MERCI_paper_zh.tex")


def fmt(x):
    return f"{x:.4f}"


def load():
    abl = json.load(open(os.path.join(RES, "ablation_multiseed.json")))
    by = {(r["config"], r["target"]): r for r in abl["summary"]}
    fam = None
    fam_path = os.path.join(RES, "feature_family_ablation.json")
    if os.path.exists(fam_path):
        fam = json.load(open(fam_path))
    return abl, fam, by


def main():
    abl, fam, by = load()

    # ---------- Table 1 / 摘要：M6_merci 主表 ----------
    m6 = {t: by[("M6_merci", t)] for t in ("wt", "esp", "hf1")}
    t1_rows = {
        "WT":  (m6["wt"],  "0.839"),
        "eSp": (m6["esp"], "0.798"),
        "HF1": (m6["hf1"], "0.817"),
    }
    # 旧占位（英文版原文数字）
    main_map = [
        (r"\FILLMAIN{0.813}", f"{m6['wt']['spearman_mean']:.3f}"),
        (r"\FILLMAIN{0.802}", f"{m6['esp']['spearman_mean']:.3f}"),
        (r"\FILLMAIN{0.804}", f"{m6['hf1']['spearman_mean']:.3f}"),
        (r"\FILLMAIN{0.002}", f"{max(m6[t]['spearman_sd'] for t in m6):.3f}"),
        (r"\FILLMAIN{0.839}", f"{m6['wt']['pearson_mean']:.3f}"),
        (r"\FILLMAIN{0.798}", f"{m6['esp']['pearson_mean']:.3f}"),
        (r"\FILLMAIN{0.817}", f"{m6['hf1']['pearson_mean']:.3f}"),
        (r"\FILLMAIN{0.0140}", f"{m6['wt']['mse_mean']:.4f}"),
        (r"\FILLMAIN{0.0128}", f"{m6['esp']['mse_mean']:.4f}"),
        (r"\FILLMAIN{0.0143}", f"{m6['hf1']['mse_mean']:.4f}"),
    ]

    # ---------- 表 2：消融 ----------
    abl_rows = {}
    for cfg in ("M0_plain", "M1_weight", "M2_mtl_direct", "M3_residual",
                "M4_bagging_avg", "M5_bagging_stack", "M6_merci"):
        abl_rows[cfg] = {t: by[(cfg, t)] for t in ("wt", "esp", "hf1")}
    ab = []
    for cfg in abl_rows:
        r = abl_rows[cfg]
        ab.append((cfg, r["wt"]["spearman_mean"], r["wt"]["spearman_sd"],
                   r["esp"]["spearman_mean"], r["esp"]["spearman_sd"],
                   r["hf1"]["spearman_mean"], r["hf1"]["spearman_sd"]))

    # ---------- Steiger：M6 vs M0 (WT) ----------
    st = [s for s in abl["steiger"] if s["vs"] == "M0_plain"]
    if st:
        sz = st[0]["z_mean"]
        sp = st[0]["p_mean"]
    else:
        sz, sp = float("nan"), float("nan")
    st_text = f"$z={sz:.2f}$, $p={sp:.3f}$"

    # ---------- 表 4：特征族 ----------
    fam_map = []
    if fam:
        for row in fam:
            for tgt, col in (("WT", "WT_rho"), ("eSp", "eSp_rho"), ("HF1", "HF1_rho")):
                fam_map.append((row["family"], tgt, row[col]))

    # 摘要 0.81/0.80/0.80 -> M6 实值
    abs_en = f"0.{int(round(m6['wt']['spearman_mean']*100)):02d}/0.{int(round(m6['esp']['spearman_mean']*100)):02d}/0.{int(round(m6['hf1']['spearman_mean']*100)):02d}"
    abs_zh = f"0.{int(round(m6['wt']['spearman_mean']*100)):02d}/0.{int(round(m6['esp']['spearman_mean']*100)):02d}/0.{int(round(m6['hf1']['spearman_mean']*100)):02d}"

    # ================= 英文版 =================
    s = open(EN, encoding="utf-8").read()
    s = s.replace("validation Spearman correlations of 0.81/0.80/0.80",
                  f"validation Spearman correlations of {abs_en}")
    for old, new in main_map:
        s = s.replace(old, new)
    # 表 2 逐行
    ab_text = {
        "M0_plain":          f"{fmt(ab[0][1])}$\\pm${fmt(ab[0][2])} & {fmt(ab[0][3])}$\\pm${fmt(ab[0][4])} & {fmt(ab[0][5])}$\\pm${fmt(ab[0][6])}",
        "M1_weight":         f"{fmt(ab[1][1])}$\\pm${fmt(ab[1][2])} & {fmt(ab[1][3])}$\\pm${fmt(ab[1][4])} & {fmt(ab[1][5])}$\\pm${fmt(ab[1][6])}",
        "M2_mtl_direct":     f"{fmt(ab[2][1])}$\\pm${fmt(ab[2][2])} & {fmt(ab[2][3])}$\\pm${fmt(ab[2][4])} & {fmt(ab[2][5])}$\\pm${fmt(ab[2][6])}",
        "M3_residual":       f"{fmt(ab[3][1])}$\\pm${fmt(ab[3][2])} & {fmt(ab[3][3])}$\\pm${fmt(ab[3][4])} & {fmt(ab[3][5])}$\\pm${fmt(ab[3][6])}",
        "M4_bagging_avg":    f"{fmt(ab[4][1])}$\\pm${fmt(ab[4][2])} & {fmt(ab[4][3])}$\\pm${fmt(ab[4][4])} & {fmt(ab[4][5])}$\\pm${fmt(ab[4][6])}",
        "M5_bagging_stack":  f"{fmt(ab[5][1])}$\\pm${fmt(ab[5][2])} & {fmt(ab[5][3])}$\\pm${fmt(ab[5][4])} & {fmt(ab[5][5])}$\\pm${fmt(ab[5][6])}",
        "M6_merci":          f"{fmt(ab[6][1])}$\\pm${fmt(ab[6][2])} & {fmt(ab[6][3])}$\\pm${fmt(ab[6][4])} & {fmt(ab[6][5])}$\\pm${fmt(ab[6][6])}",
    }
    for cfg in ab_text:
        s = s.replace(r"\FILLABL{}" + " & " + r"\FILLABL{}" + " & " + r"\FILLABL{}",
                      ab_text[cfg], 1) if False else s
    # 逐行替换：第 i 个 "\FILLABL{} & \FILLABL{} & \FILLABL{}" 出现
    lines = s.split("\n")
    fill_abl = 0
    for i, ln in enumerate(lines):
        if r"\FILLABL{}" in ln and "&" in ln and r"$\pm$" not in ln:
            fill_abl += 1
            if fill_abl <= 6:  # 前 6 个占位行(不含 M0 已填行)
                cfg = list(ab_text.keys())[fill_abl]
                ln = ln.replace(r"\FILLABL{} & \FILLABL{} & \FILLABL{}", ab_text[cfg])
            elif fill_abl == 7:
                ln = ln.replace(r"\FILLABL{} & \FILLABL{} & \FILLABL{}", ab_text["M6_merci"])
            lines[i] = ln
    s = "\n".join(lines)
    # 第 3.2 节叙述：HF1 对比 与 Steiger
    s = s.replace(r"\FILLABL{0.804}$\pm$\FILLABL{0.002}",
                  f"{m6['hf1']['spearman_mean']:.3f}$\\pm${m6['hf1']['spearman_sd']:.3f}")
    s = s.replace(r"\FILLABL{0.797}$\pm$\FILLABL{0.002}",
                  f"{by[('M0_plain','hf1')]['spearman_mean']:.3f}$\\pm${by[('M0_plain','hf1')]['spearman_sd']:.3f}")
    s = s.replace(r"\FILLABL{$z=1.8$, $p=0.07$}", st_text)
    # ---------- 表 4：特征族（缺省时跳过） ----------
    fam_keys = ["sequence", "+thermodynamic", "+DNA-shape", "+interaction (full)"]
    if fam:
        lines = s.split("\n")
        fi = 0
        for i, ln in enumerate(lines):
            if r"\FILLFAM{}" in ln:
                row = next(r for r in fam if r["family"] == fam_keys[fi])
                ln = ln.replace(r"\FILLFAM{} & \FILLFAM{} & \FILLFAM{}",
                                f"{row['WT_rho']:.4f} & {row['eSp_rho']:.4f} & {row['HF1_rho']:.4f}")
                fi += 1
                lines[i] = ln
        s = "\n".join(lines)
        open(EN, "w", encoding="utf-8", newline="").write(s)
        print("EN updated (with feature family).")
    else:
        open(EN, "w", encoding="utf-8", newline="").write(s)
        print("EN updated (feature family pending).")

    # ================= 中文版 =================
    z = open(ZH, encoding="utf-8").read()
    z = z.replace("分别达到 0.81/0.80/0.80 的 Spearman 相关",
                  f"分别达到 {abs_zh} 的 Spearman 相关")
    for old, new in main_map:
        z = z.replace(old, new)
    z = z.replace(r"\FILLABL{0.804}$\pm$\FILLABL{0.002}",
                  f"{m6['hf1']['spearman_mean']:.3f}$\\pm${m6['hf1']['spearman_sd']:.3f}")
    z = z.replace(r"\FILLABL{0.797}$\pm$\FILLABL{0.002}",
                  f"{by[('M0_plain','hf1')]['spearman_mean']:.3f}$\\pm${by[('M0_plain','hf1')]['spearman_sd']:.3f}")
    z = z.replace(r"\FILLABL{$z=1.8$, $p=0.07$}", st_text)
    zlines = z.split("\n")
    fill_abl = 0
    for i, ln in enumerate(zlines):
        if r"\FILLABL{}" in ln and "&" in ln and r"$\pm$" not in ln:
            fill_abl += 1
            cfg = list(ab_text.keys())[fill_abl] if fill_abl <= 7 else None
            if cfg:
                ln = ln.replace(r"\FILLABL{} & \FILLABL{} & \FILLABL{}", ab_text[cfg])
                zlines[i] = ln
    z = "\n".join(zlines)
    fi = 0
    zlines = z.split("\n")
    if fam:
        for i, ln in enumerate(zlines):
            if r"\FILLFAM{}" in ln:
                row = next(r for r in fam if r["family"] == fam_keys[fi])
                ln = ln.replace(r"\FILLFAM{} & \FILLFAM{} & \FILLFAM{}",
                                f"{row['WT_rho']:.4f} & {row['eSp_rho']:.4f} & {row['HF1_rho']:.4f}")
                fi += 1
                zlines[i] = ln
    z = "\n".join(zlines)
    open(ZH, "w", encoding="utf-8", newline="").write(z)
    print("ZH updated.")

    # ---------- 打印回填摘要 ----------
    print("\n=== M6 (full MERCI) ===")
    for t in ("wt", "esp", "hf1"):
        r = m6[t]
        print(f"  {t}: rho {r['spearman_mean']:.4f}+-{r['spearman_sd']:.4f}  "
              f"pearson {r['pearson_mean']:.4f}  mse {r['mse_mean']:.4f}")
    print("\n=== ablation table ===")
    for cfg, *v in ab:
        print(f"  {cfg:16s} WT {v[0]:.4f}+-{v[1]:.4f}  eSp {v[2]:.4f}+-{v[3]:.4f}  "
              f"HF1 {v[4]:.4f}+-{v[5]:.4f}")
    print("\n=== Steiger M6 vs M0 (WT) ===")
    print(f"  z={sz:.3f}  p={sp:.4f}")
    print("\n=== feature family ===")
    if fam:
        for row in fam:
            print(f"  {row['family']:20s} n={row['n_features']:3d}  WT {row['WT_rho']:.4f}  "
                  f"eSp {row['eSp_rho']:.4f}  HF1 {row['HF1_rho']:.4f}")
    else:
        print("  (pending)")


if __name__ == "__main__":
    main()
