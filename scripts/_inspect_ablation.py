# -*- coding: utf-8 -*-
import json

d = json.load(open(r"E:\claude\deepmeans_repo\research\results\revision\ablation_multiseed.json"))
print("keys:", list(d.keys()))
for r in d["summary"]:
    print("%-16s %-4s rho=%.4f±%.4f pearson=%.4f mse=%.4f"
          % (r["config"], r["target"], r["spearman_mean"], r["spearman_sd"],
             r["pearson_mean"], r["mse_mean"]))
print()
for s in d.get("steiger", []):
    print("steiger %s vs %s target=%s z=%.3f p=%.4f"
          % (s["a"], s["vs"], s.get("target", "wt"), s["z_mean"], s["p_mean"]))
