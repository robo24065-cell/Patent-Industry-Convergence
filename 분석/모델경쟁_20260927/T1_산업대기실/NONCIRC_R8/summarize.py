# -*- coding: utf-8 -*-
"""A·C 결과 원점 평균 표 → res/A_summary.csv, res/C_summary.csv, res/C_delta.csv"""
import json, os, numpy as np, pandas as pd
R = "res"; C = [2008, 2009, 2010, 2011, 2012]
rows = []
for c in C:
    j = json.load(open(f"{R}/A_c{c}.json", encoding="utf-8"))
    for a, v in j["arms"].items():
        rows.append({"c": c, "arm": a, **v, "n_eval": j["n_eval"], "pos_eval": j["pos_eval"]})
A = pd.DataFrame(rows)
S = A.groupby("arm")[["auc", "auc_far", "top1000"]].mean().round(4)
print("A mean over 5 origins\n", S.loc[["NONE", "CW27", "P27", "P50", "P50CW"]].to_string())
for a, b in (("P50", "CW27"), ("P27", "CW27"), ("CW27", "NONE"), ("P50", "NONE"), ("P50CW", "CW27"), ("P50", "P27")):
    x = A[A.arm == a].set_index("c"); y = A[A.arm == b].set_index("c")
    print(f"{a}-{b}: auc wins {(x.auc > y.auc).sum()}/5, far mean d {round((x.auc_far - y.auc_far).mean(), 4)} wins {(x.auc_far > y.auc_far).sum()}/5, top1000 d {round((x.top1000 - y.top1000).mean(), 4)}")
A.to_csv(f"{R}/A_summary.csv", index=False, encoding="utf-8-sig")
print("A n_eval/pos", A.drop_duplicates("c")[["c", "n_eval", "pos_eval"]].values.tolist())
rows, drows = [], []
for c in C:
    j = json.load(open(f"{R}/C_c{c}.json", encoding="utf-8"))
    for Y in ("Y_P", "Y_C"):
        for a, v in j[Y]["arms"].items():
            rows.append({"c": c, "Y": Y, "arm": a, **v, "n_eval": j[Y]["n_eval"], "pos_eval": j[Y]["pos_eval"]})
        for k, v in j[Y]["delta"].items():
            drows.append({"c": c, "Y": Y, "pair": k, "d": v["d"], "lo": v["ci"][0], "hi": v["ci"][1]})
Cd = pd.DataFrame(rows); Dd = pd.DataFrame(drows)
print("\nC mean AUC (rows=features, cols=outcome)\n", Cd.pivot_table(index="arm", columns="Y", values="auc", aggfunc="mean").round(4).to_string())
print("C mean top500\n", Cd.pivot_table(index="arm", columns="Y", values="top500", aggfunc="mean").round(4).to_string())
g = Dd.groupby(["Y", "pair"]).agg(mean_d=("d", "mean"), wins=("d", lambda s: int((s > 0).sum())), lo_min=("lo", "min"), hi_max=("hi", "max"),
                                  sig_pos=("lo", lambda s: int((s > 0).sum())))
print(g.round(4).to_string())
print("C n_eval/pos", Cd.drop_duplicates(["c", "Y"])[["c", "Y", "n_eval", "pos_eval"]].values.tolist())
Cd.to_csv(f"{R}/C_summary.csv", index=False, encoding="utf-8-sig"); Dd.to_csv(f"{R}/C_delta.csv", index=False, encoding="utf-8-sig")
