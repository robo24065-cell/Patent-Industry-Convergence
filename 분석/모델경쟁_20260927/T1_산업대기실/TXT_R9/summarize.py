# -*- coding: utf-8 -*-
"""res/C_c*.json → res/C_summary.csv(원점별·평균), res/C_pool_summary.csv"""
from r9_common import *
import glob
rows = []
for c in [2008, 2009, 2010, 2011, 2012]:
    j = json.load(open(os.path.join(RES, f"C_c{c}.json"), encoding="utf-8"))
    for a, v in j["arms"].items():
        rows.append({"c": c, "arm": a, **v, "n_eval": j["n_eval"], "pos_eval": j["pos_eval"]})
S = pd.DataFrame(rows); S.to_csv(os.path.join(RES, "C_summary.csv"), index=False, encoding="utf-8-sig")
print(S.groupby("arm", sort=False)[["auc", "auc_far", "top1000"]].mean().round(4).to_string())
R8 = pd.read_csv(os.path.join(NC8, "res", "A_summary.csv"))
m = S.merge(R8, on=["c", "arm"], suffixes=("", "_r8"))
print("R8 재현 최대 |ΔAUC|:", float((m.auc - m.auc_r8).abs().max()), "팔", sorted(m.arm.unique()))
pp = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(RES, "C_pool_*.json")))]
Q = pd.DataFrame([{"pair": r["pair"], "mean_d": round(r["mean_d"], 5), "ci_lo": round(r["ci95"][0], 5), "ci_hi": round(r["ci95"][1], 5),
                   "wins": f'{r["wins"]}/5', "per_origin": " ".join(f"{x:+.4f}" for x in r["per_origin_d"])} for r in pp])
Q.to_csv(os.path.join(RES, "C_pool_summary.csv"), index=False, encoding="utf-8-sig"); print(Q.to_string())
