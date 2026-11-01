# -*- coding: utf-8 -*-
"""I1 견고성(개발 원점 2017–19만): (a) 더 촘촘한 비특허 기준선(업종 6자리·시군구 코드를 수치로 투입) 위에서도 특허 침투 피처가 더하는가
(b) 묶음 분해: 업종 침투 vs 지역 침투  (c) 상위 1,000개 기업 적중.  사후 점검용이며 채택 판단에는 쓰지 않는다."""
import os
import pandas as pd, numpy as np
from i00_common import *

D = pd.read_parquet(os.path.join(CACHE, "I1_panel.parquet"))
D["c6num"] = pd.to_numeric(D.indcd, errors="coerce")
D["sggnum"] = pd.to_numeric(D.sgg, errors="coerce")
F0 = ["log_emp", "g_emp1", "age", "hire_rate", "loss_rate", "wage", "n_wkpl", "div2", "sido"]
IND = ["pen_c6", "ent_c6", "pen_c4", "ent_c4"]; LOC = ["pen_sgg", "ent_sgg", "pen_sggd", "ent_sggd"]
V = [2017, 2018, 2019]
cfg = {"M0": F0, "M0fine": F0 + ["c6num", "sggnum"], "M0fine+IND+LOC": F0 + ["c6num", "sggnum"] + IND + LOC,
       "M0+IND": F0 + IND, "M0+LOC": F0 + LOC, "M1(IND+LOC)": F0 + IND + LOC}
P = {}
rows = []
for name, feats in cfg.items():
    r = {"model": name}
    for v in V:
        tr = D[D.t0 <= v - 2]; te = D[D.t0 == v]
        p = fit_pred(tr[feats].astype(float), tr.y, te[feats].astype(float))
        P[(name, v)] = p
        r[f"auc_{v}"] = auc(te.y, p)
        k = 1000
        r[f"top1000_hit_{v}"] = int(te.y.values[np.argsort(-p)[:k]].sum())
    rows.append(r)
R = pd.DataFrame(rows)
for ref, new in [("M0", "M1(IND+LOC)"), ("M0fine", "M0fine+IND+LOC"), ("M0", "M0+IND"), ("M0", "M0+LOC")]:
    a = R.set_index("model")
    sk = [skill(a.loc[new, f"auc_{v}"], a.loc[ref, f"auc_{v}"]) for v in V]
    print(f"{new} vs {ref}: skill", [round(x, 4) for x in sk], "mean", round(np.mean(sk), 4))
    for v in V:
        te = D[D.t0 == v]
        lo, hi = boot_ci_skill(te.y.values, P[(new, v)], P[(ref, v)], n=200)
        print("   ", v, "CI", round(lo, 4), round(hi, 4))
pos = D[D.t0.isin(V)].groupby("t0").y.sum()
print(R.round(4).to_string(index=False))
print("양성 수", pos.to_dict(), "상위1000 무작위 기대", {v: round(1000 * D[D.t0 == v].y.mean(), 1) for v in V})
R.to_csv(os.path.join(OUT, "I1_robust.csv"), index=False, encoding="utf-8-sig")
