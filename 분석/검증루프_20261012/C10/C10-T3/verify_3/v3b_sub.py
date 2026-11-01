# -*- coding: utf-8 -*-
# verify_3 보조: 보조창(2015–17) FREE 제공−대체 점추정(C-RCA-SUB), 판정 문구 점검
import os, sys, json
sys.dont_write_bytecode = True
import pandas as pd
from scipy.stats import spearmanr
V = os.path.dirname(os.path.abspath(__file__)); MC = os.path.abspath(os.path.join(V, *[".."] * 4, "모델경쟁_20260927"))
A = pd.read_parquet(os.path.join(V, "cache_A.parquet"))
km = pd.read_csv(os.path.join(MC, "T3_기술셀부상예측", "ESSENTIAL_R7", "conc_ksic_map.csv"), dtype=str); K50 = sorted(km.ksic)
ek = pd.read_csv(os.path.join(MC, "_shared_report", "fusion_advantage", "map5", "m4_exports_ksic_5rep.csv"), dtype={"k5": str})
v = ek[ek.year.isin([2015, 2016, 2017])].groupby(["k5", "rep"]).v.sum().unstack(); s_ = v / v.sum(); RCA = s_.div(s_.mean(1), axis=0).KR.reindex(K50)
P2 = A.appl_year.between(2015, 2017)
def ab(lab):
    d = pd.DataFrame({"k": lab, "o": A.kr_origin})[P2 & lab.notna()]
    k = d[d.o].groupby("k").size().reindex(K50).fillna(0); a = d.groupby("k").size().reindex(K50)
    return (k / a) / (k.sum() / a.sum())
def r(x): d = pd.concat([x, RCA], axis=1).dropna(); return spearmanr(d.iloc[:, 0], d.iloc[:, 1]).statistic
rp = r(ab(A.ksic_prov)); out = {"rho_prov_P2": round(rp, 4)}
for k in ("r8", "reg", "pure"):
    out[k] = dict(rho=round(r(ab(A["free_" + k])), 4), d=round(rp - r(ab(A["free_" + k])), 4))
print(json.dumps(out)); json.dump(out, open(os.path.join(V, "v3b_out.json"), "w"), indent=1)
