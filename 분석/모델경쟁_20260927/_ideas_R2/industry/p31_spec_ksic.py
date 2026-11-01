# -*- coding: utf-8 -*-
"""아이디어 A 재정식화: KSIC 단위(특허 해상도와 일치)로 집계한 국가별 수출 점유 변화 vs 특허 점유 변화.
단위 (국가 c, KSIC k, t0). y = Δlog(수출점유_c,k) over t0→t0+3 − (c,t0) 중앙값. 무적합 단일신호 ρ + 기준선(과거 수출점유 변화) 비교.
사용 가능 연도 2004·2006·2009·2012(완결) — 2015는 부분 수집이라 제외."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from ilib import *
import sys; LAG = int(sys.argv[1]) if len(sys.argv) > 1 else 1
X = pd.read_csv(os.path.join(B, "_shared", "ext", "R2ind_comtrade_exports_5c_hs6_2004_2021.csv"), dtype={"hs6": str})
X = X[X.year.isin([2004, 2006, 2009, 2012])]; X["hs6"] = X.hs6.str.zfill(6)
X["c"] = X.reporter.map({410: "KR", 156: "CN", 392: "JP", 842: "US", 276: "EP"})
mp = load_map(); w = mp.groupby("hs6").k5.transform("count"); mp = mp.assign(w=1/w)
A = X.merge(mp, on="hs6"); A["v"] = A.export_usd*A.w
K = A.pivot_table(index=["k5", "year"], columns="c", values="v", aggfunc="sum").fillna(0)
S = K.div(K.sum(1), axis=0)
pv = load_pat(); out = []
for t0, t3, tm in [(2006, 2009, 2004), (2009, 2012, 2006)]:
    pf = pat_feats(pv, t0, lag=LAG)
    for c in ["KR", "CN", "JP", "US", "EP"]:
        s0, s3, sm = S.xs(t0, level=1)[c], S.xs(t3, level=1)[c], S.xs(tm, level=1)[c]
        d = pd.DataFrame({"y": np.log((s3+1e-4)/(s0+1e-4)), "past": np.log((s0+1e-4)/(sm+1e-4)), "lvl": np.log(s0+1e-4)})
        d = d.join(pf[[f"dsh_{c}", f"mom_{c}", f"sh_{c}"]].set_axis(["p_dsh", "p_mom", "p_sh"], axis=1), how="inner")
        d["p_rta"] = np.log((d.p_sh+1e-3)/(d.p_sh.mean()+1e-3))
        for s in ["p_dsh", "p_mom", "p_rta", "past", "lvl"]:
            dd = d[["y", s]].dropna(); out.append((t0, c, s, spearmanr(dd.y, dd[s])[0], len(dd)))
O = pd.DataFrame(out, columns=["t0", "c", "sig", "rho", "n"])
print(O.pivot_table(index="sig", columns=["t0", "c"], values="rho").round(2).to_string())
print(O.groupby("sig").rho.agg(["mean", lambda s: (s > 0).mean()]).round(3))
