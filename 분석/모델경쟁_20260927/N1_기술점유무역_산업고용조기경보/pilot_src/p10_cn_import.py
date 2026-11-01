# -*- coding: utf-8 -*-
"""아이디어 C 파일럿: 중국 특허 추격 → 한국의 HS6 품목 대중 수입의존도 3년 변화 예측.
단위 = (HS6, 원점 t0). y = CN점유율(t0+3) − CN점유율(t0) (%p). 결과 ≤2021 → t0 ≤ 2018. 평가 원점 v=2011..2018, 학습 t0 ≤ v−3.
기준선 B = 무역만(점유율 수준·과거 3년 변화·수입규모·수입성장) 모형. 모형 M = B + KSIC 특허 피처."""
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.linear_model import HuberRegressor
from ilib import *
pv = load_pat(); mp = load_map(); imp = load_imports()
single = mp.groupby("hs6").k5.nunique(); mp1 = mp[mp.hs6.isin(single[single == 1].index)]
W = imp["WLD"].unstack(); C = imp["CN"].unstack().reindex_like(W).fillna(0)
S = (C / W.where(W > 0)) * 100
rows = []
for t0 in range(2006, 2019):
    if t0+3 > DEV_MAX: break
    pf = pat_feats(pv, t0)
    d = pd.DataFrame({"hs6": W.index})
    d["t0"] = t0
    g = lambda M, y: M[y].reindex(d.hs6).values if y in M.columns else np.nan
    d["w0"], d["wm3"], d["w3"] = g(W, t0), g(W, t0-3), g(W, t0+3)
    d["s0"], d["sm3"], d["s3"] = g(S, t0), g(S, t0-3), g(S, t0+3)
    d = d[(d.w0 >= 1e6) & (d.w3 >= 1e6) & (d.wm3 >= 1e6)].dropna(subset=["s0", "s3", "sm3"])
    d["y"] = d.s3 - d.s0
    d["ds_past"] = d.s0 - d.sm3; d["lw0"] = np.log(d.w0); d["gw_past"] = np.log(d.w0/d.wm3)
    d = d.merge(mp1.rename(columns={"k5": "ksic"}), on="hs6").merge(pf, left_on="ksic", right_index=True, how="left")
    rows.append(d)
P = pd.concat(rows, ignore_index=True)
P.to_parquet(f"{CACHE}/panel_cnimp.parquet")
print("표본", len(P), "HS6", P.hs6.nunique(), "KSIC", P.ksic.nunique(), "원점", P.t0.min(), P.t0.max())
print("y 요약", P.y.describe().round(2).to_dict())
BF = ["s0", "ds_past", "lw0", "gw_past"]
PF = ["mom_CN", "mom_KR", "gap_CN_KR", "sh_CN", "dsh_CN", "mom_ALL", "lvl_CN", "frgn_mom"]
def fit_pred(tr, te, F, kind):
    if kind == "huber":
        mu, sd = tr[F].mean(), tr[F].std()+1e-9
        m = HuberRegressor(max_iter=500).fit((tr[F]-mu)/sd, tr.y); return m.predict((te[F]-mu)/sd)
    m = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                         objective="reg:absoluteerror", n_jobs=2, random_state=0)
    m.fit(tr[F], tr.y); return m.predict(te[F])
res = []
for v in range(2011, 2019):
    tr, te = P[P.t0 <= v-3], P[P.t0 == v]
    r = {"v": v, "n": len(te), "zero": mae(te.y, 0)}
    for kind in ["huber", "xgb"]:
        r[f"B_{kind}"] = mae(te.y, fit_pred(tr, te, BF, kind))
        r[f"M_{kind}"] = mae(te.y, fit_pred(tr, te, BF+PF, kind))
    res.append(r)
R = pd.DataFrame(res); print(R.round(3).to_string())
for kind in ["huber", "xgb"]:
    sk = 1 - R[f"M_{kind}"]/R[f"B_{kind}"]; sk0 = 1 - R[f"B_{kind}"]/R.zero
    print(kind, "기준선(무역만) vs 0: skill", round(sk0.mean(), 4), "| 특허 추가 skill vs 무역만:", round(sk.mean(), 4), "승률", (sk > 0).mean())
R.to_csv(f"{HERE}/p10_result.csv", index=False)
