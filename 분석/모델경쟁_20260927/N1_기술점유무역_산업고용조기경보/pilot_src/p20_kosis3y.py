# -*- coding: utf-8 -*-
"""아이디어 B 파일럿: 50개 산업의 3년 누적 상대성장(원점 내 중앙값 차감) 예측.
타깃: 종사자(T02)·출하(T04)·부가가치(T06) 3년 로그성장 − 그 원점 50개 중앙값. 결과 ≤2021 → t0 ≤ 2018.
원점 t0=2008..2018 구축, 평가 v=2012..2018, 학습 t0 ≤ v−3. 지표: MAE skill(기준선 0=중앙값 예측), 원점별 Spearman ρ."""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
from ilib import *
K = load_kosis50(); pv = load_pat()
rows = []
for t0 in range(2008, 2019):
    if t0 == 2010: continue
    pf = pat_feats(pv, t0)
    d = pd.DataFrame(index=K["T04"].index); d["t0"] = t0
    for it, nm in [("T02", "emp"), ("T04", "shp"), ("T06", "va")]:
        M = K[it]
        d[f"y_{nm}"] = lg(M, t0+3, t0)
        d[f"g3_{nm}"] = lg(M, t0, t0-3) if t0-3 >= 2008 else np.nan
        d[f"g1_{nm}"] = lg(M, t0, t0-1) if t0-1 >= 2008 else np.nan
        d[f"lv_{nm}"] = np.log(M[t0 if t0 != 2010 else 2009])
    d["prod"] = d.lv_va - d.lv_emp
    d = d.join(pf)
    for c in [c for c in d.columns if c.startswith("y_")]:
        d[c] = d[c] - d[c].median()
    for c in [c for c in d.columns if c.startswith("g3_") or c.startswith("g1_")]:
        d[c] = d[c] - d[c].median()
    rows.append(d.reset_index())
P = pd.concat(rows, ignore_index=True); P.to_parquet(f"{CACHE}/panel_kosis3y.parquet")
print("행", len(P), "원점", sorted(P.t0.unique()))
PF = ["mom_KR", "mom_ALL", "dsh_KR", "frgn_mom", "mom_CN", "lvl_KR"]
for tg in ["emp", "shp", "va"]:
    BF = [f"g3_{tg}", f"g1_{tg}", f"lv_{tg}", "prod"]
    res = []
    for v in range(2012, 2019):
        if v == 2010: continue
        tr, te = P[(P.t0 <= v-3)].dropna(subset=[f"y_{tg}"]), P[P.t0 == v].dropna(subset=[f"y_{tg}"])
        r = {"v": v, "n": len(te), "zero": mae(te[f"y_{tg}"], 0)}
        for lab, F in [("B", BF), ("M", BF+PF)]:
            trf = tr[F].fillna(0); mu, sd = trf.mean(), trf.std()+1e-9
            m = Ridge(alpha=10).fit((trf-mu)/sd, tr[f"y_{tg}"])
            p = m.predict((te[F].fillna(0)-mu)/sd)
            r[lab] = mae(te[f"y_{tg}"], p); r[f"rho_{lab}"] = spearmanr(te[f"y_{tg}"], p)[0]
        res.append(r)
    R = pd.DataFrame(res)
    skB, skM = 1-R.B/R.zero, 1-R.M/R.zero; skMB = 1-R.M/R.B
    print(f"[{tg}] 0기준 MAE {R.zero.mean():.4f} | 기준선(자기추세) skill {skB.mean():.4f} ρ {R.rho_B.mean():.3f} | +특허 skill {skM.mean():.4f} ρ {R.rho_M.mean():.3f} | 특허 증분 skill {skMB.mean():.4f} 승률 {(skMB>0).mean():.2f}")
    print(R.round(3).to_string())
