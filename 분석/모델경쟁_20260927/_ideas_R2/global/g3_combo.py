# -*- coding: utf-8 -*-
"""G3 3차 — 견고 결합: 국가·시차별 수축 가중 w (naive ↔ theta/ens3) 를 과거 원점에서만 학습.
 C1: r̂ = w_cc,h · theta_r      (w 격자 0..1, 과거 목표연도 ≤T 행의 절대오차 최소)
 C2: r̂ = w_cc,h · ens3_r
 C3: C2 + 국가 전체 가속 가드(tot_acc<0 이면 추세 절반)
 C4: 계층형 — KSIC 점유율(국가 내) naive × 국가 총량 ens3  (산업 구성은 고정, 총량 추세만)
 C5: 계층형 — 점유율 theta × 총량 ens3
개발 목표연도 2014–2023, 🔒 2024 계산 안 함.
"""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from statsmodels.tsa.forecasting.theta import ThetaModel

D = pd.read_parquet(os.path.join(CACHE, "g3_global_panel.parquet"))
D = D[D.cc.isin(["KR", "US", "EP", "CN"]) & D.theta.notna()].copy()
D["ens3_r"] = np.log1p(np.median(np.c_[D.naive, D.theta, D.ets_d], axis=1)) - D.lv
GRID = np.linspace(0, 1.2, 25)


def shrink(col, guard=False):
    out = []
    for T in range(2011, 2023):
        tr = D[D.Y <= T]; te = D[(D["T"] == T) & (D.Y >= 2014)].copy()
        te["rhat"] = np.nan
        for (cc, h), g in te.groupby(["cc", "h"]):
            t = tr[(tr.cc == cc) & (tr.h == h)]
            w = GRID[np.argmin([np.abs(t.r - a * t[col]).mean() for a in GRID])] if len(t) else 0.0
            rh = w * g[col]
            if guard:
                rh = np.where(g.tot_acc < 0, rh * 0.5, rh)
            te.loc[g.index, "rhat"] = rh; te.loc[g.index, "w"] = w
        out.append(te)
    return pd.concat(out)


# 계층형: 국가 총량 ens3 예측 × 점유율 예측
pan = pd.read_parquet(os.path.join(CACHE, "pat_panel.parquet"))
P = pan[pan.cc.isin(["KR", "US", "EP", "CN"]) & (pan.dy >= 2005)].groupby(["ksic", "cc", "dy"]).n.sum().unstack(fill_value=0).reindex(columns=range(2005, 2024), fill_value=0)
TOT = P.groupby(level="cc").sum()
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def ens3_series(y, H=3):
    ly = np.log(y)
    try:
        e = ExponentialSmoothing(ly, trend="add", damped_trend=True, initialization_method="estimated").fit().forecast(H)
    except Exception:
        e = np.repeat(ly[-1], H)
    t = np.asarray(ThetaModel(ly, period=1, deseasonalize=False).fit().forecast(H))
    return np.exp(np.median(np.c_[np.repeat(ly[-1], H), e, t], axis=1))


def hier(share_theta=False):
    out = []
    for T in range(2011, 2023):
        te = D[(D["T"] == T) & (D.Y >= 2014)].copy()
        tf = {cc: ens3_series(TOT.loc[cc, 2005:T].values.astype(float)) for cc in TOT.index}
        rh = []
        for _, r in te.iterrows():
            tot_hat = tf[r.cc][r.h - 1]
            sh = P.loc[(r.ksic, r.cc), T] / TOT.loc[r.cc, T]
            if share_theta:
                shs = (P.loc[(r.ksic, r.cc), 2005:T] + 0.5) / TOT.loc[r.cc, 2005:T]
                try:
                    sh = float(np.exp(np.asarray(ThetaModel(np.log(shs.values.astype(float)), period=1, deseasonalize=False).fit().forecast(3))[r.h - 1]))
                except Exception:
                    pass
            rh.append(np.log1p(tot_hat * sh) - r.lv)
        te["rhat"] = rh; out.append(te)
    return pd.concat(out)


def score(E, name):
    E = E.copy(); E["e_m"] = np.abs(E.r - E.rhat); E["e_n"] = np.abs(E.r); E["e_e"] = np.abs(E.r - E.ens3_r)
    g = E.groupby("Y")[["e_m", "e_n", "e_e"]].mean()
    res = {"name": name, "skill_vs_naive": round(1 - g.e_m.mean() / g.e_n.mean(), 4), "win_vs_naive": round(float((g.e_m < g.e_n).mean()), 2),
           "skill_vs_ens3": round(1 - g.e_m.mean() / g.e_e.mean(), 4), "win_vs_ens3": round(float((g.e_m < g.e_e).mean()), 2)}
    for c in ["KR", "US", "EP", "CN"]:
        gg = E[E.cc == c].groupby("Y")[["e_m", "e_n"]].mean()
        res["sk_" + c] = round(1 - gg.e_m.mean() / gg.e_n.mean(), 4); res["win_" + c] = round(float((gg.e_m < gg.e_n).mean()), 2)
    res["y2023_sk"] = round(1 - g.loc[2023, "e_m"] / g.loc[2023, "e_n"], 4)
    print(json.dumps(res, ensure_ascii=False)); return res


R = []
E = D[D.Y >= 2014].copy(); E["rhat"] = E.ens3_r; R.append(score(E, "ens3(기준모형)"))
E = D[D.Y >= 2014].copy(); E["rhat"] = E.theta_r; R.append(score(E, "theta"))
e1 = shrink("theta_r"); R.append(score(e1, "C1_shrink_theta"))
e2 = shrink("ens3_r"); R.append(score(e2, "C2_shrink_ens3"))
print(e2.groupby(["cc", "h"]).w.mean().unstack().round(2))
R.append(score(shrink("ens3_r", guard=True), "C3_shrink_ens3+guard"))
R.append(score(hier(False), "C4_hier_naiveShare"))
R.append(score(hier(True), "C5_hier_thetaShare"))
pd.DataFrame(R).to_csv(os.path.join(HERE, "g3_combo_result.csv"), index=False, encoding="utf-8-sig")
