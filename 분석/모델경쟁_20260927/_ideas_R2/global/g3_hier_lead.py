# -*- coding: utf-8 -*-
"""G3 4차 — 계층형(C4) 위에 '국가 간 선행' 점유율 조정 + (실험) 무역 조정.
 share_hat = share_T · exp(b_h · x),  x = 타국(해당국 제외 KR/US/EP/JP) 같은 KSIC 점유율의 3년 로그변화 평균
 b_h 는 원점 T 이전 목표연도(≤T) 행에서 LAD 격자로 학습(국가 공통 / 국가별 두 가지).
 무역 조정 xt = KSIC 수입 점유율 3년 로그변화(≤2021만) — 원점 ≤2021 비교.
 총량은 C4와 동일(국가 총량 ens3). 🔒 목표 2024 계산 안 함.
"""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing

pan = pd.read_parquet(os.path.join(CACHE, "pat_panel.parquet"))
C5 = ["KR", "US", "EP", "CN", "JP"]
P = pan[pan.cc.isin(C5) & (pan.dy >= 2005)].groupby(["ksic", "cc", "dy"]).n.sum().unstack(fill_value=0).reindex(columns=range(2005, 2024), fill_value=0)
TOT = P.groupby(level="cc").sum()
SH = (P + 0.5).div(TOT, level="cc")
LSH = np.log(SH)
m = load_map(); imp = load_imports()
t = imp[imp.partner == "WLD"].merge(m, on="hs6"); t["value"] = t.value / t.groupby(["year", "hs6"]).k5.transform("count")
TRM = t.groupby(["k5", "year"]).value.sum().unstack()
TRS = np.log(TRM.div(TRM.sum()))            # KSIC 수입 점유율(50산업 내)


def ens3_series(y, H=3):
    ly = np.log(y)
    try:
        e = ExponentialSmoothing(ly, trend="add", damped_trend=True, initialization_method="estimated").fit().forecast(H)
    except Exception:
        e = np.repeat(ly[-1], H)
    tt = np.asarray(ThetaModel(ly, period=1, deseasonalize=False).fit().forecast(H))
    return np.exp(np.median(np.c_[np.repeat(ly[-1], H), e, tt], axis=1))


rows = []
TF = {}
for T in range(2008, 2023):
    TF[T] = {cc: ens3_series(TOT.loc[cc, 2005:T].values.astype(float)) for cc in ["KR", "US", "EP", "CN"]}
    d3 = LSH[T] - LSH[T - 3]
    d3w = d3.unstack("cc")
    for cc in ["KR", "US", "EP", "CN"]:
        others = [c for c in ["KR", "US", "EP", "JP"] if c != cc]
        x = d3w[others].mean(axis=1)
        for h in [1, 2, 3]:
            if T + h > 2023:
                continue
            idx = [(k, cc) for k in d3w.index]
            b = pd.DataFrame({"ksic": d3w.index, "cc": cc, "T": T, "h": h, "Y": T + h, "x": x.values, "own3": d3w[cc].values})
            b["lv"] = np.log1p(P.loc[idx, T].values)
            b["base_r"] = np.log1p(TF[T][cc][h - 1] * (P.loc[idx, T].values / TOT.loc[cc, T])) - b.lv
            b["r"] = np.log1p(P.loc[idx, T + h].values) - b.lv
            b["xt"] = (TRS[T] - TRS[T - 3]).reindex(d3w.index).values if T <= 2021 else np.nan
            rows.append(b)
D = pd.concat(rows, ignore_index=True)
GRID = np.linspace(-1.0, 1.5, 51)


def run(var, per_cc=False, Tmax=2022, base="base_r"):
    out = []
    for T in range(2011, Tmax + 1):
        tr = D[(D.Y <= T) & D[var].notna()]; te = D[(D["T"] == T) & (D.Y >= 2014)].copy()
        te["rhat"] = te[base]
        keys = ["cc", "h"] if per_cc else ["h"]
        for k, g in te.groupby(keys):
            kk = k if isinstance(k, tuple) else (k,)
            t = tr
            for kn, kv in zip(keys, kk):
                t = t[t[kn] == kv]
            if len(t) == 0:
                continue
            bb = GRID[np.argmin([np.abs(t.r - t[base] - a * t[var]).mean() for a in GRID])]
            te.loc[g.index, "rhat"] = g[base] + bb * g[var].fillna(0); te.loc[g.index, "b"] = bb
        out.append(te)
    return pd.concat(out)


def score(E, name):
    E = E.copy(); E["e_m"] = np.abs(E.r - E.rhat); E["e_n"] = np.abs(E.r); E["e_c4"] = np.abs(E.r - E.base_r)
    g = E.groupby("Y")[["e_m", "e_n", "e_c4"]].mean()
    res = {"name": name, "skill_vs_naive": round(1 - g.e_m.mean() / g.e_n.mean(), 4),
           "skill_vs_C4": round(1 - g.e_m.mean() / g.e_c4.mean(), 4), "win_vs_C4": round(float((g.e_m < g.e_c4).mean()), 2)}
    for c in ["KR", "US", "EP", "CN"]:
        gg = E[E.cc == c].groupby("Y")[["e_m", "e_n", "e_c4"]].mean()
        res["sk_" + c] = round(1 - gg.e_m.mean() / gg.e_n.mean(), 4); res["dC4_" + c] = round(1 - gg.e_m.mean() / gg.e_c4.mean(), 4)
    if "b" in E:
        res["b_mean"] = round(float(E.b.mean()), 3)
    print(json.dumps(res, ensure_ascii=False)); return res


R = []
E = D[D.Y >= 2014].copy(); E["rhat"] = E.base_r; R.append(score(E, "C4_hier(챔피언)"))
R.append(score(run("x"), "C6_lead_share_common"))
R.append(score(run("x", per_cc=True), "C6b_lead_share_percc"))
R.append(score(run("own3"), "C7_own_share_momentum"))
e = D[(D.Y >= 2014) & (D["T"] <= 2021)].copy(); e["rhat"] = e.base_r; R.append(score(e, "T21_C4"))
R.append(score(run("xt", Tmax=2021), "T21_C8_import_share_mom"))
# C9: C7(자기 점유율 모멘텀) 채택 후 그 위에 국가 간 선행 x 추가 (2단계: C7 계수는 원점별 과거 학습값)
c7 = run("own3")
D.loc[c7.index, "c7_r"] = c7.rhat
# C7 계수 b7 은 원점 T에서 과거로만 학습된 값을 그대로 재사용
bmap = c7.groupby(["T", "h"]).b.first()
E9 = []
for T in range(2011, 2023):
    tr = D[D.Y <= T].copy(); te = D[(D["T"] == T) & (D.Y >= 2014)].copy()
    for h in [1, 2, 3]:
        b7 = bmap.get((T, h), 0.0)
        tr.loc[tr.h == h, "c7b"] = tr.base_r + b7 * tr.own3; te.loc[te.h == h, "c7b"] = te.base_r + b7 * te.own3
        t = tr[tr.h == h]; g = te[te.h == h]
        bb = GRID[np.argmin([np.abs(t.r - t.c7b - a * t.x).mean() for a in GRID])]
        te.loc[g.index, "rhat"] = g.c7b + bb * g.x; te.loc[g.index, "b"] = bb
    E9.append(te)
E9 = pd.concat(E9); E9["base_r"] = E9.c7b
r9 = score(E9, "C9_C7+lead_x (base=C7로 비교)"); R.append(r9)
pd.DataFrame(R).to_csv(os.path.join(HERE, "g3_hier_lead_result.csv"), index=False, encoding="utf-8-sig")
