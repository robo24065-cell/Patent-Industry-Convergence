# -*- coding: utf-8 -*-
"""G3 파일럿 — T2 재도전: KSIC × 국가 '최초공표연도' 특허 건수 1~3년 앞 예측, 다원점 전진 검증.
계열: KSIC 50 × {KR, US, EP, CN} = 200 (JP는 공표연도 추이 불규칙 → 제외, 별도 점검).
이력: 2005~원점. 개발 목표연도 Y = 2014..2023, h=1..3 (원점 T = Y−h).
🔒 최종시험: 목표연도 Y=2024 (원점 2023/2022/2021, h=1/2/3), F=2024. 패널은 dy≤2023만 존재(build_panel.py).
지표: 로그오차 |log1p(ŷ)−log1p(y)| 평균, skill = 1 − err/err_naive. 보조: MASE(원점까지 1-step naive 척도).
"""
import sys, os, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.forecasting.theta import ThetaModel

CCS = ["KR", "US", "EP", "CN"]
pan = pd.read_parquet(os.path.join(CACHE, "pat_panel.parquet"))
P = pan[pan.cc.isin(CCS) & (pan.dy >= 2005)].groupby(["ksic", "cc", "dy"]).n.sum().unstack(fill_value=0)
assert P.columns.max() <= 2023
P = P.reindex(columns=range(2005, 2024), fill_value=0)
print("계열", P.shape)


def f_naive(y, H): return np.repeat(y[-1], H)
def f_mean3(y, H): return np.repeat(y[-3:].mean(), H)


def f_drift_damped(y, H, k=5, phi=0.8):
    ly = np.log1p(y[-k:]); g = (ly[-1] - ly[0]) / (k - 1)
    return np.expm1(ly[-1] + g * np.cumsum(phi ** np.arange(1, H + 1)))


def f_ets(y, H):
    try:
        m = ExponentialSmoothing(np.log1p(y), trend="add", damped_trend=True, initialization_method="estimated").fit()
        return np.expm1(m.forecast(H))
    except Exception:
        return f_naive(y, H)


def f_theta(y, H):
    try:
        m = ThetaModel(np.log1p(y).astype(float), period=1, deseasonalize=False).fit()
        return np.expm1(np.asarray(m.forecast(H)))
    except Exception:
        return f_naive(y, H)


METHODS = {"naive": f_naive, "mean3": f_mean3, "drift_d": f_drift_damped, "ets_d": f_ets, "theta": f_theta}
H = 3
cache_f = os.path.join(CACHE, "g3_fc.parquet")
if os.path.exists(cache_f):
    FC = pd.read_parquet(cache_f)
else:
    out = []
    for T in range(2011, 2023):          # 원점: 2011..2022 → 목표 ≤2023(개발). 2023 원점은 최종시험 전용이라 계산 안 함
        hist = P.loc[:, 2005:T].values.astype(float)
        for i, (ks, cc) in enumerate(P.index):
            y = hist[i]
            scale = np.mean(np.abs(np.diff(y))) + 1e-9
            for mn, f in METHODS.items():
                fc = f(y, H)
                for h in range(1, H + 1):
                    if T + h <= 2023:
                        out.append((ks, cc, T, h, T + h, mn, max(fc[h - 1], 0.0), scale))
    FC = pd.DataFrame(out, columns=["ksic", "cc", "T", "h", "Y", "m", "yhat", "scale"])
    FC.to_parquet(cache_f, index=False)
act = P.stack().rename("y").reset_index().rename(columns={"dy": "Y"})
FC = FC.merge(act, on=["ksic", "cc", "Y"])
W = FC.pivot_table(index=["ksic", "cc", "T", "h", "Y", "y", "scale"], columns="m", values="yhat").reset_index()
W["ens3"] = np.median(W[["naive", "ets_d", "theta"]].values, axis=1)
W["ens_all"] = np.exp(np.log1p(W[["naive", "mean3", "drift_d", "ets_d", "theta"]]).mean(axis=1)) - 1
W.to_parquet(os.path.join(CACHE, "g3_wide.parquet"), index=False)
MS = ["naive", "mean3", "drift_d", "ets_d", "theta", "ens3", "ens_all"]


def report(W, MS, tag):
    rows = []
    for Y, g in W[W.Y >= 2014].groupby("Y"):
        r = {"Y": Y, "n": len(g)}
        for mth in MS:
            r["le_" + mth] = np.abs(np.log1p(g[mth]) - np.log1p(g.y)).mean()
            r["mase_" + mth] = (np.abs(g[mth] - g.y) / g.scale).mean()
        rows.append(r)
    R = pd.DataFrame(rows)
    s = {}
    for mth in MS:
        s[mth] = {"logerr": round(R["le_" + mth].mean(), 4),
                  "skill_log": round(1 - R["le_" + mth].mean() / R.le_naive.mean(), 4),
                  "skill_mase": round(1 - R["mase_" + mth].mean() / R.mase_naive.mean(), 4),
                  "win_Y": round(float((R["le_" + mth] < R.le_naive).mean()), 2)}
    print(tag); print(pd.DataFrame(s).T.to_string())
    return R, s


R, s = report(W, MS, "== 전체 (목표연도 2014–2023, h=1..3) ==")
for h in [1, 2, 3]:
    report(W[W.h == h], MS, f"-- h={h}")
for cc in CCS:
    report(W[W.cc == cc], MS, f"-- cc={cc}")
R.to_csv(os.path.join(HERE, "g3_result_byY.csv"), index=False, encoding="utf-8-sig")
