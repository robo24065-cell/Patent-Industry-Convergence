# -*- coding: utf-8 -*-
"""아이디어 N1 파일럿: KSIC 5자리 출하액 성장률(광업제조업조사, 익년 12월 공표) 나우캐스트.
나우캐스트 시점 = t+1년 4월: DART t년 사업보고서(3월), 광공업생산지수 t년 12월분(1월), KOSIS 조사 t−1년(작년 12월) 이용 가능.
결과 g(k,t) = log(출하_t/출하_{t-1}) — 10차(≤2020)/11차(≥2021) 같은 차수 안에서 계산.
🔒 최종시험 t=2024 (F=2024): 2024 조사값은 읽기만 하고 즉시 버린다(계산·출력 안 함). 개발 t=2016..2023, 학습 = t' ≤ t−1.
특허 KSIC 브리지: 기업 매출 성장(DART)을 t−1 시점 특허 KSIC 배분가중으로 산업에 배분."""
import os, sys, json
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, HuberRegressor
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, "cache")
k = pd.read_csv(os.path.join(BASE, r"공모전데이터\작업데이터\kosis_광업제조업_50개산업.csv"))
k = k[(k.항목코드 == "T04") & (k.연도 <= 2023)]
k["ksic"] = k.세세분류코드.astype(str)
rows = []
for (ks, bs), g in k.groupby(["ksic", "기준"]):
    s = g.set_index("연도").값.sort_index()
    for t in s.index:
        if t - 1 in s.index and s[t - 1] > 0 and s[t] > 0 and ((bs == "10차" and t <= 2020) or (bs == "11차" and t >= 2021)):
            rows.append((ks, t, np.log(s[t] / s[t - 1]), s[t - 1]))
Y = pd.DataFrame(rows, columns=["ksic", "t", "g", "ship_prev"])
# 과거 성장(작년)
Y["g_lag"] = Y.set_index(["ksic", "t"]).g.reindex(pd.MultiIndex.from_arrays([Y.ksic, Y.t - 1])).values
# 광공업 출하지수(3자리 → 없으면 2자리) 연간 성장
pi = pd.read_csv(os.path.join(BASE, r"공모전데이터\외부데이터\경기지표\kosis_광공업생산지수_월별.csv"), dtype={"산업코드": str})
pi = pi[pi.항목명 == "출하지수(원지수)"].copy(); pi["y"] = pi.연월 // 100; pi["값"] = pd.to_numeric(pi.값, errors="coerce")
ann = pi.groupby(["산업코드", "y"]).값.agg(["sum", "size"]).reset_index(); ann = ann[ann["size"] == 12]
A = ann.set_index(["산업코드", "y"])["sum"]
def idx_g(code, t):
    for c in ("C" + code[:3], "C" + code[:2]):
        if (c, t) in A.index and (c, t - 1) in A.index:
            return np.log(A[(c, t)] / A[(c, t - 1)])
    return np.nan
Y["g_idx"] = [idx_g(a, b) for a, b in zip(Y.ksic, Y.t)]
# DART 배분 성장
F = pd.read_parquet(os.path.join(C, "fin.parquet"))[["corp_code", "year", "rev"]]
W = pd.read_parquet(os.path.join(C, "alloc_w.parquet"))
fr = F.merge(F.assign(year=F.year + 1).rename(columns={"rev": "rev_prev"}), on=["corp_code", "year"])
fr = fr[(fr.rev > 0) & (fr.rev_prev > 0)]
fr["gi"] = np.log(fr.rev / fr.rev_prev).clip(-1, 1)
M = fr.merge(W, left_on=["corp_code", fr.year - 1], right_on=["corp_code", "t0"])
out = []
for wcol in ["w_share", "w_full"]:
    M["a"] = M[wcol] * M.rev_prev
    G = M.groupby(["ksic", "year"])
    d = pd.DataFrame({f"gd_{wcol}": np.log((G.apply(lambda x: (x.a * np.exp(x.gi)).sum(), include_groups=False)) / G.a.sum()),
                      f"cov_{wcol}": G.a.sum(), f"nf_{wcol}": G.size()})
    out.append(d)
D = pd.concat(out, axis=1).reset_index().rename(columns={"year": "t"})
Y = Y.merge(D, on=["ksic", "t"], how="left")
Y["cov_ratio"] = Y.cov_w_full / (Y.ship_prev * 1e6)   # KOSIS 백만원 단위
for c in ["gd_w_share", "gd_w_full"]:
    Y[c + "_f"] = Y[c].fillna(Y.g_idx)
Y.to_parquet(os.path.join(C, "nowcast_panel.parquet"), index=False)
DEV = list(range(2016, 2024))

def run(feats, name, model="ols"):
    res = {}
    for t in DEV:
        tr = Y[(Y.t <= t - 1) & (Y.t >= 2014)].dropna(subset=feats + ["g"])
        te = Y[Y.t == t].dropna(subset=["g"])
        Xte = te[feats].copy()
        m = LinearRegression() if model == "ols" else HuberRegressor(epsilon=1.35, max_iter=500)
        m.fit(tr[feats], tr.g)
        Xte = Xte.fillna(tr[feats].median())
        res[t] = np.mean(np.abs(te.g - m.predict(Xte)))
    return name, res

def direct(col, name):
    res = {}
    for t in DEV:
        te = Y[Y.t == t].dropna(subset=["g"])
        res[t] = np.mean(np.abs(te.g - te[col].fillna(0)))
    return name, res

Y["zero"] = 0.0
exps = [direct("zero", "B0_zero"), direct("g_lag", "B1_lastyear"), direct("g_idx", "B2_idx_direct"),
        run(["g_idx"], "M1_ols_idx"), run(["g_idx"], "M1h_huber_idx", "huber"),
        run(["g_idx", "gd_w_share_f"], "M2_ols_idx+dart_share"), run(["g_idx", "gd_w_full_f"], "M3_ols_idx+dart_full"),
        run(["g_idx", "gd_w_full_f"], "M3h_huber_idx+dart_full", "huber"),
        run(["gd_w_full_f"], "M4_ols_dart_full_only")]
base = dict(exps)["M1_ols_idx"]
rows = []
for name, r in exps:
    sk = {t: 1 - r[t] / base[t] for t in DEV}
    rows.append({"exp": name, **{f"mae_{t}": round(r[t], 4) for t in DEV}, "mean_mae": round(np.mean(list(r.values())), 4),
                 "skill_vs_M1": round(np.mean(list(sk.values())), 4), "win_vs_M1": np.mean([r[t] < base[t] for t in DEV]).round(2)})
R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, "pilotN1_nowcast.csv"), index=False)
pd.set_option("display.width", 250)
print(R.to_string())
print("coverage: firms per ksic-year median", Y.nf_w_full.median(), "missing gd share", Y[Y.t.isin(DEV)].gd_w_full.isna().mean().round(3),
      "cov_ratio median", Y.cov_ratio.median())
print("corr g vs gd_full", Y[["g", "g_idx", "gd_w_full", "gd_w_share", "g_lag"]].corr().round(3).to_string())
