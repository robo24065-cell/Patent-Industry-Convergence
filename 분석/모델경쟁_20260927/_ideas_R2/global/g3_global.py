# -*- coding: utf-8 -*-
"""G3 2차 — 전역(풀링) 모형: 계열 자기이력 + 국가 간 동행/선행 + 국가 전체 국면 + (실험) 무역 외부데이터.
타깃 r = log1p(y[T+h]) − log1p(y[T]). 원점 T 학습 = 목표연도 ≤ T 인 과거 행만.
개발 목표연도 2014–2023. 🔒 Y=2024 계산 안 함. 무역 피처는 ≤2021 값만 사용(G1/G2 최종시험 봉인) → 원점 T≤2021 행으로 비교.
"""
import sys, os, json, warnings
import numpy as np, pandas as pd, xgboost as xgb
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

pan = pd.read_parquet(os.path.join(CACHE, "pat_panel.parquet"))
ALLC = ["KR", "US", "EP", "CN", "JP"]
P = pan[(pan.dy >= 2005)].groupby(["ksic", "cc", "dy"]).n.sum().unstack(fill_value=0).reindex(columns=range(2005, 2024), fill_value=0)
LP = np.log1p(P)
Wd = pd.read_parquet(os.path.join(CACHE, "g3_wide.parquet"))
th = Wd[["ksic", "cc", "T", "h", "theta", "drift_d", "ets_d", "y", "naive"]]

# 무역: KSIC 수입(세계·중국) — 균등배분, ≤2021
m = load_map(); imp = load_imports()
t = imp.merge(m, on="hs6"); t["value"] = t.value / t.groupby(["year", "partner", "hs6"]).k5.transform("count")
TR = t.groupby(["k5", "partner", "year"]).value.sum().unstack().reindex(columns=range(2000, 2022))
try:
    ex = load_exports().merge(m, on="hs6"); ex["value"] = ex.value / ex.groupby(["year", "hs6"]).k5.transform("count")
    EX = ex.groupby(["k5", "year"]).value.sum().unstack()
except Exception:
    EX = None

rows = []
for T in range(2008, 2023):
    lvT = LP[T]
    g1 = LP[T] - LP[T - 1]; g1b = LP[T - 1] - LP[T - 2]; g3 = (LP[T] - LP[T - 3]) / 3
    tot = P.groupby(level="cc").sum(); ltot = np.log1p(tot)
    tg1 = ltot[T] - ltot[T - 1]; tg3 = (ltot[T] - ltot[T - 3]) / 3
    base = pd.DataFrame({"lv": lvT, "g1": g1, "g1b": g1b, "g3": g3}).reset_index()
    base["T"] = T
    base["tot_g1"] = base.cc.map(tg1); base["tot_g3"] = base.cc.map(tg3); base["tot_acc"] = base.tot_g1 - base.tot_g3
    g3w = g3.unstack("cc").reindex(columns=ALLC).add_prefix("x_g3_")
    g1w = g1.unstack("cc").reindex(columns=ALLC).add_prefix("x_g1_")
    base = base.merge(g3w, left_on="ksic", right_index=True).merge(g1w, left_on="ksic", right_index=True)
    if T <= 2021:
        wl = np.log1p(TR.xs("WLD", level="partner")); cl = np.log1p(TR.xs("CN", level="partner"))
        tf = pd.DataFrame({"tr_m_g3": (wl[T] - wl[T - 3]) / 3, "tr_m_g1": wl[T] - wl[T - 1],
                           "tr_cn_g3": (cl[T] - cl[T - 3]) / 3, "tr_cnsh": TR.xs("CN", level="partner")[T] / TR.xs("WLD", level="partner")[T]})
        if EX is not None and T in EX.columns and (T - 3) in EX.columns:
            el = np.log1p(EX); tf["tr_x_g3"] = (el[T] - el[T - 3]) / 3; tf["tr_x_g1"] = el[T] - el[T - 1]
            tf["tr_nx"] = np.log((EX[T] + 1) / (TR.xs("WLD", level="partner")[T] + 1))
        base = base.merge(tf, left_on="ksic", right_index=True, how="left")
    for h in [1, 2, 3]:
        if T + h > 2023:
            continue
        b = base.copy(); b["h"] = h; b["Y"] = T + h
        b["r"] = (LP[T + h] - LP[T]).reindex(pd.MultiIndex.from_frame(b[["ksic", "cc"]])).values
        rows.append(b)
D = pd.concat(rows, ignore_index=True)
D = D.merge(th, on=["ksic", "cc", "T", "h"], how="left")
D["theta_r"] = np.log1p(D.theta) - D.lv; D["drift_r"] = np.log1p(D.drift_d) - D.lv; D["ets_r"] = np.log1p(D.ets_d) - D.lv
for c in ALLC:
    D["is_" + c] = (D.cc == c).astype(int)
D.to_parquet(os.path.join(CACHE, "g3_global_panel.parquet"), index=False)

OWN = ["lv", "g1", "g1b", "g3", "h", "is_KR", "is_US", "is_EP", "is_CN", "is_JP"]
CROSS = [f"x_g3_{c}" for c in ALLC] + [f"x_g1_{c}" for c in ALLC]
REG = ["tot_g1", "tot_g3", "tot_acc"]
UNI = ["theta_r", "drift_r", "ets_r"]
TRA = ["tr_m_g3", "tr_m_g1", "tr_cn_g3", "tr_cnsh"]
TRX = ["tr_x_g3", "tr_x_g1", "tr_nx"]


def run(cols, D, ccs=("KR", "US", "EP", "CN"), Ymin=2014, Ymax=2023, Tmax=2022, seed=0):
    out = []
    for T in range(2011, Tmax + 1):
        tr = D[(D.Y <= T) & D.r.notna()]
        te = D[(D["T"] == T) & D.cc.isin(ccs) & (D.Y >= Ymin) & (D.Y <= Ymax) & D.theta.notna()]
        if len(te) == 0:
            continue
        mdl = xgb.XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                               min_child_weight=20, objective="reg:absoluteerror", nthread=2, random_state=seed)
        mdl.fit(tr[cols], tr.r)
        te = te.copy(); te["rhat"] = mdl.predict(te[cols]); out.append(te)
    return pd.concat(out)


def score(E, name, by="Y"):
    E = E.copy()
    E["e_m"] = np.abs(E.r - E.rhat); E["e_naive"] = np.abs(E.r)
    E["e_theta"] = np.abs(E.r - E.theta_r)
    E["e_ens3"] = np.abs(np.log1p(np.median(np.c_[E.naive, E.theta, E.ets_d], axis=1)) - E.lv - E.r)
    g = E.groupby(by)[["e_m", "e_naive", "e_theta", "e_ens3"]].mean()
    res = {"name": name, "n": len(E), "err": round(g.e_m.mean(), 4), "skill_vs_naive": round(1 - g.e_m.mean() / g.e_naive.mean(), 4),
           "skill_theta": round(1 - g.e_theta.mean() / g.e_naive.mean(), 4), "skill_ens3": round(1 - g.e_ens3.mean() / g.e_naive.mean(), 4),
           "win_vs_naive": round(float((g.e_m < g.e_naive).mean()), 2), "win_vs_ens3": round(float((g.e_m < g.e_ens3).mean()), 2)}
    for c in ["KR", "US", "EP", "CN"]:
        s = E[E.cc == c]
        if len(s):
            gg = s.groupby(by)[["e_m", "e_naive"]].mean()
            res["skill_" + c] = round(1 - gg.e_m.mean() / gg.e_naive.mean(), 4)
    print(json.dumps(res, ensure_ascii=False))
    return res, g


if __name__ == "__main__":
    allres = []
    for name, cols in [("G_own", OWN), ("G_own+uni", OWN + UNI), ("G_own+uni+cross", OWN + UNI + CROSS),
                       ("G_own+uni+cross+reg", OWN + UNI + CROSS + REG)]:
        allres.append(score(run(cols, D), name)[0])
    # 무역 외부데이터 실험 (원점 ≤2021)
    for name, cols in [("T21_base", OWN + UNI + CROSS + REG), ("T21_+importKSIC", OWN + UNI + CROSS + REG + TRA)] + \
                      ([("T21_+import+export", OWN + UNI + CROSS + REG + TRA + TRX)] if "tr_x_g3" in D else []):
        allres.append(score(run(cols, D, Tmax=2021), name)[0])
    pd.DataFrame(allres).to_csv(os.path.join(HERE, "g3_global_result.csv"), index=False, encoding="utf-8-sig")
