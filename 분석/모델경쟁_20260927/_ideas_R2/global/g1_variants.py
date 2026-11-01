# -*- coding: utf-8 -*-
"""G1 정식화 변경 파일럿.
 V1: 대중(對中) 수입 점유율의 3년 변화 예측 (HS6) — 중국 기술 추격(CN 특허 모멘텀·CN 기원 해외출원) → 수입선 전환
 V2: KSIC5 산업 단위 수입 3년 상대성장 (48개 산업) — 특허 피처와 같은 해상도
개발 원점 v=2010..2018, 학습 원점 ≤ v−3. 🔒 2022+ 무역 데이터 로드 안 함(common.TRADE_MAX_YEAR).
"""
import sys, json, os
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

m = load_map(); imp = load_imports()
W = trade_panel(imp, "WLD"); C = trade_panel(imp, "CN").reindex_like(W).fillna(0)
pan = pd.read_parquet(os.path.join(CACHE, "pat_panel.parquet"))
hk = m.merge(pd.Series(W.index, name="hs6"), on="hs6")
PATCORE = ["pm_KR", "pm_CN", "pm_FOR", "pm_US", "pm_JP", "pm_EP", "p_gap_for_kr", "p_domshare", "p_domshare_chg",
           "pm_KRin", "pm_KRout", "pm_CNout", "pl_CNout", "p_outshare", "p_krshare5", "pl_KR", "pl_CN"]


def xgbfit(tr, te, cols, obj="reg:absoluteerror"):
    mdl = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                           colsample_bytree=0.8, objective=obj, nthread=2, random_state=0)
    mdl.fit(tr[cols], tr.y); return mdl.predict(te[cols])


def ridgefit(tr, te, cols):
    mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12)))
    mdl.fit(tr[cols].fillna(0), tr.y); return mdl.predict(te[cols].fillna(0))


def evaluate(D, sets, name, fitters):
    res = []
    for v in range(2010, 2019):
        tr, te = D[D.t0 <= v - 3], D[D.t0 == v]
        r = {"v": v, "n": len(te), "mae_B0": np.abs(te.y).mean()}
        for fn, f in fitters.items():
            for k, cols in sets.items():
                p = f(tr, te, cols)
                r[f"mae_{fn}_{k}"] = np.abs(te.y - p).mean(); r[f"rho_{fn}_{k}"] = spearman(p, te.y)
        res.append(r)
    R = pd.DataFrame(res)
    out = {"name": name, "n_mean": float(R.n.mean()), "mae_B0": round(float(R.mae_B0.mean()), 4)}
    for fn in fitters:
        base = R[f"mae_{fn}_B1"]
        for k in sets:
            mk = R[f"mae_{fn}_{k}"]
            out[f"{fn}_{k}"] = {"mae": round(float(mk.mean()), 4), "skill_vsB0": round(float(1 - mk.mean() / R.mae_B0.mean()), 4),
                                "skill_vsB1": round(float((1 - mk / base).mean()), 4), "win_vsB1": round(float((mk < base).mean()), 3),
                                "rho": round(float(R[f"rho_{fn}_{k}"].mean()), 4)}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    R.to_csv(os.path.join(HERE, f"g1_{name}.csv"), index=False, encoding="utf-8-sig")
    return out


# ---------- V1: 대중 수입 점유율 변화 ----------
rows = []
for t0 in range(2005, 2019):
    A_, B_, F_ = [t0 - 2, t0 - 1, t0], [t0 - 5, t0 - 4, t0 - 3], [t0 + 1, t0 + 2, t0 + 3]
    sub = W[B_ + A_ + F_]; s = sub[(sub > 0).all(axis=1)]
    A, B, Fw = s[A_].sum(1), s[B_].sum(1), s[F_].sum(1)
    cA, cB, cF = C.loc[s.index, A_].sum(1), C.loc[s.index, B_].sum(1), C.loc[s.index, F_].sum(1)
    df = pd.DataFrame({"hs6": s.index, "t0": t0, "lvl": np.log(A.values), "mom3": np.log(A / B).values,
                       "cn_sh": (cA / A).clip(0, 1).values, "cn_sh_chg": ((cA / A) - (cB / B)).clip(-1, 1).values,
                       "cn_mom3": np.log((cA + 1) / (cB + 1)).values})
    df["y"] = ((cF / Fw).clip(0, 1) - (cA / A).clip(0, 1)).values
    pf = patent_feats(t0, pan)
    ff = hk.merge(pf, on="k5").drop(columns="k5").groupby("hs6").mean()
    rows.append(df.merge(ff, left_on="hs6", right_index=True, how="left"))
D1 = pd.concat(rows, ignore_index=True)
OWN1 = ["lvl", "mom3", "cn_sh", "cn_sh_chg", "cn_mom3"]
evaluate(D1, {"B1": OWN1, "M1": OWN1 + PATCORE}, "V1_cnshare", {"xgb": xgbfit, "ridge": ridgefit})

# ---------- V2: KSIC5 단위 수입 상대성장 ----------
rows = []
for t0 in range(2005, 2019):
    A_, B_, F_ = [t0 - 2, t0 - 1, t0], [t0 - 5, t0 - 4, t0 - 3], [t0 + 1, t0 + 2, t0 + 3]
    sub = W[B_ + A_ + F_]; s = sub[(sub > 0).all(axis=1)]   # 창 전체 관측 품목만(HS 개정 단절 제거)
    t = s.reset_index().merge(hk, on="hs6")
    w = t.groupby("hs6").k5.transform("count")
    for c in B_ + A_ + F_:
        t[c] = t[c] / w                                    # 다중 KSIC 품목은 균등 배분
    g = t.groupby("k5")[B_ + A_ + F_].sum()
    gc = C.loc[s.index].reset_index().merge(hk, on="hs6")
    for c in B_ + A_:
        gc[c] = gc[c] / gc.groupby("hs6").k5.transform("count")
    gc = gc.groupby("k5")[B_ + A_].sum()
    A, B, Fw = g[A_].sum(1), g[B_].sum(1), g[F_].sum(1)
    df = pd.DataFrame({"k5": g.index, "t0": t0, "lvl": np.log(A.values), "mom3": np.log(A / B).values,
                       "mom1": np.log(g[t0] / g[t0 - 1]).values,
                       "cn_sh": (gc.loc[g.index, A_].sum(1) / A).values})
    df["y_raw"] = np.log(Fw / A).values
    rows.append(df.merge(patent_feats(t0, pan), on="k5", how="left"))
D2 = pd.concat(rows, ignore_index=True)
D2["y"] = D2.y_raw - D2.groupby("t0").y_raw.transform("median")
OWN2 = ["lvl", "mom3", "mom1", "cn_sh"]
evaluate(D2, {"B1": OWN2, "M1": OWN2 + PATCORE, "M2": OWN2 + ["pm_KR", "pm_FOR", "p_gap_for_kr", "p_domshare_chg", "pm_KRin"]},
         "V2_ksic5", {"xgb": xgbfit, "ridge": ridgefit})
D1.to_parquet(os.path.join(CACHE, "g1_V1.parquet")); D2.to_parquet(os.path.join(CACHE, "g1_V2.parquet"))
