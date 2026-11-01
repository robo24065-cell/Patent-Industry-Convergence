# -*- coding: utf-8 -*-
"""G1 파일럿 — 특허 선행 신호로 한국 HS6 품목 수입의 다음 3년 상대 성장 예측 (수입대체/기술의존 조기경보).
단위: HS6 품목 × 원점 t0. 타깃: y = log(Σ수입[t0+1..t0+3] / Σ수입[t0-2..t0]) − 원점 내 중앙값.
개발 원점 v = 2010..2018 (레이블 ≤2021), 학습 원점 ≤ v−3. 🔒 최종시험 t0=2021 (F=2022) — 여기서 절대 계산 안 함.
비교: B0=0(모든 품목 동일 성장), B1=품목 자기이력 XGB, M1=B1+KSIC 특허 피처(같은 학습기).
"""
import sys, json, os
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

FLOW = sys.argv[1] if len(sys.argv) > 1 else "M"   # M=수입(WLD), X=수출
HZ = int(sys.argv[2]) if len(sys.argv) > 2 else 3   # 결과 창 길이(년). 레이블 끝 ≤2021 유지
m = load_map()
if FLOW == "M":
    imp = load_imports()
    W = trade_panel(imp, "WLD"); C = trade_panel(imp, "CN").reindex_like(W)
else:
    ex = load_exports()
    W = ex.pivot_table(index="hs6", columns="year", values="value", aggfunc="sum")
    imp = load_imports(); C = trade_panel(imp, "WLD").reindex_like(W)   # 수출 모형에서 C=같은 품목 수입
pan = pd.read_parquet(os.path.join(CACHE, "pat_panel.parquet"))
hk = m.merge(pd.Series(W.index, name="hs6"), on="hs6")
k_of = hk.groupby("hs6").k5.apply(list)

rows = []
for t0 in range(2005, 2022 - HZ):
    yrsA, yrsB, yrsF = [t0 - 2, t0 - 1, t0], [t0 - 5, t0 - 4, t0 - 3], list(range(t0 + 1, t0 + HZ + 1))
    need = yrsB + yrsA + yrsF
    if min(need) < W.columns.min():
        continue
    sub = W[need]
    ok = (sub > 0).all(axis=1)
    s = sub[ok]
    A, B, Fw = s[yrsA].sum(1), s[yrsB].sum(1), s[yrsF].sum(1)
    df = pd.DataFrame({"hs6": s.index, "t0": t0})
    df["lvl"] = np.log(A.values)
    df["mom3"] = np.log(A / B).values
    df["mom1"] = np.log(s[t0] / s[t0 - 1]).values
    df["vol"] = np.log(s[yrsB + yrsA]).diff(axis=1).std(axis=1).values
    cA = C.loc[s.index, yrsA].fillna(0).sum(1); cB = C.loc[s.index, yrsB].fillna(0).sum(1)
    if FLOW == "M":
        df["cn_sh"] = (cA / A).clip(0, 1).values; df["cn_sh_chg"] = (cA / A - cB / B).clip(-1, 1).values
    else:
        df["imp_ratio"] = np.log((cA + 1) / A).values; df["imp_mom"] = np.log((cA + 1) / (cB + 1)).values
    df["y_raw"] = np.log(Fw / A).values
    # KSIC 수준 무역 모멘텀
    tmp = pd.DataFrame({"hs6": s.index, "A": A.values, "B": B.values}).merge(hk, on="hs6")
    kt = tmp.groupby("k5")[["A", "B"]].sum(); kt["k_mom3"] = np.log(kt.A / kt.B)
    pf = patent_feats(t0, pan).merge(kt[["k_mom3"]].reset_index(), on="k5", how="left")
    ff = hk.merge(pf, on="k5").drop(columns="k5").groupby("hs6").mean()
    df = df.merge(ff, left_on="hs6", right_index=True, how="left")
    rows.append(df)
D = pd.concat(rows, ignore_index=True)
D["y"] = D.y_raw - D.groupby("t0").y_raw.transform("median")
os.makedirs(CACHE, exist_ok=True)
D.to_parquet(os.path.join(CACHE, f"g1_panel_{FLOW}{'' if HZ == 3 else HZ}.parquet"), index=False)
print("표본", D.groupby("t0").size().to_dict())

OWN = [c for c in ["lvl", "mom3", "mom1", "vol", "cn_sh", "cn_sh_chg", "imp_ratio", "imp_mom", "k_mom3"] if c in D]
PAT = [c for c in D.columns if c.startswith("pm_") or c.startswith("pl_") or c.startswith("p_")]
SETS = {"B1_own": OWN, "M1_own+pat": OWN + PAT,
        "M2_own+patcore": OWN + ["pm_KR", "pm_FOR", "p_gap_for_kr", "p_domshare", "p_domshare_chg", "pm_KRin", "pm_KRout", "p_outshare", "p_outshare_chg"]}


def fit_pred(tr, te, cols, seed=0):
    mdl = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                           colsample_bytree=0.8, objective="reg:absoluteerror", nthread=2, random_state=seed)
    mdl.fit(tr[cols], tr.y)
    return mdl.predict(te[cols])


res = []
for v in range(2010, 2022 - HZ):
    tr, te = D[D.t0 <= v - HZ], D[D.t0 == v]
    top = (te.y >= te.y.quantile(0.8)).astype(int)
    r = {"v": v, "n": len(te), "mae_B0": np.abs(te.y).mean()}
    for k, cols in SETS.items():
        p = fit_pred(tr, te, cols)
        r["mae_" + k] = np.abs(te.y - p).mean(); r["rho_" + k] = spearman(p, te.y); r["auc_" + k] = roc_auc_score(top, p)
    res.append(r)
R = pd.DataFrame(res)
for k in SETS:
    R["skill_" + k + "_vsB1"] = 1 - R["mae_" + k] / R["mae_B1_own"]
    R["skill_" + k + "_vsB0"] = 1 - R["mae_" + k] / R["mae_B0"]
pd.set_option("display.width", 250)
print(R.round(4).to_string())
summ = {c: round(float(R[c].mean()), 4) for c in R.columns if c != "v"}
for k in SETS:
    summ["win_vsB1_" + k] = float((R["mae_" + k] < R["mae_B1_own"]).mean())
print(json.dumps(summ, ensure_ascii=False, indent=1))
R.to_csv(os.path.join(HERE, f"g1_result_{FLOW}{'' if HZ == 3 else HZ}.csv"), index=False, encoding="utf-8-sig")
