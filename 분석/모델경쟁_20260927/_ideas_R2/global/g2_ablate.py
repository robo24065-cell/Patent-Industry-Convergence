# -*- coding: utf-8 -*-
"""G2 2차 — 수출 3년 상대성장(HS6) 특허 기여의 정체 확인 + 개선 실험 (챔피언/도전자).
 A0 B1_own                       : 품목 자기이력 + 같은 품목 수입 + KSIC 수출 모멘텀
 A1 M1 = B1 + 전 특허 피처       : 1차 파일럿 챔피언
 A2 B1 + KSIC 고유효과(산업 더미) : '특허=산업 정체성 대리변수'인지 검증용 대조군
 A3 B1 + 특허 '변화량'만(pm_*, *_chg) : 수준(pl_*) 제거 → 산업 정체성 효과 배제
 A4 B1 + 특허 수준만(pl_*, share 수준)
 A5 M1 + 5시드 평균
 A6 M1 + 특허 피처 연도 내 순위화(원점 간 분포이동 제거)
 개발 원점 v=2010..2018 (레이블 ≤2021). 🔒 t0=2021 최종시험 미계산.
"""
import sys, os, json
import numpy as np, pandas as pd, xgboost as xgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

FLOW = sys.argv[1] if len(sys.argv) > 1 else "X"
D = pd.read_parquet(os.path.join(CACHE, f"g1_panel_{FLOW}.parquet"))
m = load_map()
k1 = m.groupby("hs6").k5.first()
D["k5"] = D.hs6.map(k1)
dum = pd.get_dummies(D.k5, prefix="K").astype(int); D = pd.concat([D, dum], axis=1)
OWN = [c for c in ["lvl", "mom3", "mom1", "vol", "cn_sh", "cn_sh_chg", "imp_ratio", "imp_mom", "k_mom3"] if c in D]
PAT = [c for c in D.columns if c.startswith(("pm_", "pl_", "p_"))]
CHG = [c for c in PAT if c.startswith("pm_") or c.endswith("_chg") or c == "p_gap_for_kr"]
LVL = [c for c in PAT if c not in CHG]
for c in PAT:
    D["rk_" + c] = D.groupby("t0")[c].rank(pct=True)
RK = ["rk_" + c for c in PAT]
SETS = {"A0_B1": (OWN, 1), "A1_M1": (OWN + PAT, 1), "A2_B1+KSICdummy": (OWN + list(dum.columns), 1),
        "A3_B1+patCHG": (OWN + CHG, 1), "A4_B1+patLVL": (OWN + LVL, 1), "A5_M1_5seed": (OWN + PAT, 5),
        "A6_M1_rank": (OWN + RK, 1), "A7_M1+KSICdummy": (OWN + PAT + list(dum.columns), 1)}


def fit(tr, te, cols, nseed):
    ps = []
    for s in range(nseed):
        mdl = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                               objective="reg:absoluteerror", nthread=2, random_state=s)
        mdl.fit(tr[cols], tr.y); ps.append(mdl.predict(te[cols]))
    return np.mean(ps, axis=0)


res = []
for v in range(2010, 2019):
    tr, te = D[D.t0 <= v - 3], D[D.t0 == v]
    r = {"v": v, "B0": np.abs(te.y).mean()}
    for k, (c, ns) in SETS.items():
        p = fit(tr, te, c, ns); r[k] = np.abs(te.y - p).mean(); r["rho_" + k] = spearman(p, te.y)
    res.append(r)
R = pd.DataFrame(res)
out = {}
for k in SETS:
    out[k] = {"mae": round(R[k].mean(), 4), "skill_vsB0": round(1 - R[k].mean() / R.B0.mean(), 4),
              "skill_vsB1": round(float((1 - R[k] / R.A0_B1).mean()), 4), "win_vsB1": round(float((R[k] < R.A0_B1).mean()), 2),
              "d_vsM1": round(float((1 - R[k] / R.A1_M1).mean()), 4), "win_vsM1": round(float((R[k] < R.A1_M1).mean()), 2),
              "rho": round(R["rho_" + k].mean(), 4)}
print(pd.DataFrame(out).T.to_string())
R.to_csv(os.path.join(HERE, f"g2_ablate_{FLOW}.csv"), index=False, encoding="utf-8-sig")
