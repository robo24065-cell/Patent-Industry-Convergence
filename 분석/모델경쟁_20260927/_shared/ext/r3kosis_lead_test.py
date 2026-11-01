# -*- coding: utf-8 -*-
"""개발 구간 선행 검정 (결과 출원연도 ≤2019만 읽음 — 최종시험 구간 미접촉).
단위: KSIC50 × 원점 t0 ∈ 2009..2016.
결과 Y = log(KR 출원 t0+1..t0+3 +1) − log(KR 출원 t0-2..t0 +1)  (보조: 5개국 합계)
통제 M = log(KR t0-2..t0 +1) − log(KR t0-5..t0-3 +1), 원점 고정효과(원점 내 순위화).
피처(원점 시점에 공표된 값만):
  F1 d_inv_ship_L2 : 품목편 짝지은 품목의 재고/출하 비율 변화(자료연도 t0-2; Y는 Y+2년 1~2월 공표)  [가설: 하락 → 특허 증가, 음(-)]
  F2 g_prod_2y_L2  : 품목편 연쇄 생산 성장 (t0-4→t0-2)                                              [양(+)]
  F3 g_ship_2y_L2  : 품목편 연쇄 출하 성장 (t0-4→t0-2)                                              [양(+)]
  F4 g3_cap_L1     : 월별 생산능력지수 연평균 3년 성장(자료연도 t0-1; 이듬해 1월 공표)             [양(+)]
  F5 g3_util_L1    : 월별 가동률지수 연평균 3년 성장(t0-1)                                           [양(+)]
  F6 inv_ship_lvl_L2: 재고/출하 비율 수준(t0-2, 비율이라 재기준 무관)                                [음(-)]
검정: 원점별 편 Spearman(M 통제) → 평균, 양/음 원점 수, 원점 내 순열(2000회) 귀무 분포; 합동 OLS(원점FE+M, KSIC 군집 SE).
"""
import os, json
import pandas as pd, numpy as np
import statsmodels.formula.api as smf
from scipy.stats import rankdata

EXT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(20260929)
P = pd.read_parquet(os.path.join(EXT, "trade_patcnt_ksic_year_cc.parquet"), filters=[("year", "<=", 2019)])
assert P.year.max() <= 2019
kr = P[P.cc == "KR"].pivot_table(index="ksic", columns="year", values="n", aggfunc="sum").fillna(0)
al = P.pivot_table(index="ksic", columns="year", values="n", aggfunc="sum").fillna(0)
def win(T, a, b): return T[[y for y in range(a, b + 1)]].sum(axis=1)

CH = pd.read_parquet(os.path.join(EXT, "r3kosis_ksic50_items_chain.parquet"))
CU = pd.read_parquet(os.path.join(EXT, "r3kosis_ksic50_capacity_util_annual.parquet"))
ch = CH.set_index(["ksic", "year"]); cu = CU.set_index(["ksic", "year"])

rows = []
for t0 in range(2009, 2017):
    for k in kr.index:
        r = {"ksic": k, "t0": t0}
        r["Y"] = np.log(win(kr, t0 + 1, t0 + 3)[k] + 1) - np.log(win(kr, t0 - 2, t0)[k] + 1)
        r["Y5"] = np.log(win(al, t0 + 1, t0 + 3)[k] + 1) - np.log(win(al, t0 - 2, t0)[k] + 1)
        r["M"] = np.log(win(kr, t0 - 2, t0)[k] + 1) - np.log(win(kr, t0 - 5, t0 - 3)[k] + 1)
        r["M5"] = np.log(win(al, t0 - 2, t0)[k] + 1) - np.log(win(al, t0 - 5, t0 - 3)[k] + 1)
        g = lambda y, c: ch[c].get((k, y), np.nan)
        r["d_inv_ship_L2"] = g(t0 - 2, "d_inv_ship")
        r["inv_ship_lvl_L2"] = g(t0 - 2, "inv_ship1")
        r["g_prod_2y_L2"] = g(t0 - 2, "g_prod") + g(t0 - 3, "g_prod")
        r["g_ship_2y_L2"] = g(t0 - 2, "g_ship") + g(t0 - 3, "g_ship")
        r["g3_cap_L1"] = cu["g3_cap"].get((k, t0 - 1), np.nan)
        r["g3_util_L1"] = cu["g3_util"].get((k, t0 - 1), np.nan)
        rows.append(r)
D = pd.DataFrame(rows)
D = D[win(kr, 2004, 2016).reindex(D.ksic).values >= 30]  # KR 출원이 극히 적은 KSIC 제외
D.to_parquet(os.path.join(EXT, "r3kosis_lead_test_panel.parquet"), index=False)

FEATS = ["d_inv_ship_L2", "inv_ship_lvl_L2", "g_prod_2y_L2", "g_ship_2y_L2", "g3_cap_L1", "g3_util_L1"]
def pcor(x, y, m):
    x, y, m = rankdata(x), rankdata(y), rankdata(m)
    X = np.c_[np.ones_like(m), m]
    rx = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]; ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])

res = []
for yv, mv in [("Y", "M"), ("Y5", "M5")]:
    for f in FEATS:
        per, groups = {}, []
        for t0, g in D.groupby("t0"):
            g = g.dropna(subset=[f, yv, mv])
            if len(g) < 15: continue
            per[t0] = pcor(g[f].values, g[yv].values, g[mv].values); groups.append(g)
        if not per: continue
        obs = np.mean(list(per.values()))
        null = []
        for _ in range(2000):
            null.append(np.mean([pcor(rng.permutation(g[f].values), g[yv].values, g[mv].values) for g in groups]))
        null = np.array(null)
        p2 = float((np.abs(null) >= abs(obs)).mean())
        G = pd.concat(groups)
        G = G.assign(fr=G.groupby("t0")[f].rank(pct=True), yr=G[yv], mr=G[mv])
        ols = smf.ols("yr ~ fr + mr + C(t0)", data=G).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(G.ksic)[0]})
        res.append({"outcome": yv, "feature": f, "origins": len(per), "n_obs": len(G),
                    "mean_partial_spearman": round(obs, 4), "null_sd": round(float(null.std()), 4),
                    "perm_p_two_sided": round(p2, 4),
                    "n_pos": sum(v > 0 for v in per.values()), "n_neg": sum(v < 0 for v in per.values()),
                    "per_origin": json.dumps({int(k): round(v, 3) for k, v in per.items()}),
                    "ols_coef_rank(top-bottom)": round(float(ols.params["fr"]), 4), "ols_p_cluster": round(float(ols.pvalues["fr"]), 4)})
R = pd.DataFrame(res)
R.to_csv(os.path.join(EXT, "r3kosis_lead_test_results.csv"), index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print(D.groupby("t0").agg(n=("ksic", "size"), inv=("d_inv_ship_L2", "count"), cap=("g3_cap_L1", "count")).to_string())
print(R.drop(columns=["per_origin"]).to_string())
for _, r in R.iterrows(): print(r.outcome, r.feature, r.per_origin)
