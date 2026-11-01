# -*- coding: utf-8 -*-
"""최종시험 이후 전용(모형 수정 없음, 해석·민감도).
(A) 사후 진단: 최종시험 타깃(11차 2021→2024)과 각 피처의 단일 Spearman — 개발 원점 평균과 비교(어떤 신호가 뒤집혔나)
(B) 주 타깃 미래 전망 민감도: t0=2024 의 frgn_mom·lvl_KR(시차1, 2023 해외출원 절단)을 시차2 값으로 바꾼 버전과의 순위상관
(C) 모듈 D(최종시험 통과) 미래 전망: D26 9피처·α=5, 학습 t0 ≤ 2021(결과 ≤2024), t0=2024 → Σ2025–27 / Σ2022–24 순위.
    HS2022 개정 때문에 수출 합계는 2019–21·2022–24 양쪽에 보고된 공통 HS6 바구니로 모든 연도 계산.
출력: R2/posthoc_final_univar.csv, future_forecast_sensitivity.csv, future_forecast_D_2024_2027.csv, R2/posthoc_future.json"""
import os, sys, json, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("N1_MODE", "dev")
from n1lib import BASE, B, CACHE, load_map, cn_share, ksic_names
import final_run_r2 as FR
F = FR.F; FD = FR.FD; res = {}
# ---------------- (A)
Pd = pd.read_parquet(os.path.join(CACHE, "panel_n1.parquet")); Pd = Pd[Pd.t0 != 2010]
y = FR.load_final_target()
te = Pd[Pd.t0 == 2021].join(y, on="ksic"); te = te[te.y.notna()]
dev = Pd[Pd.t0.between(2011, 2018)].dropna(subset=["y_emp"])
rows = []
for c in F + ["g3_emp", "tsi", "pi_g1", "xg3", "mom_KR", "dsh_KR"]:
    dv = dev.groupby("t0").apply(lambda g: spearmanr(g[c], g.y_emp, nan_policy="omit")[0])
    rows.append({"feature": c, "dev_mean_rho": round(dv.mean(), 3), "dev_pos_share": round((dv > 0).mean(), 3),
                 "final_rho": round(spearmanr(te[c], te.y, nan_policy="omit")[0], 3)})
U = pd.DataFrame(rows); U["sign_flip"] = np.sign(U.dev_mean_rho) != np.sign(U.final_rho)
U.to_csv(os.path.join(HERE, "R2", "posthoc_final_univar.csv"), index=False, encoding="utf-8-sig"); print(U.to_string())
# 차수 차이: 2020 중복연도에서 10차 대 11차 종사자 순위 일치도
k = pd.read_csv(os.path.join(BASE, "공모전데이터", "작업데이터", "kosis_광업제조업_50개산업.csv"), dtype={"세세분류코드": str})
e = k[(k.항목코드 == "T02") & (k.연도 == 2020)].copy(); e.loc[e.마스킹 == 1, "값"] = np.nan
w = e.pivot_table(index="세세분류코드", columns="기준", values="값").dropna()
res["emp2020_10th_vs_11th"] = {"n": int(len(w)), "spearman_level": round(spearmanr(w["10차"], w["11차"])[0], 3),
                               "ratio_11_over_10_median": round(float((w["11차"] / w["10차"]).median()), 3),
                               "ratio_iqr": [round(float(q), 3) for q in (w["11차"] / w["10차"]).quantile([.25, .75])]}
# ---------------- (B)
Pf = pd.read_parquet(os.path.join(CACHE, "panel_n1_future.parquet")); Pf = Pf[Pf.t0 != 2010]
tr = Pf[Pf.t0.between(2008, 2021)].dropna(subset=["y_emp"]); t4 = Pf[Pf.t0 == 2024].copy()
p1, _ = FR.fit_predict(tr, t4, F, "y_emp", FR.ALPHA)
t4b = t4.copy(); t4b["frgn_mom"] = t4b["frgn_mom_L2"]; t4b["lvl_KR"] = t4b["lvl_KR_L2"]
p2, _ = FR.fit_predict(tr, t4b, F, "y_emp", FR.ALPHA)
S = pd.DataFrame({"ksic": t4.ksic.values, "name": t4.ksic.map(ksic_names()).values, "pred_spec": p1, "pred_lag2_fix": p2})
S["rank_spec"] = S.pred_spec.rank(ascending=False).astype(int); S["rank_lag2_fix"] = S.pred_lag2_fix.rank(ascending=False).astype(int)
S.sort_values("rank_spec").to_csv(os.path.join(HERE, "future_forecast_sensitivity.csv"), index=False, encoding="utf-8-sig")
top = lambda c: set(S.nsmallest(10, c).ksic)
res["future_main"] = {"n": int(len(S)), "feature_nan_share_t0_2024": {c: round(float(t4[c].isna().mean()), 3) for c in F},
                      "rank_corr_spec_vs_lag2fix": round(spearmanr(p1, p2)[0], 3), "top10_overlap": len(top("rank_spec") & top("rank_lag2_fix"))}
# ---------------- (C) 모듈 D 미래
x0 = pd.read_csv(os.path.join(B, "_shared", "ext", "R2global_kr_export_hs6.csv"), dtype={"hs6": str}); x0 = x0[(x0.flow == "X") & (x0.year <= 2021)]
x1 = pd.read_csv(os.path.join(B, "_shared", "ext", "N1final_kr_export_hs6_2019_2024.csv"), dtype={"hs6": str}); x1 = x1[x1.year >= 2022]
x = pd.concat([x0[["year", "hs6", "value"]], x1[["year", "hs6", "value"]]]); x["hs6"] = x.hs6.str.zfill(6)
common = set(x[x.year.between(2019, 2021)].hs6) & set(x[x.year >= 2022].hs6)
mp = load_map(); X = x[x.hs6.isin(common)].merge(mp, on="hs6").assign(v=lambda z: z.value * z.w).pivot_table(index="k5", columns="year", values="v", aggfunc="sum")
rows = []
for t0 in range(2005, 2025):
    now = X[[t0-2, t0-1, t0]].sum(1); past = X[[t0-5, t0-4, t0-3]].sum(1)
    fut = X[[t0+1, t0+2, t0+3]].sum(1) if t0 + 3 <= 2024 else pd.Series(np.nan, index=X.index)
    rows.append(pd.DataFrame({"ksic": X.index, "t0": t0, "yD": np.log(fut / now).values, "momX": np.log(now / past).values, "xlvD": np.log(now).values}))
D = pd.concat(rows); D["yD"] = D.yD - D.groupby("t0").yD.transform("median")
D = D.merge(Pf.drop(columns=[c for c in Pf.columns if c.startswith("y_")]), on=["ksic", "t0"], how="left")
D = D[D.ksic.isin(cn_share().ksic.unique())]
trD = D[D.t0 <= 2021].dropna(subset=["yD"]); teD = D[D.t0 == 2024].copy()
assert trD.t0.max() + 3 <= 2024
pD, mD = FR.fit_predict(trD, teD, FD, "yD", FR.ALPHA_D)
trD18 = trD[trD.t0 <= 2018]; pD18, _ = FR.fit_predict(trD18, teD, FD, "yD", FR.ALPHA_D)
R = pd.DataFrame({"ksic": teD.ksic.values, "name": teD.ksic.map(ksic_names()).values, "pred": pD, "pred_train_le2018": pD18,
                  "past_export_growth_2019_21_to_2022_24": teD.momX.values}).sort_values("pred", ascending=False)
R["rank"] = range(1, len(R) + 1); R.to_csv(os.path.join(HERE, "future_forecast_D_2024_2027.csv"), index=False, encoding="utf-8-sig")
res["future_D"] = {"n": int(len(R)), "train_t0_max": int(trD.t0.max()), "common_hs6": len(common), "coef": dict(zip(FD, mD.coef_.round(4).tolist())),
                   "rank_corr_vs_train_le2018": round(spearmanr(pD, pD18)[0], 3), "rank_corr_vs_B0_past_growth": round(spearmanr(pD, teD.momX, nan_policy="omit")[0], 3),
                   "feature_nan_share": {c: round(float(teD[c].isna().mean()), 3) for c in FD}}
json.dump(res, open(os.path.join(HERE, "R2", "posthoc_future.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1)); print(R.head(12).to_string()); print(S.sort_values("rank_spec").head(12).to_string())
