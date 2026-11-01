# -*- coding: utf-8 -*-
"""아이디어 A 파일럿: 5개국 '기술 특화(특허 점유) 변화 → 수출 특화(수출 점유) 변화'.
단위 = (HS6 h, 국가 c ∈ KR·CN·JP·US·DE(특허는 EP 대응), 원점 t0). 수출 점유 s = X_c,h / Σ_5국 X_h.
y = [log s(t0+3) − log s(t0)] − 같은 (c,t0) 중앙값  → '그 나라 평균 대비 이 품목에서 점유를 더 늘렸나'.
특허: 같은 KSIC(h)의 국가 c 특허 점유 변화(Δlog share), 특화도(RTA) — (c,t0) 내 중앙값 차감. 출원연도 ≤ t0−1.
기준선 B = 무역만(과거 3년 점유변화, 점유 수준, 수출 규모). 모형 M = B + 특허. 원점: 결과 ≤2021 → t0 ≤ 2018.
지표: (c,t0) 내 Spearman ρ 평균, MAE skill(1−MAE_M/MAE_B), 상위 1/3 점유확대 AUC."""
import sys, numpy as np, pandas as pd, xgboost as xgb
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from ilib import *
STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 3
X = pd.read_csv(os.path.join(B, "_shared", "ext", "R2ind_comtrade_exports_5c_hs6_2004_2021.csv"), dtype={"hs6": str})
X = X[X.year <= DEV_MAX]; X["hs6"] = X.hs6.str.zfill(6)
RC = {410: "KR", 156: "CN", 392: "JP", 842: "US", 276: "EP"}
X["c"] = X.reporter.map(RC)
XV = X.pivot_table(index=["hs6", "year"], columns="c", values="export_usd", aggfunc="sum")
yrs = sorted(X.year.unique()); print("수집 연도", yrs)
mp = load_map(); pv = load_pat()
# 특허 국가 점유 (KSIC, t0) — 다중 KSIC HS6는 대응 KSIC 평균
def pshare(t0, lag=1):
    e = t0-lag; idx = pv.index.get_level_values(1)
    w1 = pv[(idx >= e-2) & (idx <= e)].groupby(level=0).sum(); w0 = pv[(idx >= e-5) & (idx <= e-3)].groupby(level=0).sum()
    s1 = (w1+1).div((w1+1).sum(1), axis=0); s0 = (w0+1).div((w0+1).sum(1), axis=0)
    tot1 = w1.sum(0); rta = np.log(s1.div((tot1/tot1.sum()), axis=1))
    out = []
    for c in ["KR", "CN", "JP", "US", "EP"]:
        out.append(pd.DataFrame({"ksic": s1.index, "c": c, "p_dls": np.log(s1[c]/s0[c]).values, "p_rta": rta[c].values,
                                 "p_mom": np.log((w1[c]+1)/(w0[c]+1)).values, "p_lvl": np.log1p(w1[c]).values}))
    return pd.concat(out)
rows = []
for t0 in yrs:
    if t0+STEP not in yrs or t0 > DEV_MAX-STEP: continue
    have_past = (t0-STEP) in yrs
    ps = pshare(t0).merge(mp.rename(columns={"k5": "ksic"}), on="ksic").groupby(["hs6", "c"])[["p_dls", "p_rta", "p_mom", "p_lvl"]].mean().reset_index()
    for c in ["KR", "CN", "JP", "US", "EP"]:
        def S(y):
            if y not in yrs: return None
            M = XV.xs(y, level="year"); return M[c], M.sum(1)
        x0, T0 = S(t0); x3, T3 = S(t0+STEP)
        d = pd.DataFrame({"x0": x0, "T0": T0}).join(pd.DataFrame({"x3": x3, "T3": T3}), how="inner")
        if have_past:
            xm, Tm = S(t0-STEP); d = d.join(pd.DataFrame({"xm": xm, "Tm": Tm}), how="left")
        else: d["xm"] = np.nan; d["Tm"] = np.nan
        d = d[(d.x0 >= 1e6) & (d.x3 >= 1e5) & (d.T0 >= 1e7)]
        d["y"] = np.log(d.x3/d.T3) - np.log(d.x0/d.T0)
        d["b_past"] = np.where(d.xm >= 1e5, np.log(d.x0/d.T0) - np.log(d.xm/d.Tm), np.nan)
        d["b_sh"] = np.log(d.x0/d.T0); d["b_lx"] = np.log(d.x0); d["b_gT"] = np.log(d.T0/d.Tm) if have_past else np.nan
        d["c"] = c; d["t0"] = t0; d = d.reset_index().rename(columns={"index": "hs6"})
        rows.append(d)
P = pd.concat(rows, ignore_index=True).merge(ps.iloc[0:0], how="left") if False else pd.concat(rows, ignore_index=True)
# 특허 결합(원점별)
PS = []
for t0 in sorted(P.t0.unique()):
    q = pshare(t0).merge(mp.rename(columns={"k5": "ksic"}), on="ksic").groupby(["hs6", "c"])[["p_dls", "p_rta", "p_mom", "p_lvl"]].mean().reset_index(); q["t0"] = t0; PS.append(q)
P = P.merge(pd.concat(PS), on=["hs6", "c", "t0"], how="left")
P = P.merge(mp.groupby("hs6").k5.first().rename("ksic").reset_index(), on="hs6", how="left")
for col in ["y", "b_past", "b_sh", "b_lx", "b_gT", "p_dls", "p_rta", "p_mom", "p_lvl"]:
    P[col] = P[col] - P.groupby(["c", "t0"])[col].transform("median")
P["y"] = P.y.clip(-3, 3)
P.to_parquet(f"{CACHE}/panel_spec_step{STEP}.parquet")
print("행", len(P), "원점", sorted(P.t0.unique()), "국가별", P.groupby("c").size().to_dict(), "HS6", P.hs6.nunique(), "KSIC", P.ksic.nunique())
BF = ["b_past", "b_sh", "b_lx", "b_gT"]; PF = ["p_dls", "p_rta", "p_mom", "p_lvl"]
origins = sorted(P.t0.unique()); res = []
for v in origins:
    tr = P[P.t0 <= v-STEP]; te = P[P.t0 == v]
    if len(tr) < 500: continue
    r = {"v": v, "n": len(te), "zero": mae(te.y, 0)}
    for lab, F in [("B", BF), ("M", BF+PF), ("P", PF)]:
        m = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                             objective="reg:absoluteerror", n_jobs=2, random_state=0).fit(tr[F], tr.y)
        p = m.predict(te[F]); r[lab] = mae(te.y, p)
        r[f"rho_{lab}"] = np.mean([spearmanr(g.y, p[g.index.map(te.index.get_loc)])[0] for _, g in te.groupby("c")])
        r[f"auc_{lab}"] = roc_auc_score((te.y >= te.groupby("c").y.transform(lambda s: s.quantile(2/3))).astype(int), p)
    res.append(r)
R = pd.DataFrame(res); print(R.round(4).to_string())
print(f"무역만 skill vs 0: {(1-R.B/R.zero).mean():.4f} | 특허 증분 skill(1−MAE_M/MAE_B): {(1-R.M/R.B).mean():.4f} 승률 {((R.M<R.B)).mean():.2f} | ρ_B {R.rho_B.mean():.3f} ρ_M {R.rho_M.mean():.3f} ρ_P {R.rho_P.mean():.3f}")
R.to_csv(f"{HERE}/p30_result_step{STEP}.csv", index=False)
