# -*- coding: utf-8 -*-
"""최종시험 독립 재계산(final_run_r2.py·b01_panel.py 함수 미사용).
(1) 타깃: KOSIS 원자료에서 11차 T02 2021→2024 직접 계산
(2) t0=2021 테스트 피처: 원천 캐시(pat_kcy, prio_kcy, kr_fam_pit)·원자료(cn_import_yearly_ksic5, 생산지수 월별)에서 새로 계산해 패널값과 대조
(3) 학습: 패널 학습행(t0 2008–2018, 결과 ≤2021) + 자체 순위화·Ridge → ρ, B0, AUC 재계산
(4) 모듈 D 타깃: 수집 수출 원자료에서 직접 재계산
출력: R2/verify_final.json"""
import os, json, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
B = os.path.join(BASE, "분석", "모델경쟁_20260927"); HERE = os.path.join(B, "N1_기술점유무역_산업고용조기경보"); CA = os.path.join(HERE, "cache")
F = ["mom_KR_L2", "dsh_KR_L2", "cnsh", "dcnsh", "lvl_KR", "frgn_mom", "kab_mom", "dffam_pit_Q2", "pi_g3"]
out = {}
# (1) 타깃
k = pd.read_csv(os.path.join(BASE, "공모전데이터", "작업데이터", "kosis_광업제조업_50개산업.csv"), dtype={"세세분류코드": str})
e = k[(k.항목코드 == "T02")].copy(); e.loc[e.마스킹 == 1, "값"] = np.nan
e11 = e[e.기준 == "11차"].pivot_table(index="세세분류코드", columns="연도", values="값")
y = np.log(e11[2024] / e11[2021]); y = (y - y.median()).dropna()
# (2) 피처 재계산
pv = pd.read_parquet(os.path.join(CA, "pat_kcy.parquet")).pivot_table(index=["ksic", "y"], columns="cc", values="n", aggfunc="sum").fillna(0)
def win(lo, hi): return pv[(pv.index.get_level_values(1) >= lo) & (pv.index.get_level_values(1) <= hi)].groupby(level=0).sum()
t0 = 2021
a1, a0 = win(2017, 2019), win(2014, 2016)              # 시차2
f = pd.DataFrame(index=a1.index)
f["mom_KR_L2"] = np.log((a1.KR + 1) / (a0.KR.reindex(a1.index).fillna(0) + 1))
f["dsh_KR_L2"] = a1.KR / a1.sum(1).clip(lower=1) - (a0.KR / a0.sum(1).clip(lower=1)).reindex(a1.index).fillna(0)
b1, b0_ = win(2018, 2020), win(2015, 2017)            # 시차1
f["lvl_KR"] = np.log1p(b1.KR.reindex(f.index))
fr = lambda w: w[["US", "JP", "EP"]].sum(1)
f["frgn_mom"] = np.log((fr(b1) + 1) / (fr(b0_).reindex(b1.index).fillna(0) + 1)).reindex(f.index)
pr = pd.read_parquet(os.path.join(CA, "prio_kcy.parquet")); pr = pr[pr.pc.notna() & (pr.pc != pr.cc) & (pr.pc == "KR")]
k1 = pr[pr.y.between(2016, 2018)].groupby("ksic").n.sum(); k0 = pr[pr.y.between(2013, 2015)].groupby("ksic").n.sum()
ix = k1.index.union(k0.index); f["kab_mom"] = np.log((k1.reindex(ix).fillna(0) + 1) / (k0.reindex(ix).fillna(0) + 1)).reindex(f.index)
fp = pd.read_parquet(os.path.join(CA, "kr_fam_pit.parquet"))
def sh(lo, hi, cut):
    w = fp[fp.y.between(lo, hi)]; n = w.groupby("ksic").n.sum()
    return (w[(w.dom == 0) | (w.fpy <= cut)].groupby("ksic").n.sum().reindex(n.index).fillna(0) + 1) / (n + 2)
f["dffam_pit_Q2"] = (sh(2017, 2019, 2021) - sh(2014, 2016, 2018)).reindex(f.index)
c = pd.read_csv(os.path.join(BASE, "분석", "fusion", "cn_import_yearly_ksic5.csv"), dtype={"ksic": str})
c = c[(c.ksic_level == 5) & (c.year <= 2021)].pivot_table(index="ksic", columns="year", values="cn_share_imp")
f["cnsh"] = c[2021].reindex(f.index); f["dcnsh"] = (c[2021] - c[2018]).reindex(f.index)
pi = pd.read_csv(os.path.join(BASE, "공모전데이터", "외부데이터", "경기지표", "kosis_광공업생산지수_월별.csv"), dtype={"산업코드": str})
pi = pi[pi.항목코드 == "T10"].copy(); pi["v"] = pd.to_numeric(pi.값, errors="coerce"); pi["yr"] = pi.연월 // 100
PI = pi[pi.yr <= 2021].groupby(["산업코드", "yr"]).v.mean().unstack()
def pick(k5):
    for cc in ["C" + k5[:3], "C" + k5[:2]]:
        if cc in PI.index: return cc
f["pi_g3"] = [np.log(PI.loc[pick(k5), 2021] / PI.loc[pick(k5), 2018]) if pick(k5) else np.nan for k5 in f.index]
# B0: 체인 2018→2021 (10차 2020 × 11차 2021/2020)
e10 = e[e.기준 == "10차"].pivot_table(index="세세분류코드", columns="연도", values="값"); e10 = e10.where(e10 > 0)
emp2021 = e10[2020] * (e11[2021] / e11[2020].where(e11[2020] > 0)).reindex(e10.index)
g3 = np.log(emp2021 / e10[2018])
P = pd.read_parquet(os.path.join(CA, "panel_n1.parquet")); P = P[P.t0 != 2010]
te_p = P[P.t0 == 2021].set_index("ksic")
diff = {cn: float(np.nanmax(np.abs(f[cn].reindex(te_p.index) - te_p[cn]))) for cn in F}
out["feature_max_abs_diff_vs_panel"] = {kk: round(v, 8) for kk, v in diff.items()}
out["B0_feature_max_abs_diff"] = float(np.nanmax(np.abs((g3 - g3.median()).reindex(te_p.index) - te_p.g3_emp)))
# (3) 재적합
tr = P[P.t0.between(2008, 2018)].dropna(subset=["y_emp"]).copy()
assert tr.t0.max() + 3 <= 2021
R = lambda s: s.rank(pct=True).fillna(0.5)
Xtr = pd.concat([tr.groupby("t0")[cn].transform(lambda s: R(s)) for cn in F], axis=1); Xtr.columns = F
ytr = tr.groupby("t0").y_emp.transform(lambda s: R(s))
te = f.reindex(y.index).copy()
Xte = pd.DataFrame({cn: R(te[cn]) for cn in F})
m = Ridge(alpha=2.0).fit(Xtr[F], ytr); p = m.predict(Xte[F])
pb = R((g3 - g3.median()).reindex(y.index)).values
lab = (y.rank(pct=True) >= 2/3).astype(int)
rho, rb = spearmanr(y.values, p)[0], spearmanr(y.values, pb)[0]
FT = ["cnsh", "dcnsh", "pi_g3"]; mT = Ridge(alpha=2.0).fit(Xtr[FT], ytr); pT = mT.predict(Xte[FT])
out.update({"n": int(len(y)), "rho": round(rho, 4), "B0_rho": round(rb, 4), "skill": round(rho - rb, 4),
            "auc_top3": round(roc_auc_score(lab, p), 4), "auc_B0": round(roc_auc_score(lab, pb), 4),
            "no_patent_rho": round(spearmanr(y.values, pT)[0], 4), "coef": dict(zip(F, m.coef_.round(4).tolist()))})
# (4) 모듈 D 타깃 재계산
x = pd.read_csv(os.path.join(B, "_shared", "ext", "N1final_kr_export_hs6_2019_2024.csv"), dtype={"hs6": str}); x["hs6"] = x.hs6.str.zfill(6)
mp = pd.read_csv(os.path.join(BASE, "분석", "fusion", "ksic_hs6_map.csv"), dtype=str); mp["hs6"] = mp.hs6.str.zfill(6); mp = mp[["k5", "hs6"]].drop_duplicates()
mp["w"] = 1 / mp.groupby("hs6").k5.transform("count")
common = set(x[x.year <= 2021].hs6) & set(x[x.year >= 2022].hs6)
kx = x[x.hs6.isin(common)].merge(mp, on="hs6").assign(v=lambda z: z.value * z.w).pivot_table(index="k5", columns="year", values="v", aggfunc="sum")
yD = np.log(kx[[2022, 2023, 2024]].sum(1) / kx[[2019, 2020, 2021]].sum(1)); yD = yD - yD.median()
out["module_D_target"] = {"n_finite": int(np.isfinite(yD).sum()), "n_total": int(len(yD)), "nonfinite_ksic": yD[~np.isfinite(yD)].index.tolist()}
json.dump(out, open(os.path.join(HERE, "R2", "verify_final.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
