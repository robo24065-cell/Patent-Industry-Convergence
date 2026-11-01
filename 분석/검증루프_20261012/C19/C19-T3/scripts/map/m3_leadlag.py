# [C19-T3 사본] 원본: C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared_report\fusion_advantage\map5\m3_leadlag.py — 경로만 바꿈(OUT/HERE→C19_OUT, META→C19_META, 상대 MC→원래 폴더)
# -*- coding: utf-8 -*-
r"""map5 3단계: 국가 간 선행-후행 (산업별 해외 → 한국, 한국 → 해외).
사전 고정 설계
- 계열: 출원연도 발명특허의 국가 내 KSIC 비중 로그 x = log((n+0.5)/Σn) 의 1차 차분 Δx (국가 총량·수록률의 균일 변화 제거, 추세 제거).
- 주 기간(개발): 출원 2004–2018 → Δ 2005–2018 (JP 2019~ 수록 감소 의심 구간과 T3 최종시험 F=2020 이전). 사후: 2004–2022.
- 산업별 시험 (KSIC 50 × 해외 4개국 = 200쌍, 방향 2개):
  (a) 교차상관 r(L) = corr(ΔX_{t−L}, ΔKR_t), L=0..3 (역방향 KR→X도 L=1..3)
  (b) 그랜저형: ΔKR_t = a + bΔKR_{t−1} + cΔX_{t−L} + e, L=1,2,3 각각 t검정 → 쌍별 최소 p×3(본페로니) → 200쌍 BH-FDR. 역방향 동일.
- 패널 시험(검정력 보강): ΔKR_{k,t} = α_k + γ_t + ρΔKR_{k,t−1} + βΔX_{k,t−L}, KSIC 군집 표준오차 + 순열(해외 계열의 KSIC 라벨을 섞음, 1000회) p.
- 포트폴리오 추종: corr_k( x_KR,t , x_X,t−L ), t=2012..2018 평균, L=0..8 (수준 비교, 해석용).
"""
import os as _os
_C19_OUT = _os.environ['C19_OUT']; _C19_META = _os.environ['C19_META']
import os, json, time
import numpy as np, pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
OUT = _C19_OUT
t0 = time.time()
rng = np.random.default_rng(20261005)
FOR = ["US", "JP", "CN", "EP"]
pa = pd.read_csv(os.path.join(OUT, "m2_panel_appl.csv"), dtype={"ksic": str})
pa = pa[pa.year.between(2004, 2022)]
pa["tot"] = pa.groupby(["cc", "year"]).n_inv.transform("sum")
pa["x"] = np.log((pa.n_inv + 0.5) / pa.tot)
X = pa.pivot_table(index=["cc", "ksic"], columns="year", values="x")
D = X.diff(axis=1)
names = pd.read_csv(os.path.join(OUT, "m2_ksic_profile.csv"), dtype={"ksic": str}).set_index("ksic")["name"]
KS = sorted(pa.ksic.unique())


def ols_t(y, Xm):
    Xm = sm.add_constant(Xm)
    r = sm.OLS(y, Xm, missing="drop").fit()
    return np.asarray(r.params)[-1], np.asarray(r.pvalues)[-1], int(r.nobs)


def pair_tests(lead, follow, years):
    """lead → follow. years: 종속변수 연도 범위."""
    res = {}
    for L in range(0, 4):
        a = [lead.get(y - L, np.nan) for y in years]; b = [follow.get(y, np.nan) for y in years]
        ok = ~(np.isnan(a) | np.isnan(b))
        res[f"ccf_L{L}"] = np.corrcoef(np.array(a)[ok], np.array(b)[ok])[0, 1] if ok.sum() > 4 else np.nan
    ps, cs = [], []
    for L in (1, 2, 3):
        rows = [(follow.get(y, np.nan), follow.get(y - 1, np.nan), lead.get(y - L, np.nan)) for y in years]
        arr = np.array(rows); arr = arr[~np.isnan(arr).any(1)]
        c, p, n = ols_t(arr[:, 0], arr[:, 1:])
        ps.append(p); cs.append(c); res[f"g_coef_L{L}"] = c; res[f"g_p_L{L}"] = p
    i = int(np.argmin(ps))
    res["g_best_L"] = i + 1; res["g_best_coef"] = cs[i]; res["g_p_bonf"] = min(1.0, ps[i] * 3); res["n_obs"] = n
    return res


def run_pairs(y_end, tag):
    years = list(range(2006, y_end + 1))  # ΔKR_{t-1}, ΔX_{t-3} 필요 → Δ는 2005부터
    rows = []
    for k in KS:
        kr = D.loc[("KR", k)].to_dict()
        for c in FOR:
            xx = D.loc[(c, k)].to_dict()
            f = pair_tests(xx, kr, years); f.update(ksic=k, foreign=c, direction=f"{c}→KR")
            r = pair_tests(kr, xx, years); r.update(ksic=k, foreign=c, direction=f"KR→{c}")
            rows += [f, r]
    df = pd.DataFrame(rows)
    for d in ["fwd", "rev"]:
        m = df.direction.str.endswith("→KR") if d == "fwd" else df.direction.str.startswith("KR→")
        df.loc[m, "q_bh"] = multipletests(df.loc[m, "g_p_bonf"], method="fdr_bh")[1]
    df["period"] = tag
    return df


pairs = pd.concat([run_pairs(2018, "dev_2004_2018"), run_pairs(2022, "posthoc_2004_2022")])
pairs["name"] = pairs.ksic.map(names)
pairs.to_csv(os.path.join(OUT, "m3_pairs_leadlag.csv"), index=False, encoding="utf-8-sig")
print("pairs done", round(time.time() - t0), flush=True)

# 패널 시험
def panel(lead_c, follow_c, L, y_end, nperm=1000):
    rows = []
    for k in KS:
        f = D.loc[(follow_c, k)]; l = D.loc[(lead_c, k)]
        for y in range(2006, y_end + 1):
            rows.append((k, y, f.get(y), f.get(y - 1), l.get(y - L)))
    df = pd.DataFrame(rows, columns=["k", "y", "dy", "dy1", "dx"]).dropna().reset_index(drop=True)
    kc = df.k.astype("category").cat.codes.values; yc = df.y.astype("category").cat.codes.values

    def dm(v):  # 이원 within 변환(불균형 거의 없음: 근사)
        return v - np.bincount(kc, v)[kc] / np.bincount(kc)[kc] - np.bincount(yc, v)[yc] / np.bincount(yc)[yc] + v.mean()
    Y = dm(df.dy.values); Z1 = dm(df.dy1.values)

    def fit_np(dx):
        Z = np.column_stack([Z1, dm(dx)])
        return np.linalg.lstsq(Z, Y, rcond=None)[0][1]
    z = pd.DataFrame({"dy": Y, "dy1": Z1, "dx": dm(df.dx.values)})
    r = sm.OLS(z.dy, z[["dy1", "dx"]]).fit(cov_type="cluster", cov_kwds={"groups": kc})
    b, se, p = r.params["dx"], r.bse["dx"], r.pvalues["dx"]
    # 순열: 해외 계열의 KSIC 라벨을 통째로 섞음(연도 구조 유지)
    piv = df.pivot(index="k", columns="y", values="dx")
    lab = piv.index.values; yrs = piv.columns.values
    yi = np.searchsorted(yrs, df.y.values); pv = piv.values
    kpos = pd.Index(lab).get_indexer(df.k.values)
    null = []
    for _ in range(nperm):
        perm = rng.permutation(len(lab))
        dx = pv[perm[kpos], yi]
        if np.isnan(dx).any():
            dx = np.where(np.isnan(dx), np.nanmean(dx), dx)
        null.append(fit_np(dx))
    null = np.array(null)
    p_perm = (np.sum(np.abs(null) >= abs(b)) + 1) / (len(null) + 1)
    return dict(lead=lead_c, follow=follow_c, L=L, beta=b, se=se, p_cluster=p, p_perm=p_perm, n=len(df))


prow = []
for y_end, tag in [(2018, "dev_2004_2018"), (2022, "posthoc_2004_2022")]:
    for c in FOR:
        for L in (1, 2, 3):
            r1 = panel(c, "KR", L, y_end, 300 if tag.startswith("post") else 1000); r1["period"] = tag; prow.append(r1)
            r2 = panel("KR", c, L, y_end, 300 if tag.startswith("post") else 1000); r2["period"] = tag; prow.append(r2)
    print("panel", tag, round(time.time() - t0), flush=True)
pn = pd.DataFrame(prow)
pn["q_bh_within_period"] = pn.groupby("period").p_perm.transform(lambda s: multipletests(s, method="fdr_bh")[1])
pn.to_csv(os.path.join(OUT, "m3_panel_leadlag.csv"), index=False, encoding="utf-8-sig")

# 포트폴리오 추종 (수준)
port = []
for c in FOR + ["KR"]:
    for L in range(0, 9):
        rs = []
        for t in range(2012, 2019):
            if t - L < 2004:
                continue
            a = X.loc["KR"][t]; b = X.loc[c][t - L]
            rs.append(np.corrcoef(a, b)[0, 1])
        port.append(dict(other=c, L=L, mean_corr=np.mean(rs), n_years=len(rs)))
port = pd.DataFrame(port)
port.to_csv(os.path.join(OUT, "m3_portfolio_follow.csv"), index=False, encoding="utf-8-sig")

dev = pairs[pairs.period == "dev_2004_2018"]
summ = {
    "dev_fwd_q<0.10": int(((dev.q_bh < 0.10) & dev.direction.str.endswith("→KR")).sum()),
    "dev_rev_q<0.10": int(((dev.q_bh < 0.10) & dev.direction.str.startswith("KR→")).sum()),
    "dev_fwd_rawp<0.05(bonf)": int(((dev.g_p_bonf < 0.05) & dev.direction.str.endswith("→KR")).sum()),
    "dev_rev_rawp<0.05(bonf)": int(((dev.g_p_bonf < 0.05) & dev.direction.str.startswith("KR→")).sum()),
    "expected_false_at_0.05_of_200": 10,
    "dev_fwd_sig_list": dev[(dev.q_bh < 0.10) & dev.direction.str.endswith("→KR")][["ksic", "foreign", "g_best_L", "g_best_coef", "q_bh"]].round(4).values.tolist(),
    "dev_rev_sig_list": dev[(dev.q_bh < 0.10) & dev.direction.str.startswith("KR→")][["ksic", "foreign", "g_best_L", "g_best_coef", "q_bh"]].round(4).values.tolist(),
    "panel_dev": pn[pn.period == "dev_2004_2018"][["lead", "follow", "L", "beta", "p_cluster", "p_perm", "q_bh_within_period"]].round(4).values.tolist(),
    "panel_posthoc": pn[pn.period == "posthoc_2004_2022"][["lead", "follow", "L", "beta", "p_cluster", "p_perm", "q_bh_within_period"]].round(4).values.tolist(),
    "mean_ccf_dev": dev.groupby("direction").apply(lambda g: g[["ccf_L0", "ccf_L1", "ccf_L2", "ccf_L3"]].mean().round(4).to_dict()).to_dict(),
    "portfolio_follow_best_L": port.loc[port.groupby("other").mean_corr.idxmax()][["other", "L", "mean_corr"]].round(4).values.tolist(),
    "runtime_s": round(time.time() - t0),
}
json.dump(summ, open(os.path.join(OUT, "m3_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(summ, ensure_ascii=False, indent=1, default=str))
