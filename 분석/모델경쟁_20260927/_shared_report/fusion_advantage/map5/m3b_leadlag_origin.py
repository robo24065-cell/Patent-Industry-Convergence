# -*- coding: utf-8 -*-
r"""map5 3b: 우선권(패밀리) 경로를 뺀 선행-후행 — m3 패널의 1년 선행이 '같은 발명의 해외 재출원(파리조약 12개월)'인지 확인.
- KR 계열: KR 관청 출원 중 우선권 첫 국가가 KR 또는 없음(국내 기원)만.
- 해외 X 계열: X 관청 출원 중 우선권 첫 국가가 KR인 것을 뺌(한국 기원 재출원 제거). X→KR 시험에서는 KR 계열에서 X 기원도 이미 빠져 있다.
- 나머지 설계는 m3 패널과 같다(Δ log 국가 내 비중, 이원 고정효과, KSIC 군집 SE, 순열 1000회). 개발 2004–2018 / 사후 2004–2022.
- 포트폴리오 추종 역방향: corr_k( x_X,t , x_KR,t−L ) (X가 한국의 과거 구성을 닮아 가는가).
"""
import os, json, time
import numpy as np, pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
OUT = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()
rng = np.random.default_rng(20261006)
FOR = ["US", "JP", "CN", "EP"]
o = pd.read_parquet(os.path.join(OUT, "m1_origin_appl.parquet"))
o["ksic"] = o.ksic.astype(str)
o = o[o.appl_year.between(2004, 2022)]
kr = o[(o.office == "KR") & o.origin.isin(["KR", "NONE"])]
fx = o[(o.office != "KR") & (o.origin != "KR")]
ser = pd.concat([kr.assign(cc="KR"), fx.assign(cc=fx.office)]).groupby(["cc", "ksic", "appl_year"]).n_inv.sum().reset_index()
idx = pd.MultiIndex.from_product([["KR"] + FOR, sorted(o.ksic.unique()), range(2004, 2023)], names=["cc", "ksic", "appl_year"])
ser = ser.set_index(["cc", "ksic", "appl_year"]).reindex(idx, fill_value=0).reset_index()
ser["x"] = np.log((ser.n_inv + 0.5) / ser.groupby(["cc", "appl_year"]).n_inv.transform("sum"))
X = ser.pivot_table(index=["cc", "ksic"], columns="appl_year", values="x")
D = X.diff(axis=1)
KS = sorted(o.ksic.unique())
share_dom = {"KR_domestic_origin_share_2004_2018": round(kr[kr.appl_year <= 2018].n_inv.sum() / o[(o.office == "KR") & (o.appl_year <= 2018)].n_inv.sum(), 4)}
for c in FOR:
    a = o[(o.office == c) & (o.appl_year <= 2018)]
    share_dom[f"{c}_kr_origin_share_2004_2018"] = round(a[a.origin == "KR"].n_inv.sum() / a.n_inv.sum(), 4)


def panel(lead_c, follow_c, L, y_end, nperm=1000):
    rows = []
    for k in KS:
        f = D.loc[(follow_c, k)]; l = D.loc[(lead_c, k)]
        for y in range(2006, y_end + 1):
            rows.append((k, y, f.get(y), f.get(y - 1), l.get(y - L)))
    df = pd.DataFrame(rows, columns=["k", "y", "dy", "dy1", "dx"]).dropna().reset_index(drop=True)
    kc = df.k.astype("category").cat.codes.values; yc = df.y.astype("category").cat.codes.values

    def dm(v):
        return v - np.bincount(kc, v)[kc] / np.bincount(kc)[kc] - np.bincount(yc, v)[yc] / np.bincount(yc)[yc] + v.mean()
    Y = dm(df.dy.values); Z1 = dm(df.dy1.values)
    z = pd.DataFrame({"dy": Y, "dy1": Z1, "dx": dm(df.dx.values)})
    r = sm.OLS(z.dy, z[["dy1", "dx"]]).fit(cov_type="cluster", cov_kwds={"groups": kc})
    b = r.params["dx"]
    piv = df.pivot(index="k", columns="y", values="dx"); yrs = piv.columns.values; pv = piv.values
    yi = np.searchsorted(yrs, df.y.values); kpos = pd.Index(piv.index.values).get_indexer(df.k.values)
    null = []
    for _ in range(nperm):
        dx = pv[rng.permutation(len(piv))[kpos], yi]
        dx = np.where(np.isnan(dx), np.nanmean(dx), dx)
        null.append(np.linalg.lstsq(np.column_stack([Z1, dm(dx)]), Y, rcond=None)[0][1])
    p_perm = (np.sum(np.abs(np.array(null)) >= abs(b)) + 1) / (nperm + 1)
    return dict(lead=lead_c, follow=follow_c, L=L, beta=b, se=r.bse["dx"], p_cluster=r.pvalues["dx"], p_perm=p_perm, n=len(df))


rows = []
for y_end, tag in [(2018, "dev_2004_2018"), (2022, "posthoc_2004_2022")]:
    for c in FOR:
        for L in (1, 2, 3):
            for a_, b_ in [(c, "KR"), ("KR", c)]:
                r = panel(a_, b_, L, y_end); r["period"] = tag; rows.append(r)
pn = pd.DataFrame(rows)
pn["q_bh_within_period"] = pn.groupby("period").p_perm.transform(lambda s: multipletests(s, method="fdr_bh")[1])
pn["series"] = "origin_purged"
pn.to_csv(os.path.join(OUT, "m3b_panel_leadlag_origin_purged.csv"), index=False, encoding="utf-8-sig")

# 포트폴리오 역방향(원 계열 m2 패널 사용)
pa = pd.read_csv(os.path.join(OUT, "m2_panel_appl.csv"), dtype={"ksic": str})
pa = pa[pa.year.between(2004, 2022)]
pa["x"] = np.log((pa.n_inv + 0.5) / pa.groupby(["cc", "year"]).n_inv.transform("sum"))
XX = pa.pivot_table(index=["cc", "ksic"], columns="year", values="x")
port = []
for c in FOR:
    for L in range(0, 9):
        rs = [np.corrcoef(XX.loc[c][t], XX.loc["KR"][t - L])[0, 1] for t in range(2012, 2019) if t - L >= 2004]
        port.append(dict(follower=c, leader="KR", L=L, mean_corr=np.mean(rs)))
port = pd.DataFrame(port)
port.to_csv(os.path.join(OUT, "m3b_portfolio_follow_reverse.csv"), index=False, encoding="utf-8-sig")
summ = dict(share_dom, panel_dev=pn[pn.period == "dev_2004_2018"][["lead", "follow", "L", "beta", "p_cluster", "p_perm", "q_bh_within_period"]].round(4).values.tolist(),
            panel_posthoc=pn[pn.period != "dev_2004_2018"][["lead", "follow", "L", "beta", "p_cluster", "p_perm", "q_bh_within_period"]].round(4).values.tolist(),
            portfolio_reverse=port.pivot(index="L", columns="follower", values="mean_corr").round(3).to_dict(), runtime_s=round(time.time() - t0))
json.dump(summ, open(os.path.join(OUT, "m3b_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(pd.Series(share_dom)); print(pn.round(4).to_string()); print(port.pivot(index="L", columns="follower", values="mean_corr").round(3))
