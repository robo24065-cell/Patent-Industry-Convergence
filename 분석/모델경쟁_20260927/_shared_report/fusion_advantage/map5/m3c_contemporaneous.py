# -*- coding: utf-8 -*-
r"""map5 3c: 같은 해 동조(L=0) — 산업별 비중 변화가 국가 간에 같은 해 함께 움직이는가 (원 계열 / 우선권 경로 제거 계열).
Δlog 국가 내 비중, 연도 고정효과 제거 후 KSIC 50개의 국가쌍 상관을 KSIC 단위로 평균, 귀무분포는 해외 계열 KSIC 라벨 순열 1000회. 개발 2005–2018.
"""
import os, json
import numpy as np, pandas as pd
OUT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(7)
FOR = ["US", "JP", "CN", "EP"]
pa = pd.read_csv(os.path.join(OUT, "m2_panel_appl.csv"), dtype={"ksic": str}); pa = pa[pa.year.between(2004, 2018)]
pa["x"] = np.log((pa.n_inv + 0.5) / pa.groupby(["cc", "year"]).n_inv.transform("sum"))
o = pd.read_parquet(os.path.join(OUT, "m1_origin_appl.parquet")); o["ksic"] = o.ksic.astype(str); o = o[o.appl_year.between(2004, 2018)]
kr = o[(o.office == "KR") & o.origin.isin(["KR", "NONE"])].assign(cc="KR"); fx = o[(o.office != "KR") & (o.origin != "KR")]; fx = fx.assign(cc=fx.office)
s2 = pd.concat([kr, fx]).groupby(["cc", "ksic", "appl_year"]).n_inv.sum().unstack(fill_value=0).stack().rename("n").reset_index()
s2["x"] = np.log((s2.n + 0.5) / s2.groupby(["cc", "appl_year"]).n.transform("sum"))
res = {}
for tag, df, yc in [("raw", pa, "year"), ("origin_purged", s2, "appl_year")]:
    X = df.pivot_table(index=["cc", "ksic"], columns=yc, values="x"); D = X.diff(axis=1).iloc[:, 1:]
    D = D - D.groupby(level=0).transform("mean")  # 연도 고정효과(국가별 KSIC 평균 변화 제거)
    for c in FOR:
        a = D.loc["KR"].values; b = D.loc[c].values
        def mc(bb):
            return np.nanmean([np.corrcoef(a[i], bb[i])[0, 1] for i in range(len(a))])
        obs = mc(b); null = np.array([mc(b[rng.permutation(len(b))]) for _ in range(1000)])
        res[f"{tag}_{c}"] = dict(mean_corr=round(obs, 4), null_mean=round(null.mean(), 4), p_perm=round((np.sum(null >= obs) + 1) / 1001, 4))
json.dump(res, open(os.path.join(OUT, "m3c_contemporaneous.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps(res, indent=1))
