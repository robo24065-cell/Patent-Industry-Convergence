# -*- coding: utf-8 -*-
"""N2 연계 산출물: ext_link_ksic_coverage.csv(산업별 특허배분 vs 업종코드 연결), ext_link_trade.csv(KSIC-HS6 수출 연계),
ext_link_pit.csv(시점판 보유 재구성 통계). 개발 구간(≤2023)만."""
import os, json
import numpy as np, pandas as pd
import n_lib as L
st = L.load_state(); spec = st["spec"]
Y = L.build({**spec, "induty": True}); d = Y[Y.t.isin(L.DEV)]
cov = d.groupby("ksic").agg(nf_patent_median=("nf_gd", "median"), patent_cov=("gd", lambda s: s.notna().mean()),
                            nf_induty_median=("nf_induty", lambda s: s.fillna(0).median()), induty_cov=("gd_induty", lambda s: s.notna().mean()),
                            corr_g_gd=("gd", lambda s: np.nan))
for k, g in d.groupby("ksic"):
    cov.loc[k, "corr_g_gd"] = g[["g", "gd"]].corr().iloc[0, 1]
cov.round(3).to_csv(os.path.join(L.HERE, "ext_link_ksic_coverage.csv"), encoding="utf-8-sig")
print("patent: coverage", round(d.gd.notna().mean(), 3), "median firms", d.nf_gd.median(),
      "| induty: coverage", round(d.gd_induty.notna().mean(), 3), "median firms (present)", d.nf_induty.dropna().median(), "median incl 0", d.nf_induty.fillna(0).median())
# 기업 수(배분 기여, 연도별)
fr = L.firm_growth(spec["fin"]); W = L.weights(spec["w"])
M = fr.merge(W, left_on=["corp_code", fr.t - 1], right_on=["corp_code", "t0"])
print("firms contributing by t", M.groupby("t").corp_code.nunique().loc[2016:2023].to_dict(), "DART firms with growth", fr.groupby("t").size().loc[2016:2023].to_dict())
# 무역
m = pd.read_csv(os.path.join(L.HERE, r"..\..\fusion\ksic_hs6_map.csv"), dtype=str)
tr = m.groupby("k5").hs6.nunique().rename("n_hs6").reset_index().rename(columns={"k5": "ksic"})
tr = tr.merge(d.groupby("ksic").g_x_hs.apply(lambda s: s.notna().mean()).rename("export_growth_avail").reset_index(), on="ksic", how="right")
tr.to_csv(os.path.join(L.HERE, "ext_link_trade.csv"), index=False, encoding="utf-8-sig")
print("trade: KSIC with HS6", int(tr.n_hs6.notna().sum()), "/50, avail", round(d.g_x_hs.notna().mean(), 3))
# 시점판
A = L.load("alloc_pit.parquet"); S = L.load("alloc_snap.parquet")
rows = []
for T in range(2015, 2023):
    a = A[A.t0 == T]; s = S[S.t0 == T]
    rows.append({"T": T, "pit_firms": a.corp_code.nunique(), "snap_firms": s.corp_code.nunique(),
                 "pit_sum_w_median": round(a.groupby("corp_code").w_pit.sum().median(), 3),
                 "snap_sum_w_median": round(s.groupby("corp_code").w_snap.sum().median(), 3),
                 "rec8_firms_with_recent": int((a.groupby("corp_code").n_k_rec8.sum() > 0).sum())})
pd.DataFrame(rows).to_csv(os.path.join(L.HERE, "ext_link_pit.csv"), index=False, encoding="utf-8-sig")
print(pd.DataFrame(rows).to_string())
