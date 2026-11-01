# -*- coding: utf-8 -*-
r"""C01-T1 s2: 커버리지 감사 (a)(b). 입력: cache\agg_ksic.parquet (s1), WIPO 공개 by technology(읽기만).
산출: coverage_cc_year.csv, doc_type_composition.csv, coverage_field_dispersion.csv, field_share_spearman_P3.csv,
      field_growth_spearman.csv, cache\cov_r.parquet(국가×공개연도×분야 r, 대체규칙 적용 전 원값)
"""
import os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
OUT = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(OUT, "cache")
MC = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927"
CCS = ["KR", "US", "JP", "CN", "EP"]
ag = pd.read_parquet(os.path.join(C, "agg_ksic.parquet"))
dist = pd.read_parquet(os.path.join(C, "agg_dist.parquet"))
wp = pd.read_csv(os.path.join(MC, r"_shared\ext\quality_raw\quality_wipo_pi\dc_indicator_patent_4_publication_by_technology.csv"))
wp = wp[wp.office.isin(CCS) & (wp.tec_id > 0)].groupby(["office", "year", "tec_id"])["count"].sum().reset_index()
wp.columns = ["cc", "year", "f", "wipo"]
wt = wp.groupby(["cc", "year"]).wipo.sum()

# ---------- (a) 국가×공개연도 총량 ----------
pub_all = ag[(ag.wipo_field > 0) & ag.pub_year.notna()].groupby(["cc", "pub_year"]).n.sum()
pub_inv = ag[(ag.wipo_field > 0) & ag.pub_year.notna() & ag.is_inv].groupby(["cc", "pub_year"]).n.sum()
appl_inv = ag[ag.is_inv].groupby(["cc", "appl_year"]).n.sum()
appl_inv_pub = ag[ag.is_inv & ag.pub_year.notna()].groupby(["cc", "appl_year"]).n.sum()
rows = []
for c in CCS:
    for y in range(2000, 2023):
        w = wt.get((c, y), np.nan)
        pa, pi = pub_all.get((c, y), 0), pub_inv.get((c, y), 0)
        ai = appl_inv.get((c, y), 0); aip = appl_inv_pub.get((c, y), 0)
        wl = np.nanmean([wt.get((c, y + 1), np.nan), wt.get((c, y + 2), np.nan)]) if y <= 2020 else np.nan
        rows.append(dict(cc=c, year=y, wipo_pub=w, prov_pub_alldocs=pa, prov_pub_inv=pi,
                         r_pub_alldocs=pa / w, r_pub_inv=pi / w, noninv_share_of_prov_pub=(pa - pi) / pa if pa else np.nan,
                         prov_appl_inv=ai, inv_appl_pubdate_missing_share=1 - aip / ai if ai else np.nan,
                         r_appl_proxy=ai / wl if wl == wl else np.nan))
cy = pd.DataFrame(rows)
# 원 파일(quality_wipo_vs_provided_pub_total.csv) 재현 대조
q0 = pd.read_csv(os.path.join(MC, r"_shared\ext\quality_wipo_vs_provided_pub_total.csv"))
cy = cy.merge(q0.rename(columns={"ratio": "r_quality_file"})[["cc", "year", "r_quality_file"]], on=["cc", "year"], how="left")
f_m2 = pd.read_csv(os.path.join(MC, r"_shared_report\fusion_advantage\map5\m2_coverage_factors_pub_total.csv"))
cy = cy.merge(f_m2[["cc", "year", "r", "f_tot"]].rename(columns={"r": "r_m2", "f_tot": "f_tot_m2"}), on=["cc", "year"], how="left")
cy.round(4).to_csv(os.path.join(OUT, "coverage_cc_year.csv"), index=False, encoding="utf-8-sig")

dtc = dist.groupby(["cc", "doctype"]).agg(n_doc=("n_doc", "sum")).reset_index()
dtc["is_inv_m1"] = dist.groupby(["cc", "doctype"]).is_inv.max().values
hp = dist[dist.pub_year.notna()].groupby(["cc", "doctype"]).n_doc.sum()
hr = dist[dist.has_rgst].groupby(["cc", "doctype"]).n_doc.sum()
idx = pd.MultiIndex.from_frame(dtc[["cc", "doctype"]])
dtc["pubdate_share"] = (hp.reindex(idx).fillna(0).values / dtc.n_doc).round(4)
dtc["rgst_share"] = (hr.reindex(idx).fillna(0).values / dtc.n_doc).round(4)
yr = dist.groupby(["cc", "doctype"]).appl_year.agg(["min", "max"]).reindex(idx)
dtc["appl_year_min"] = yr["min"].values; dtc["appl_year_max"] = yr["max"].values
pyr = dist[dist.pub_year.notna()].groupby(["cc", "doctype"]).pub_year.agg(["min", "max"]).reindex(idx)
dtc["pub_year_min"] = pyr["min"].values; dtc["pub_year_max"] = pyr["max"].values
dtc.to_csv(os.path.join(OUT, "doc_type_composition.csv"), index=False, encoding="utf-8-sig")

# ---------- (b) 분야 커버리지 ----------
pf = ag[ag.is_inv & ag.pub_year.notna() & (ag.wipo_field > 0)].groupby(["cc", "pub_year", "wipo_field"]).n.sum().reset_index()
pf.columns = ["cc", "year", "f", "prov"]
m = wp.merge(pf, on=["cc", "year", "f"], how="left").fillna({"prov": 0})
m["r"] = m.prov / m.wipo
m["small"] = (m.wipo < 300) | (m.prov < 30)
m.to_parquet(os.path.join(C, "cov_r.parquet"), index=False)
# 기하평균 g(y,f) — 5개국 모두 비소형 칸에서만
big = m[~m.small].copy()
g = big.groupby(["year", "f"]).agg(g=("r", lambda s: np.exp(np.log(s).mean())), k=("r", "size")).reset_index()
big = big.merge(g, on=["year", "f"]); big["q"] = big.r / big.g
rows = []
for (c, y), d in m.groupby(["cc", "year"]):
    if y < 2003 or (c == "CN" and y < 2010):
        continue
    b = d[~d.small]; bq = big[(big.cc == c) & (big.year == y) & (big.k == 5)]
    rows.append(dict(cc=c, year=y, n_fields_ok=len(b), r_tot=d.prov.sum() / d.wipo.sum(),
                     r_cv=b.r.std() / b.r.mean() if len(b) > 1 else np.nan, r_min=b.r.min(), r_max=b.r.max(),
                     q_n=len(bq), q_cv=bq.q.std() / bq.q.mean() if len(bq) > 1 else np.nan, q_min=bq.q.min(), q_max=bq.q.max(),
                     q_sd_log=np.log(bq.q).std() if len(bq) > 1 else np.nan))
disp = pd.DataFrame(rows)
disp.round(4).to_csv(os.path.join(OUT, "coverage_field_dispersion.csv"), index=False, encoding="utf-8-sig")

# P3 분야 점유율: 제공 vs WIPO (공개 2017–19), 국가별 Spearman + 분야 커버리지 프로파일의 국가간 상관
p3 = m[m.year.between(2017, 2019)].groupby(["cc", "f"])[["prov", "wipo"]].sum().reset_index()
p3["s_prov"] = p3.prov / p3.groupby("cc").prov.transform("sum"); p3["s_wipo"] = p3.wipo / p3.groupby("cc").wipo.transform("sum")
p3["r"] = p3.prov / p3.wipo
R = p3.pivot(index="f", columns="cc", values="r")
rows = []
for c in CCS:
    d = p3[p3.cc == c]
    others = np.exp(np.log(R[[o for o in CCS if o != c]]).mean(1))
    rows.append(dict(cc=c, spearman_share_prov_vs_wipo=spearmanr(d.s_prov, d.s_wipo).statistic,
                     spearman_rprofile_vs_others=spearmanr(R[c], others).statistic,
                     pearson_logr_vs_others=np.corrcoef(np.log(R[c]), np.log(others))[0, 1]))
fs = pd.DataFrame(rows)
fs.round(4).to_csv(os.path.join(OUT, "field_share_spearman_P3.csv"), index=False, encoding="utf-8-sig")
R.round(4).to_csv(os.path.join(OUT, "coverage_r_by_field_P3.csv"), encoding="utf-8-sig")

# 분야별 3년 성장 순위상관(제공 발명 vs WIPO): 원 문서의 2008–16 창(CN 2014–16)과 깨끗한 창 2019–22
rows = []
for c in CCS:
    x = m[m.cc == c].pivot_table(index="f", columns="year", values=["prov", "wipo"])
    for tag, ys in [("orig_window", range(2014 if c == "CN" else 2008, 2017)), ("clean_2019_22", range(2019, 2023))]:
        rs = []
        for y in ys:
            a = np.log((x["prov"][y] + 1) / (x["prov"][y - 3] + 1)); b = np.log(x["wipo"][y] / x["wipo"][y - 3])
            bg = x["prov"][y - 3] >= 200
            rs.append(spearmanr(a[bg], b[bg]).statistic)
        rows.append(dict(cc=c, window=tag, years=f"{min(ys)}-{max(ys)}", mean_rho=np.nanmean(rs), min_rho=np.nanmin(rs)))
fg = pd.DataFrame(rows)
fg.round(4).to_csv(os.path.join(OUT, "field_growth_spearman.csv"), index=False, encoding="utf-8-sig")
print(cy[["cc", "year", "r_pub_alldocs", "r_quality_file", "r_pub_inv", "r_m2", "noninv_share_of_prov_pub", "inv_appl_pubdate_missing_share", "r_appl_proxy"]].round(3).to_string())
print(disp.round(3).to_string()); print(fs.round(3).to_string()); print(fg.round(3).to_string())
