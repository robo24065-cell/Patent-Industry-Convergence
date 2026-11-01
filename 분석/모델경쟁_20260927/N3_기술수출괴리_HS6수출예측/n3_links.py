# -*- coding: utf-8 -*-
"""데이터 연계 프레임워크 산출물 — 특허(KSIC) ↔ HS6 무역 ↔ 5개국 수출 ↔ KOSIS ↔ 산업연관표 연계표·연계율.
출력: ext_link_ksic_hs6.csv, ext_link_ksic_kosis.csv, ext_link_ksic_io.csv, ext_link_5c_coverage.csv, ext_link_summary.json
(개발 구간 정보만: 무역 ≤2021, 특허 최초공표 ≤2018 집계)"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_n3 import *
m = load_map(); pan = load_pat_panel(); ex = load_exports(); imp = load_imports()
x5 = load_5c_exports(2021); K = load_kosis(); Z, io_exp, ksub = load_io()
D = pd.read_parquet(os.path.join(CACHE, "panel_X.parquet"))
ksic50 = sorted(pan.ksic.unique())
# 1) KSIC ↔ HS6
nk = m.groupby("hs6").k5.nunique()
pat = pan[pan.dy <= 2018].groupby(["ksic", "cc"]).n.sum().unstack(fill_value=0)
rows = []
for k in ksic50:
    hs = m[m.k5 == k].hs6
    rows.append({"k5": k, "n_hs6_linked": len(hs), "n_hs6_single_ksic": int((nk.reindex(hs) == 1).sum()),
                 "n_hs6_with_kr_export": int(hs.isin(ex.hs6).sum()),
                 "n_hs6_in_dev_sample_any": int(hs.isin(D.hs6).sum()),
                 "kr_export_2021_usd_mn": round(ex[(ex.year == 2021) & ex.hs6.isin(hs)].value.sum() / 1e6, 1),
                 **{f"patents_{c}_dy2018": int(pat.loc[k, c]) if k in pat.index and c in pat.columns else 0 for c in ["KR", "US", "JP", "EP", "CN"]}})
L1 = pd.DataFrame(rows); L1.to_csv(os.path.join(HERE, "ext_link_ksic_hs6.csv"), index=False, encoding="utf-8-sig")
# 2) KSIC ↔ KOSIS
rows = []
for k in ksic50:
    r = {"k5": k}
    if k in K.index.get_level_values(0):
        s = K.xs(k, level=0).loc["ship"] if "ship" in K.xs(k, level=0).index else None
        r["kosis_years_ship"] = int(s.notna().sum()) if s is not None else 0
        r["ship_2018_bn_krw"] = round(float(s.get(2018, np.nan)) / 1e3, 1) if s is not None else np.nan
    else:
        r["kosis_years_ship"] = 0
    rows.append(r)
L2 = pd.DataFrame(rows); L2.to_csv(os.path.join(HERE, "ext_link_ksic_kosis.csv"), index=False, encoding="utf-8-sig")
# 3) KSIC ↔ IO 소분류, 하류 연결
rows = []
Zm = Z.copy(); np.fill_diagonal(Zm.values, 0)
for k in ksic50:
    for _, r in ksub[ksub.k5 == k].iterrows():
        wd = Zm.loc[r["sub"]]
        top = wd.sort_values(ascending=False).head(3)
        rows.append({"k5": k, "io_sub": r["sub"], "w": r.w, "intermediate_sales_mn_krw_2010": round(float(wd.sum()), 0),
                     "top3_downstream_subs": "|".join(top.index), "top3_share": round(float(top.sum() / max(wd.sum(), 1)), 3),
                     "export_share_of_output": round(float(io_exp.get(r["sub"], np.nan) / (Z.loc[r["sub"]].sum() + io_exp.get(r["sub"], 0) + 1e-9)), 3)})
L3 = pd.DataFrame(rows); L3.to_csv(os.path.join(HERE, "ext_link_ksic_io.csv"), index=False, encoding="utf-8-sig")
# 4) 5개국 수출 커버리지
rows = []
for y in sorted(x5.year.unique()):
    s = x5[x5.year == y]
    rows.append({"year": y, "n_hs6_any": s.hs6.nunique(), **{f"n_hs6_{c}": s[s.cc == c].hs6.nunique() for c in COUNTRIES.values()},
                 "share_of_1019": round(s.hs6.nunique() / m.hs6.nunique(), 3), "sparse_used": y in [2004, 2006, 2009, 2012, 2015, 2018, 2021]})
L4 = pd.DataFrame(rows); L4.to_csv(os.path.join(HERE, "ext_link_5c_coverage.csv"), index=False, encoding="utf-8-sig")
S = {"ksic_provided": len(ksic50), "ksic_linked_hs6": int((L1.n_hs6_linked > 0).sum()),
     "hs6_linked": int(m.hs6.nunique()), "hs6_single_ksic_share": round(float((nk == 1).mean()), 3),
     "hs6_with_kr_export_2000_2021": int(m.hs6.drop_duplicates().isin(ex.hs6).sum()),
     "hs6_in_dev_panel_any_origin": int(D.hs6.nunique()), "dev_panel_rows": len(D),
     "dev_eval_rows_2010_2018": int(D.t0.between(2010, 2018).sum()),
     "ksic_linked_kosis": int((L2.kosis_years_ship > 0).sum()), "ksic_linked_io": int(L3.k5.nunique()), "io_subs_used": int(L3.io_sub.nunique()),
     "hs6_5c_coverage_2012": float(L4.set_index("year").share_of_1019.get(2012, np.nan)),
     "dev_rows_with_gap_feature": round(float(D[D.t0.between(2010, 2018)].gap_kr_h.notna().mean()), 3),
     "kr_export_2021_linked_usd_bn": round(ex[(ex.year == 2021) & ex.hs6.isin(m.hs6)].value.sum() / 1e9, 1)}
json.dump(S, open(os.path.join(HERE, "ext_link_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(S, ensure_ascii=False, indent=1)); print(L4.to_string())
