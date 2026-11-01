# -*- coding: utf-8 -*-
"""s1: B0–B4 재현 + 수집본 완결성·중복·이상값(D2, D3)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *

mp = load_map()
ex = load_ex()
res = {}
# ---------- 재현 ----------
w = mp.copy(); w["w"] = 1 / w.groupby("hs6").k5.transform("nunique"); w = w.drop_duplicates(["k5", "hs6"])
e0 = ex.drop_duplicates(["year", "reporter", "hs6"]).rename(columns={"export_usd": "v"})[["year", "rep", "hs6", "v"]]
ek = ksic_value(e0, w)
ek.to_csv(os.path.join(CACHE, "ek_base.csv"), index=False)
rta = patent_rta()
p2, v2 = rca_from(ek, [2015, 2016, 2017]); p1, _ = rca_from(ek, [2008, 2009, 2010]); p4, _ = rca_from(ek, [2021])
out, q = evaluate(rta, p2, p1, p4, "B_reproduce")
orig = pd.read_csv(os.path.join(MAP5, "m4_quadrant_table.csv"), dtype={"ksic": str}, encoding="utf-8-sig").set_index("ksic")
res["reproduce"] = out
res["max_abs_diff_rca_P2_vs_m4"] = float((p2.reindex(orig.index) - orig.exp_rca_P2).abs().max())
res["max_abs_diff_rca_P1_vs_m4"] = float((p1.reindex(orig.index) - orig.exp_rca_P1).abs().max())
res["max_abs_diff_rta_P2_vs_m4"] = float((rta["P2"].reindex(orig.index) - orig.pat_rta_P2).abs().max())
# m4_exports_ksic_5rep.csv 대조
m4e = pd.read_csv(os.path.join(MAP5, "m4_exports_ksic_5rep.csv"), dtype={"k5": str})
cmp = m4e.merge(ek, on=["k5", "rep", "year"], how="outer", suffixes=("_m4", "_re"))
res["m4_exports_rows"] = [int(len(m4e)), int(len(ek))]
res["m4_exports_max_reldiff"] = float(((cmp.v_m4 - cmp.v_re).abs() / cmp.v_m4.abs()).max())

# ---------- D2 완결성 ----------
codes = sorted(mp.hs6.unique()); chunks = [codes[i:i + 100] for i in range(0, len(codes), 100)]
res["n_map_codes"] = len(codes); res["n_chunks"] = len(chunks)
d_ind = set(open(os.path.join(EXT, "R2ind_comtrade_exports_5c_hs6_2004_2021_done_chunks.txt")).read().split())
d_n3 = set(open(os.path.join(EXT, "R2N3_comtrade_exports_5c_hs6_done.txt")).read().split())
done = d_ind | d_n3
yrs = list(range(2004, 2022))
missing = {y: [c for c in range(len(chunks)) if f"{y}_{c}" not in done] for y in yrs}
res["done_missing_chunks"] = {y: m for y, m in missing.items() if m}
res["done_overlap_ind_n3"] = sorted(d_ind & d_n3)
# 청크별 행 수(500 상한)
c_of = {h: i // 100 for i, h in enumerate(codes)}
ex["chunk"] = ex.hs6.map(c_of)
rows_chunk = ex.groupby(["year", "chunk"]).size()
res["chunk_rows_max"] = int(rows_chunk.max()); res["chunk_rows_ge_490"] = rows_chunk[rows_chunk >= 490].reset_index().values.tolist()
res["codes_not_in_map"] = int((~ex.hs6.isin(codes)).sum())
# 보고국×연도 코드 수·합계
ry = ex.drop_duplicates(["year", "reporter", "hs6"]).groupby(["rep", "year"]).agg(n_codes=("hs6", "nunique"), sum_usd=("export_usd", "sum"),
                                                                                    hs_rev=("hs_rev", lambda s: ",".join(sorted(s.dropna().unique()))))
ry.to_csv(os.path.join(CACHE, "s1_rep_year.csv"))
# ---------- D3 중복·이상 ----------
dup = ex.groupby(["year", "reporter", "hs6"]).agg(n=("export_usd", "size"), nuniq=("export_usd", "nunique"), srcs=("src", lambda s: ",".join(sorted(set(s)))))
res["dup_keys"] = int((dup.n > 1).sum()); res["dup_keys_value_conflict"] = int((dup.nuniq > 1).sum())
res["dup_by_src"] = dup[dup.n > 1].srcs.value_counts().to_dict()
res["neg_or_zero"] = int((ex.export_usd <= 0).sum()); res["na_value"] = int(ex.export_usd.isna().sum())
res["special_codes"] = int(ex.hs6.str.match(r"^(9999|TOTAL|99)").sum())
res["hs_rev_by_year"] = ex.groupby("year").hs_rev.apply(lambda s: s.value_counts().to_dict()).dropna().to_dict()
res["hs_rev_by_year"] = {str(k): v for k, v in res["hs_rev_by_year"].items()}
# 보고국별 연도별 hs_rev
hr = ex.groupby(["rep", "year"]).hs_rev.agg(lambda s: ",".join(sorted(s.dropna().unique()))).unstack()
hr.to_csv(os.path.join(CACHE, "s1_hsrev_rep_year.csv"))
json.dump(res, open(os.path.join(CACHE, "s1_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(res, ensure_ascii=False, indent=1, default=str)[:6000])
print(hr.to_string())
