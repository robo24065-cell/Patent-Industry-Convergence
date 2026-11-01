# -*- coding: utf-8 -*-
import pandas as pd, pyarrow.parquet as pq, os, re
MC = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927"
EXT = os.path.join(MC, r"_shared\ext")
print(pq.read_schema(os.path.join(EXT, "firm_nps_wkpl_201912.parquet")))
print(pq.read_schema(os.path.join(EXT, "firm_nps_applicant_link.parquet")))
lk = pd.read_parquet(os.path.join(EXT, "firm_nps_applicant_link.parquet"))
print(len(lk), lk.nps_hit.mean() if "nps_hit" in lk else None)
for c in lk.columns:
    if lk[c].nunique() < 20: print(c, lk[c].value_counts(dropna=False).to_dict())
w = pd.read_parquet(os.path.join(EXT, "firm_nps_wkpl_201912.parquet"), columns=["indcd", "indnm"]).head(5); print(w)
L = pd.read_csv(os.path.join(EXT, "r7_data_ksic_ksco_revision_link_raw.csv"), encoding="cp949", dtype=str)
print(L.columns.tolist()); print(L.head(3).to_string())
for c in L.columns:
    if L[c].nunique() < 30: print(c, L[c].value_counts(dropna=False).to_dict())
LO = os.path.join(MC, r"_shared_report\label_origin_R10\cache")
print(os.listdir(LO))
print(pq.read_schema(os.path.join(LO, "fam_reg.parquet")))
K = pd.read_parquet(os.path.join(MC, r"_ideas_R7\cache\kind_linked.parquet")); print(K.columns.tolist(), len(K))
print(K["업종"].head(10).tolist())
