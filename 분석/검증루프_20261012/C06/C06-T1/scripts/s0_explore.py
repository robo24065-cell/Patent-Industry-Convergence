# -*- coding: utf-8 -*-
"""C06-T1 탐색: DART 원천 구조 확인(읽기만)."""
import sqlite3, pandas as pd, os
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\공모전데이터\외부데이터\DART"
c = sqlite3.connect("file:" + os.path.join(B, "dart_company.sqlite") + "?mode=ro", uri=True)
print(pd.read_sql("select name, sql from sqlite_master", c).to_string())
for t in pd.read_sql("select name from sqlite_master where type='table'", c).name:
    d = pd.read_sql(f"select * from [{t}] limit 3", c); print(t, d.columns.tolist()); print(d.drop(columns=[x for x in d.columns if x in ('corp_name','corp_name_eng','ceo_nm','adres','hm_url','ir_url','phn_no','fax_no','stock_name')], errors='ignore').to_string())
    print(t, pd.read_sql(f"select count(*) n from [{t}]", c).n[0])
co = pd.read_sql("select * from company", c)
print(co.dtypes)
for col in ["corp_cls", "status", "fetched_at", "acc_mt"]:
    if col in co: print(col, co[col].value_counts(dropna=False).head(15).to_dict())
print("induty len", co.induty_code.fillna("").str.len().value_counts().to_dict())
print("jurir dup", co.jurir_no.duplicated().sum(), "corp_code dup", co.corp_code.duplicated().sum() if "corp_code" in co else None)
t = pd.read_csv(os.path.join(B, "dart_targets.csv"), dtype=str, nrows=5); print("targets cols", t.columns.tolist())
t = pd.read_csv(os.path.join(B, "dart_targets.csv"), dtype=str); print("targets", len(t))
for col in t.columns:
    if t[col].nunique() < 30: print(col, t[col].value_counts(dropna=False).to_dict())
k = pd.read_csv(os.path.join(B, "dart_corpcode.csv"), dtype=str, nrows=3); print("corpcode cols", k.columns.tolist())
k = pd.read_csv(os.path.join(B, "dart_corpcode.csv"), dtype=str); print("corpcode", len(k), "stock_code 있음", k.get("stock_code", pd.Series(dtype=str)).fillna("").str.strip().ne("").sum())
p = pd.read_parquet(os.path.join(B, "corp_applicant.parquet")); print("corp_applicant", p.shape, p.columns.tolist())
