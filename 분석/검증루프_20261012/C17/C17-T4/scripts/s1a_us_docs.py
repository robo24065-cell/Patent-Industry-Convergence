# C17-T4 s1a: 제공 US 문서 표(읽기 전용) → cache/us_docs.parquet
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib17 import *
import duckdb
con = duckdb.connect(); con.execute("set threads=2; set memory_limit='3GB'")
df = con.execute(f"""
select lpad(appl_no, 8, '0') appl8, appl_no, appl_date, appl_year, rgst_no, rgst_date, pub_date,
       applicant, main_ipc, ipc_all, main_cpc, family, prio_cc, prio_date, ksic
from read_parquet('{META}') where cc='US'""").fetchdf()
# 7자리 appl_no(앞 0 누락)와 8자리 판이 같은 문서로 중복 수록된 경우가 있다 → 8자리 원판 우선 1건
n_raw = len(df)
dup = df[df.appl8.duplicated(keep=False)]
ksic_conf = int((dup.groupby('appl8').ksic.nunique() > 1).sum())
df['_p'] = (df.appl_no.str.len() == 8).astype(int)
df = df.sort_values(['appl8', '_p'], ascending=[True, False]).drop_duplicates('appl8').drop(columns='_p')
assert df.appl8.is_unique
df.to_parquet(C + r'\us_docs.parquet', index=False)
res = {'n_raw': n_raw, 'n': len(df), 'dup_groups': int(dup.appl8.nunique()), 'dup_ksic_conflict': ksic_conf,
       'series': df.appl8.str[:2].value_counts().to_dict(), 'n_rgst': int(df.rgst_no.notna().sum())}
json.dump(res, open(C + r'\s1a_log.json', 'w'), indent=1)
out(res)
