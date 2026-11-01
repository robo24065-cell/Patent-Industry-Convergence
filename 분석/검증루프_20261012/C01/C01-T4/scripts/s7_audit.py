# s7: (a) 총량 대조 — 등록원부 vs 제공 메타 KR 등록, 연도별 등록 수 / 만료일 결측
import os, json, duckdb, pandas as pd
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
OUTD = os.path.join(ROOT, r"분석\검증루프_20261012\C01\C01-T4\cache")
META = os.path.join(ROOT, r"분석\모델경쟁_20260927\_shared\meta\*.parquet").replace("\\", "/")
df = pd.read_parquet(os.path.join(OUTD, "pat_base.parquet"), columns=["rgst_no", "ksic", "rg", "cndrtExptnDate_d", "applDate_d", "lapsed", "paid_thru"])
con = duckdb.connect(); con.sql("set threads=2"); con.sql("set memory_limit='3GB'")
m = con.sql(f"select count(distinct rgst_no) n, count(distinct appl_no) na from read_parquet('{META}') where cc='KR' and rgst_no is not null and rgst_no<>''").fetchall()
my = con.sql(f"select substr(rgst_date,1,4) y, count(distinct appl_no) from read_parquet('{META}') where cc='KR' and rgst_date is not null and rgst_date<>'' and rgst_date>='2012' group by 1 order by 1").fetchall()
mk = con.sql(f"select count(*) from (select appl_no from read_parquet('{META}') where cc='KR' and rgst_no is not null and rgst_no<>'' group by appl_no having count(distinct ksic)>1)").fetchall()
con.close()
res = {"meta_kr_registered_distinct_rgst_no": m[0][0], "meta_kr_registered_distinct_appl_no": m[0][1], "registry_patents": len(df),
       "meta_kr_registered_multi_ksic_appl": mk[0][0],
       "meta_rgst_year": my, "registry_rgst_year": df.rg.dt.year.value_counts().sort_index().loc[2012:].to_dict(),
       "expiry_missing": int(df.cndrtExptnDate_d.isna().sum()),
       "lapsed_by_paidthru_year_dev": df[df.lapsed].paid_thru.dt.year.value_counts().sort_index().loc[2016:2024].to_dict()}
json.dump(res, open(os.path.join(OUTD, "s7_audit.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
