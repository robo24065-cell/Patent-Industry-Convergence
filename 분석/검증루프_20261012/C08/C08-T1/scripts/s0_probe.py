# -*- coding: utf-8 -*-
# 사전 점검(PREREG 전): applicant 필드 형식만 본다(결과 지표 계산 없음). 읽기 전용.
import duckdb, os, sys, json
MC = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927"
META = os.path.join(MC, r"_shared\meta\*.parquet").replace("\\", "/")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
con = duckdb.connect(); con.sql("set memory_limit='3GB'; set threads=2; set enable_progress_bar=false")
r = con.sql(f"""select cc, count(*) n, sum(case when applicant is null or trim(applicant)='' then 1 else 0 end) n_null,
 count(distinct applicant) n_names, sum(case when applicant like '%|%' then 1 else 0 end) n_pipe,
 sum(case when applicant like '%;%' then 1 else 0 end) n_semi from read_parquet('{META}') group by 1 order by 1""").df()
print(r.to_string())
s = con.sql(f"""select cc, applicant, count(*) n from read_parquet('{META}') where appl_year between 2017 and 2019 group by 1,2
 qualify row_number() over (partition by cc order by count(*) desc) <= 25 order by cc, n desc""").df()
s.to_csv(os.path.join(OUT, "s0_top_names_probe.csv"), index=False, encoding="utf-8-sig")
print(s.to_string())
x = con.sql(f"""select cc, applicant from read_parquet('{META}') using sample 40 rows""").df()
print(x.to_string())
