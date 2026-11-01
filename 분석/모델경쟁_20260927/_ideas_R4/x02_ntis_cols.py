import duckdb
E=r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/모델경쟁_20260927/_shared/ext"
con=duckdb.connect()
P=f"read_parquet('{E}/ntis_project_raw.parquet')"
print(con.sql(f"select pjt_no,year,research_agency,org_pno,corp_rgst_no,perform_agent,perform_grp,dev_phase,period_start,period_end,total_start,total_end,gov_funds,sbiz_funds,continuous,cls_L_canon from {P} where year=2016 limit 5").df().T.to_string())
print(con.sql(f"select perform_grp, count(*) n, count(distinct corp_rgst_no) nc, avg(length(corp_rgst_no)) L, avg(case when corp_rgst_no is null or corp_rgst_no='' then 1 else 0 end) miss from {P} where year between 2010 and 2022 group by 1 order by 2 desc").df())
print(con.sql(f"select year, count(*) n, count(distinct corp_rgst_no) nc from {P} where perform_grp in ('중소기업','중견기업','대기업') or perform_agent like '%기업%' group by 1 order by 1").df().to_string())
