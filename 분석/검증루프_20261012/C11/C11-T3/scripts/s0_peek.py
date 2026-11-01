import duckdb
con=duckdb.connect()
B=r'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전'
M=B+'/분석/모델경쟁_20260927/_shared/meta/20202.parquet'
for cc in ['US','KR','EP','JP']:
    for r in con.sql(f"select appl_no, appl_date, rgst_no, pub_date, rgst_date, prio_cc, prio_date, left(main_ipc,20), left(title,60), left(family,250) from read_parquet('{M}') where cc='{cc}' and family like '%US%' order by hash(appl_no) limit 2").fetchall(): print(cc, r)
print(con.sql(f"describe select * from read_parquet('{B}/공모전데이터/작업데이터/kr_patents.parquet')").fetchall())
X=B+'/분석/모델경쟁_20260927/_shared/ext/'
print(con.sql(f"describe select * from read_parquet('{X}quality_kpss_2025.parquet')").fetchall())
print(con.sql(f"select min(patent_num),max(patent_num),min(issue_date),max(issue_date),count(*) from read_parquet('{X}quality_kpss_2025.parquet')").fetchall())
print(con.sql(f"describe select * from read_parquet('{X}r3ros_family_us_pairs.parquet')").fetchall())
print(con.sql(f"describe select * from read_parquet('{X}quality_kpss_family_link.parquet')").fetchall())
