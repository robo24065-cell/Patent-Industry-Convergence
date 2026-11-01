# 제공 융합데이터(5개국 1,514만행) KSIC×국가×출원연도 고유 출원 수 → cache/ksic_cc_year.parquet
import duckdb, sys
sys.stdout.reconfigure(encoding='utf-8')
M = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/모델경쟁_20260927/_shared/meta/*.parquet"
con = duckdb.connect(); con.execute("SET threads=2; SET memory_limit='3GB'")
df = con.execute(f"select ksic, cc, appl_year y, count(distinct appl_no) n from read_parquet('{M}') where appl_year between 1995 and 2016 group by 1,2,3").df()
df.to_parquet('cache/ksic_cc_year.parquet', index=False)
print(df.pivot_table(index='y', columns='cc', values='n', aggfunc='sum').to_string())
