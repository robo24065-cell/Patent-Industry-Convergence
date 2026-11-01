# KSIC x 출원연도 x 국가 특허 건수(고유 cc+appl_no) 집계 — 외부 전송 없음, 로컬 집계만
import duckdb
M=r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\meta\*.parquet'
con=duckdb.connect(); con.sql("SET memory_limit='3GB'; SET threads=2")
df=con.sql(f"""select ksic, appl_year::INT as year, cc, count(distinct appl_no) n
 from read_parquet('{M}') where appl_year between 1995 and 2022 group by 1,2,3""").df()
df.to_parquet('trade_patcnt_ksic_year_cc.parquet',index=False)
print(df.shape, df.ksic.nunique()); print(df.groupby('cc').n.sum())
