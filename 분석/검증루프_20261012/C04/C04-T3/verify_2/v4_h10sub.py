import sys, duckdb
sys.stdout.reconfigure(encoding='utf-8')
BR = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/모델경쟁_20260927/T3_기술셀부상예측/R3/cache_pub_final/base_rows.parquet"
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2")
print(con.sql(f"""select cc, substr(replace(ipc_raw,' ',''),1,4) sc, year, count(distinct appl_no) n from read_parquet('{BR}')
 where substr(replace(ipc_raw,' ',''),1,4) in ('H10D','H10F','H10H','H10B','H10K') and year>=2017 group by all order by cc, sc, year""").df().pivot_table(index=['cc','sc'], columns='year', values='n', fill_value=0).to_string())
# H01L 전체(대표 IPC) KR 연도별: H10* 신설로 빠져나간 몫
print(con.sql(f"""select year, count(distinct case when replace(ipc_raw,' ','') like 'H01L%' then appl_no end) h01l,
 count(distinct case when replace(ipc_raw,' ','') like 'H10%' then appl_no end) h10 from read_parquet('{BR}') where cc='KR' and year>=2015 group by 1 order by 1""").df().to_string())
