# -*- coding: utf-8 -*-
# 반박 검증자(2차) 스키마 점검. 읽기만.
import duckdb
B = 'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전'
T3 = B + '/분석/모델경쟁_20260927/T3_기술셀부상예측'
W = B + '/분석/검증루프_20261012/C11/C11-T2'
C10 = B + '/분석/검증루프_20261012/C10/C10-T3'
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2")
for f in [T3 + '/R3/cache_pub/base_rows.parquet', C10 + '/cache/links.parquet', W + '/cache/txt_assign_reg.parquet',
          W + '/cache/txt_reg/panel_ccy.parquet', W + '/cache/ctx_txt_reg.parquet', W + '/res/runs/mg_ctx-txt__reg_preds.parquet']:
    print(f.split('/')[-2:], [(r[0], r[1]) for r in con.sql(f"describe select * from read_parquet('{f}')").fetchall()])
print(con.sql(f"select link, cc, count(*), count(distinct cc||appl_no||kr_appl_no) from read_parquet('{C10}/cache/links.parquet') group by all order by all").fetchall())
print(con.sql(f"select count(*), count(distinct cc||'|'||appl_no) from read_parquet('{T3}/R3/cache_pub/base_rows.parquet')").fetchall())
