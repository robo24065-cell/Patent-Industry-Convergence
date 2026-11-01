# -*- coding: utf-8 -*-
"""제공 융합데이터(5개국 KSIC별 특허 메타)에서 기술셀 × 국가 × 출원연도 건수 집계 → R2/cache/meta_*.parquet
- IPC 메인그룹(예 'A43B 13'), IPC 서브클래스('A43B'), KSIC 세세분류 3가지 단위.
- 같은 출원이 여러 KSIC 파일에 중복 → (cc, appl_no) 고유로 센다(IPC 단위). KSIC 단위는 (cc, appl_no, ksic) 고유.
- 피처에 쓸 때는 기준일 L 시점에 공개 완결된 출원연도만 쓴다(r2ext.meta_feats).
"""
import os, duckdb, time
R2 = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(os.path.dirname(os.path.dirname(R2)), '_shared', 'meta', '*.parquet').replace('\\', '/')
OUT = os.path.join(R2, 'cache')
t = time.time()
con = duckdb.connect()
con.execute("SET threads=2; SET memory_limit='3GB'")
con.execute(f"""create temp table m as
  select distinct cc, appl_no, cast(appl_year as int) yr, ksic,
         regexp_replace(trim(split_part(main_ipc, '/', 1)), '\\s+', ' ', 'g') mg
  from read_parquet('{META}') where appl_year is not null and main_ipc is not null""")
print('rows', con.execute('select count(*) from m').fetchone(), f'{time.time() - t:.0f}s', flush=True)
con.execute(f"""copy (select cc, mg, yr, count(distinct appl_no) n from m group by 1,2,3) to '{OUT}/meta_mg.parquet' (format parquet)""")
con.execute(f"""copy (select cc, substr(mg,1,4) sub, yr, count(distinct appl_no) n from m group by 1,2,3) to '{OUT}/meta_sub.parquet' (format parquet)""")
con.execute(f"""copy (select cc, ksic, yr, count(distinct appl_no) n from m group by 1,2,3) to '{OUT}/meta_ksic.parquet' (format parquet)""")
print('done', f'{time.time() - t:.0f}s')
