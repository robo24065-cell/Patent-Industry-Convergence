# C18-T2 s3t_base: T3 R3/build_base_pub.py 와 같은 SQL로 변형 base_rows 생성 (쓰기: cache/t3_<V>_<dev|fin>/)
# 사용: python s3t_base.py V1|V1E|V2|V12 dev|fin
#  V1 = meta_V1(US 7자리 쌍 행 제거), V1E = meta_V1E(+EP 9자 쌍 행 제거), V2 = 원본 메타 + US 공개연도 = pub_date만, V12 = meta_V1 + V2 규칙
import os, sys, time, duckdb
V, mode = sys.argv[1], sys.argv[2]
W = os.environ['C18_W']; C = os.path.join(W, 'cache')
ORIG = r'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/모델경쟁_20260927/_shared/meta/*.parquet'
M = {'V1': f'{C}/meta_V1.parquet', 'V1E': f'{C}/meta_V1E.parquet', 'V2': ORIG, 'V12': f'{C}/meta_V1.parquet'}[V].replace('\\', '/')
us_pub_only = V in ('V2', 'V12')
MAXYEAR = 2021 if mode == 'dev' else 2024
OUT = os.path.join(C, f't3_{V}_{mode}'); os.makedirs(OUT, exist_ok=True)
t = time.time()
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false")
con.execute(f"set temp_directory='{C.replace(chr(92), '/')}/duck_tmp'")
# 원본: py = left(coalesce(least(nullif(pub_date,''), nullif(rgst_date,'')), nullif(pub_date,''), nullif(rgst_date,'')),4)
py_orig = "left(coalesce(least(nullif(pub_date,''), nullif(rgst_date,'')), nullif(pub_date,''), nullif(rgst_date,'')),4)"
py_expr = f"case when cc='US' then left(nullif(pub_date,''),4) else {py_orig} end" if us_pub_only else py_orig
con.execute(rf"""
copy (
 select ksic, cc, py::int as year, appl_year::int as appl_year, appl_no,
   regexp_replace(trim(coalesce(nullif(main_ipc,''), split_part(ipc_all,'|',1))), '\s+', ' ', 'g') as ipc_raw,
   trim(split_part(applicant,'|',1)) as app1,
   case when prio_cc is null then 0
        when regexp_replace(replace(prio_cc,cc,''),'\|','','g')<>'' then 1 else 0 end as prio_for,
   ipc_all, title
 from (select *, {py_expr} as py
       from read_parquet('{M}') where appl_year >= 2000)
 where py is not null and py::int between 2000 and {MAXYEAR}
) to '{OUT.replace(chr(92), '/')}/base_rows.parquet' (format parquet, compression zstd)""")
print(V, mode, 'base done', round(time.time() - t, 1))
print(con.sql(f"select cc, count(*) n, min(year), max(year) from read_parquet('{OUT.replace(chr(92), '/')}/base_rows.parquet') group by cc order by cc").df().to_string())
