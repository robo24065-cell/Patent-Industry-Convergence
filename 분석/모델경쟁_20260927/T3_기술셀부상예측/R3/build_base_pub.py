"""T3 3라운드 공개연도 변형 — 기본 행 캐시.
공개일(public date) = least(pub_date, rgst_date) (둘 중 먼저 공개된 날짜; 등록공고·CN 실용신안·US 2001 이전 출원은 등록일이 첫 공개).
year 컬럼 = 공개연도. 1라운드 build_panel.py / build_feat2.py(수정 없음)가 이 year 로 패널을 만든다 → 모든 셀 집계가 '그 해 말까지 공개된 문서'.
🔒 3라운드 최종시험 F'=2022 (KR 공개연도 2022–24). 개발 캐시는 공개연도 ≤2021 만 읽는다.
사용: T3_CACHE=R3/cache_pub T3_MAXYEAR=2021 python R3/build_base_pub.py   (cwd = T3 폴더)
"""
import os, duckdb, time
CACHE = os.environ.get('T3_CACHE', 'R3/cache_pub'); MAXYEAR = int(os.environ.get('T3_MAXYEAR', '2021'))
if MAXYEAR >= 2022 and os.environ.get('T3R3_FINAL_UNLOCK') != '1':
    raise RuntimeError('🔒 공개연도 2022 이후는 최종시험 전용(T3R3_FINAL_UNLOCK=1)')
os.makedirs(CACHE, exist_ok=True)
t = time.time()
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false")
M = "../_shared/meta/*.parquet"
con.execute(rf"""
copy (
 select ksic, cc, py::int as year, appl_year::int as appl_year, appl_no,
   regexp_replace(trim(coalesce(nullif(main_ipc,''), split_part(ipc_all,'|',1))), '\s+', ' ', 'g') as ipc_raw,
   trim(split_part(applicant,'|',1)) as app1,
   case when prio_cc is null then 0
        when regexp_replace(replace(prio_cc,cc,''),'\|','','g')<>'' then 1 else 0 end as prio_for,
   ipc_all, title
 from (select *, left(coalesce(least(nullif(pub_date,''), nullif(rgst_date,'')), nullif(pub_date,''), nullif(rgst_date,'')),4) as py
       from read_parquet('{M}') where appl_year >= 2000)
 where py is not null and py::int between 2000 and {MAXYEAR}
) to '{CACHE}/base_rows.parquet' (format parquet, compression zstd)""")
print('base done', round(time.time() - t, 1))
print(con.sql(f"select cc, count(*) n, min(year), max(year), sum((year-appl_year>=3)::int)*1.0/count(*) lag3p from read_parquet('{CACHE}/base_rows.parquet') group by cc order by cc").df().to_string())
print(con.sql(f"select year, count(*) filter (where cc='KR') kr, count(*) filter (where cc='US') us, count(*) filter (where cc='JP') jp, count(*) filter (where cc='EP') ep, count(*) filter (where cc='CN') cn from read_parquet('{CACHE}/base_rows.parquet') group by year order by year").df().to_string())
