# -*- coding: utf-8 -*-
r"""map5 1단계: 제공 5개국 메타 → KSIC×국가×연도 집계 (읽기 전용, 쓰기는 map5 폴더에만).
산출
- m1_counts_appl.parquet : ksic, cc, appl_year, n_all, n_inv(발명특허만), n_inv_jp8(JP 8자리만; 타국은 n_inv와 같음)
- m1_counts_pub.parquet  : ksic, cc, pub_year, n_inv (공개일 있는 발명특허)
- m1_origin_appl.parquet : ksic, office(cc), appl_year, origin(우선권 첫 국가: KR/US/JP/CN/EP계(DE·FR·GB·EP 등)/기타/없음), n_inv
- m1_month_jp.parquet    : JP 출원·공개 월별 (커버리지 진단)
발명특허 정의(사전 고정): KR appl_no 앞2='10'(20=실용신안 제외), CN 5번째 자리 in (1,8)(2=실용신안 제외, 7자리 등 구형식 제외),
JP 8자리+17자리(PCT), 10자리(공개일 64% 없음, 실용신안·특원 혼재로 추정) 제외, US·EP 전부.
"""
import os, time, duckdb
OUT = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(OUT, r"..\..\..\_shared\meta\*.parquet").replace("\\", "/")
t = time.time()
con = duckdb.connect()
con.sql("SET memory_limit='3GB'; SET threads=2; SET enable_progress_bar=false")
inv = """case when cc='KR' then substr(appl_no,1,2)='10'
 when cc='CN' then length(appl_no) in (12,14) and substr(appl_no,5,1) in ('1','8')
 when cc='JP' then length(appl_no) in (8,17)
 else true end"""
con.sql(f"""create temp table m as select ksic, cc, appl_year, try_cast(substr(pub_date,1,4) as int) pub_year,
 substr(pub_date,1,6) pub_ym, substr(appl_date,1,6) appl_ym, length(appl_no) L, ({inv}) is_inv,
 split_part(coalesce(prio_cc,''),'|',1) p1
 from read_parquet('{META}')""")
print('loaded', round(time.time() - t), flush=True)
con.sql("""copy (select ksic, cc, appl_year, count(*) n_all, sum(is_inv::int) n_inv,
 sum(case when is_inv and (cc<>'JP' or L=8) then 1 else 0 end) n_inv_jp8
 from m where appl_year between 1995 and 2025 group by 1,2,3) to '""" + os.path.join(OUT, 'm1_counts_appl.parquet').replace('\\', '/') + "' (format parquet)")
con.sql("""copy (select ksic, cc, pub_year, count(*) n_inv from m where is_inv and pub_year between 1995 and 2025
 group by 1,2,3) to '""" + os.path.join(OUT, 'm1_counts_pub.parquet').replace('\\', '/') + "' (format parquet)")
con.sql("""copy (select ksic, cc office, appl_year,
 case when p1 in ('KR','US','JP','CN') then p1
      when p1 in ('EP','DE','FR','GB','IT','NL','SE','CH','AT','BE','DK','FI','ES','IE','NO','PL','PT','LU','WO') and p1<>'WO' then 'EU'
      when p1='' then 'NONE' else 'OTH' end origin, count(*) n_inv
 from m where is_inv and appl_year between 1995 and 2025 group by 1,2,3,4) to '""" + os.path.join(OUT, 'm1_origin_appl.parquet').replace('\\', '/') + "' (format parquet)")
con.sql("""copy (select appl_ym, pub_ym, L, count(*) n from m where cc='JP' and appl_year>=2015 group by 1,2,3)
 to '""" + os.path.join(OUT, 'm1_month_jp.parquet').replace('\\', '/') + "' (format parquet)")
print('done', round(time.time() - t), flush=True)
