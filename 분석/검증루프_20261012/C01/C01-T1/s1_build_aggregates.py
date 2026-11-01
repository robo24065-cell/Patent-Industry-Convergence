# -*- coding: utf-8 -*-
r"""C01-T1 s1: 원천 meta(읽기만)에서 감사용 집계를 만든다. 쓰기는 이 폴더(cache\)에만.
산출(cache\)
- agg_ksic.parquet : ksic, cc, appl_year, pub_year, wipo_field, is_inv, n  (행=출원×KSIC, 원 m1 정의와 같은 KSIC 중복 유지)
- agg_dist.parquet : cc, appl_year, pub_year, wipo_field, doctype, is_inv, n_doc (cc+appl_no 고유 문서)
- qa_counts.parquet: cc, 항목별 이상치 수
- ksic_mult.parquet: cc, appl_year, 문서당 KSIC 수 분포
"""
import os, time, duckdb
OUT = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(OUT, "cache"); os.makedirs(C, exist_ok=True)
MC = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927"
META = os.path.join(MC, r"_shared\meta\*.parquet").replace("\\", "/")
CW = os.path.join(MC, r"_shared\ext\quality_wipo_ipc_field_concordance.csv").replace("\\", "/")
t = time.time()
con = duckdb.connect()
con.sql("SET memory_limit='3GB'; SET threads=2; SET enable_progress_bar=false")
con.sql(f"SET temp_directory='{C}/duck_tmp'".replace("\\", "/"))
con.sql(f"create temp table cw as select * from read_csv_auto('{CW}', all_varchar=true)")
inv = """case when cc='KR' then substr(appl_no,1,2)='10'
 when cc='CN' then length(appl_no) in (12,14) and substr(appl_no,5,1) in ('1','8')
 when cc='JP' then length(appl_no) in (8,17) else true end"""
dt = """case when cc='KR' then 'KR'||substr(appl_no,1,2)
 when cc='CN' then 'CN'||length(appl_no)::varchar||'_'||substr(appl_no,5,1)
 when cc='JP' then 'JP'||length(appl_no)::varchar
 when cc='US' then 'US'||length(appl_no)::varchar
 else 'EP'||length(appl_no)::varchar end"""
con.sql(f"""create temp table m as
 select p.ksic, p.cc, p.appl_no, p.appl_year, p.pub_year, p.has_pub, p.has_rgst, p.is_inv, p.doctype,
   coalesce(try_cast(g.wipo_field as int), try_cast(s.wipo_field as int), 0) wipo_field
 from (select ksic, cc, appl_no, appl_year, try_cast(substr(pub_date,1,4) as int) pub_year,
        (pub_date is not null and pub_date<>'') has_pub, (rgst_no is not null and rgst_no<>'') has_rgst,
        ({inv}) is_inv, ({dt}) doctype,
        substr(main_ipc,1,4) sub, trim(split_part(substr(main_ipc,5),'/',1)) grp
       from read_parquet('{META}')) p
 left join (select * from cw where main_group is not null and main_group<>'') g on p.sub=g.subclass and p.grp=g.main_group
 left join (select * from cw where main_group is null or main_group='') s on p.sub=s.subclass""")
print("loaded", con.sql("select count(*) from m").fetchone(), round(time.time() - t), flush=True)
con.sql(f"copy (select ksic, cc, appl_year, pub_year, wipo_field, is_inv, count(*) n from m group by all) to '{C}/agg_ksic.parquet' (format parquet)".replace("\\", "/"))
con.sql(f"""copy (select cc, appl_year, pub_year, wipo_field, doctype, is_inv, has_rgst, count(*) n_doc from
  (select cc, appl_no, any_value(appl_year) appl_year, any_value(pub_year) pub_year, any_value(wipo_field) wipo_field,
          any_value(doctype) doctype, bool_or(is_inv) is_inv, bool_or(has_rgst) has_rgst from m group by cc, appl_no)
  group by all) to '{C}/agg_dist.parquet' (format parquet)""".replace("\\", "/"))
print("dist", round(time.time() - t), flush=True)
con.sql(f"""copy (select cc,
  count(*) n_rows,
  sum((ksic is null or ksic='')::int) ksic_missing,
  sum((appl_no is null or appl_no='')::int) applno_missing,
  sum((appl_year is null)::int) appl_year_null,
  sum((appl_year < 1990 or appl_year > 2025)::int) appl_year_out,
  sum((pub_year is not null and (pub_year < 1990 or pub_year > 2026))::int) pub_year_out,
  sum((pub_year is not null and pub_year < appl_year)::int) pub_before_appl,
  sum((not has_pub)::int) pub_missing,
  sum((wipo_field=0)::int) field_unmapped
  from m group by cc) to '{C}/qa_counts.parquet' (format parquet)""".replace("\\", "/"))
con.sql(f"""copy (select cc, count(*) dup_pairs_extra from (select cc, appl_no, ksic, count(*) k from m group by all having count(*)>1) group by cc)
  to '{C}/qa_dup.parquet' (format parquet)""".replace("\\", "/"))
con.sql(f"""copy (select cc, appl_year, k, count(*) n_doc from (select cc, appl_no, any_value(appl_year) appl_year, count(distinct ksic) k from m group by cc, appl_no)
  group by all) to '{C}/ksic_mult.parquet' (format parquet)""".replace("\\", "/"))
# 한 문서가 여러 분야/공개연도를 갖는지(any_value 사용 타당성)
con.sql(f"""copy (select cc, count(*) docs, sum((nf>1)::int) multi_field, sum((npy>1)::int) multi_pubyear, sum((nay>1)::int) multi_applyear from
  (select cc, appl_no, count(distinct wipo_field) nf, count(distinct pub_year) npy, count(distinct appl_year) nay from m group by cc, appl_no) group by cc)
  to '{C}/qa_docconsistency.parquet' (format parquet)""".replace("\\", "/"))
print("done", round(time.time() - t), flush=True)
