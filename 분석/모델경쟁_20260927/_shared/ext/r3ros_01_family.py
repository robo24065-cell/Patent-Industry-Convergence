# 제공 데이터(5개국) 레코드 -> US 등록특허번호 목록 (family 필드 US 토큰 + US 자국 rgst_no)
import os, time, duckdb
EXT = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, '/')
META = EXT + '/../meta/*.parquet'
t = time.time()
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false")
con.execute(r"""create macro mgn(s) as (case
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{1,4}/') then substr(replace(s,' ',''),1,4)||' '||cast(cast(regexp_extract(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z]([0-9]{1,4})/',1) as int) as varchar)
  when regexp_matches(replace(s,' ',''),'^[A-H][0-9]{2}[A-Z][0-9]{3,6}$') then substr(replace(s,' ',''),1,4)||' '||cast(cast(substr(replace(s,' ',''),5,length(replace(s,' ',''))-6) as int) as varchar)
  else null end)""")
con.execute(f"""create table rec as select cc, appl_no, any_value(appl_year)::int appl_year, ksic,
   any_value(mgn(regexp_replace(trim(coalesce(nullif(main_ipc,''), split_part(ipc_all,'|',1))), '\s+', ' ', 'g'))) mg,
   any_value(family) famstr, any_value(rgst_no) rgst_no
  from read_parquet('{META}') group by cc, appl_no, ksic""")
con.execute("""create table fam as
  select distinct cc, appl_no, ksic, try_cast(substr(m,3) as bigint) us_pn, 'family' src
  from (select cc, appl_no, ksic, unnest(string_split(famstr,'|')) m from rec where famstr is not null)
  where regexp_full_match(m,'US[0-9]{7,8}')
  union
  select cc, appl_no, ksic, try_cast(rgst_no as bigint), 'own' from rec
  where cc='US' and regexp_full_match(coalesce(rgst_no,''),'[0-9]{7,8}')""")
con.execute(f"copy (select cc, appl_no, ksic, appl_year, mg from rec) to '{EXT}/r3ros_work/rec.parquet' (format parquet)")
con.execute(f"copy (select cc, appl_no, ksic, us_pn, max(src) src from fam group by all) to '{EXT}/r3ros_work/fam_us.parquet' (format parquet)")
print(con.sql("select count(*) n_rec, count(distinct cc||appl_no) n_app from rec").fetchall())
print(con.sql(f"select count(*), count(distinct us_pn) from read_parquet('{EXT}/r3ros_work/fam_us.parquet')").fetchall())
print('secs', round(time.time() - t))
