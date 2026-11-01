# JP·EP 행 추출(읽기만) → cache/jp_rows.parquet, ep_rows.parquet ; 번호 형식 가설 검증
import sys, os; sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib19 import *
con = con3()
py = "left(coalesce(least(nullif(pub_date,''), nullif(rgst_date,'')), nullif(pub_date,''), nullif(rgst_date,'')),4)"
con.execute(f"""copy (select appl_no, appl_date, appl_year, nullif(rgst_no,'') rgst_no, nullif(rgst_date,'') rgst_date, nullif(pub_date,'') pub_date,
  title, applicant, main_ipc, ipc_all, family, prio_cc::varchar prio_cc, ksic::varchar ksic, {py} py, {JPKEY} jkey
  from read_parquet('{META}') where cc='JP') to '{ddb(C)}/jp_rows.parquet' (format parquet, compression zstd)""")
con.execute(f"""copy (select appl_no, appl_date, appl_year, nullif(rgst_no,'') rgst_no, nullif(rgst_date,'') rgst_date, nullif(pub_date,'') pub_date, ksic::varchar ksic, main_ipc, {py} py
  from read_parquet('{META}') where cc='EP') to '{ddb(C)}/ep_rows.parquet' (format parquet, compression zstd)""")
J = f"read_parquet('{ddb(C)}/jp_rows.parquet')"
R = {}
R['n_jp'] = con.sql(f"select count(*), count(distinct appl_no), count(distinct jkey) from {J}").fetchone()
# 8자리 연호 가설: 앞 2자리+1988 == appl_year 비율 ; 10자리: 앞 4자리 == appl_year
R['era8_match'] = con.sql(f"""select count(*), avg((1988+cast(substr(appl_no,1,2) as int)=appl_year)::int) from {J} where regexp_matches(appl_no,'^[0-9]{{8}}U?$')""").fetchone()
R['y10_match'] = con.sql(f"""select count(*), avg((cast(substr(appl_no,1,4) as int)=appl_year)::int) from {J} where regexp_matches(appl_no,'^[0-9]{{10}}U?$')""").fetchone()
R['u9_examples'] = con.sql(f"select appl_no, appl_date, appl_year, rgst_no, rgst_date from {J} where regexp_matches(appl_no,'^[0-9]{{9}}U$') limit 12").fetchall()
# 10자리 rgst_no 가설(공개번호): 앞 4자리 == rgst_date 연도 비율, 출원→rgst 시차
R['rg10'] = con.sql(f"""select count(*), avg((substr(rgst_no,1,4)=substr(rgst_date,1,4))::int),
   median(date_diff('day', strptime(appl_date,'%Y%m%d'), strptime(rgst_date,'%Y%m%d'))) from {J} where regexp_matches(rgst_no,'^[0-9]{{10}}$')""").fetchone()
R['rg78'] = con.sql(f"""select count(*), median(date_diff('day', strptime(appl_date,'%Y%m%d'), strptime(rgst_date,'%Y%m%d'))) from {J} where regexp_matches(rgst_no,'^[0-9]{{7,8}}$')""").fetchone()
R['fmt_cross'] = con.sql(f"""select regexp_replace(appl_no,'[0-9]','9','g') ap, regexp_replace(coalesce(rgst_no,'-'),'[0-9]','9','g') rg, (pub_date is null) nopub, (applicant is null) noappl, count(*) n,
   min(coalesce(pub_date,rgst_date)) dmin, max(coalesce(pub_date,rgst_date)) dmax from {J} group by all order by n desc""").fetchall()
jdump(R, 's1_extract.json')
for k, v in R.items():
    out(k)
    if isinstance(v, list):
        [out('  ', r) for r in v]
    else:
        out('  ', v)
