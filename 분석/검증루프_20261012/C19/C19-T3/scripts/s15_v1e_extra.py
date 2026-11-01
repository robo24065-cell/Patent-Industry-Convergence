# C19-T3 s15: data_check 보강 — V1E에 남은 이중 수록 0 확인, 제거 행의 KR 연결 비중(선택 편향), 국가·연도 비중
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib19 import *
con = con3()
V1E = ddb(C18T2 + r'\cache\meta_V1E.parquet')
r = {}
r['remaining_us_pairs'] = con.sql(f"with u as (select appl_no from read_parquet('{V1E}') where cc='US') select count(*) from u s join u l on length(s.appl_no)=7 and length(l.appl_no)=8 and lpad(s.appl_no,8,'0')=l.appl_no").fetchone()[0]
r['remaining_ep_dupkeys'] = con.sql(f"select count(*) from (select ltrim(appl_no,'0') k from read_parquet('{V1E}') where cc='EP' group by 1 having count(distinct appl_no)>1)").fetchone()[0]
# 제거 행 중 KR 등록번호 토큰 연결이 있는 비중 vs 전체
for cc in ('US', 'EP'):
    q = f"""with d as (select appl_no, family from read_parquet('{META}') where cc='{cc}'),
      v as (select appl_no from read_parquet('{V1E}') where cc='{cc}'),
      f as (select d.appl_no, (d.appl_no not in (select appl_no from v)) dropped, regexp_matches(coalesce(d.family,''), 'KR(10|20)[0-9]{{7}}') haskr from d)
      select dropped, count(*), avg(haskr::int) from f group by 1 order by 1"""
    r[f'{cc}_krlink_share'] = con.sql(q).fetchall()
jdump(r, 's15_v1e_extra.json'); out(r)
