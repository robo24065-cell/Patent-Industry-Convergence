# -*- coding: utf-8 -*-
r"""C19-T2 s7: 제공 메타 KR 기록 family 필드의 총량·연도별 보유율·이중 표기(F1·F1x) 비율(읽기만) → cache/s7_meta_cov.json, cache/s7_kr_family_by_year.csv
 (a) 총량: KR 행·고유 출원·family 보유 수, 전체 토큰 수와 T4 연도 추정 토큰(s2_fam_tokens) 대조
 (b) 출원연도별 family 보유율, 기록당 토큰 수, F1·F1x 이중 표기 보유 기록 비율(모든 토큰 기준)"""
import os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib19 import *
import pandas as pd

con = con3()
R = {}
R['kr_rows'], R['kr_appl'], R['kr_appl_with_family'] = con.execute(f"""select count(*), count(distinct appl_no), count(distinct case when family is not null and family<>'' then appl_no end)
   from read_parquet('{META}') where cc='KR'""").fetchone()
con.execute(f"""create table k as select appl_no, any_value(appl_year)::int ay, any_value(family) fam from read_parquet('{META}') where cc='KR' group by appl_no""")
con.execute("""create table t as select appl_no, ay, list_distinct(string_split(fam,'|')) toks from k where fam is not null and fam<>''""")
con.execute(r"""create table t2 as select appl_no, ay, len(toks) n,
   len(list_distinct(list_transform(toks, x -> substr(x,1,2) || ltrim(regexp_replace(substr(x,3), '[A-Z][0-9]?$', ''),'0')))) n1,
   len(list_distinct(list_transform(toks, x -> case
      when substr(x,1,2)='US' and regexp_full_match(substr(x,3),'[0-9]{11}') and substr(x,7,1)='0' then 'US'||substr(x,3,4)||substr(x,8)
      when substr(x,1,2)='JP' and regexp_full_match(substr(x,3),'[1-9][0-9]{7}') then 'JP'||cast(1988+cast(substr(x,3,2) as int) as varchar)||substr(x,5)
      else substr(x,1,2) || ltrim(regexp_replace(substr(x,3), '[A-Z][0-9]?$', ''),'0') end))) nx,
   len(list_filter(toks, x -> not regexp_full_match(substr(x,3),'[0-9]+'))) n_nondigit
   from t""")
R['tokens_all_distinct_exact'] = int(con.execute("select sum(n) from t2").fetchone()[0])
R['tokens_nondigit(suffix등)'] = int(con.execute("select sum(n_nondigit) from t2").fetchone()[0])
R['records_dup_F1_share'] = float(con.execute("select avg((n1<n)::int) from t2").fetchone()[0])
R['records_dup_F1x_share'] = float(con.execute("select avg((nx<n)::int) from t2").fetchone()[0])
R['tokens_dup_F1_share'] = float(con.execute("select sum(n-n1)/sum(n) from t2").fetchone()[0])
R['tokens_dup_F1x_share'] = float(con.execute("select sum(n-nx)/sum(n) from t2").fetchone()[0])
by = con.execute("""select k.ay, count(*) n_appl, avg((k.fam is not null and k.fam<>'')::int) fam_share,
     avg(t2.n) tok_per_rec, avg((t2.n1<t2.n)::int) rec_dup_F1, avg((t2.nx<t2.n)::int) rec_dup_F1x, sum(t2.n-t2.n1)/sum(t2.n) tok_dup_F1, sum(t2.n-t2.nx)/sum(t2.n) tok_dup_F1x
     from k left join t2 using(appl_no) where k.ay between 1995 and 2025 group by k.ay order by k.ay""").df()
by.to_csv(J(C, 's7_kr_family_by_year.csv'), index=False, encoding='utf-8-sig')
R['by_year'] = by.round(4).to_dict('records')
R['t4_s2_tokens_rows'] = 6471841
jdump(R, J(C, 's7_meta_cov.json'))
out({k: v for k, v in R.items() if k != 'by_year'})
print(by.round(4).to_string())
