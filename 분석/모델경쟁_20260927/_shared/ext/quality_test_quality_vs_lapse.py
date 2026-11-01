# 해석 검정: KR 등록특허의 '국제 품질 표지'(ETSI SEP 선언 / US 패밀리의 KPSS 가치 보유)와 등록 후 8년 내 소멸 비율 (등록 2005-2012)
import duckdb, pandas as pd, numpy as np
R=r'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/수집파이프라인/registry_full.sqlite'
con=duckdb.connect(); con.sql("SET memory_limit='3GB'; SET threads=2"); con.sql("INSTALL sqlite; LOAD sqlite")
con.sql(f"""create table reg as select p.rgst_no, p.ksic, p.applNo, try_strptime(p.rgstDate,'%Y%m%d')::date rd,
   min(case when r.rgstCsName='소멸등록' then try_strptime(r.rgstCsDate,'%Y%m%d')::date end) lapse
 from sqlite_scan('{R}','patent') p left join sqlite_scan('{R}','right') r on p.rgst_no=r.rgst_no
 group by all""")
print(con.sql("select count(*), count(rd), count(lapse), min(rd), max(rd) from reg").fetchall())
con.sql("create table sep as select distinct appl_no from 'quality_etsi_sep_link.parquet' where cc='KR'")
con.sql("create table kp as select distinct appl_no from 'quality_kpss_family_link.parquet' where cc='KR'")
d=con.sql("""select reg.ksic, (sep.appl_no is not null)::int is_sep, (kp.appl_no is not null)::int has_kpss,
   (lapse is not null and lapse < rd + interval 8 year)::int lapse8
 from reg left join sep on reg.applNo=sep.appl_no left join kp on reg.applNo=kp.appl_no
 where year(rd) between 2005 and 2012""").df()
print('n',len(d))
print(d.groupby('is_sep').lapse8.agg(['mean','size']))
print(d.groupby('has_kpss').lapse8.agg(['mean','size']))
# KSIC 고정: KSIC 내 차이의 가중평균
def within(col):
    g=d.groupby(['ksic',col]).lapse8.agg(['mean','size']).unstack(col).dropna()
    w=g[('size',1)]; return float(((g[('mean',1)]-g[('mean',0)])*w).sum()/w.sum()), len(g)
print('KSIC 내 소멸률 차이 SEP:',within('is_sep'),' KPSS:',within('has_kpss'))
# 추가: 제공데이터 내부 family에 US 특허가 있는 KR 특허로 한정(내부 신호 통제) 후 외부 신호의 추가 효과
con.sql("create table usfam as select distinct appl_no from read_parquet('../meta/*.parquet') where cc='KR' and regexp_matches(coalesce(family,''),'(^|[|])US[0-9]{7,8}($|[|])')")
d2=con.sql("""select reg.ksic, (sep.appl_no is not null)::int is_sep, (kp.appl_no is not null)::int has_kpss, (u.appl_no is not null)::int us_fam,
   (lapse is not null and lapse < rd + interval 8 year)::int lapse8
 from reg left join sep on reg.applNo=sep.appl_no left join kp on reg.applNo=kp.appl_no left join usfam u on reg.applNo=u.appl_no
 where year(rd) between 2005 and 2012""").df()
print(d2.groupby('us_fam').lapse8.agg(['mean','size']))
x=d2[d2.us_fam==1]
print('US패밀리 보유 KR 내: KPSS', x.groupby('has_kpss').lapse8.agg(['mean','size']).to_dict(), ' SEP', x.groupby('is_sep').lapse8.agg(['mean','size']).to_dict())
d=x; print('KSIC 내 (US패밀리 보유 한정) SEP:',within('is_sep'),' KPSS:',within('has_kpss'))
