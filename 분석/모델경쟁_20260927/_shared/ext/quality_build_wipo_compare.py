# 제공 데이터(국가x연도xWIPO35분야) vs WIPO 공식 공개건수(관청x연도x분야) 비교
import pandas as pd, duckdb, re
R='quality_raw/'
c=pd.read_excel(R+'quality_wipo_ipc_technology.xlsx',skiprows=6)[['Field_number','Field_en','IPC_code']].dropna()
rows=[]
for _,r in c.iterrows():
    s=r.IPC_code.replace('%','').strip()
    sub=s[:4]; grp=s[4:].replace('/','').strip()
    rows.append((sub, grp if grp else '', int(r.Field_number), r.Field_en))
cc=pd.DataFrame(rows,columns=['subclass','main_group','wipo_field','field_en']).drop_duplicates()
cc.to_csv('quality_wipo_ipc_field_concordance.csv',index=False,encoding='utf-8-sig')
con=duckdb.connect(); con.sql("SET memory_limit='3GB'; SET threads=2")
con.register('cc',cc)
q="""
with p as (
 select distinct cc, appl_no, appl_year, try_cast(substr(pub_date,1,4) as int) pub_year,
   substr(main_ipc,1,4) sub, trim(split_part(substr(main_ipc,5),'/',1)) grp
 from read_parquet('../meta/*.parquet') where main_ipc is not null)
select p.cc, p.appl_year, p.pub_year, coalesce(g.wipo_field, s.wipo_field, 0) wipo_field, count(*) n
from p left join (select * from cc where main_group<>'') g on p.sub=g.subclass and p.grp=g.main_group
       left join (select * from cc where main_group='') s on p.sub=s.subclass
group by all
"""
d=con.sql(q).df()
# 같은 출원이 여러 KSIC에 있어도 distinct라 1회. 단, 한 출원이 main_ipc 두 개면 중복 가능 → 무시(극소)
d.to_parquet('quality_provided_cc_year_wipo35.parquet',index=False)
print(d.groupby('cc').n.sum()); print('unmapped share', d[d.wipo_field==0].n.sum()/d.n.sum())
