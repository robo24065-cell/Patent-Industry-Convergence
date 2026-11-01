import sys, json; sys.path.insert(0, sys.argv[1]); sys.stdout.reconfigure(encoding='utf-8')
from _paths import *
import duckdb
M=(MC+r"\_shared\meta\*.parquet").replace("\\","/")
con=duckdb.connect(); con.execute("set memory_limit='2GB'; set threads=2")
res={}
for t0 in (2016,2019,2020,2021):
    q=f"""with d as (select distinct appl_no, nullif(pub_date,'') pd, nullif(rgst_date,'') rd from read_parquet('{M}') where cc='KR' and appl_year between {t0-2} and {t0})
    select count(*) n, avg(case when least(coalesce(pd,'99999999'),coalesce(rd,'99999999'))>'{t0}1231' then 1 else 0 end) unseen_min_pub_rgst,
           avg(case when coalesce(pd,'99999999')>'{t0}1231' then 1 else 0 end) unseen_pub_only from d"""
    r=con.sql(q).fetchone(); res[t0]=dict(n=r[0],unseen_min_pub_rgst=round(r[1],4),unseen_pub_only=round(r[2],4))
json.dump(res,open(OUT+r"\_s8_unseen.json","w"),indent=1); print(res)
