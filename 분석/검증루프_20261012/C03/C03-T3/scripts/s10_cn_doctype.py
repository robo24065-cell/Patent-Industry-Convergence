import sys, json; sys.path.insert(0, sys.argv[1]); sys.stdout.reconfigure(encoding='utf-8')
from _paths import *
import duckdb
M=(MC+r"\_shared\meta\*.parquet").replace("\\","/")
con=duckdb.connect(); con.execute("set memory_limit='2GB'; set threads=2")
r=con.sql(f"""select length(appl_no) L, substr(appl_no,5,1) d5, count(*) n from read_parquet('{M}') where cc='CN' group by all order by n desc limit 12""").df()
print(r.to_string())
tot=int(r.n.sum()) if len(r)<12 else None
q=con.sql(f"""select count(*) n, sum(case when length(appl_no) in (12,14) and substr(appl_no,5,1) in ('1','8') then 1 else 0 end) inv,
 sum(case when length(appl_no) in (12,14) and substr(appl_no,5,1) in ('2','9') then 1 else 0 end) um from read_parquet('{M}') where cc='CN'""").fetchone()
out=dict(cn_rows=q[0],invention=q[1],utility=q[2],utility_share=round(q[2]/q[0],4),invention_share=round(q[1]/q[0],4))
json.dump(out,open(OUT+r"\_s10_cn_doctype.json","w"),indent=1); print(out)
