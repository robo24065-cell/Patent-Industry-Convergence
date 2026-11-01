# T1 R3 B단계 — 제공 KR 출원(공개·등록) IPC 표 캐시: appl_no, ksic, ay, pub_date, rgst_date, main_ipc, ipc_all → R3/cache/src_kr_ipc.parquet
import sys, pathlib, duckdb
R3 = pathlib.Path(__file__).resolve().parent; T1 = R3.parent; sys.path.insert(0, str(T1))
from common import B
M = (B/'분석/모델경쟁_20260927/_shared/meta/*.parquet').as_posix()
out = (R3/'cache'/'src_kr_ipc.parquet').as_posix()
duckdb.sql("SET memory_limit='2GB'; SET threads=2")
duckdb.sql(f"""copy (select appl_no, ksic, appl_year as ay, coalesce(pub_date,'') pub_date, coalesce(rgst_date,'') rgst_date, main_ipc, ipc_all
             from read_parquet('{M}') where cc='KR' and appl_year>=2000) to '{out}' (format parquet)""")
print(duckdb.sql(f"select count(*), count(distinct appl_no), avg(len(ipc_all)) from read_parquet('{out}')"))
