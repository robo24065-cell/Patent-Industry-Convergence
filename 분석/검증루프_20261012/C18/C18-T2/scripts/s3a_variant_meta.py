# C18-T2 s3a: 변형 메타 parquet (V1: US 7자리 쌍 행 제거, V1E: + EP 9자 쌍 행 제거) → cache/meta_V1.parquet, cache/meta_V1E.parquet, cache/s3a.json
# 원본 _shared/meta는 읽기만. 열·값은 그대로(행만 뺌).
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib18 import *
con = con3()
con.execute(f"create table us7 as select a7 appl_no from read_parquet('{ddb(C)}/us_dup_pairs.parquet')")
con.execute(f"""create table ep9 as select case when L1<L2 then n1 else n2 end appl_no from read_parquet('{ddb(C)}/other_dup_pairs.parquet') where cc='EP'""")
res = {'n_us7': con.sql("select count(*), count(distinct appl_no) from us7").fetchone(), 'n_ep9': con.sql("select count(*), count(distinct appl_no) from ep9").fetchone()}
res['ep9_len'] = con.sql("select length(appl_no), count(*) from ep9 group by 1").fetchall()
for v, cond in [('V1', "not (cc='US' and appl_no in (select appl_no from us7))"),
                ('V1E', "not (cc='US' and appl_no in (select appl_no from us7)) and not (cc='EP' and appl_no in (select appl_no from ep9))")]:
    con.execute(f"copy (select * from read_parquet('{META}') where {cond}) to '{ddb(C)}/meta_{v}.parquet' (format parquet, compression zstd, row_group_size 500000)")
    res[v] = con.sql(f"select cc, count(*) from read_parquet('{ddb(C)}/meta_{v}.parquet') group by 1 order by 1").fetchall()
res['P0'] = con.sql(f"select cc, count(*) from read_parquet('{META}') group by 1 order by 1").fetchall()
res['schema_same'] = [str(x) for x in con.sql(f"describe select * from read_parquet('{ddb(C)}/meta_V1E.parquet')").fetchall()] == [str(x) for x in con.sql(f"describe select * from read_parquet('{META}')").fetchall()]
jdump(res, 's3a.json'); out(res)
