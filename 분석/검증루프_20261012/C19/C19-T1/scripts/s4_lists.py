# C19-T1 s4: 변형별 제거 JP 행 목록(appl_no) — VJ1·VJ1s·VJ3·VRND-JP(s=0,1,2) ; VJ2는 base SQL에서 py 재정의
import sys, os; sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib19 import *
con = con3()
NP = f"read_parquet('{ddb(C)}/jp_nopub_rows.parquet')"
J = f"read_parquet('{ddb(C)}/jp_rows.parquet')"
L = os.path.join(C, 'lists'); os.makedirs(L, exist_ok=True)
# T3 범위: appl_year>=2000, py 2000–2024 (fin 최대)
con.execute(f"create table np as select * from {NP} where py is not null and py::int between 2000 and 2024")
con.execute(f"copy (select appl_no from np) to '{ddb(L)}/VJ1.parquet' (format parquet)")
con.execute(f"copy (select appl_no from np where py in ('2021','2022')) to '{ddb(L)}/VJ1s.parquet' (format parquet)")
con.execute(f"copy (select appl_no from np where dup) to '{ddb(L)}/VJ3.parquet' (format parquet)")
# 위약: 공개일 있는 JP 행 중 쌍(이중 수록 짝)도 아닌 행에서 py별 VJ1 제거 수만큼, hash 순
con.execute(f"""create or replace table pool as select appl_no, ksic, py from {J}
  where pub_date is not null and appl_year>=2000 and py is not null and py::int between 2000 and 2024""")
# 짝 목록: s3에서 tw_appl 을 저장하지 않았으므로 jkey/제목+출원일 기준으로 다시 구한다
con.execute(f"""create table jj as select appl_no, jkey, lower(regexp_replace(coalesce(title,''),'[[:space:]]','','g')) tnorm, appl_date, pub_date from {J}""")
con.execute(f"""create table dupnp as select j1.appl_no, j1.jkey, j1.tnorm, j1.appl_date from jj j1 join np on np.appl_no=j1.appl_no where np.dup""")
con.execute("""create table twinset as select distinct b.appl_no from jj b join dupnp a on b.pub_date is not null and b.appl_no<>a.appl_no and (b.jkey=a.jkey or (a.tnorm<>'' and b.tnorm=a.tnorm and b.appl_date=a.appl_date))""")
q = con.sql("select py, count(*) n from np group by py").fetchall()
quota = {p: n for p, n in q}
res = {'quota_total': sum(quota.values()), 'twins_excluded': con.sql("select count(*) from twinset").fetchone()[0]}
for s in (0, 1, 2):
    con.execute(f"""copy (select appl_no from (select appl_no, py, row_number() over (partition by py order by hash(appl_no || ksic || 'c19s{s}')) rn
        from pool where appl_no not in (select appl_no from twinset)) x join (select * from (values {','.join(f"('{p}',{n})" for p, n in quota.items())}) t(py, n)) qq using(py) where rn <= qq.n)
        to '{ddb(L)}/VRNDJP{s}.parquet' (format parquet)""")
    res[f'VRNDJP{s}'] = con.sql(f"select count(*) from read_parquet('{ddb(L)}/VRNDJP{s}.parquet')").fetchone()[0]
for v in ('VJ1', 'VJ1s', 'VJ3'):
    res[v] = con.sql(f"select count(*) from read_parquet('{ddb(L)}/{v}.parquet')").fetchone()[0]
res['quota_by_py'] = dict(sorted(quota.items()))
res['overlap_rnd0_vj1'] = con.sql(f"select count(*) from read_parquet('{ddb(L)}/VRNDJP0.parquet') a join read_parquet('{ddb(L)}/VJ1.parquet') b using(appl_no)").fetchone()[0]
jdump(res, 's4_lists.json'); out(res)
