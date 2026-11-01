# C19-T1 s3: 이중 수록 세부(연·월별, 짝의 공개일 = 이 행 rgst_date 여부), 포착률 판정 수치, VJ 변형용 행 목록
import sys, os; sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib19 import *
import pandas as pd, numpy as np
con = con3()
J = f"read_parquet('{ddb(C)}/jp_rows.parquet')"
con.execute(f"""create table j as select *, (pub_date is null) nopub, appl_no like '%U' u,
  case when rgst_no is null then 'none' when regexp_matches(rgst_no,'^[0-9]{{10}}$') then 'pubno10' when regexp_matches(rgst_no,'^[0-9]{{7,8}}$') then 'patno' else 'other' end rgfmt,
  lower(regexp_replace(coalesce(title,''),'[[:space:]]','','g')) tnorm from {J}""")
con.execute("create table np as select * from j where nopub and rgst_date is not null")
con.execute("create table pb as select appl_no, jkey, tnorm, appl_date, pub_date, ksic, py from j where not nopub")
# 짝 찾기: k1 = 같은 jkey(다른 appl_no), k2 = 같은 (tnorm, appl_date)
con.execute("""create table tw as
  select np.appl_no, any_value(pb.appl_no) tw_appl, any_value(pb.pub_date) tw_pub, any_value(pb.py) tw_py, 'k1' how from np join pb on pb.jkey=np.jkey and pb.appl_no<>np.appl_no group by np.appl_no
  union all
  select np.appl_no, any_value(pb.appl_no), any_value(pb.pub_date), any_value(pb.py), 'k2' from np join pb on pb.tnorm=np.tnorm and pb.appl_date=np.appl_date and pb.appl_no<>np.appl_no and np.tnorm<>'' group by np.appl_no""")
con.execute("create table tw1 as select appl_no, min(how) how, any_value(tw_appl) tw_appl, any_value(tw_pub) tw_pub, any_value(tw_py) tw_py from tw group by appl_no")
con.execute("""create table npx as select np.*, tw1.how is not null dup, tw1.how, tw1.tw_appl, tw1.tw_pub, tw1.tw_py from np left join tw1 using(appl_no)""")
R = {}
R['dup_by_year_fmt'] = con.sql("""select left(rgst_date,4) y, rgfmt, u, count(*) n, sum(dup::int) dup, sum((dup and tw_pub=rgst_date)::int) same_date,
   sum((dup and left(tw_pub,4)=left(rgst_date,4))::int) same_year, sum((dup and tw_pub<rgst_date)::int) twin_earlier
   from npx where left(rgst_date,4) between '2012' and '2024' group by all order by 1,2,3""").fetchall()
R['pubno10_dup_by_month'] = con.sql("""select left(rgst_date,6) m, count(*) n, sum(dup::int) dup, sum((dup and tw_pub=rgst_date)::int) same_date from npx where rgfmt='pubno10' group by 1 order by 1""").fetchall()
# 포착률 판정 수치: 초과분 = 관측 − r̄ × 공식(특허공보), r̄ = 2016–2020 평균 (s2 jp_capture.csv)
cap = pd.read_csv(os.path.join(W, 'jp_capture.csv'))
rb = cap[(cap.year >= 2016) & (cap.year <= 2020)].cap_nopub_pat_vs_koho.mean()
ex = {}
for y in (2021, 2022):
    r = cap[cap.year == y].iloc[0]
    exp = rb * r.jpo_tokkyo_koho; obs = r.nopub_pat
    d = con.sql(f"select count(*) from npx where not u and dup and left(rgst_date,4)='{y}'").fetchone()[0]
    ex[y] = dict(observed=int(obs), expected=float(exp), excess=float(obs - exp), dup_rows=int(d), rel=float(r.rel_nopub_pat_vs_koho))
tot_ex = sum(v['excess'] for v in ex.values()); tot_dup = sum(v['dup_rows'] for v in ex.values())
R['capture_judgement'] = dict(r_bar_2016_2020=float(rb), by_year=ex, excess_2y=tot_ex, dup_2y=tot_dup, dup_share_of_excess=tot_dup / tot_ex)
# T3 범위(appl_year>=2000) py별: 행 종류 구성
R['t3_py_composition'] = con.sql("""select py, count(*) n_all, sum((not nopub)::int) pubdate, sum((nopub and rgfmt='pubno10')::int) pubno10, sum((nopub and rgfmt<>'pubno10' and rgst_date is not null)::int) realreg_nopub
  from j where appl_year>=2000 and py between '2016' and '2024' group by 1 order by 1""").fetchall()
# 변형용 행 목록 (T3 범위 appl_year>=2000, 키 = appl_no; JP 안에서 appl_no 고유)
con.execute(f"copy (select appl_no, py, rgfmt, u, dup, how, appl_date, rgst_date from npx where appl_year>=2000) to '{ddb(C)}/jp_nopub_rows.parquet' (format parquet)")
R['n_nopub_t3'] = con.sql("select count(*), sum(dup::int), sum((py in ('2021','2022'))::int), sum((dup and py in ('2021','2022'))::int) from npx where appl_year>=2000").fetchone()
jdump(R, 's3_dup_detail.json')
for k, v in R.items():
    out(k)
    if isinstance(v, list):
        [out('  ', r) for r in v]
    else:
        out('  ', v)
