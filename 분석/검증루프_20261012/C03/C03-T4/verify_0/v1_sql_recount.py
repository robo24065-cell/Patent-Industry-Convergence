# verify_0/v1: C03-C 제공 사건을 duckdb SQL로 독립 재구현해 C_dev_pos.parquet와 집합 비교(개발 원점만)
import sys, duckdb, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
B = 'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전'
KRP = B + '/공모전데이터/작업데이터/kr_patents.parquet'
DOCS = B + '/분석/검증루프_20261012/C02/C02-T4/cache/docs.parquet'
POS = B + '/분석/검증루프_20261012/C03/C03-T4/cache/C_dev_pos.parquet'
d = duckdb.connect(); d.execute("set memory_limit='3GB'; set threads=2")
d.execute(f"""create table x as select distinct dc.cd, kp.출원번호 a, kp.공개일자 p, cast(kp.세세분류코드 as varchar) k
  from read_parquet('{KRP}') kp join (select distinct cd, appl_no from read_parquet('{DOCS}') where cd like '1%') dc on kp.출원번호=dc.appl_no
  where length(kp.공개일자)=8 and kp.공개일자 < '20250101'""")
print('max p', d.sql('select max(p) from x').fetchone())
tot = []
for t0 in range(2012, 2024):
    q = f"""with b as (select * from x where p between '{t0-2}0101' and '{t0}1231'),
                 w as (select * from x where p between '{t0+1}0101' and '{t0+1}0630'),
                 nb as (select cd, count(distinct a) n from b group by 1), nw as (select cd, count(distinct a) n from w group by 1),
                 f as (select nb.cd from nb join nw using(cd) where nb.n>=3 and nw.n>=3),
                 sb as (select b.cd, k, count(distinct a)*1.0/any_value(nb.n) s from b join nb using(cd) where cd in (select cd from f) group by 1,2),
                 sw as (select w.cd, k, count(distinct a)*1.0/any_value(nw.n) s from w join nw using(cd) where cd in (select cd from f) group by 1,2)
            select {t0} t0, sb.cd, sb.k from sb left join sw using(cd,k) where coalesce(sw.s,0)-sb.s >= 0.2-1e-12"""
    tot.append(d.sql(q).df())
S = pd.concat(tot)
P = pd.read_parquet(POS)
a = set(zip(S.t0, S.cd, S.k)); b = set(zip(P.t0, P.cd, P.k.astype(str)))
print('sql', len(a), 'orig', len(b), 'both', len(a & b), 'only_sql', len(a - b), 'only_orig', len(b - a))
print(S.groupby('t0').size().to_dict())
