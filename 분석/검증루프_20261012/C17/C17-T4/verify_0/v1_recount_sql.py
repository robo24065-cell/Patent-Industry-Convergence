# C17-T4 verify v1: P0(a)·d2·기저율을 duckdb SQL만으로 독립 재계산(s2·s3b 코드 미사용). LOCKED 금지.
import duckdb, json
W = r'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/검증루프_20261012/C17/C17-T4'
EV = W + '/cache/ev_dev_20250303.parquet'
DOCS = W + '/cache/us_docs.parquet'
c = duckdb.connect(); c.execute("set threads=2; set memory_limit='3GB'")
c.execute(f"""create temp table ev as select distinct pat, appl8, filing, G, e, code,
  strptime(cast(G as varchar), '%Y%m%d')::date gd, strptime(cast(e as varchar), '%Y%m%d')::date ed,
  strptime(cast(filing as varchar), '%Y%m%d')::date fd
  from '{EV}' where regexp_matches(pat, '^[0-9]+$')""")
# 단계 2·3 단위: D = G + 4s년, t0 = D - 12개월, 직전 단계 납부가 t0 이전
pay = {1: "('M1551','M2551','M3551','M183','M283','M170','M173','M273','F170','F173','F273')",
       2: "('M1552','M2552','M3552','M184','M284','M171','M174','M274','F171','F174','F274')"}
out = {}
for s in (2, 3):
    c.execute(f"""create or replace temp table u{s} as
      with p as (select pat, any_value(gd) gd, any_value(fd) fd from ev group by pat),
      u as (select pat, gd, fd, (gd + interval '{4*s} years')::date D from p),
      u2 as (select *, (D - interval '12 months')::date t0 from u where fd + interval '20 years' > D)
      select u2.* from u2 where exists (select 1 from ev where ev.pat=u2.pat and ev.code in {pay[s-1]} and ev.ed < u2.t0)""")
    c.execute(f"""create or replace temp table y{s} as select u.pat, u.D,
       exists (select 1 from ev where ev.pat=u.pat and ev.code in ('EXP.','EXPX') and ev.ed between u.D - interval '7 days' and u.D + interval '45 days') y
       from u{s} u where u.D < date '2025-01-01'""")
    out[f'lock_n_s{s}'] = c.execute(f"select count(*) from u{s} where D between date '2025-01-01' and date '2025-01-17'").fetchone()[0]
    out[f'base_s{s}'] = c.execute(f"select avg(y::int) from y{s} where D between date '2023-01-01' and date '2024-11-15'").fetchone()[0]
    out[f'rate2023_s{s}'] = c.execute(f"select avg(y::int), count(*) from y{s} where year(D)=2023").fetchone()
out['expected_pos'] = out['lock_n_s2'] * out['base_s2'] + out['lock_n_s3'] * out['base_s3']
out['d2_eq'] = c.execute(f"""with p as (select appl8, any_value(pat) pat from ev group by appl8)
   select avg((cast(d.rgst_no as bigint) = cast(p.pat as bigint))::int), count(*) from '{DOCS}' d join p using(appl8)
   where d.rgst_no is not null and regexp_matches(d.rgst_no, '^[0-9]{{8}}$')""").fetchone()
json.dump(out, open(W + '/verify_0/v1_out.json', 'w'), indent=1, default=str)
print(out)
