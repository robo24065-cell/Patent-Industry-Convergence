# verify_0 V1: P0·P1 독립 재집계(duckdb SQL, 작업자 캐시·lib11 미사용). 원점 2020·2024와 잠금 2025 위험집합(레이블 없음).
import sys, json, time, duckdb
sys.stdout.reconfigure(encoding='utf-8')
META = 'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/모델경쟁_20260927/_shared/meta/*.parquet'
OUT = 'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/검증루프_20261012/C11/C11-T4/verify_0'
t = time.time()
con = duckdb.connect(); con.execute('set threads=2'); con.execute("set memory_limit='3GB'"); con.execute('set enable_progress_bar=false')
suf_cjk = "股份有限公司|有限责任公司|有限公司|股份公司|株式会社|有限会社"
suf_lat = r"\b(CO|COMPANY|LTD|LIMITED|INC|INCORPORATED|CORP|CORPORATION|LLC|L\.L\.C|GMBH|AG|SA|SAS|SARL|BV|B\.V|NV|N\.V|AB|OY|OYJ|SPA|S\.P\.A|PLC|KK|KABUSHIKI|KAISHA|KG|SE|AS|A/S)\b"
keep = "[^0-9A-Z\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7a3]"
con.execute(f"""create table d as
 select cc, appl_no, cast(substr(cast(pub_date as varchar),1,8) as int) pub, ksic, trim(nm) nm
 from (select cc, appl_no, pub_date, ksic, unnest(regexp_split_to_array(coalesce(applicant,''), '[|;]')) nm
       from read_parquet('{META}') where cc in ('CN','EP') and pub_date is not null and pub_date < '20250101')""")
con.execute(f"""create table da as select cc, appl_no, pub, ksic,
  regexp_replace(regexp_replace(regexp_replace(upper(nm), '{suf_cjk}', ' ', 'g'), '{suf_lat}', ' ', 'g'), '{keep}', '', 'g') fkey
  from d where nm <> '' and lower(nm) not in ('nan','none','-','null')""")
con.execute("delete from da where length(fkey) < 2")
con.execute("create table dk as select distinct cc, appl_no, pub, ksic, fkey as k from da")
print('rows', con.execute("select cc, count(*) from dk group by 1").fetchall(), round(time.time() - t, 1), flush=True)
res = {}
for Y in [2020, 2024, 2025]:
    t0 = Y * 10000 + 101
    lab = 1 if Y < 2025 else 0
    q = f"""
    with act as (
      select cc, k from dk where pub >= {t0 - 30000} and pub < {t0}
      group by 1,2 having count(distinct appl_no) >= 10 and count(distinct case when pub >= {t0 - 10000} then appl_no end) >= 1),
    held as (select distinct cc, k, ksic from dk where pub >= {t0 - 50000} and pub < {t0}),
    ks as (select distinct ksic from dk),
    allp as (select a.cc, a.k, ks.ksic from act a cross join ks),
    risk as (select p.* from allp p anti join held h on h.cc=p.cc and h.k=p.k and h.ksic=p.ksic),
    ev as (select distinct cc, k, ksic from dk where pub >= {t0} and pub <= {Y * 10000 + 630} and {lab} = 1)
    select r.cc, count(*) units, count(distinct r.k) firms,
           sum(case when e.k is not null then 1 else 0 end) pos,
           count(distinct case when e.k is not null then r.k end) pos_firms
    from risk r left join ev e on e.cc=r.cc and e.k=r.k and e.ksic=r.ksic group by 1 order by 1"""
    rows = con.execute(q).fetchall()
    res[Y] = {r[0]: dict(units=r[1], firms=r[2], pos=(r[3] if lab else None), pos_firms=(r[4] if lab else None)) for r in rows}
    print(Y, res[Y], round(time.time() - t, 1), flush=True)
json.dump(res, open(OUT + '/v1_out.json', 'w'), indent=1)
