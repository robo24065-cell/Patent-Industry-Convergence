# verify_0 V1: P0(a)·P1 독립 재집계(duckdb SQL, 작업자 캐시·lib12 미사용). 원점 2020·2024와 잠금 2025 위험집합(레이블 없음).
import sys, json, time, duckdb
sys.stdout.reconfigure(encoding='utf-8')
META = 'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/모델경쟁_20260927/_shared/meta/*.parquet'
OUT = 'C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전/분석/검증루프_20261012/C12/C12-T4/verify_0'
t = time.time()
con = duckdb.connect(); con.execute('set threads=2'); con.execute("set memory_limit='3GB'"); con.execute('set enable_progress_bar=false')
suf = "股份有限公司|有限责任公司|有限公司|股份公司|株式会社|有限会社|주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|\\(주\\)|㈜|（주）|\\(유\\)|\\(株\\)"
suf_lat = r"\b(CO|COMPANY|LTD|LIMITED|INC|INCORPORATED|CORP|CORPORATION|LLC|L\.L\.C|GMBH|AG|SA|SAS|SARL|BV|B\.V|NV|N\.V|AB|OY|OYJ|SPA|S\.P\.A|PLC|KK|KABUSHIKI|KAISHA|KG|SE|AS|A/S)\b\.?"
keep = "[^0-9A-Z\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7a3]"
con.execute(f"""create table doc as select cc, appl_no, cast(substr(cast(pub_date as varchar),1,8) as int) pub,
   nullif(regexp_extract(upper(trim(coalesce(main_ipc,''))), '^([A-H][0-9][0-9][A-Z])', 1), '') sc, applicant
   from read_parquet('{META}') where cc in ('CN','EP') and pub_date is not null and pub_date < '20250101'""")
con.execute(f"""create table da as select cc, appl_no, pub, sc,
  regexp_replace(regexp_replace(regexp_replace(upper(trim(nm)), '{suf}', ' ', 'g'), '{suf_lat}', ' ', 'g'), '{keep}', '', 'g') k
  from (select cc, appl_no, pub, sc, unnest(regexp_split_to_array(coalesce(applicant,''), '[|;]')) nm from doc)
  where trim(nm) <> '' and lower(trim(nm)) not in ('nan','none','-','null')""")
con.execute("delete from da where length(k) < 2")
con.execute("create table dk as select distinct cc, appl_no, pub, sc, k from da")
print('rows', con.execute("select cc, count(*) from dk group by 1 order by 1").fetchall(), round(time.time() - t, 1), flush=True)
res = {}
for Y in [2020, 2024, 2025]:
    t0 = Y * 10000 + 101
    lab = 1 if Y < 2025 else 0
    q = f"""
    with act as (
      select cc, k from dk where pub >= {t0 - 30000} and pub < {t0}
      group by 1,2 having count(*) >= 10 and count(case when pub >= {t0 - 10000} then 1 end) >= 1),
    cand as (select cc, sc from doc where sc is not null and pub >= {t0 - 50000} and pub < {t0} group by 1,2 having count(*) >= 30),
    held as (select distinct cc, k, sc from dk where sc is not null and pub >= {t0 - 100000} and pub < {t0}),
    allp as (select a.cc, a.k, c.sc from act a join cand c on a.cc = c.cc),
    risk as (select p.* from allp p anti join held h on h.cc=p.cc and h.k=p.k and h.sc=p.sc),
    ev as (select distinct cc, k, sc from dk where sc is not null and pub >= {t0} and pub <= {Y * 10000 + 630} and {lab} = 1)
    select r.cc, count(*) units, count(distinct r.k) firms,
           sum(case when e.k is not null then 1 else 0 end) pos,
           count(distinct case when e.k is not null then r.k end) pos_firms
    from risk r left join ev e on e.cc=r.cc and e.k=r.k and e.sc=r.sc group by 1 order by 1"""
    rows = con.execute(q).fetchall()
    res[Y] = {r[0]: dict(units=r[1], firms=r[2], pos=(r[3] if lab else None), pos_firms=(r[4] if lab else None)) for r in rows}
    print(Y, res[Y], round(time.time() - t, 1), flush=True)
json.dump(res, open(OUT + '/v1_out.json', 'w'), indent=1)
