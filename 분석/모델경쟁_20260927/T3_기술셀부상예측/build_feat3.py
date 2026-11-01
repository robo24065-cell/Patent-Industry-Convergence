import os
CACHE=os.environ.get("T3_CACHE","cache"); MAXYEAR=int(os.environ.get("T3_MAXYEAR","2018"))
os.makedirs(CACHE,exist_ok=True)
# [2단계] 국가 간 우선권 흐름 패널: (ksic, mg, cc, year) × 우선권 국가별 출원수. 출원연도 2000~MAXYEAR만 (검증 단계 2018, 🔒 2019~ 미접근)
#  n_pKR: 해당 관청 출원 중 KR 우선권(=한국발 해외출원, cc<>KR일 때 의미), n_pUS/JP/CN: 각국 우선권, n_pEU: 유럽(EP·DE·FR·GB·IT·NL·CH·SE·AT·BE·DK·FI·ES) 우선권
import duckdb, time
t=time.time()
con=duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false")
src=open('build_panel.py',encoding='utf-8').read()
exec(src[src.index('con.execute(r"""create macro'):src.index('con.execute(rf"""create view b')])
M="../_shared/meta/*.parquet"
con.execute(rf"""create temp table a as select ksic, mg, cc, year, appl_no, max(pKR) pKR, max(pUS) pUS, max(pJP) pJP, max(pCN) pCN, max(pEU) pEU from (
  select ksic, cc, appl_year::int as year, appl_no,
    mgn(regexp_replace(trim(coalesce(nullif(main_ipc,''), split_part(ipc_all,'|',1))), '\s+', ' ', 'g')) mg,
    (coalesce(prio_cc,'') like '%KR%')::int pKR, (coalesce(prio_cc,'') like '%US%')::int pUS, (coalesce(prio_cc,'') like '%JP%')::int pJP,
    (coalesce(prio_cc,'') like '%CN%')::int pCN, regexp_matches(coalesce(prio_cc,''),'EP|DE|FR|GB|IT|NL|CH|SE|AT|BE|DK|FI|ES')::int pEU
  from read_parquet('{M}') where appl_year between 2000 and {MAXYEAR}) where mg is not null group by all""")
print('a',con.sql("select count(*), max(year) from a").fetchall(), round(time.time()-t,1),flush=True)
con.execute(f"""copy (select ksic, mg, cc, year, sum(pKR) n_pKR, sum(pUS) n_pUS, sum(pJP) n_pJP, sum(pCN) n_pCN, sum(pEU) n_pEU, count(*) n
  from a group by all) to '{CACHE}/feat3_panel.parquet' (format parquet)""")
print(con.sql(f"select cc, sum(n), sum(n_pKR), sum(n_pUS), sum(n_pJP), sum(n_pCN), sum(n_pEU) from read_parquet('{CACHE}/feat3_panel.parquet') group by cc").df().to_string())
print('done',round(time.time()-t,1))
