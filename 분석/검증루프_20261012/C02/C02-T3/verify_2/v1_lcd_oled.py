import sys, json, math, duckdb
sys.stdout.reconfigure(encoding="utf-8")
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
META = (B + r"\모델경쟁_20260927\_shared\meta\*.parquet").replace("\\", "/")
OUT = B + r"\검증루프_20261012\C02\C02-T3\verify_2\v1_out.json"
con = duckdb.connect(); con.sql("SET memory_limit='3GB'; SET threads=2; SET enable_progress_bar=false")
R = {}
# all 262xx codes present in KR
R["ksic_262_list"] = con.sql(f"""select ksic, count(*) from read_parquet('{META}') where cc='KR' and ksic like '262%' group by 1 order by 1""").fetchall()
def g(a, b): return {"cagr%": round(((b/a)**(1/8)-1)*100, 2), "log%(+1)": round((math.log(b+1)-math.log(a+1))/8*100, 2), "a": a, "b": b}
for name, cond in [("26211", "ksic='26211'"), ("26212", "ksic='26212'"), ("26211|26212", "ksic in ('26211','26212')"), ("C2620(262*)", "ksic like '262%'")]:
    for mode in ["rows", "distinct"]:
        sel = "count(*)" if mode == "rows" else "count(distinct appl_no)"
        r = dict(con.sql(f"""select case when appl_year in (2008,2009) then 'a' else 'b' end p, {sel}
            from read_parquet('{META}') where cc='KR' and ({cond}) and appl_year in (2008,2009,2016,2017) group by 1""").fetchall())
        R[f"{name}_{mode}"] = g(r['a']/2, r['b']/2)
# overlap between 26211 and 26212
R["overlap_26211_26212_KR_0817"] = con.sql(f"""select count(*) from (select appl_no from read_parquet('{META}') where cc='KR' and ksic in ('26211','26212') and appl_year in (2008,2009,2016,2017) group by appl_no having count(distinct ksic)>1)""").fetchone()[0]
# yearly series 2005-2022 for path shape
R["yearly"] = con.sql(f"""select appl_year, sum((ksic='26211')::int), sum((ksic='26212')::int), count(distinct case when ksic like '262%' then appl_no end) from read_parquet('{META}') where cc='KR' and ksic like '262%' and appl_year between 2005 and 2022 group by 1 order by 1""").fetchall()
json.dump(R, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for k, v in R.items(): print(k, v)
