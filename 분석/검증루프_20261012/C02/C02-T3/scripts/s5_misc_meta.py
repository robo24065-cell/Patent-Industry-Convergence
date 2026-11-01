# 2.5 공개 지연 46%, 3.1.3/4.1 C2620 합산 추세, CN 공개연도 구성(E-C01-04), README 2025 공개 월별(X4)
import sys, json, duckdb
sys.stdout.reconfigure(encoding="utf-8")
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
MC = B + r"\모델경쟁_20260927"; OUT = B + r"\검증루프_20261012\C02\C02-T3"
META = (MC + r"\_shared\meta\*.parquet").replace("\\", "/")
con = duckdb.connect(); con.sql("SET memory_limit='3GB'; SET threads=2; SET enable_progress_bar=false")
R = {}
# (1) 공개 지연: KR 출원 window [t0-2, t0], 원점 연말까지 공개·등록 안 된 비율 (발명 10, 전체)
for t0 in [2019, 2020]:
    for inv in ["all", "inv10"]:
        w = "" if inv == "all" else "and substr(appl_no,1,2)='10'"
        r = con.sql(f"""select avg(case when least(coalesce(nullif(pub_date,''),'99999999'), coalesce(nullif(rgst_date,''),'99999999')) > '{t0}1231' then 1 else 0 end), count(*)
            from read_parquet('{META}') where cc='KR' and appl_year between {t0-2} and {t0} {w}""").fetchone()
        R[f"kr_unpub_share_t0{t0}_{inv}"] = [round(r[0], 4), r[1]]
# (2) KR 출원 26211/26212/C2620 전체: 2008–09 vs 2016–17 연평균 성장
c2620 = ["26211", "26212", "26223", "26224", "26291"]
rows = con.sql(f"""select ksic, appl_year, count(*) n, sum((substr(appl_no,1,2)='10')::int) n10 from read_parquet('{META}')
   where cc='KR' and ksic in ({','.join("'"+k+"'" for k in c2620)}) and appl_year in (2008,2009,2016,2017) group by all""").df()
def cagr(df, col):
    a = df[df.appl_year.isin([2008, 2009])][col].sum() / 2; b = df[df.appl_year.isin([2016, 2017])][col].sum() / 2
    return round(((b / a) ** (1 / 8) - 1) * 100, 2), int(a), int(b)
for col in ["n", "n10"]:
    R[f"cagr_{col}"] = {k: cagr(rows[rows.ksic == k], col) for k in c2620}
    R[f"cagr_{col}"]["26211+26212"] = cagr(rows[rows.ksic.isin(["26211", "26212"])], col)
    R[f"cagr_{col}"]["C2620_all"] = cagr(rows, col)
# (3) CN 공개연도별 문서종류
R["cn_pub_by_type"] = con.sql(f"""select try_cast(substr(pub_date,1,4) as int) py,
   sum((length(appl_no) in (12,14) and substr(appl_no,5,1) in ('1','8'))::int) inv,
   sum((length(appl_no) in (12,14) and substr(appl_no,5,1) in ('2','9'))::int) um, count(*) n
   from read_parquet('{META}') where cc='CN' and pub_date is not null and pub_date<>'' and substr(pub_date,1,4) between '2008' and '2016' group by 1 order by 1""").fetchall()
R["cn_inv_appl_pubmissing"] = con.sql(f"""select appl_year, avg((pub_date is null or pub_date='')::int) from read_parquet('{META}')
   where cc='CN' and length(appl_no) in (12,14) and substr(appl_no,5,1) in ('1','8') and appl_year between 2008 and 2016 group by 1 order by 1""").fetchall()
# (4) 공개 월별 2024-07 ~ 2025-12 (cc별)
R["pub_month"] = con.sql(f"""select cc, substr(pub_date,1,6) ym, count(*) from read_parquet('{META}')
   where pub_date >= '20240601' group by all order by cc, ym""").fetchall()
json.dump(R, open(OUT + r"\s5_misc_meta_out.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
for k, v in R.items():
    if k != "pub_month": print(k, v)
import collections
pm = collections.defaultdict(dict)
for cc, ym, n in R["pub_month"]: pm[cc][ym] = n
for cc in pm: print(cc, pm[cc])
