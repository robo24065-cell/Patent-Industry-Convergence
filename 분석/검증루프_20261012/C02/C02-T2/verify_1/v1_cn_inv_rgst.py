# verify_1 v1: CN 발명 등록일 기준 건수를 meta에서 독립 재계산 (읽기 전용)
import duckdb, sys, os, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 400)
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
META = os.path.join(ROOT, r"분석\모델경쟁_20260927\_shared\meta\*.parquet").replace(os.sep, '/')
OUT = os.path.join(ROOT, r"분석\검증루프_20261012\C02\C02-T2\verify_1")
con = duckdb.connect(); con.sql("set threads=2"); con.sql("set memory_limit='3GB'")
print(con.sql(f"select length(appl_no) L, substr(appl_no,5,1) k, count(*) n from read_parquet('{META}') where cc='CN' group by all order by all").df().to_string())
con.sql(f"""create temp table a as select appl_no,
  min(nullif(rgst_date,'')) rd, count(distinct nullif(rgst_date,'')) n_rd,
  min(nullif(rgst_no,'')) rn, min(nullif(pub_date,'')) pd, min(appl_year) ay,
  bool_and(coalesce(nullif(main_ipc,''), nullif(split_part(ipc_all,'|',1),'')) is null) miss_all
 from read_parquet('{META}') where cc='CN' and length(appl_no) in (12,14) and substr(appl_no,5,1) in ('1','8') group by appl_no""")
print(con.sql("select count(*) n, sum((n_rd>1)::int) multi_rd, sum((rd is null and rn is not null)::int) rn_no_rd, sum((rd is not null and rn is null)::int) rd_no_rn, sum((pd is null)::int) no_pd, sum((rd is not null and pd is null)::int) rd_no_pd from a").df().to_string())
ry = con.sql("select left(rd,4)::int y, count(*) n_inv_rgst, sum((pd is null)::int) n_no_pd, sum(miss_all::int) n_ipcmiss from a where rd is not null group by 1 order by 1").df()
print(ry.to_string()); ry.to_csv(os.path.join(OUT, "v1_cn_inv_rgst_year.csv"), index=False)
m = con.sql("select left(rd,6) ym, count(*) n from a where rd>='2019' and rd<'2026' group by 1 order by 1").df()
m.to_csv(os.path.join(OUT, "v1_cn_inv_rgst_month.csv"), index=False); print(m.to_string())
c = con.sql("""select left(pd,4)::int py, count(*) n, avg((rd is not null)::int) sh_rgst,
   avg(case when rd is not null then (left(rd,4)::int - left(pd,4)::int) end) mean_lag,
   sum((left(rd,4)='2020')::int) g2020, sum((left(rd,4)='2021')::int) g2021, sum((left(rd,4)='2022')::int) g2022,
   sum((left(rd,4)='2023')::int) g2023, sum((left(rd,4)='2024')::int) g2024
   from a where pd is not null and pd>='2012' and pd<'2026' group by 1 order by 1""").df()
c.to_csv(os.path.join(OUT, "v1_cn_inv_pubcohort.csv"), index=False); print(c.round(4).to_string())
c2 = con.sql("select ay, count(*) n, avg((rd is not null)::int) sh_rgst from a where ay between 2010 and 2024 group by 1 order by 1").df()
print(c2.round(4).to_string())
