# JP·KR 공개연도 구성 진단 (공개연도 ≤2021만; 🔒 2022+ 미접근)
import duckdb
con=duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2")
M="../../_shared/meta/*.parquet"
q=f"""select cc, left(coalesce(least(nullif(pub_date,''),nullif(rgst_date,'')),nullif(pub_date,''),nullif(rgst_date,'')),4)::int py,
  count(distinct appl_no) n_all, count(distinct appl_no) filter (where pub_date<>'') n_pubdate, count(distinct appl_no) filter (where coalesce(pub_date,'')='') n_rgst_only
  from read_parquet('{M}') where appl_year>=2000 and cc in ('JP','KR')
  group by 1,2 having py between 2015 and 2021 order by cc, py"""
print(con.sql(q).df().to_string())
