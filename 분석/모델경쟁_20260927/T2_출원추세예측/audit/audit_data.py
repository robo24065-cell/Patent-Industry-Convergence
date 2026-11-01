# 2단계 감사: 데이터 점검 (출원연도 <=2018 만 조회; 홀드아웃 목표연도 결과변수 미조회)
import duckdb, os
T = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(T, '..', '_shared', 'meta', '*.parquet').replace(os.sep, '/')
con = duckdb.connect(); con.sql("SET threads=2; SET memory_limit='3GB'")
print(con.sql(f"""select count(*) n, count(distinct cc||'|'||appl_no) nu, count(distinct cc||'|'||appl_no||'|'||ksic) nuk,
 sum((try_cast(substr(appl_date,1,4) as int) <> appl_year)::int) ymis, sum((appl_year is null)::int) ynull
 from read_parquet('{P}') where appl_year between 2000 and 2018"""))
# 국가별 공개지연(개월) 분포: 출원 2014-2018 만 (공개일-출원일)
print(con.sql(f"""select cc, count(*) n, sum((pub_date is null or pub_date='')::int) nopub,
  quantile_cont(date_diff('month', strptime(appl_date,'%Y%m%d'), try_strptime(pub_date,'%Y%m%d')), [0.5,0.9,0.99]) q
 from read_parquet('{P}') where appl_year between 2014 and 2018 group by 1 order by 1"""))
# 국가별 최대 공개일 (출원연도 필터 없음이나 건수는 조회하지 않음)
print(con.sql(f"select cc, max(pub_date) maxpub, max(appl_date) maxappl from read_parquet('{P}') group by 1 order by 1"))
