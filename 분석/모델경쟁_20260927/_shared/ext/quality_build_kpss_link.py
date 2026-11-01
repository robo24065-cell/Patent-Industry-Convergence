# 제공 데이터 family 필드의 US 특허번호(7~8자리) -> KPSS(경제적 가치 xi, 피인용) 연결
import duckdb
con=duckdb.connect(); con.sql("SET memory_limit='3GB'; SET threads=2; SET preserve_insertion_order=false")
con.sql("""create table fam as
 select distinct cc, appl_no, appl_year, ksic, main_ipc,
   try_cast(substr(m,3) as bigint) us_pn
 from (select cc, appl_no, appl_year, ksic, main_ipc, unnest(string_split(family,'|')) m
       from read_parquet('../meta/*.parquet') where family is not null) where regexp_full_match(m,'US[0-9]{7,8}')""")
print(con.sql("select count(*), count(distinct us_pn) from fam").fetchall())
con.sql("""create table link as select f.cc, f.appl_no, f.appl_year, f.ksic, f.main_ipc, f.us_pn, k.permno, k.issue_date, k.xi_real, k.cites
 from fam f join 'quality_kpss_2025.parquet' k on f.us_pn=k.patent_num""")
con.sql("copy link to 'quality_kpss_family_link.parquet' (format parquet)")
# 연계율: (cc, appl_year) 별 출원 중 KPSS US 패밀리 보유 비율
r=con.sql("""with base as (select cc, appl_year, count(distinct appl_no) n from read_parquet('../meta/*.parquet') group by 1,2),
 l as (select cc, appl_year, count(distinct appl_no) nl from link group by 1,2)
 select base.cc, base.appl_year, n, coalesce(nl,0) nl, coalesce(nl,0)/n rate from base left join l using(cc,appl_year) order by 1,2""").df()
r.to_csv('quality_kpss_link_rate_cc_year.csv',index=False)
print(r.pivot(index='appl_year',columns='cc',values='rate').round(3).to_string())
k=con.sql("""select ksic, count(distinct cc||appl_no) nl from link where appl_year<=2016 group by 1""").df()
print('KSIC linked(>=1):',len(k),' >=100:',(k.nl>=100).sum(),' min',k.nl.min(),' median',k.nl.median())
k.to_csv('quality_kpss_link_by_ksic.csv',index=False)
