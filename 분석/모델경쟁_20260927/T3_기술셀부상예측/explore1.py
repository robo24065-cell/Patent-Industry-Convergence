import duckdb
con=duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2")
M="../_shared/meta/*.parquet"
df=con.sql(f"select cc, applicant, ipc_all, prio_cc, family from read_parquet('{M}') where appl_year between 2010 and 2012 using sample 40").df()
print(df.to_string())
print(con.sql(f"select cc, avg((applicant like '%|%')::int) multi_app, avg((prio_cc is not null)::int) has_prio, avg((family is not null)::int) fam from read_parquet('{M}') where appl_year<=2018 group by cc").df())
print(con.sql(f"select cc, prio_cc, count(*) n from read_parquet('{M}') where appl_year between 2005 and 2018 and cc='KR' group by 1,2 order by n desc limit 15").df())
