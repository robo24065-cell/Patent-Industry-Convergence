import duckdb
con=duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2")
M="../_shared/meta/*.parquet"

print(con.sql(f"select * from read_parquet('{M}') limit 3").df().T)
# year coverage per country (rows, <=2018 only)
print(con.sql(f"select cc, min(appl_year) mn, count(*) filter (where appl_year<2000) pre2000, count(*) filter(where appl_year between 2000 and 2018) y00_18 from read_parquet('{M}') group by cc").df())
print(con.sql(f"select appl_year, cc, count(distinct appl_no) n from read_parquet('{M}') where appl_year between 1995 and 2018 group by 1,2 order by 1,2").df().pivot(index='appl_year',columns='cc',values='n'))
print(con.sql(f"select cc, count(*) filter(where main_ipc is null or main_ipc='') nul, count(*) n from read_parquet('{M}') where appl_year<=2018 group by cc").df())
print(con.sql(f"select cc, main_ipc from read_parquet('{M}') using sample 20").df())
