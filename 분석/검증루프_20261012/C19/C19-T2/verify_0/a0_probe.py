import duckdb, sys
sys.stdout.reconfigure(encoding='utf-8')
T4=r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\T4_특허포기예측\cache'
con=duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2")
for f in ['s2_fam_tokens.parquet','patent.parquet']:
    p=(T4+'\\'+f).replace('\\','/')
    print(f, con.execute(f"describe select * from read_parquet('{p}')").fetchall())
    print(con.execute(f"select count(*) from read_parquet('{p}')").fetchone())
p=(T4+'\\s2_fam_tokens.parquet').replace('\\','/')
print(con.execute(f"select * from read_parquet('{p}') limit 5").fetchall())
