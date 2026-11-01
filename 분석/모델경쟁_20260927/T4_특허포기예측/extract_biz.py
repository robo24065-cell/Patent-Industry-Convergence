import sqlite3, pandas as pd, time
DB=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\business_rights.sqlite"
con=sqlite3.connect(f"file:{DB}?mode=ro",uri=True)
t=time.time()
q=pd.read_sql_query("select applicant_cd, applicant_name, n_in_fusion, total_count, fetched, truncated, status from applicant_query",con)
q.to_parquet('cache/biz_query.parquet',index=False)
r=pd.read_sql_query("select applicant_cd, substr(rgst_no,1,2) t, substr(rgst_date,1,4) y, substr(appl_date,1,4) ay, count(*) n from right_list group by 1,2,3,4",con)
r.to_parquet('cache/biz_counts.parquet',index=False)
print(len(q),len(r),time.time()-t)
print(q.truncated.value_counts(), (q.total_count>q.fetched).sum())
print(q.groupby('status').n_in_fusion.describe())
