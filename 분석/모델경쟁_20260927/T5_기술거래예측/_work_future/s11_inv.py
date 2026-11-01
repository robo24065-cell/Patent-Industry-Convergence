import sqlite3,sys,pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
DB=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
con=sqlite3.connect('file:'+DB+'?mode=ro',uri=True)
print(con.execute("select count(*),count(inventorCd),count(inventorName),count(distinct rgst_no) from inventor").fetchall())
df=pd.read_sql("select rgst_no,count(inventorName) n_inv,sum(coalesce(inventorNatl,'')<>'대한민국' and inventorName is not null) n_inv_for,group_concat(inventorName,'|') inv_names from inventor group by rgst_no",con)
df.to_parquet('cache/inv.parquet',index=False); print(df.shape, df.n_inv.describe())
