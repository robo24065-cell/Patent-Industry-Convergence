import sqlite3, pandas as pd, time
DB=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
con=sqlite3.connect(f"file:{DB}?mode=ro",uri=True)
t=time.time()
d=pd.read_sql_query("select rgst_no, inventorCd, inventorName, inventorNatl from inventor where inventorName is not null",con)
print(len(d), d.inventorCd.isna().mean(), d.inventorCd.nunique(), time.time()-t)
d.to_parquet('cache/inventor.parquet',index=False)
