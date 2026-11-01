import sqlite3, json, time
DB=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\business_rights.sqlite"
con=sqlite3.connect(f"file:{DB}?mode=ro",uri=True)
for (t,sql) in con.execute("select name,sql from sqlite_master where type='table'"): print(t,sql)
cur=con.execute("select * from right_list limit 2"); print([d[0] for d in cur.description])
for r in cur.fetchall(): print([str(x)[:300] for x in r])
print(con.execute("select status,count(*) from applicant_query group by 1").fetchall())
print(con.execute("select substr(rgst_no,1,2),count(*) from right_list group by 1").fetchall())
