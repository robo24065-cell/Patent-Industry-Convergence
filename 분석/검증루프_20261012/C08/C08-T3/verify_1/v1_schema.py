import sqlite3, sys
sys.stdout.reconfigure(encoding='utf-8')
con=sqlite3.connect(r'file:C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite?mode=ro',uri=True)
for (n,s) in con.execute("select name,sql from sqlite_master where type in ('table','index')"): print(n,'|',s)
print(con.execute("select rgstCsName, count(*) from right group by 1 order by 2 desc limit 25").fetchall())
