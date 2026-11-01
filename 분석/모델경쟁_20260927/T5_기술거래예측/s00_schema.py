import sqlite3
DB=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
con=sqlite3.connect('file:'+DB+'?mode=ro',uri=True)
for (t,s) in con.execute("select name,sql from sqlite_master where type='table'"):
    n=con.execute(f"select count(*) from [{t}]").fetchone()[0]
    print('==',t,n); print(s)
    for r in con.execute(f"select * from [{t}] limit 2"): print('  ',r)
