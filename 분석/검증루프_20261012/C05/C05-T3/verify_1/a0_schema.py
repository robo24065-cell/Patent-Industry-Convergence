import sqlite3,sys,io
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
p=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\business_rights.sqlite"
c=sqlite3.connect('file:'+p+'?mode=ro',uri=True)
for n,s in c.execute("select name,sql from sqlite_master"): print(n,'|',s)
for (t,) in c.execute("select name from sqlite_master where type='table'").fetchall():
    print(t, c.execute(f'select count(*) from "{t}"').fetchone())
