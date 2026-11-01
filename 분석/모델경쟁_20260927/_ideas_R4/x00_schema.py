import sqlite3
R=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
con=sqlite3.connect(f"file:{R}?mode=ro",uri=True)
for (n,s) in con.execute("select name,sql from sqlite_master where type in ('table','index')"):
    print(n, '|', s)
