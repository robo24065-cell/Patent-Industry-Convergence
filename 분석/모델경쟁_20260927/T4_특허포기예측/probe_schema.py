import sqlite3
DB=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
con=sqlite3.connect(f"file:{DB}?mode=ro",uri=True)
for (t,sql) in con.execute("select name,sql from sqlite_master where type in ('table','index')"):
    print(t, sql)
for t in ['patent','pay','right','owner','applicant','cofpr','inventor','elsLcnsList','noelsLcnsList']:
    try:
        cur=con.execute(f"select * from [{t}] limit 3")
        print('==',t,[d[0] for d in cur.description])
        for r in cur.fetchall(): print('  ',r)
    except Exception as e: print(t,e)
