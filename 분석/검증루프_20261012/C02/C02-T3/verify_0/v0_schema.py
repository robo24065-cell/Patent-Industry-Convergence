import sqlite3, sys, time
t=time.time()
db=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
c=sqlite3.connect("file:"+db.replace("\\","/")+"?mode=ro", uri=True)
q=lambda s: c.execute(s).fetchall()
print("tables", q("select name from sqlite_master where type='table'"))
for tb in ["owner","right","patent"]:
    print(tb, q(f"pragma table_info('{tb}')"))
print("owner rows", q("select count(*), count(distinct rgst_no) from owner"))
print("patent rows", q("select count(*), count(distinct rgst_no) from patent"))
print("ownerRgstCsName top", q("select ownerRgstCsName, count(*) from owner group by 1 order by 2 desc limit 40"))
print("finalOwnerYn", q("select finalOwnerYn, count(*) from owner group by 1"))
print("time", time.time()-t)
