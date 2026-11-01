import sqlite3, time
DB=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
con=sqlite3.connect(f"file:{DB}?mode=ro",uri=True)
t=time.time()
def q(s):
    print('>>',s); 
    for r in con.execute(s).fetchall(): print('   ',r)
q("select rgstCsName, count(*) from right group by 1 order by 2 desc limit 30")
q("select rgstCsReason, count(*) from right where rgstCsName like '소멸%' group by 1 order by 2 desc limit 30")
q("select lastDspst, count(*) from patent group by 1 order by 2 desc limit 20")
q("select applTpcd, count(*) from patent group by 1 order by 2 desc limit 20")
q("select ownerRgstCsReason, count(*) from owner group by 1 order by 2 desc limit 20")
q("select ownerRgstCsName, count(*) from owner group by 1 order by 2 desc limit 20")
q("select substr(rgstDate,1,4), count(*) from patent group by 1 order by 1")
q("select count(distinct ksic) from patent")
q("select statAnnl=lastAnnl, count(*) from pay group by 1")
q("select count(*), min(payDate), max(payDate) from pay")
q("select max(rgstCsDate) from right")
print(time.time()-t)
