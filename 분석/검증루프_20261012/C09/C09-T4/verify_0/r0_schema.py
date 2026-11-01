# C09-T4 verify_0 (반박 검증 2차): 스키마·기본 분포 확인. sqlite mode=ro.
import sqlite3, sys
sys.stdout.reconfigure(encoding='utf-8')
DB = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite'
c = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
for n, s in c.execute("select name, sql from sqlite_master where type in ('table','index')"):
    print(n, '|', s)
print(c.execute("select seq, count(*) from pay group by seq order by seq limit 8").fetchall())
print(c.execute("select statAnnl, count(*) from pay where seq>0 group by statAnnl order by count(*) desc limit 25").fetchall())
print(c.execute("select substr(rgst_no,1,2), count(*) from pay where seq>0 group by 1").fetchall())
print(c.execute("select * from pay where seq>0 limit 3").fetchall())
