import sqlite3,sys
sys.stdout.reconfigure(encoding='utf-8')
DB=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
con=sqlite3.connect('file:'+DB+'?mode=ro',uri=True)
q=lambda s: con.execute(s).fetchall()
print('right names'); [print(r) for r in q("select rgstCsName,count(*),count(distinct rgst_no) from right group by 1 order by 2 desc")]
print('right reasons for 권리이전 names'); [print(r) for r in q("select rgstCsName,rgstCsReason,count(*) from right where rgstCsName like '%이전%' group by 1,2 order by 3 desc limit 40")]
print('owner reasons'); [print(r) for r in q("select ownerRgstCsName,ownerRgstCsReason,count(*) from owner group by 1,2 order by 3 desc limit 40")]
print('applTpcd'); [print(r) for r in q("select applTpcd,count(*) from patent group by 1")]
print('lastDspst'); [print(r) for r in q("select lastDspst,count(*) from patent group by 1")]
print('rgst year'); [print(r) for r in q("select substr(rgstDate,1,4),count(*) from patent group by 1")]
print('right purpose'); [print(r) for r in q("select rgstCsPupos,count(*) from right group by 1 order by 2 desc limit 20")]
