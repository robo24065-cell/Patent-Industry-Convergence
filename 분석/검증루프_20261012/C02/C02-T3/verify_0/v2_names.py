import sqlite3, time, io
db=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
c=sqlite3.connect("file:"+db.replace("\\","/")+"?mode=ro", uri=True)
q=lambda s: c.execute(s).fetchall()
out=io.open(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C02\C02-T3\verify_0\v2_names.txt","w",encoding="utf-8")
p=lambda *a: print(*a, file=out)
p("== owner ownerRgstCsName (rows, patents)")
for r in q("select ownerRgstCsName, count(*), count(distinct rgst_no) from owner where ownerRgstCsName is not null group by 1 order by 2 desc limit 60"): p(r)
p("== right rgstCsName (rows, patents) all")
for r in q("select rgstCsName, count(*), count(distinct rgst_no) from right group by 1 order by 2 desc"): p(r)
p("== sample ownerRgstCsReason for top names")
for r in q("select ownerRgstCsName, ownerRgstCsReason, count(*) from owner where ownerRgstCsName is not null group by 1,2 order by 3 desc limit 25"): p(r)
p("== sample multi-row no-reason patent")
ids=[x[0] for x in q("select rgst_no from owner group by rgst_no having count(*)>1 and sum(case when ownerRgstCsName is not null and trim(ownerRgstCsName)<>'' then 1 else 0 end)=0 limit 3")]
for i in ids:
    for r in q(f"select rgst_no, seq, finalOwnerYn, ownerName, ownerRgstCsDate, ownerRgstCsName from owner where rgst_no='{i}' order by seq"): p(r)
p("== sample with reason")
ids=[x[0] for x in q("select rgst_no from owner where ownerRgstCsName is not null group by rgst_no limit 3")]
for i in ids:
    for r in q(f"select rgst_no, seq, finalOwnerYn, ownerName, ownerRgstCsDate, ownerRgstCsName from owner where rgst_no='{i}' order by seq"): p(r)
out.close()
